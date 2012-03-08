#include "vtkCudaSegmentor.h"

//VTK compatibility
#include "vtkObjectFactory.h"
#include "vtkCellArray.h"
#include "vtkPoints.h"
#include "vtkPointData.h"

//Cuda kernels for segmentation
extern "C" {
	#include "CUDA_segmentAlgo.h"
}

//miscellaneous
#include "vtkTimerLog.h"
#include <deque>
#include <vector>

vtkStandardNewMacro(vtkCudaSegmentor);

vtkCudaSegmentor::vtkCudaSegmentor(){
	this->padder = vtkImageMirrorPad::New();
	this->numberOfClassifications = 1;
	this->numSmoothingIterations = 40;
	this->constraintThreshold = 1.0;
	this->constraintType = 1;
	this->smoothingCoefficient = 0.25;
	this->meshCase = 64;
	this->input = 0;
	this->funct = 0;
	this->output = new std::vector<vtkPolyData*>;
}

vtkCudaSegmentor::~vtkCudaSegmentor(){
	this->padder->Delete();
	
	//delete the output information
	for(std::vector<vtkPolyData*>::iterator it = this->output->begin(); it != this->output->end(); it++){
		(*it)->Delete();
	}
	delete this->output;
}

void vtkCudaSegmentor::SetConstraintThreshold(double constraintThreshold){
	
	//return if the threshold is invalid
	if(constraintThreshold < 0.0) return;

	this->constraintThreshold = constraintThreshold;

}

void vtkCudaSegmentor::SetSmoothingCoefficient(double smoothingCoefficient){

	//return if the coefficient is invalid
	if(smoothingCoefficient < 0.0 || smoothingCoefficient > 1.0) return;

	this->smoothingCoefficient = smoothingCoefficient;

}

void vtkCudaSegmentor::SetConstraintType(int constraintType){
	if(constraintType == 0 || constraintType == 1 || constraintType == 2){
		this->constraintType = constraintType;
	}
}

void vtkCudaSegmentor::SetMeshingType(int meshType){
	if(meshType == 64 || meshType == 12){
		this->meshCase = meshType;
	}
}

void vtkCudaSegmentor::SetInput(vtkImageData* in){
	this->input = in;
}

void vtkCudaSegmentor::SetFunction(vtkCuda2DTransferClassificationFunction* f){
	this->funct = f;
}

vtkPolyData* vtkCudaSegmentor::GetOutput(int n){
	if( n > -1 && n < this->output->size() ){
		return this->output->at(n);
	}
	return 0;
}


//pre:	- the data is stored as shorts
//post:	- the polydata has been reduced to prevent any redundancy of surface information (necessary due to high information density)
void vtkCudaSegmentor::Update()
{

	//delete the previous output information
	for(std::vector<vtkPolyData*>::iterator it = this->output->begin(); it != this->output->end(); it++){
		(*it)->Delete();
	}

	//grab the dimensions of the input data (needed to determine buffer size, and whether padding is necessary
    int dims[3];
	this->input->GetDimensions(dims);

	//copy the input data into the mask for input into CUDA (pad if x-length is not a multiple of 4)
	vtkImageData* maskData = 0;
	if( dims[0] & 3 ){
		this->padder->SetInput(this->input);
		dims[0] += 4 - (dims[0] & 3);
		this->padder->SetOutputWholeExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1);
		this->padder->Modified();
		this->padder->Update();
		maskData = padder->GetOutput();
		maskData->Register(NULL);
		maskData->SetSource(NULL);
	}else{
		maskData = vtkImageData::New();
		maskData->DeepCopy(this->input);
		maskData->Register(NULL);
		maskData->SetSource(NULL);
	}

	//grab the spacing to allow for gradient calculation to happen correctly
	float spaces[3] = {maskData->GetSpacing()[0], maskData->GetSpacing()[1], maskData->GetSpacing()[2]};
	float origin[3] = {maskData->GetOrigin()[0], maskData->GetOrigin()[1], maskData->GetOrigin()[2]};

	//create a table to store the classification function
	short* functionTable = new short[256*256];
	memset((void*) functionTable,0,256*256*sizeof(short));
	double range[2];
	maskData->GetPointData()->GetScalars()->GetRange(range);
	float lowI = range[0];
	float highI = range[1];
	float lowG = 0.0;
	float highG = 52.0;
	this->funct->GetClassifyTable(functionTable, 256, 256, lowI, highI, lowG, highG);
	this->numberOfClassifications = this->funct->GetNumberOfClassifications();

	//create a buffer to store the outputted mask in (as the current mask is filled with the padded input data)
	short* buffer = new short[dims[0]*dims[1]*dims[2]];

	//run the voxel classification algorithm
	CUDAsegmentAlgo_doSegment(buffer, maskData->GetScalarPointer(), dims, spaces, functionTable, lowI, highI, lowG, highG );
	maskData->Delete();

	//generate a surface net (polydata) from the outputted mask
	this->ComputeElasticSurfaceNet(buffer, dims, spaces, origin);
	delete buffer;

}

vtkCudaSegmentorGraphPoint* findProperVertex( vtkCudaSegmentorGraphPoint* start, short id ){
	vtkCudaSegmentorGraphPoint* curr = start;
	while( curr != 0 ){
		if( curr->type == id ){
			return curr;
		}
		curr = curr->next;
	}
	return 0;
}

void vtkCudaSegmentor::ComputeElasticSurfaceNet(short* mask, int* dims, float* spacing, float* origin){


	//create polyhedra for each of the classifications
	std::vector<vtkCellArray*>* output_triangles = new std::vector<vtkCellArray*>();
	std::vector<vtkPoints*>* output_points = new std::vector<vtkPoints*>();
	for(int c = 0; c < this->numberOfClassifications+1; c++){
		output_triangles->push_back(vtkCellArray::New());
		output_points->push_back(vtkPoints::New());
	}

	//store the graph vertices in a vector for later iteration
	std::vector<vtkCudaSegmentorGraphPoint*>* vertices = new std::vector<vtkCudaSegmentorGraphPoint*>();

	//keep a full map to determine vertex existance
	vtkCudaSegmentorGraphPoint** map = new vtkCudaSegmentorGraphPoint*[ (dims[0]-1)*(dims[1]-1)*(dims[2]-1) ];
	memset( (void*) map, 0, (dims[0]-1)*(dims[1]-1)*(dims[2]-1)*sizeof(vtkCudaSegmentorGraphPoint*) );

	//make the graph from the series of points
	for(int c = 1; c < this->numberOfClassifications+1; c++){
		int index = 0;
		int mapIndex = 0;
		for(int z = 0; z < dims[2]-1; z++){
			for(int y = 0; y < dims[1]-1; y++){
				for(int x = 0; x < dims[0]-1; x++){

					//determine if this particular voxel is on the edge of the organ identified by c
					bool allTrue = true;
					bool allFalse = true;
					(c == mask[index]) ? allFalse = false : allTrue = false ;
					(c == mask[index + 1]) ? allFalse = false : allTrue = false ;
					(c == mask[index + dims[0]]) ? allFalse = false : allTrue = false ;
					(c == mask[index + dims[0] + 1]) ? allFalse = false : allTrue = false ;
					(c == mask[index + dims[0]*dims[1]]) ? allFalse = false : allTrue = false ;
					(c == mask[index + dims[0]*dims[1] + 1]) ? allFalse = false : allTrue = false ;
					(c == mask[index + dims[0]*dims[1] + dims[0]]) ? allFalse = false : allTrue = false ;
					(c == mask[index + dims[0]*dims[1] + dims[0] + 1]) ? allFalse = false : allTrue = false ;

					//if it is on the surface, create a vertex to put in the graph
					if( !allTrue && !allFalse ){
						vtkCudaSegmentorGraphPoint* vertex = new vtkCudaSegmentorGraphPoint();
						vertex->type = c;
						vertex->visited = false;
						vertex->originalPositionX = (double) x + 0.5;
						vertex->activationXEven = (double) x + 0.5;
						vertex->activationXOdd = (double) x + 0.5;
						vertex->originalPositionY = (double) y + 0.5;
						vertex->activationYEven = (double) y + 0.5;
						vertex->activationYOdd = (double) y + 0.5;
						vertex->originalPositionZ = (double) z + 0.5;
						vertex->activationZEven = (double) z + 0.5;
						vertex->activationZOdd = (double) z + 0.5;
						vertex->next = 0;

						//connect to the downward vertex in the x direction
						vtkCudaSegmentorGraphPoint* neighbourPoint = x > 0 ? findProperVertex(map[mapIndex-1], c) : 0;
						if( neighbourPoint ){
							(neighbourPoint->edges)[1] = vertex;
							vertex->edges[0] = neighbourPoint;
						}else{
							vertex->edges[0] = 0;
						}

						//connect to the upward vertex in the x direction
						neighbourPoint = x < dims[0] - 2 ? findProperVertex(map[mapIndex+1], c) : 0;
						if( neighbourPoint ){
							(neighbourPoint->edges)[0] = vertex;
							vertex->edges[1] = neighbourPoint;
						}else{
							vertex->edges[1] = 0;
						}

						//connect to the downward vertex in the y direction
						neighbourPoint = y > 0 ? findProperVertex(map[mapIndex-dims[0]+1], c) : 0;
						if( neighbourPoint ){
							(neighbourPoint->edges)[3] = vertex;
							vertex->edges[2] = neighbourPoint;
						}else{
							vertex->edges[2] = 0;
						}

						//connect to the upward vertex in the y direction
						neighbourPoint = y < dims[1] - 2 ? findProperVertex(map[mapIndex+dims[0]-1], c) : 0;
						if( neighbourPoint ){
							(neighbourPoint->edges)[2] = vertex;
							vertex->edges[3] = neighbourPoint;
						}else{
							vertex->edges[3] = 0;
						}

						//connect to the downward vertex in the z direction
						neighbourPoint = z > 0 ? findProperVertex(map[mapIndex-(dims[0]-1)*(dims[1]-1)], c) : 0;
						if( neighbourPoint ){
							(neighbourPoint->edges)[5] = vertex;
							vertex->edges[4] = neighbourPoint;
						}else{
							vertex->edges[4] = 0;
						}
						
						//connect to the upward vertex in the z direction
						neighbourPoint = z < dims[2] - 2 ? findProperVertex(map[mapIndex+(dims[0]-1)*(dims[1]-1)], c) : 0;
						if( neighbourPoint ){
							(neighbourPoint->edges)[4] = vertex;
							vertex->edges[5] = neighbourPoint;
						}else{
							vertex->edges[5] = 0;
						}

						//add new vertex to the set of vertices
						vertices->push_back( vertex );
						vertex->next = map[mapIndex];
						map[mapIndex] = vertex;
					}

					//increment to the next index in the image
					index++;
					mapIndex++;

				} // for x
				index++; //extra index for x = dims[0]-1
			} // for y
			index+=dims[0]; //extra index for y = dims[1]-1
		} // for z
	} // for c
	delete map;

	//use a BFST (breadth-first search) to filter out surfaces with less than 10 vertices (assumed to be false positive noise)
	//containers required for the BFS
	std::vector<vtkCudaSegmentorGraphPoint*>* consumed = new std::vector<vtkCudaSegmentorGraphPoint*>();
	std::deque<vtkCudaSegmentorGraphPoint*>* queue = new std::deque<vtkCudaSegmentorGraphPoint*>();
	
	//run the BFS search checking all vertices and their visited status, allowing for a multiple tree forest to be created
	for( std::vector<vtkCudaSegmentorGraphPoint*>::iterator it = vertices->begin(); it != vertices->end(); it++ ){\

		//if we haven't already included this node in a tree, select it as the root of a new tree
		if( !(*it)->visited ){

			short classifier = (*it)->type;

			//perform the single-tree breadth-first search
			(*it)->visited = true;
			queue->push_back( (*it) );
			while(!queue->empty()){
				vtkCudaSegmentorGraphPoint* t = queue->front();
				queue->pop_front();
				consumed->push_back(t);
				for(int i = 0; i < 6; i++){
					if( t->edges[i] && !t->edges[i]->visited ){
						t->edges[i]->visited = true;
						queue->push_back(t->edges[i]);
					}
				}
			}

			//relax athe nodes on the tree if it is large enough, and then triangulate the vertices to add to the solid
			if( consumed->size() > 1024 ){
				switch(this->constraintType){
					case(0):
						relaxWithNoConstraints( consumed );
						break;
					case(1):
						relaxWithImplicitConstraints( consumed );
						break;
					case(2):
						relaxWithExplicitConstraints( consumed );
						break;
					default:
						relaxWithNoConstraints( consumed );
				}

				//actual face-generation process
				triangulate( consumed, output_triangles->at(classifier), output_points->at(classifier), spacing, origin );

			}

			//empty the set containing the tree for further use
			consumed->clear();
		}

		//delete this graph node (not necessary anymore and can no longer be visited)
		delete *it;
	}

	//populate the output
	for(int c = 0; c < this->numberOfClassifications+1; c++){
		vtkPolyData* tempPoly = vtkPolyData::New();
		this->output->push_back(tempPoly);
	}
	for(int c = 0; c < this->numberOfClassifications+1; c++){
		this->output->at(c)->SetPoints(output_points->at(c));
		this->output->at(c)->SetPolys(output_triangles->at(c));
		output_points->at(c)->Delete();
		output_triangles->at(c)->Delete();
	}
	output_triangles->clear();
	output_points->clear();


	//remove all data structures except the returnable one
	delete output_triangles;
	delete output_points;
	delete consumed;
	delete queue;
	delete vertices;
}

void vtkCudaSegmentor::triangulate( std::vector<vtkCudaSegmentorGraphPoint*>* vertices,  vtkCellArray* triangleArray, vtkPoints* points,
								    float* Spacing, float* Origin){

	//populate the points array, saving the index numbers for when we have to create the triangle's faces
	std::vector<vtkCudaSegmentorGraphPoint*>::iterator it;
	if(this->numSmoothingIterations & 1){ //final is even
		for( it = vertices->begin(); it != vertices->end(); it++ ){
			vtkCudaSegmentorGraphPoint* v = *it;
			v->index = points->InsertNextPoint( v->activationXEven * Spacing[0] + Origin[0], v->activationYEven * Spacing[1] + Origin[1], v->activationZEven * Spacing[2] + Origin[2]);
		}
	}else{ //final is odd
		for( it = vertices->begin(); it != vertices->end(); it++ ){
			vtkCudaSegmentorGraphPoint* v = *it;
			v->index = points->InsertNextPoint( v->activationXOdd * Spacing[0] + Origin[0], v->activationYOdd * Spacing[1] + Origin[1], v->activationZOdd * Spacing[2] + Origin[2]);
		}
	}

	//create the triangle's faces
	for( it = vertices->begin(); it != vertices->end(); it++ ){
		vtkCudaSegmentorGraphPoint* v = *it;
		if(this->meshCase == 12){
			use12CaseMesh(v, triangleArray);
		}else{
			use64CaseMesh(v, triangleArray);
		}
	}
}

void vtkCudaSegmentor::relaxWithNoConstraints( std::vector<vtkCudaSegmentorGraphPoint*>* vertices ){

	//relax the edges in the graph by changing the activation of each node to be the average of the adjacent node's activation
	//repeat a set number of times (this->numSmoothingIterations)
	std::vector<vtkCudaSegmentorGraphPoint*>::iterator relaxingIterator;
	for( int rounds = 0; rounds < this->numSmoothingIterations; rounds++){
		for( relaxingIterator = vertices->begin(); relaxingIterator != vertices->end(); relaxingIterator++ ){

			//values needed to take the average of the adjacent vertices
			int adjacentCardinality = 0;
			double accumX = 0.0f;
			double accumY = 0.0f;
			double accumZ = 0.0f;

			//accumulate the activation of the adjacent vertices
			for(int i = 0; i < 6; i++){
				if( (*relaxingIterator)->edges[i] ){
					adjacentCardinality++;
					if( rounds & 1){
						accumX += (*relaxingIterator)->edges[i]->activationXEven;
						accumY += (*relaxingIterator)->edges[i]->activationYEven;
						accumZ += (*relaxingIterator)->edges[i]->activationZEven;
					}else{
						accumX += (*relaxingIterator)->edges[i]->activationXOdd;
						accumY += (*relaxingIterator)->edges[i]->activationYOdd;
						accumZ += (*relaxingIterator)->edges[i]->activationZOdd;
					}
				}
			}

			//assert: the vertex has at least 1 adjacent vertex (else it would be a component of its own, and separated during the cutting of the BFST forest)
			//divide by the number of adjacent vertices to get the average
			accumX /= adjacentCardinality;
			accumY /= adjacentCardinality;
			accumZ /= adjacentCardinality;

			//set the new value to be the average of the adjacent vertices' values without any constraint
			if(rounds & 1){
				accumX = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationXEven + this->smoothingCoefficient * accumX;
				accumY = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationYEven + this->smoothingCoefficient * accumY;
				accumZ = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationZEven + this->smoothingCoefficient * accumZ;
				(*relaxingIterator)->activationXOdd = accumX;
				(*relaxingIterator)->activationYOdd = accumY;
				(*relaxingIterator)->activationZOdd = accumZ;
			}else{
				accumX = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationXOdd + this->smoothingCoefficient * accumX;
				accumY = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationYOdd + this->smoothingCoefficient * accumY;
				accumZ = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationZOdd + this->smoothingCoefficient * accumZ;
				(*relaxingIterator)->activationXEven = accumX;
				(*relaxingIterator)->activationYEven = accumY;
				(*relaxingIterator)->activationZEven = accumZ;
			}
		}
	}
}

void vtkCudaSegmentor::relaxWithExplicitConstraints( std::vector<vtkCudaSegmentorGraphPoint*>* vertices ){

	//relax the edges in the graph by changing the activation of each node to be the average of the adjacent node's activation

	double threshold = this->constraintThreshold / 2.0;

	//repeat a set number of times (this->numSmoothingIterations)
	std::vector<vtkCudaSegmentorGraphPoint*>::iterator relaxingIterator;
	for( int rounds = 0; rounds < this->numSmoothingIterations; rounds++){
		for( relaxingIterator = vertices->begin(); relaxingIterator != vertices->end(); relaxingIterator++ ){

			//values needed to take the average of the adjacent vertices
			int adjacentCardinality = 0;
			double accumX = 0.0f;
			double accumY = 0.0f;
			double accumZ = 0.0f;

			//accumulate the activation of the adjacent vertices
			for(int i = 0; i < 6; i++){
				if( (*relaxingIterator)->edges[i] ){
					adjacentCardinality++;
					if( rounds & 1){
						accumX += (*relaxingIterator)->edges[i]->activationXEven;
						accumY += (*relaxingIterator)->edges[i]->activationYEven;
						accumZ += (*relaxingIterator)->edges[i]->activationZEven;
					}else{
						accumX += (*relaxingIterator)->edges[i]->activationXOdd;
						accumY += (*relaxingIterator)->edges[i]->activationYOdd;
						accumZ += (*relaxingIterator)->edges[i]->activationZOdd;
					}
				}
			}

			//assert: the vertex has at least 1 adjacent vertex (else it would be a component of its own, and separated during the cutting of the BFST forest)
			//divide by the number of adjacent vertices to get the average
			accumX /= adjacentCardinality;
			accumY /= adjacentCardinality;
			accumZ /= adjacentCardinality;

			//if the new value doesn't fit inside the bounds, set it to the bounds
			if(rounds & 1){
				accumX = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationXEven + this->smoothingCoefficient * accumX;
				accumY = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationYEven + this->smoothingCoefficient * accumY;
				accumZ = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationZEven + this->smoothingCoefficient * accumZ;
				(*relaxingIterator)->activationXOdd = ( accumX > (*relaxingIterator)->originalPositionX + threshold ? (*relaxingIterator)->originalPositionX + threshold : ( accumX < (*relaxingIterator)->originalPositionX - threshold ? (*relaxingIterator)->originalPositionX - threshold : accumX) );
				(*relaxingIterator)->activationYOdd = ( accumY > (*relaxingIterator)->originalPositionY + threshold ? (*relaxingIterator)->originalPositionY + threshold : ( accumY < (*relaxingIterator)->originalPositionY - threshold ? (*relaxingIterator)->originalPositionY - threshold : accumY) );
				(*relaxingIterator)->activationZOdd = ( accumZ > (*relaxingIterator)->originalPositionZ + threshold ? (*relaxingIterator)->originalPositionZ + threshold : ( accumZ < (*relaxingIterator)->originalPositionZ - threshold ? (*relaxingIterator)->originalPositionZ - threshold : accumZ) );
			}else{
				accumX = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationXOdd + this->smoothingCoefficient * accumX;
				accumY = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationYOdd + this->smoothingCoefficient * accumY;
				accumZ = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationZOdd + this->smoothingCoefficient * accumZ;
				(*relaxingIterator)->activationXEven = ( accumX > (*relaxingIterator)->originalPositionX + threshold ? (*relaxingIterator)->originalPositionX + threshold : ( accumX < (*relaxingIterator)->originalPositionX - threshold ? (*relaxingIterator)->originalPositionX - threshold : accumX) );
				(*relaxingIterator)->activationYEven = ( accumY > (*relaxingIterator)->originalPositionY + threshold ? (*relaxingIterator)->originalPositionY + threshold : ( accumY < (*relaxingIterator)->originalPositionY - threshold ? (*relaxingIterator)->originalPositionY - threshold : accumY) );
				(*relaxingIterator)->activationZEven = ( accumZ > (*relaxingIterator)->originalPositionZ + threshold ? (*relaxingIterator)->originalPositionZ + threshold : ( accumZ < (*relaxingIterator)->originalPositionZ - threshold ? (*relaxingIterator)->originalPositionZ - threshold : accumZ) );
			}
		}
	}
}

void vtkCudaSegmentor::relaxWithImplicitConstraints( std::vector<vtkCudaSegmentorGraphPoint*>* vertices ){

	//relax the edges in the graph by changing the activation of each node to be the average of the adjacent node's activation
	//the points are implicitly constrained by applying a linear regression model to themselves, mapping them to their original locations
		//applying the linear regression should keep the volume relatively consistant
		//instead of using a set number of iterations, the process stops after the error deviation exceeds 1

	//variables used for the linear regression
	const double n = vertices->size(); //the number of samples
	double rX = 1.0;
	double rY = 1.0;
	double rZ = 1.0;
	double bX = 0.0;
	double bY = 0.0;
	double bZ = 0.0;

	//for the linear regression vectors
	long double totOX = 0.0; //sum of the original x positons (constant)
	long double totOY = 0.0; //sum of the original y positons (constant)
	long double totOZ = 0.0; //sum of the original z positons (constant)

	long double totOXX = 0.0; //sum of the original x positons times the new x position
	long double totOYY = 0.0; //sum of the original y positons times the new y position
	long double totOZZ = 0.0; //sum of the original z positons times the new z position

	long double totX; //the sum of the modified x positions
	long double totY; //the sum of the modified y positions
	long double totZ; //the sum of the modified z positions

	long double totXX; //the sum of the modified x positions squared
	long double totYY; //the sum of the modified y positions squared
	long double totZZ; //the sum of the modified z positions squared

	//calculate the constants
	std::vector<vtkCudaSegmentorGraphPoint*>::iterator relaxingIterator;
	for( relaxingIterator = vertices->begin(); relaxingIterator != vertices->end(); relaxingIterator++ ){
		totOX += (*relaxingIterator)->originalPositionX;
		totOY += (*relaxingIterator)->originalPositionY;
		totOZ += (*relaxingIterator)->originalPositionZ;
	}

	//variables for the error deviation
	long double Err2 = 0.0;
	double threshold = (this->constraintThreshold * this->constraintThreshold) / 4.0;

	//repeat a set number of times (this->numSmoothingIterations)
	int rounds = 0;
	do{
		
		//reset the linear regression and error accumulators
		Err2 = 0.0;
		totX = totY = totZ = 0.0;
		totXX = totYY = totZZ = 0.0;
		totOXX = totOYY = totOZZ = 0.0;

		for( relaxingIterator = vertices->begin(); relaxingIterator != vertices->end(); relaxingIterator++ ){

			//values needed to take the average of the adjacent vertices
			int adjacentCardinality = 0;
			double accum[3] = {0.0, 0.0, 0.0};

			//accumulate the activation of the adjacent vertices
			for(int i = 0; i < 6; i++){
				if( (*relaxingIterator)->edges[i] ){
					adjacentCardinality++;
					if( rounds & 1){
						accum[0] += (*relaxingIterator)->edges[i]->activationXEven;
						accum[1] += (*relaxingIterator)->edges[i]->activationYEven;
						accum[2] += (*relaxingIterator)->edges[i]->activationZEven;
					}else{
						accum[0] += (*relaxingIterator)->edges[i]->activationXOdd;
						accum[1] += (*relaxingIterator)->edges[i]->activationYOdd;
						accum[2] += (*relaxingIterator)->edges[i]->activationZOdd;
					}
				}
			}

			//assert: the vertex has at least 1 adjacent vertex (else it would be a component of its own, and separated during the cutting of the BFST forest)
			//divide by the number of adjacent vertices to get the average
			accum[0] /= adjacentCardinality;
			accum[1] /= adjacentCardinality;
			accum[2] /= adjacentCardinality;
			
			//apply the previous linear regression
			accum[0] = rX*accum[0] + bX;
			accum[1] = rY*accum[1] + bY;
			accum[2] = rZ*accum[2] + bZ;

			//set the new edge, applying the previous linear regression
			if(rounds & 1){
				accum[0] = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationXEven + this->smoothingCoefficient * accum[0];
				accum[1] = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationYEven + this->smoothingCoefficient * accum[1];
				accum[2] = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationZEven + this->smoothingCoefficient * accum[2];
				(*relaxingIterator)->activationXOdd = accum[0];
				(*relaxingIterator)->activationYOdd = accum[1];
				(*relaxingIterator)->activationZOdd = accum[2];
			}else{
				accum[0] = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationXOdd + this->smoothingCoefficient * accum[0];
				accum[1] = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationYOdd + this->smoothingCoefficient * accum[1];
				accum[2] = (1-this->smoothingCoefficient) * (*relaxingIterator)->activationZOdd + this->smoothingCoefficient * accum[2];
				(*relaxingIterator)->activationXEven = accum[0];
				(*relaxingIterator)->activationYEven = accum[1];
				(*relaxingIterator)->activationZEven = accum[2];
			}

			//add to the linear regression accumulators
			totX += accum[0];
			totY += accum[1];
			totZ += accum[2];
			totXX += accum[0] * accum[0];
			totYY += accum[1] * accum[1];
			totZZ += accum[2] * accum[2];
			totOXX += accum[0] * (*relaxingIterator)->originalPositionX;
			totOYY += accum[1] * (*relaxingIterator)->originalPositionY;
			totOZZ += accum[2] * (*relaxingIterator)->originalPositionZ;

			//add to the error accumulators
			accum[0] = abs( accum[0] - (*relaxingIterator)->originalPositionX );
			accum[1] = abs( accum[1] - (*relaxingIterator)->originalPositionY );
			accum[2] = abs( accum[2] - (*relaxingIterator)->originalPositionZ );
			Err2 += accum[0] > 0.5 ? (accum[0]-0.5)*(accum[0]-0.5):0;
			Err2 += accum[1] > 0.5 ? (accum[1]-0.5)*(accum[1]-0.5):0;
			Err2 += accum[2] > 0.5 ? (accum[2]-0.5)*(accum[2]-0.5):0;

		}

		//calculate the linear regession for this round to be applied in the next round
		rX = ( n * totOXX - totOX * totX ) / ( n * totXX - totX * totX);
		bX = ( totOX - rX * totX ) / n;
		rY = ( n * totOYY - totOY * totY ) / ( n * totYY - totY * totY);
		bY = ( totOY - rY * totY ) / n;
		rZ = ( n * totOZZ - totOZ * totZ ) / ( n * totZZ - totZ * totZ);
		bZ = ( totOZ - rZ * totZ ) / n;

		//calculate the average squared error for this round
		Err2 /= n;

		rounds++;
	}while( Err2 < threshold && rounds < 1000 );

	//apply the previous linear regression
	for( relaxingIterator = vertices->begin(); relaxingIterator != vertices->end(); relaxingIterator++ ){
		if(rounds & 1){
			(*relaxingIterator)->activationXOdd = rX * (*relaxingIterator)->activationXOdd + bX;
			(*relaxingIterator)->activationYOdd = rY * (*relaxingIterator)->activationYOdd + bY;
			(*relaxingIterator)->activationZOdd = rZ * (*relaxingIterator)->activationZOdd + bZ;
		}else{
			(*relaxingIterator)->activationXEven = rX * (*relaxingIterator)->activationXEven + bX;
			(*relaxingIterator)->activationYEven = rY * (*relaxingIterator)->activationYEven + bY;
			(*relaxingIterator)->activationZEven = rZ * (*relaxingIterator)->activationZEven + bZ;
		}
	}

	//set the number of iterations to the minimum (for triangulation)
	this->numSmoothingIterations = rounds;

}

void vtkCudaSegmentor::use64CaseMesh(vtkCudaSegmentorGraphPoint* v, vtkCellArray* triangleArray){
	vtkIdType pts[3];

	//get a numeric description of which edges are available
	short edgeDescriptor = 0;
	int i = 1;
	for(int j = 0; j < 6; j++){
		edgeDescriptor += v->edges[j] ? i:0;
		i+=i;
	}

	//create the triangles on a case by case basis
	switch(edgeDescriptor){
		case 5: // y- to x-
			pts[0] = v->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 6: // x+ to y-
			pts[0] = v->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 7: // x+ to y- and y- to x-
			pts[0] = v->index;;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 9: // x- to y+
			pts[0] = v->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 10: // y+ to x+
			pts[0] = v->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 11: // x- to y+ and y+ to x+
			pts[0] = v->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 13: // y- to x- and x- to y+
			pts[0] = v->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 14: // y+ to x+ and x+ to y-
			pts[0] = v->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 15: //full xy plane
			pts[0] = v->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 17: // x- to z-
			pts[0] = v->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 18: // z- to x+
			pts[0] = v->index;
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 19: //x- to z- and z- to x+
			pts[0] = v->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 20: //z- to y-
			pts[0] = v->index;
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 21: //z- to y- to x-
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 22: // z- to x+ to y-
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 23: //both case 21 and 22
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 24: //y+ to z-
			pts[0] = v->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 25: // y+ to x- to z-
			pts[0] = v->edges[3]->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 26: // z- to x+ to y+
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 27: //both cases 25 and 26
			pts[0] = v->edges[3]->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 28: //y+ to z- and z- to y-
			pts[0] = v->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 29: // both cases 25 and 21
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[3]->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 30: // both cases 26 and 22
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 31: // all cases 21, 22, 25, 26 and the xy plane
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[3]->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);

			pts[0] = v->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);

			break;
		case 33: // z+ to x-
			pts[0] = v->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 34: // x+ to z+
			pts[0] = v->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 35: // x+ to z+ and z+ to x-
			pts[0] = v->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 36: // y- to z+
			pts[0] = v->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 37: //x- to z+ to y-
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 38: //y- to z+ to x+
			pts[0] = v->edges[2]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 39: // both cases 37 and 38
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[2]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 40: // z+ to y+
			pts[0] = v->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 41: // x- to y+ to z+
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 42: // z+ to y+ to x+
			pts[0] = v->edges[5]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 43: // both cases 41 and 42
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[5]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 44: // y- to z+ and z+ to y+
			pts[0] = v->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 45: //both cases 41 and 37
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 46: //both cases 42 and 38
			pts[0] = v->edges[5]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[2]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 47: //all cases 37, 38, 41 and 42 and the xy plane
			pts[0] = v->edges[5]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[2]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);

			pts[0] = v->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);

			break;
		case 49: // z+ to x- and x- to z-
			pts[0] = v->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 50: // z- to x+ and x+ to z+
			pts[0] = v->index;
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 51: //full xz plane
			pts[0] = v->index;
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 52: //z- to y- and y - to z+
			pts[0] = v->index;
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 53: //both cases 37 and 21
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 54: //both cases 22 and 38
			pts[0] = v->edges[2]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 55: // all cases 21, 22, 37 and 38 and the xz plane
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[2]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			
			pts[0] = v->index;
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);

			break;
		case 56: // z+ to y+ and y+ to z-
			pts[0] = v->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 57: // both cases 41 and 25
			pts[0] = v->edges[3]->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 58: //both cases 42 and 26
			pts[0] = v->edges[5]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 59: // all cases 25, 26, 41 and 42 and xz plane
			pts[0] = v->edges[5]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[3]->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);

			pts[0] = v->index;
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);

			break;
		case 60: // full yz plane
			pts[0] = v->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 61: //all cases 21, 25, 37 and 41 and the yz plane
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[3]->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);

			pts[0] = v->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 62: //all cases 22, 26, 38 and 42 and the yz plane
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[2]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[5]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);

			pts[0] = v->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
		case 63: //all 8 faces non-orthogonal to any axis
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[3]->index;
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[0]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[4]->index;
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[2]->index;
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			pts[0] = v->edges[5]->index;
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
			break;
	}

}

void vtkCudaSegmentor::use12CaseMesh(vtkCudaSegmentorGraphPoint* v, vtkCellArray* triangleArray){
	vtkIdType pts[3];

	pts[0] = v->index;
	if( v->edges[0] ){
		if( v->edges[2] ){ //y- to x-
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
		}
		if( v->edges[3] ){ //x- to y+
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
		}
		if( v->edges[4] ){ //x- to z-
			pts[1] = v->edges[0]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
		}
		if( v->edges[5] ){ //z+ to z-
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[0]->index;
			triangleArray->InsertNextCell(3, pts);
		}
	}

	if( v->edges[1] ){
		if( v->edges[2] ){ //x+ to y-
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
		}
		if( v->edges[3] ){ //y+ to x+
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
		}
		if( v->edges[4] ){ //z- to x+
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[1]->index;
			triangleArray->InsertNextCell(3, pts);
		}
		if( v->edges[5] ){ //x+ to z+
			pts[1] = v->edges[1]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
		}
	}

	if( v->edges[2] ){
		if( v->edges[4] ){ //z- to y-
			pts[1] = v->edges[4]->index;
			pts[2] = v->edges[2]->index;
			triangleArray->InsertNextCell(3, pts);
		}
		if( v->edges[5] ){ //y- to z+
			pts[1] = v->edges[2]->index;
			pts[2] = v->edges[5]->index;
			triangleArray->InsertNextCell(3, pts);
		}
	}

	if( v->edges[3] ){
		if( v->edges[4] ){ //y+ to z-
			pts[1] = v->edges[3]->index;
			pts[2] = v->edges[4]->index;
			triangleArray->InsertNextCell(3, pts);
		}
		if( v->edges[5] ){ //z+ to y+
			pts[1] = v->edges[5]->index;
			pts[2] = v->edges[3]->index;
			triangleArray->InsertNextCell(3, pts);
		}
	}

}