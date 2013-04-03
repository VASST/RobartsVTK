#include "vtkHierarchicalMaxFlowSegmentation.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkTreeDFSIterator.h"

#include <assert.h>
#include <math.h>
#include <float.h>

#define SQR(X) X*X

vtkStandardNewMacro(vtkHierarchicalMaxFlowSegmentation);

vtkHierarchicalMaxFlowSegmentation::vtkHierarchicalMaxFlowSegmentation(){
	
	//configure the IO ports
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(0);

	//set algorithm mathematical parameters to defaults
	this->NumberOfIterations = 100;
	this->StepSize = 0.075;
	this->CC = 0.75;

	//set up the input mapping structure
	this->InputPortMapping.clear();
	this->BackwardsInputPortMapping.clear();
	this->FirstUnusedPort = 0;

	//set the other values to defaults
	this->Hierarchy = 0;
	this->SmoothnessScalars.clear();
	this->OutputPortMapping.clear();
	this->InputPortMapping.clear();
	this->IntermediateBufferMapping.clear();

}

vtkHierarchicalMaxFlowSegmentation::~vtkHierarchicalMaxFlowSegmentation(){
	if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
	this->SmoothnessScalars.clear();
	this->OutputPortMapping.clear();
	this->InputPortMapping.clear();
	this->BackwardsInputPortMapping.clear();
	this->IntermediateBufferMapping.clear();
}

//------------------------------------------------------------

void vtkHierarchicalMaxFlowSegmentation::SetHierarchy(vtkTree* graph){
	if( graph != this->Hierarchy ){
		if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
		this->Hierarchy = graph;
		if( this->Hierarchy ) this->Hierarchy->Register(this);
		this->Modified();

		//update output mapping
		this->OutputPortMapping.clear();
		vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
		iterator->SetTree(this->Hierarchy);
		iterator->SetStartVertex(this->Hierarchy->GetRoot());
		while( iterator->HasNext() ){
			vtkIdType node = iterator->Next();
			if( this->Hierarchy->IsLeaf(node) )
				this->OutputPortMapping.insert(std::pair<vtkIdType,int>(node,(int) this->OutputPortMapping.size()));
		}
		iterator->Delete();

		//update number of output ports
		this->SetNumberOfOutputPorts((int) this->OutputPortMapping.size());

	}
}

vtkTree* vtkHierarchicalMaxFlowSegmentation::GetHierarchy(){
	return this->Hierarchy;
}

//------------------------------------------------------------

void vtkHierarchicalMaxFlowSegmentation::AddSmoothnessScalar(vtkIdType node, double value){
	if( value >= 0.0 ){
		this->SmoothnessScalars.insert(std::pair<vtkIdType,double>(node,value));
		this->Modified();
	}else{
		vtkErrorMacro(<<"Cannot use a negative smoothness value.");
	}
}

//------------------------------------------------------------

int vtkHierarchicalMaxFlowSegmentation::FillInputPortInformation(int i, vtkInformation* info){
	info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
	info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
	info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
	return this->Superclass::FillInputPortInformation(i,info);
}

void vtkHierarchicalMaxFlowSegmentation::SetInput(int idx, vtkDataObject *input)
{
	//we are adding/switching an input, so no need to resort list
	if( input != NULL ){
	
		//if their is no pair in the mapping, create one
		if( this->InputPortMapping.find(idx) == this->InputPortMapping.end() ){
			int portNumber = this->FirstUnusedPort;
			this->FirstUnusedPort++;
			this->InputPortMapping.insert(std::pair<vtkIdType,int>(idx,portNumber));
			this->BackwardsInputPortMapping.insert(std::pair<vtkIdType,int>(portNumber,idx));
		}
		this->SetNthInputConnection(0, this->InputPortMapping.find(idx)->second, input->GetProducerPort() );

	}else{
		//if their is no pair in the mapping, just exit, nothing to do
		if( this->InputPortMapping.find(idx) == this->InputPortMapping.end() ) return;

		int portNumber = this->InputPortMapping.find(idx)->second;
		this->InputPortMapping.erase(this->InputPortMapping.find(idx));
		this->BackwardsInputPortMapping.erase(this->BackwardsInputPortMapping.find(portNumber));

		//if we are the last input, no need to reshuffle
		if(portNumber == this->FirstUnusedPort - 1){
			this->SetNthInputConnection(0, portNumber,  0);
		
		//if we are not, move the last input into this spot
		}else{
			vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedPort - 1));
			this->SetNthInputConnection(0, portNumber, swappedInput->GetProducerPort() );
			this->SetNthInputConnection(0, this->FirstUnusedPort - 1, 0 );

			//correct the mappings
			vtkIdType swappedId = this->BackwardsInputPortMapping.find(this->FirstUnusedPort - 1)->second;
			this->InputPortMapping.erase(this->InputPortMapping.find(swappedId));
			this->BackwardsInputPortMapping.erase(this->BackwardsInputPortMapping.find(this->FirstUnusedPort - 1));
			this->InputPortMapping.insert(std::pair<vtkIdType,int>(swappedId,portNumber) );
			this->BackwardsInputPortMapping.insert(std::pair<int,vtkIdType>(portNumber,swappedId) );

		}

		//decrement the number of unused ports
		this->FirstUnusedPort--;

	}
}

vtkDataObject *vtkHierarchicalMaxFlowSegmentation::GetInput(int idx)
{
	if( this->InputPortMapping.find(idx) == this->InputPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputPortMapping.find(idx)->second));
}

vtkDataObject *vtkHierarchicalMaxFlowSegmentation::GetOutput(int idx)
{
	//look up port in mapping
	std::map<vtkIdType,int>::iterator port = this->OutputPortMapping.find(idx);
	if( port == this->OutputPortMapping.end() )
		return 0;

	return vtkImageData::SafeDownCast(this->GetExecutive()->GetOutputData(port->second));
}

//----------------------------------------------------------------------------

int vtkHierarchicalMaxFlowSegmentation::CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges ){
	
	//check to make sure that the hierarchy is specified
	if( !this->Hierarchy ){
		vtkErrorMacro(<<"Hierarchy must be provided.");
		return -1;
	}

	//check to make sure that there is an image associated with each leaf node
	NumLeaves = 0;
	NumNodes = 0;
	Extent[0] = -1;
	vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	iterator->SetStartVertex(this->Hierarchy->GetRoot());
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		NumNodes++;


		//make sure all leaf nodes have a data term
		if( this->Hierarchy->IsLeaf(node) ){
			NumLeaves++;
			
			if( this->InputPortMapping.find(node) == this->InputPortMapping.end() ){
				vtkErrorMacro(<<"Missing data prior for leaf node.");
				return -1;
			}

			int inputPortNumber = this->InputPortMapping.find(node)->second;

			if( !(inputVector[0])->GetInformationObject(inputPortNumber) && (inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ){
				vtkErrorMacro(<<"Missing data prior for leaf node.");
				return -1;
			}

			//make sure the term is non-negative
			vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
			if( CurrImage->GetScalarRange()[0] < 0.0 ){
				vtkErrorMacro(<<"Data prior must be non-negative.");
				return -1;
			}
			
		}
		
		//if we get here, any missing mapping implies just no smoothness term (default intended)
		if( this->InputPortMapping.find(node) == this->InputPortMapping.end() ) continue;
		int inputPortNumber = this->InputPortMapping.find(node)->second;
		if( ! (inputVector[0])->GetInformationObject(inputPortNumber) ||
			! (inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ) continue;

		//check to make sure the datatype is float
		vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
		if( CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1 ){
			vtkErrorMacro(<<"Data type must be FLOAT and only have one component.");
			return -1;
		}

		//check to make sure that the sizes are consistant
		if( Extent[0] == -1 ){
			vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
			CurrImage->GetExtent(Extent);
		}else{
			int CurrExtent[6];
			vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
			CurrImage->GetExtent(CurrExtent);
			if( CurrExtent[0] != Extent[0] || CurrExtent[1] != Extent[1] || CurrExtent[2] != Extent[2] ||
				CurrExtent[3] != Extent[3] || CurrExtent[4] != Extent[4] || CurrExtent[5] != Extent[5] ){
				vtkErrorMacro(<<"Inconsistant object extent.");
				return -1;
			}
		}

	}
	iterator->Delete();

	NumEdges = NumNodes - 1;

	return 0;
}

int vtkHierarchicalMaxFlowSegmentation::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	//check input for consistancy
	int Extent[6]; int NumNodes; int NumLeaves; int NumEdges;
	int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
	if( result || NumNodes == 0 ) return -1;
	
	//set the number of output ports
	outputVector->SetNumberOfInformationObjects(NumLeaves);
	this->SetNumberOfOutputPorts(NumLeaves);

	return 1;
}

int vtkHierarchicalMaxFlowSegmentation::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	//check input for consistancy
	int Extent[6]; int NumNodes; int NumLeaves; int NumEdges;
	int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
	if( result || NumNodes == 0 ) return -1;

	//set up the extents
	for(int i = 0; i < NumLeaves; i++ ){
		vtkInformation *outputInfo = outputVector->GetInformationObject(i);
		vtkImageData *outputBuffer = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
		outputInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),Extent,6);
		outputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),Extent,6);
	}

	return 1;
}

void vtkHierarchicalMaxFlowSegmentation::PropogateLabels( vtkIdType currNode, float** branchLabels, float** leafLabels, int size ){
	
	//update graph for all kids
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int kid = 0; kid < NumKids; kid++)
		PropogateLabels( this->Hierarchy->GetChild(currNode,kid), branchLabels, leafLabels, size );

	//sum value into parent (if parent exists and is not the root)
	if( currNode == this->Hierarchy->GetRoot() ) return;
	vtkIdType parent = 	this->Hierarchy->GetParent(currNode);
	if( parent == this->Hierarchy->GetRoot() ) return;
	int parentIndex = this->IntermediateBufferMapping.find(parent)->second;
	float* currVal =   this->Hierarchy->IsLeaf(currNode) ?
		currVal = leafLabels[this->OutputPortMapping.find(currNode)->second] :
		currVal = branchLabels[this->IntermediateBufferMapping.find(currNode)->second];
	for( int x = 0; x < size; x++ )
		(branchLabels[parentIndex])[x] += currVal[x];
	
}


void vtkHierarchicalMaxFlowSegmentation::PropogateFlows( vtkIdType currNode, float* sourceSinkFlow, float** branchSinkFlows, float** leafSinkFlows,
																			 float** branchIncFlows, float** branchDivFlows, float** leafDivFlows,
																			 float** branchLabels, float** leafLabels, int size ){

	//update graph for all kids
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int kid = 0; kid < NumKids; kid++)
		PropogateFlows( this->Hierarchy->GetChild(currNode,kid), sourceSinkFlow, branchSinkFlows, leafSinkFlows, branchIncFlows, branchDivFlows, leafDivFlows, branchLabels, leafLabels, size );
	
	//if we are a leaf, we have nothing to do, since sink flow is already up-to-date
	if( NumKids == 0 ) return;

	//if we are the root, we need to be treated differently
	if( currNode == this->Hierarchy->GetRoot() ){

		//initialize the flow
		for( int x = 0; x < size; x++ )
			sourceSinkFlow[x] = 1.0f / this->CC;

		//get required flow from children
		for(int kid = 0; kid < NumKids; kid++){
			vtkIdType kidNode = this->Hierarchy->GetChild(currNode,kid);
			float* kidSinkFlow = (this->Hierarchy->IsLeaf(kidNode)) ?
				leafSinkFlows[this->OutputPortMapping.find(kidNode)->second] :
				branchSinkFlows[this->IntermediateBufferMapping.find(kidNode)->second] ;
			float* kidDivFlow = (this->Hierarchy->IsLeaf(kidNode)) ?
				leafDivFlows[this->OutputPortMapping.find(kidNode)->second] :
				branchDivFlows[this->IntermediateBufferMapping.find(kidNode)->second] ;
			float* kidLabels = (this->Hierarchy->IsLeaf(kidNode)) ?
				leafLabels[this->OutputPortMapping.find(kidNode)->second] :
				branchLabels[this->IntermediateBufferMapping.find(kidNode)->second] ;
			for( int x = 0; x < size; x++ )
				sourceSinkFlow[x] += kidSinkFlow[x] + kidDivFlow[x] - kidLabels[x] / this->CC;
		}
		
		//implement the normalizing factor
		for( int x = 0; x < size; x++ )
			sourceSinkFlow[x] /= (float) NumKids;

		return;

	//if we are a branch node, the process is more complicated
	}else{
		int currNodeIndex = this->IntermediateBufferMapping.find(currNode)->second;

		//start at 0
		for( int x = 0; x < size; x++ )
			(branchSinkFlows[currNodeIndex])[x] = 0.0f;

		//get required flow from children
		for(int kid = 0; kid < NumKids; kid++){
			vtkIdType kidNode = this->Hierarchy->GetChild(currNode,kid);
			float* kidSinkFlow = (this->Hierarchy->IsLeaf(kidNode)) ?
				leafSinkFlows[this->OutputPortMapping.find(kidNode)->second] :
				branchSinkFlows[this->IntermediateBufferMapping.find(kidNode)->second] ;
			float* kidDivFlow = (this->Hierarchy->IsLeaf(kidNode)) ?
				leafDivFlows[this->OutputPortMapping.find(kidNode)->second] :
				branchDivFlows[this->IntermediateBufferMapping.find(kidNode)->second] ;
			float* kidLabels = (this->Hierarchy->IsLeaf(kidNode)) ?
				leafLabels[this->OutputPortMapping.find(kidNode)->second] :
				branchLabels[this->IntermediateBufferMapping.find(kidNode)->second] ;
			for( int x = 0; x < size; x++ )
				(branchSinkFlows[currNodeIndex])[x] += kidSinkFlow[x] + kidDivFlow[x] - kidLabels[x] / this->CC;
		}

		//get required flow from self
		for( int x = 0; x < size; x++ )
			(branchSinkFlows[currNodeIndex])[x] += (branchIncFlows[currNodeIndex])[x] - (branchDivFlows[currNodeIndex])[x] + (branchLabels[currNodeIndex])[x] / this->CC;

		//implement the normalizing factor
		for( int x = 0; x < size; x++ )
			(branchSinkFlows[currNodeIndex])[x] /= (float) (NumKids + 1);
	}

}

int vtkHierarchicalMaxFlowSegmentation::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){
								
	//check input consistancy
	int Extent[6]; int NumNodes; int NumLeaves; int NumEdges;
	int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
	if( result || NumNodes == 0 ) return -1;
	int NumBranches = NumNodes - NumLeaves - 1;

	//set the number of output ports
	outputVector->SetNumberOfInformationObjects(NumLeaves);
	this->SetNumberOfOutputPorts(NumLeaves);

	//find the size of the volume
	int X = (Extent[1] - Extent[0] + 1);
	int Y = (Extent[3] - Extent[2] + 1);
	int Z = (Extent[5] - Extent[4] + 1);
	int VolumeSize =  X * Y * Z;

	//create relevant node identifier to buffer mappings
	this->IntermediateBufferMapping.clear();
	vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	iterator->SetStartVertex(this->Hierarchy->GetRoot());
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( node == this->Hierarchy->GetRoot() ) continue;
		if( !this->Hierarchy->IsLeaf(node) )
			IntermediateBufferMapping.insert(std::pair<vtkIdType,int>(node,(int) this->IntermediateBufferMapping.size()));
	}
	iterator->Delete();

	//get the data term buffers
	float** leafDataTermBuffers = new float* [NumLeaves];
	iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( this->Hierarchy->IsLeaf(node) ){
			int inputNumber = this->InputPortMapping.find(node)->second;
			vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputNumber)->Get(vtkDataObject::DATA_OBJECT()));
			leafDataTermBuffers[this->OutputPortMapping.find(node)->second] = (float*) CurrImage->GetScalarPointer();
		}
	}
	iterator->Delete();
	
	//get the smoothness term buffers
	float** leafSmoothnessTermBuffers = new float* [NumLeaves];
	float** branchSmoothnessTermBuffers = new float* [NumBranches];
	iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( node == this->Hierarchy->GetRoot() ) continue;
		vtkImageData* CurrImage = 0;
		if( this->InputPortMapping.find(this->Hierarchy->GetParent(node)) != this->InputPortMapping.end() ){
			int parentInputNumber = this->InputPortMapping.find(this->Hierarchy->GetParent(node))->second;
			vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(parentInputNumber)->Get(vtkDataObject::DATA_OBJECT()));
		}
		if( this->Hierarchy->IsLeaf(node) )
			leafSmoothnessTermBuffers[this->OutputPortMapping.find(node)->second] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
		else
			branchSmoothnessTermBuffers[this->IntermediateBufferMapping.find(node)->second] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
	}
	iterator->Delete();

	//get the output buffers
	float** leafLabelBuffers = new float* [NumLeaves];
	for(int i = 0; i < NumLeaves; i++ ){
		vtkInformation *outputInfo = outputVector->GetInformationObject(i);
		vtkImageData *outputBuffer = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
		outputBuffer->SetExtent(Extent);
		outputBuffer->Modified();
		outputBuffer->AllocateScalars();
		leafLabelBuffers[i] = (float*) outputBuffer->GetScalarPointer();
	}

	//allocate source flow buffer
	float* sourceFlow = new float[VolumeSize];
	for(int x = 0; x < VolumeSize; x++)
		sourceFlow[x] = 0.0f;

	//allocate required memory buffers for the brach nodes
	//		buffers needed:
	//			per brach node (not the root):
	//				1 label buffer
	//				3 spatial flow buffers
	//				1 divergence buffer
	//				1 outgoing flow buffer
	//				1 working temp buffer (ie: guk)
	float* branchNodeBuffer = new float[6*VolumeSize*NumBranches];
	float** branchFlowXBuffers = new float* [NumBranches];
	float** branchFlowYBuffers = new float* [NumBranches];
	float** branchFlowZBuffers = new float* [NumBranches];
	float** branchSinkBuffers = new float* [NumBranches];
	float** branchLabelBuffers = new float* [NumBranches];
	float** branchWorkingBuffers = new float* [NumBranches];
	float* ptr = branchNodeBuffer;
	for(int i = 0; i < NumBranches; i++ ){
		branchFlowXBuffers[i] = ptr; ptr += VolumeSize;
		branchFlowYBuffers[i] = ptr; ptr += VolumeSize;
		branchFlowZBuffers[i] = ptr; ptr += VolumeSize;
		branchLabelBuffers[i] = ptr; ptr += VolumeSize;
		branchSinkBuffers[i] = ptr; ptr += VolumeSize;
		branchWorkingBuffers[i] = ptr; ptr += VolumeSize;
	}

	//and for the leaf nodes
	//			per leaf node (note label buffer is in output)
	//				3 spatial flow buffers
	//				1 divergence buffer
	//				1 sink flow buffer
	//				1 working temp buffer (ie: guk)
	float*	leafNodeBuffer = new float[5*VolumeSize*NumLeaves];
	float** leafFlowXBuffers = new float* [NumLeaves];
	float** leafFlowYBuffers = new float* [NumLeaves];
	float** leafFlowZBuffers = new float* [NumLeaves];
	float** leafSinkBuffers = new float* [NumLeaves];
	float** leafWorkingBuffers = new float* [NumLeaves];
	ptr = leafNodeBuffer;
	for(int i = 0; i < NumLeaves; i++ ){
		leafFlowXBuffers[i] = ptr; ptr += VolumeSize;
		leafFlowYBuffers[i] = ptr; ptr += VolumeSize;
		leafFlowZBuffers[i] = ptr; ptr += VolumeSize;
		leafSinkBuffers[i] = ptr; ptr += VolumeSize;
		leafWorkingBuffers[i] = ptr; ptr += VolumeSize;
	}

	//initalize all spatial flows and divergences to zero
	for(int x = 0; x < VolumeSize; x++){
		for(int i = 0; i < NumBranches; i++ ){
			(branchFlowXBuffers[i])[x] = 0.0f;
			(branchFlowYBuffers[i])[x] = 0.0f;
			(branchFlowZBuffers[i])[x] = 0.0f;
			(branchLabelBuffers[i])[x] = 0.0f;
			(branchWorkingBuffers[i])[x] = 0.0f;
		}
		for(int i = 0; i < NumLeaves; i++ ){
			(leafFlowXBuffers[i])[x] = 0.0f;
			(leafFlowYBuffers[i])[x] = 0.0f;
			(leafFlowZBuffers[i])[x] = 0.0f;
			(leafWorkingBuffers[i])[x] = 0.0f;
		}
	}

	//create pointers to parent outflow
	float** leafIncBuffers = new float* [NumLeaves];
	float** branchIncBuffers = new float* [NumBranches];
	iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	iterator->Next();
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		vtkIdType parent = this->Hierarchy->GetParent(node);
		if( this->Hierarchy->IsLeaf(node) ){
			if( parent == this->Hierarchy->GetRoot() )
				leafIncBuffers[this->OutputPortMapping.find(node)->second] = sourceFlow;
			else
				leafIncBuffers[this->OutputPortMapping.find(node)->second] = branchSinkBuffers[this->IntermediateBufferMapping.find(parent)->second];
		}else{
			if( parent == this->Hierarchy->GetRoot() )
				branchIncBuffers[this->IntermediateBufferMapping.find(node)->second] = sourceFlow;
			else
				branchIncBuffers[this->IntermediateBufferMapping.find(node)->second] = branchSinkBuffers[this->IntermediateBufferMapping.find(parent)->second];
		}
	}
	iterator->Delete();
	

	//initalize all labels to most likely result from data prior at the leaf nodes
	for(int x = 0; x < VolumeSize; x++){
		float maxProbValue = FLT_MAX;
		int maxProbLabel = 0;
		for(int i = 0; i < NumLeaves; i++){
			maxProbLabel = (maxProbValue > (leafDataTermBuffers[i])[x]) ? i : maxProbLabel;
			maxProbValue = (maxProbValue > (leafDataTermBuffers[i])[x]) ? (leafDataTermBuffers[i])[x] : maxProbLabel; ;
		}
		for(int i = 0; i < NumLeaves; i++){
			(leafLabelBuffers[i])[x] = (i == maxProbLabel) ? 1.0f : 0.0f;
			(leafSinkBuffers[i])[x] = (leafDataTermBuffers[maxProbLabel])[x];
		}
		for(int i = 0; i < NumBranches; i++)
			(branchSinkBuffers[i])[x] = (leafDataTermBuffers[maxProbLabel])[x];
	}
	PropogateLabels( this->Hierarchy->GetRoot(), branchLabelBuffers, leafLabelBuffers, VolumeSize );

	//convert smoothness constants mapping to two mappings
	iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	float* leafSmoothnessConstants = new float[NumLeaves];
	float* branchSmoothnessConstants = new float[NumBranches];
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( node == this->Hierarchy->GetRoot() ) continue;
		if( this->Hierarchy->IsLeaf(node) )
			if( this->SmoothnessScalars.find(this->Hierarchy->GetParent(node)) != this->SmoothnessScalars.end() )
				leafSmoothnessConstants[this->OutputPortMapping.find(node)->second] = this->SmoothnessScalars.find(this->Hierarchy->GetParent(node))->second;
			else
				leafSmoothnessConstants[this->OutputPortMapping.find(node)->second] = 1.0f;
		else
			if( this->SmoothnessScalars.find(this->Hierarchy->GetParent(node)) != this->SmoothnessScalars.end() )
				branchSmoothnessConstants[this->IntermediateBufferMapping.find(node)->second] = this->SmoothnessScalars.find(this->Hierarchy->GetParent(node))->second;
			else
				branchSmoothnessConstants[this->IntermediateBufferMapping.find(node)->second] = 1.0f;
	}
	iterator->Delete();

	for( int iteration = 0; iteration < this->NumberOfIterations; iteration++ ){

		//compute the flow conservation error at each point and store in working buffer
		//before, working buffer held the divergence
		for(int i = 0; i < NumLeaves; i++ )
			for( int x = 0; x < VolumeSize; x++ )
				(leafWorkingBuffers[i])[x] = (leafWorkingBuffers[i])[x] - (leafIncBuffers[i])[x] + (leafSinkBuffers[i])[x] - (leafLabelBuffers[i])[x] / this->CC;
		for(int i = 0; i < NumBranches; i++ )
			for( int x = 0; x < VolumeSize; x++ )
				(branchWorkingBuffers[i])[x] = (branchWorkingBuffers[i])[x] - (branchIncBuffers[i])[x] + (branchSinkBuffers[i])[x] - (branchLabelBuffers[i])[x] / this->CC;

		//Do the gradient-descent based update of the spatial flows
		for(int i = 0; i < NumLeaves; i++ )
			for( int idx = 0, z = 0; z < Z; z++ ) for( int y = 0; y < Y; y++ ) for( int x = 0; x < X; x++, idx++ ){
				if( x == 0 || y == 0 || z == 0 ) continue;  
				(leafFlowXBuffers[i])[idx] += this->StepSize * ((leafWorkingBuffers[i])[idx] - (leafWorkingBuffers[i])[idx-1] );
				(leafFlowYBuffers[i])[idx] += this->StepSize * ((leafWorkingBuffers[i])[idx] - (leafWorkingBuffers[i])[idx-X] );
				(leafFlowZBuffers[i])[idx] += this->StepSize * ((leafWorkingBuffers[i])[idx] - (leafWorkingBuffers[i])[idx-X*Y] );
			}
		for(int i = 0; i < NumBranches; i++ )
			for( int idx = 0, z = 0; z < Z; z++ ) for( int y = 0; y < Y; y++ ) for( int x = 0; x < X; x++, idx++ ){
				if( x == 0 || y == 0 || z == 0 ) continue;  
				(branchFlowXBuffers[i])[idx]	+= this->StepSize * ((branchWorkingBuffers[i])[idx]	-(branchWorkingBuffers[i])[idx-1] );
				(branchFlowYBuffers[i])[idx]	+= this->StepSize * ((branchWorkingBuffers[i])[idx]	-(branchWorkingBuffers[i])[idx-X] );
				(branchFlowZBuffers[i])[idx]	+= this->StepSize * ((branchWorkingBuffers[i])[idx]	-(branchWorkingBuffers[i])[idx-X*Y] );
			}

		//calculate the magnitude of spatial flows through each voxel and place this magnitude in the working buffer
		for(int i = 0; i < NumLeaves; i++ )
			for( int idx = 0, z = 0; z < Z; z++ ) for( int y = 0; y < Y; y++ ) for( int x = 0; x < X; x++, idx++ ){
				float upX = (x == X-1) ? 0.0f : (leafFlowXBuffers[i])[idx+1];
				float upY = (y == Y-1) ? 0.0f : (leafFlowYBuffers[i])[idx+X];
				float upZ = (z == Z-1) ? 0.0f : (leafFlowZBuffers[i])[idx+X*Y];
				(leafWorkingBuffers[i])[x] = sqrt(0.5f *	(SQR(upX) + SQR((leafFlowXBuffers[i])[x]) +
																 SQR(upY) + SQR((leafFlowYBuffers[i])[x]) +
																 SQR(upZ) + SQR((leafFlowZBuffers[i])[x]) ));
			}
		for(int i = 0; i < NumBranches; i++ )
			for( int idx = 0, z = 0; z < Z; z++ ) for( int y = 0; y < Y; y++ ) for( int x = 0; x < X; x++, idx++ ){
				float upX = (x == X-1) ? 0.0f : (branchFlowXBuffers[i])[idx+1];
				float upY = (y == Y-1) ? 0.0f : (branchFlowYBuffers[i])[idx+X];
				float upZ = (z == Z-1) ? 0.0f : (branchFlowZBuffers[i])[idx+X*Y];
				(branchWorkingBuffers[i])[x] = sqrt(0.5f * (SQR(upX) + SQR((branchFlowXBuffers[i])[x]) +
																 SQR(upY) + SQR((branchFlowYBuffers[i])[x]) +
																 SQR(upZ) + SQR((branchFlowZBuffers[i])[x]) ));
			}

		//calculate the multiplying for bringing it to within the specified bound (smoothness term)
		for(int i = 0; i < NumLeaves; i++ )
			for( int x = 0; x < VolumeSize; x++ ){
				if( (leafWorkingBuffers[i])[x] > leafSmoothnessConstants[i] * (leafSmoothnessTermBuffers[i] ? (leafSmoothnessTermBuffers[i])[x] : 1.0f) )
					(leafWorkingBuffers[i])[x] = leafSmoothnessConstants[i] * (leafSmoothnessTermBuffers[i] ? (leafSmoothnessTermBuffers[i])[x] : 1.0f) / (leafWorkingBuffers[i])[x];
				else
					(leafWorkingBuffers[i])[x] = 1.0f;
			}
		for(int i = 0; i < NumBranches; i++ )
			for( int x = 0; x < VolumeSize; x++ ){
				if( (branchWorkingBuffers[i])[x] > branchSmoothnessConstants[i] * (branchSmoothnessTermBuffers[i] ? (branchSmoothnessTermBuffers[i])[x] : 1.0f) )
					(branchWorkingBuffers[i])[x] = branchSmoothnessConstants[i] * (branchSmoothnessTermBuffers[i] ? (branchSmoothnessTermBuffers[i])[x] : 1.0f) / (branchWorkingBuffers[i])[x];
				else
					(branchWorkingBuffers[i])[x] = 1.0f;
			}

		//adjust the spatial flows by the multipliers
		for(int i = 0; i < NumLeaves; i++ )
			for( int idx = 0, z = 0; z < Z; z++ ) for( int y = 0; y < Y; y++ ) for( int x = 0; x < X; x++, idx++ ){
				if( x == X-1 || y == Y-1 || z == Z-1 ) continue;  
				(leafFlowXBuffers[i])[idx+1] *= 0.5f * ((leafWorkingBuffers[i])[idx] + (leafWorkingBuffers[i])[idx+1]);
				(leafFlowYBuffers[i])[idx+X] *= 0.5f * ((leafWorkingBuffers[i])[idx] + (leafWorkingBuffers[i])[idx+X]);
				(leafFlowZBuffers[i])[idx+X*Y] *= 0.5f * ((leafWorkingBuffers[i])[idx] + (leafWorkingBuffers[i])[idx+X*Y]);
			}
		for(int i = 0; i < NumBranches; i++ )
			for( int idx = 0, z = 0; z < Z; z++ ) for( int y = 0; y < Y; y++ ) for( int x = 0; x < X; x++, idx++ ){
				if( x == X-1 || y == Y-1 || z == Z-1 ) continue;  
				(branchFlowXBuffers[i])[idx+1] *= 0.5f * ((branchWorkingBuffers[i])[idx] + (branchWorkingBuffers[i])[idx+1]);
				(branchFlowYBuffers[i])[idx+X] *= 0.5f * ((branchWorkingBuffers[i])[idx] + (branchWorkingBuffers[i])[idx+X]);
				(branchFlowZBuffers[i])[idx+X*Y] *= 0.5f * ((branchWorkingBuffers[i])[idx] + (branchWorkingBuffers[i])[idx+X*Y]);
			}

		//calculate the divergence since the source flows won't change for the rest of this iteration
		for(int i = 0; i < NumLeaves; i++ )
			for( int x = 0; x < VolumeSize; x++ )
				(leafWorkingBuffers[i])[x] = (leafFlowXBuffers[i])[x+1] - (leafFlowXBuffers[i])[x] +
										(leafFlowYBuffers[i])[x+X] - (leafFlowYBuffers[i])[x] +
										(leafFlowZBuffers[i])[x+X*Y] - (leafFlowZBuffers[i])[x];
		for(int i = 0; i < NumBranches; i++ )
			for( int x = 0; x < VolumeSize; x++ )
				(branchWorkingBuffers[i])[x] = (branchFlowXBuffers[i])[x+1] - (branchFlowXBuffers[i])[x] +
										 (branchFlowYBuffers[i])[x+X] - (branchFlowYBuffers[i])[x] +
										 (branchFlowZBuffers[i])[x+X*Y] - (branchFlowZBuffers[i])[x];
		
		//update the leaf sink flows
		for(int i = 0; i < NumLeaves; i++ ){
			for( int x = 0; x < VolumeSize; x++ ){
				float potential = (leafIncBuffers[i])[x] - (leafWorkingBuffers[i])[x] + (leafLabelBuffers[i])[x] / this->CC;
				(leafSinkBuffers[i])[x] = std::max( 0.0f, std::min( potential, (leafDataTermBuffers[i])[x] ) );  
			}
		}

		//compute sink flow (store in working buffer) and update source flows bottom up
		PropogateFlows( this->Hierarchy->GetRoot(), sourceFlow, branchSinkBuffers, leafSinkBuffers, branchIncBuffers,
						branchWorkingBuffers, leafWorkingBuffers, branchLabelBuffers, leafLabelBuffers, VolumeSize );
		
		//update labels (multipliers) at the leaves
		for(int i = 0; i < NumLeaves; i++ )
			for( int x = 0; x < VolumeSize; x++ ){
				(leafLabelBuffers[i])[x] -= this->CC * ((leafWorkingBuffers[i])[x] - (leafIncBuffers[i])[x] + (leafSinkBuffers[i])[x]);
				(leafLabelBuffers[i])[x] = std::min(1.0f, std::max( 0.0f, (leafLabelBuffers[i])[x] ) );
			}
		
		//update labels (multipliers) at the branches (incrementally, not push-up)
		for(int i = 0; i < NumBranches; i++ )
			for( int x = 0; x < VolumeSize; x++ ){
				(branchLabelBuffers[i])[x] -= this->CC * ((branchWorkingBuffers[i])[x] - (branchIncBuffers[i])[x] + (branchSinkBuffers[i])[x]);
				(branchLabelBuffers[i])[x] = std::min(1.0f, std::max( 0.0f, (branchLabelBuffers[i])[x] ) );
			}
	}
	
	//deallocate branch temporary buffers
	delete[] branchNodeBuffer;
	delete[] branchFlowXBuffers;
	delete[] branchFlowYBuffers;
	delete[] branchFlowZBuffers;
	delete[] branchIncBuffers;
	delete[] branchLabelBuffers;
	delete[] branchWorkingBuffers;
	delete[] branchSmoothnessTermBuffers;
	delete[] branchSmoothnessConstants;

	//deallocate leaf temporary buffers
	delete[] leafNodeBuffer;
	delete[] leafFlowXBuffers;
	delete[] leafFlowYBuffers;
	delete[] leafFlowZBuffers;
	delete[] leafIncBuffers;
	delete[] leafSinkBuffers;
	delete[] leafWorkingBuffers;
	delete[] leafLabelBuffers;
	delete[] leafDataTermBuffers;
	delete[] leafSmoothnessTermBuffers;
	delete[] leafSmoothnessConstants;

	return 1;
}


int vtkHierarchicalMaxFlowSegmentation::RequestDataObject(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector ,
  vtkInformationVector* outputVector){

	vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
	if (!inInfo)
		return 0;
	vtkImageData *input = vtkImageData::SafeDownCast(inInfo->Get(vtkImageData::DATA_OBJECT()));
 
	if (input) {
		std::cout << "There are " << outputVector->GetNumberOfInformationObjects() << " output info objects" << std::endl;
		for(int i=0; i < outputVector->GetNumberOfInformationObjects(); ++i) {
			vtkInformation* info = outputVector->GetInformationObject(0);
			vtkDataSet *output = vtkDataSet::SafeDownCast(
			info->Get(vtkDataObject::DATA_OBJECT()));
 
			if (!output || !output->IsA(input->GetClassName())) {
				vtkImageData* newOutput = input->NewInstance();
				newOutput->SetPipelineInformation(info);
				newOutput->Delete();
			}
			return 1;
		}
	}
	return 0;
}
