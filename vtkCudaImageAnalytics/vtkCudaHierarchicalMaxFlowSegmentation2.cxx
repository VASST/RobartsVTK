#include "vtkCudaHierarchicalMaxFlowSegmentation2.h"
#include "vtkCudaHierarchicalMaxFlowSegmentation2Task.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTreeDFSIterator.h"

#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include <set>
#include <list>

#include "CUDA_hierarchicalmaxflow.h"
#include "vtkCudaDeviceManager.h"
#include "vtkCudaObject.h"

#define SQR(X) X*X

vtkStandardNewMacro(vtkCudaHierarchicalMaxFlowSegmentation2);

vtkCudaHierarchicalMaxFlowSegmentation2::vtkCudaHierarchicalMaxFlowSegmentation2(){
	
	//configure the IO ports
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(0);

	//set algorithm mathematical parameters to defaults
	this->NumberOfIterations = 100;
	this->StepSize = 0.1;
	this->CC = 0.25;
	this->MaxGPUUsage = 0.75;

	//set up the input mapping structure
	this->InputPortMapping.clear();
	this->BackwardsInputPortMapping.clear();
	this->FirstUnusedPort = 0;

	//set the other values to defaults
	this->Hierarchy = 0;
	this->SmoothnessScalars.clear();
	this->LeafMap.clear();
	this->InputPortMapping.clear();
	this->BranchMap.clear();

}

vtkCudaHierarchicalMaxFlowSegmentation2::~vtkCudaHierarchicalMaxFlowSegmentation2(){
	if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
	this->SmoothnessScalars.clear();
	this->LeafMap.clear();
	this->InputPortMapping.clear();
	this->BackwardsInputPortMapping.clear();
	this->BranchMap.clear();

}

//------------------------------------------------------------

void vtkCudaHierarchicalMaxFlowSegmentation2::AddGPU(int GPU){
	if( GPU > 0 && GPU < vtkCudaDeviceManager::Singleton()->GetNumberOfDevices() )
		this->GPUsUsed.insert(GPU);
}

void vtkCudaHierarchicalMaxFlowSegmentation2::RemoveGPU(int GPU){
	if( this->GPUsUsed.find(GPU) != this->GPUsUsed.end() )
		this->GPUsUsed.erase(this->GPUsUsed.find(GPU));
}
//------------------------------------------------------------

void vtkCudaHierarchicalMaxFlowSegmentation2::SetHierarchy(vtkTree* graph){
	if( graph != this->Hierarchy ){
		if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
		this->Hierarchy = graph;
		if( this->Hierarchy ) this->Hierarchy->Register(this);
		this->Modified();

		//update output mapping
		this->LeafMap.clear();
		vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
		iterator->SetTree(this->Hierarchy);
		iterator->SetStartVertex(this->Hierarchy->GetRoot());
		while( iterator->HasNext() ){
			vtkIdType node = iterator->Next();
			if( this->Hierarchy->IsLeaf(node) )
				this->LeafMap.insert(std::pair<vtkIdType,int>(node,(int) this->LeafMap.size()));
		}
		iterator->Delete();

		//update number of output ports
		this->SetNumberOfOutputPorts((int) this->LeafMap.size());

	}
}

vtkTree* vtkCudaHierarchicalMaxFlowSegmentation2::GetHierarchy(){
	return this->Hierarchy;
}

//------------------------------------------------------------

void vtkCudaHierarchicalMaxFlowSegmentation2::AddSmoothnessScalar(vtkIdType node, double value){
	if( value >= 0.0 ){
		this->SmoothnessScalars.insert(std::pair<vtkIdType,double>(node,value));
		this->Modified();
	}else{
		vtkErrorMacro(<<"Cannot use a negative smoothness value.");
	}
}

//------------------------------------------------------------

int vtkCudaHierarchicalMaxFlowSegmentation2::FillInputPortInformation(int i, vtkInformation* info){
	info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
	info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
	info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
	return this->Superclass::FillInputPortInformation(i,info);
}

void vtkCudaHierarchicalMaxFlowSegmentation2::SetInput(int idx, vtkDataObject *input)
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

vtkDataObject *vtkCudaHierarchicalMaxFlowSegmentation2::GetInput(int idx)
{
	if( this->InputPortMapping.find(idx) == this->InputPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputPortMapping.find(idx)->second));
}

vtkDataObject *vtkCudaHierarchicalMaxFlowSegmentation2::GetOutput(int idx)
{
	//look up port in mapping
	std::map<vtkIdType,int>::iterator port = this->LeafMap.find(idx);
	if( port == this->LeafMap.end() )
		return 0;

	return vtkImageData::SafeDownCast(this->GetExecutive()->GetOutputData(port->second));
}

//----------------------------------------------------------------------------

int vtkCudaHierarchicalMaxFlowSegmentation2::CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges ){
	
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

int vtkCudaHierarchicalMaxFlowSegmentation2::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	//check input for consistancy
	int Extent[6];
	int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
	if( result || NumNodes == 0 ) return -1;
	
	//set the number of output ports
	outputVector->SetNumberOfInformationObjects(NumLeaves);
	this->SetNumberOfOutputPorts(NumLeaves);

	return 1;
}

int vtkCudaHierarchicalMaxFlowSegmentation2::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	//check input for consistancy
	int Extent[6];
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

//-----------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------//

int vtkCudaHierarchicalMaxFlowSegmentation2::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){
								
	//check input consistancy
	int Extent[6];
	int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
	if( result || NumLeaves < 2 ) return -1;
	NumBranches = NumNodes - NumLeaves - 1;

	if( this->Debug )
		vtkDebugMacro(<< "Starting input data preparation." );

	//set the number of output ports
	outputVector->SetNumberOfInformationObjects(NumLeaves);
	this->SetNumberOfOutputPorts(NumLeaves);

	//find the size of the volume
	VX = (Extent[1] - Extent[0] + 1);
	VY = (Extent[3] - Extent[2] + 1);
	VZ = (Extent[5] - Extent[4] + 1);
	VolumeSize = VX * VY * VZ;

	//make a container for the total number of memory buffers
	TotalNumberOfBuffers = 0;

	//create relevant node identifier to buffer mappings
	this->BranchMap.clear();
	vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	iterator->SetStartVertex(this->Hierarchy->GetRoot());
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( node == this->Hierarchy->GetRoot() ) continue;
		if( !this->Hierarchy->IsLeaf(node) )
			BranchMap.insert(std::pair<vtkIdType,int>(node,(int) this->BranchMap.size()));
	}
	iterator->Delete();

	//get the data term buffers
	this->leafDataTermBuffers = new float* [NumLeaves];
	iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( this->Hierarchy->IsLeaf(node) ){
			int inputNumber = this->InputPortMapping[node];
			vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputNumber)->Get(vtkDataObject::DATA_OBJECT()));
			leafDataTermBuffers[this->LeafMap[node]] = (float*) CurrImage->GetScalarPointer();

			//add the data term buffer in and set it to read only
			TotalNumberOfBuffers++;
			ReadOnly.insert( (float*) CurrImage->GetScalarPointer());

		}
	}
	iterator->Delete();
	
	//get the smoothness term buffers
	this->leafSmoothnessTermBuffers = new float* [NumLeaves];
	this->branchSmoothnessTermBuffers = new float* [NumBranches];
	iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( node == this->Hierarchy->GetRoot() ) continue;
		vtkImageData* CurrImage = 0;
		if( this->InputPortMapping.find(this->Hierarchy->GetParent(node)) != this->InputPortMapping.end() ){
			int parentInputNumber = this->InputPortMapping[this->Hierarchy->GetParent(node)];
			CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(parentInputNumber)->Get(vtkDataObject::DATA_OBJECT()));
		}
		if( this->Hierarchy->IsLeaf(node) )
			leafSmoothnessTermBuffers[this->LeafMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
		else
			branchSmoothnessTermBuffers[this->BranchMap.find(node)->second] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
		
		// add the smoothness buffer in as read only
		if( CurrImage ){
			TotalNumberOfBuffers++;
			ReadOnly.insert( (float*) CurrImage->GetScalarPointer());
		}
	}
	iterator->Delete();

	//get the output buffers
	this->leafLabelBuffers = new float* [NumLeaves];
	for(int i = 0; i < NumLeaves; i++ ){
		vtkInformation *outputInfo = outputVector->GetInformationObject(i);
		vtkImageData *outputBuffer = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
		outputBuffer->SetExtent(Extent);
		outputBuffer->Modified();
		outputBuffer->AllocateScalars();
		leafLabelBuffers[i] = (float*) outputBuffer->GetScalarPointer();
		TotalNumberOfBuffers++;
	}
	
	//convert smoothness constants mapping to two mappings
	iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	leafSmoothnessConstants = new float[NumLeaves];
	branchSmoothnessConstants = new float[NumBranches];
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( node == this->Hierarchy->GetRoot() ) continue;
		if( this->Hierarchy->IsLeaf(node) )
			if( this->SmoothnessScalars.find(this->Hierarchy->GetParent(node)) != this->SmoothnessScalars.end() )
				leafSmoothnessConstants[this->LeafMap.find(node)->second] = this->SmoothnessScalars.find(this->Hierarchy->GetParent(node))->second;
			else
				leafSmoothnessConstants[this->LeafMap.find(node)->second] = 1.0f;
		else
			if( this->SmoothnessScalars.find(this->Hierarchy->GetParent(node)) != this->SmoothnessScalars.end() )
				branchSmoothnessConstants[this->BranchMap.find(node)->second] = this->SmoothnessScalars.find(this->Hierarchy->GetParent(node))->second;
			else
				branchSmoothnessConstants[this->BranchMap.find(node)->second] = 1.0f;
	}
	iterator->Delete();
	
	//if verbose, print progress
	if( this->Debug ){
		vtkDebugMacro(<<"Starting CPU buffer acquisition");
	}

	//source flow and working buffers
	std::list<float**> BufferPointerLocs;
	int NumberOfAdditionalCPUBuffersNeeded = 2;
	TotalNumberOfBuffers += 2;
	BufferPointerLocs.push_front(&sourceFlowBuffer);
	BufferPointerLocs.push_front(&sourceWorkingBuffer);

	//allocate required memory buffers for the brach nodes
	//		buffers needed:
	//			per brach node (not the root):
	//				1 label buffer
	//				3 spatial flow buffers
	//				1 divergence buffer
	//				1 outgoing flow buffer
	//				1 working temp buffer (ie: guk)
	//and for the leaf nodes
	//			per leaf node (note label buffer is in output)
	//				3 spatial flow buffers
	//				1 divergence buffer
	//				1 sink flow buffer
	//				1 working temp buffer (ie: guk)
	NumberOfAdditionalCPUBuffersNeeded += 7 * NumBranches;
	TotalNumberOfBuffers += 7*NumBranches;
	NumberOfAdditionalCPUBuffersNeeded += 5 * NumLeaves;
	TotalNumberOfBuffers += 5 * NumLeaves;

	//allocate those buffer pointers and put on list
	float** bufferPointers = new float* [7 * NumBranches + 5 * NumLeaves];
	float** tempPtr = bufferPointers;
	this->branchFlowXBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchFlowYBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchFlowZBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchDivBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchSinkBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchLabelBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchWorkingBuffers =	tempPtr; tempPtr += NumBranches;
	for(int i = 0; i < NumBranches; i++ )
		BufferPointerLocs.push_front(&(branchFlowXBuffers[i]));
	for(int i = 0; i < NumBranches; i++ )
		BufferPointerLocs.push_front(&(branchFlowYBuffers[i]));
	for(int i = 0; i < NumBranches; i++ )
		BufferPointerLocs.push_front(&(branchFlowZBuffers[i]));
	for(int i = 0; i < NumBranches; i++ )
		BufferPointerLocs.push_front(&(branchDivBuffers[i]));
	for(int i = 0; i < NumBranches; i++ )
		BufferPointerLocs.push_front(&(branchSinkBuffers[i]));
	for(int i = 0; i < NumBranches; i++ )
		BufferPointerLocs.push_front(&(branchLabelBuffers[i]));
	for(int i = 0; i < NumBranches; i++ )
		BufferPointerLocs.push_front(&(branchWorkingBuffers[i]));
	this->leafFlowXBuffers =		tempPtr; tempPtr += NumLeaves;
	this->leafFlowYBuffers =		tempPtr; tempPtr += NumLeaves;
	this->leafFlowZBuffers =		tempPtr; tempPtr += NumLeaves;
	this->leafDivBuffers =			tempPtr; tempPtr += NumLeaves;
	this->leafSinkBuffers =			tempPtr; tempPtr += NumLeaves;
	for(int i = 0; i < NumLeaves; i++ )
		BufferPointerLocs.push_front(&(leafFlowXBuffers[i]));
	for(int i = 0; i < NumLeaves; i++ )
		BufferPointerLocs.push_front(&(leafFlowYBuffers[i]));
	for(int i = 0; i < NumLeaves; i++ )
		BufferPointerLocs.push_front(&(leafFlowZBuffers[i]));
	for(int i = 0; i < NumLeaves; i++ )
		BufferPointerLocs.push_front(&(leafDivBuffers[i]));
	for(int i = 0; i < NumLeaves; i++ )
		BufferPointerLocs.push_front(&(leafSinkBuffers[i]));

	//try to obtain required CPU buffers
	std::list<float*> CPUBuffersAcquired;
	std::list<int> CPUBuffersSize;
	while( NumberOfAdditionalCPUBuffersNeeded > 0 ){
		int NumBuffersAcquired = (NumberOfAdditionalCPUBuffersNeeded < INT_MAX / VolumeSize) ?
			NumberOfAdditionalCPUBuffersNeeded : INT_MAX / VolumeSize;
		for( ; NumBuffersAcquired > 0; NumBuffersAcquired--){
			try{
				float* NewCPUBuffer = new float[VolumeSize*NumBuffersAcquired];
				if( !NewCPUBuffer ) continue;
				CPUBuffersAcquired.push_front( NewCPUBuffer );
				CPUBuffersSize.push_front( NumBuffersAcquired );
				NumberOfAdditionalCPUBuffersNeeded -= NumBuffersAcquired;
				break;
			} catch( ... ) { };
		}
		if( NumBuffersAcquired == 0 ) break;
	}

	//if we cannot obtain all required buffers, return an error and exit
	if( NumberOfAdditionalCPUBuffersNeeded > 0 ){
		while( CPUBuffersAcquired.size() > 0 ){
			float* tempBuffer = CPUBuffersAcquired.front();
			delete[] tempBuffer;
			CPUBuffersAcquired.pop_front();
		}
		vtkErrorMacro(<<"Not enough CPU memory. Cannot run algorithm.");
		return -1;
	}

	//put buffer pointers into given structures
	std::list<float**>::iterator bufferNameIt = BufferPointerLocs.begin();
	std::list<float*>::iterator bufferAllocIt = CPUBuffersAcquired.begin();
	std::list<int>::iterator bufferSizeIt = CPUBuffersSize.begin();
	for( ; bufferAllocIt != CPUBuffersAcquired.end(); bufferAllocIt++, bufferSizeIt++ ){
		for( int i = 0; i < *bufferSizeIt; i++ ){
			*(*bufferNameIt) = (*bufferAllocIt) + VolumeSize*i;
			bufferNameIt++;
		}
	}
	
	//if verbose, print progress
	if( this->Debug ){
		vtkDebugMacro(<<"Relate parent sink with child source buffer pointers.");
	}

	//create pointers to parent outflow
	leafIncBuffers = new float* [NumLeaves];
	branchIncBuffers = new float* [NumBranches];
	iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	iterator->Next();
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		vtkIdType parent = this->Hierarchy->GetParent(node);
		if( this->Hierarchy->IsLeaf(node) ){
			if( parent == this->Hierarchy->GetRoot() )
				leafIncBuffers[this->LeafMap.find(node)->second] = sourceFlowBuffer;
			else
				leafIncBuffers[this->LeafMap.find(node)->second] = branchSinkBuffers[this->BranchMap.find(parent)->second];
		}else{
			if( parent == this->Hierarchy->GetRoot() )
				branchIncBuffers[this->BranchMap.find(node)->second] = sourceFlowBuffer;
			else
				branchIncBuffers[this->BranchMap.find(node)->second] = branchSinkBuffers[this->BranchMap.find(parent)->second];
		}
	}
	iterator->Delete();
	
	//if verbose, print progress
	if( this->Debug ){
		vtkDebugMacro(<<"Building workers.");
	}
	for(std::set<int>::iterator gpuIterator = GPUsUsed.begin(); gpuIterator != GPUsUsed.end(); gpuIterator++)
		this->Workers.insert( new Worker(*gpuIterator,this) );

	//if verbose, print progress
	if( this->Debug ){
		vtkDebugMacro(<<"Find priority structures.");
	}

	//create LIFO priority queue (priority stack) data structure
	FigureOutBufferPriorities( this->Hierarchy->GetRoot() );
	
	//add all the working buffers from the branches to the garbage (no copy necessary) list
	NoCopyBack.insert( sourceWorkingBuffer );
	for(int i = 0; i < NumBranches; i++ )
		NoCopyBack.insert( branchWorkingBuffers[i] );
	
	
	//if verbose, print progress
	if( this->Debug ){
		vtkDebugMacro(<<"Starting initialization");
	}
	NumMemCpies = 0;

	//initialize solution
	//initalize all spatial flows and divergences to zero

	//initialize all leak sink flows to their constraints

	//find the minimum sink flow

	//copy minimum sink flow over all leaves and sum the resulting labels into the source working buffer

	//divide the labels out to constrain them to validity

	//apply minimal sink flow over the remaining hierarchy

	//propogate labels up the hierarchy

	//synchronize works

	if( this->Debug )
		vtkDebugMacro(<< "Finished initialization with a total of " << NumMemCpies << " memory transfers.");

	//Solve maximum flow problem in an iterative bottom-up manner
	if( this->Debug )
		vtkDebugMacro(<<"Starting max-flow iterations.");
	NumMemCpies = 0;
	NumKernelRuns = 0;

	//TODO organize iterations

	if( this->Debug )
		vtkDebugMacro(<< "Finished all iterations with a total of " << NumMemCpies << " memory transfers.");
	
	//remove workers
	if( this->Debug )
		vtkDebugMacro(<< "Deallocating workers" );
	for(std::set<Worker*>::iterator workerIterator = Workers.begin(); workerIterator != Workers.end(); workerIterator++)
		delete *workerIterator;
	Workers.clear();

	//deallocate CPU buffers
	if( this->Debug )
		vtkDebugMacro(<< "Deallocating CPU buffers" );
	while( CPUBuffersAcquired.size() > 0 ){
		float* tempBuffer = CPUBuffersAcquired.front();
		delete[] tempBuffer;
		CPUBuffersAcquired.pop_front();
	}

	//deallocate structure that holds the pointers to the buffers
	delete[] bufferPointers;

	return 1;
}

void vtkCudaHierarchicalMaxFlowSegmentation2::FigureOutBufferPriorities( vtkIdType currNode ){
	
	//Propogate down the tree
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int kid = 0; kid < NumKids; kid++)
		FigureOutBufferPriorities( this->Hierarchy->GetChild(currNode,kid) );

	//if we are the root, figure out the buffers
	if( this->Hierarchy->GetRoot() == currNode ){
		this->CPU2PriorityMap.insert(std::pair<float*,int>(sourceFlowBuffer,NumKids+2));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(sourceWorkingBuffer,NumKids+3));

	//if we are a leaf, handle separately
	}else if( NumKids == 0 ){
		int Number = LeafMap.find(currNode)->second;
		this->CPU2PriorityMap.insert(std::pair<float*,int>(leafDivBuffers[Number],3));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(leafFlowXBuffers[Number],2));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(leafFlowYBuffers[Number],2));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(leafFlowZBuffers[Number],2));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(leafSinkBuffers[Number],3));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(leafDataTermBuffers[Number],1));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(leafLabelBuffers[Number],3));
		if( leafSmoothnessTermBuffers[Number] )
			this->CPU2PriorityMap[leafSmoothnessTermBuffers[Number]]++;

	//else, we are a branch
	}else{
		int Number = BranchMap.find(currNode)->second;
		this->CPU2PriorityMap.insert(std::pair<float*,int>(branchDivBuffers[Number],3));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(branchFlowXBuffers[Number],2));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(branchFlowYBuffers[Number],2));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(branchFlowZBuffers[Number],2));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(branchSinkBuffers[Number],NumKids+4));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(branchLabelBuffers[Number],3));
		this->CPU2PriorityMap.insert(std::pair<float*,int>(branchWorkingBuffers[Number],NumKids+3));
		if( branchSmoothnessTermBuffers[Number] )
			this->CPU2PriorityMap[branchSmoothnessTermBuffers[Number]]++;
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation2::ReturnBufferGPU2CPU(float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream){
	if( ReadOnly.find(CPUBuffer) != ReadOnly.end() ) return;
	if( NoCopyBack.find(CPUBuffer) != NoCopyBack.end() ) return;
	CUDA_CopyBufferToCPU( GPUBuffer, CPUBuffer, VolumeSize, stream);
	NumMemCpies++;
}

void vtkCudaHierarchicalMaxFlowSegmentation2::MoveBufferCPU2GPU(float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream){
	if( NoCopyBack.find(CPUBuffer) != NoCopyBack.end() ) return;
	CUDA_CopyBufferToGPU( GPUBuffer, CPUBuffer, VolumeSize, stream);
	NumMemCpies++;
}

int vtkCudaHierarchicalMaxFlowSegmentation2::RequestDataObject(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector ,
  vtkInformationVector* outputVector){

	vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
	if (!inInfo)
		return 0;
	vtkImageData *input = vtkImageData::SafeDownCast(inInfo->Get(vtkImageData::DATA_OBJECT()));
 
	if (input) {
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