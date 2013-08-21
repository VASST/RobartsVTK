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
#include <vector>

#include "CUDA_hierarchicalmaxflow.h"
#include "vtkCudaDeviceManager.h"
#include "vtkCudaObject.h"

#define SQR(X) X*X

vtkStandardNewMacro(vtkCudaHierarchicalMaxFlowSegmentation2);

vtkCudaHierarchicalMaxFlowSegmentation2::vtkCudaHierarchicalMaxFlowSegmentation2(){
	
	//configure the IO ports
	this->SetNumberOfInputPorts(2);
	this->SetNumberOfOutputPorts(0);

	//set algorithm mathematical parameters to defaults
	this->NumberOfIterations = 100;
	this->StepSize = 0.1;
	this->CC = 0.25;
	this->MaxGPUUsage = 0.75;
	this->ReportRate = 100;

	//set up the input mapping structure
	this->InputDataPortMapping.clear();
	this->BackwardsInputDataPortMapping.clear();
	this->FirstUnusedDataPort = 0;
	this->InputSmoothnessPortMapping.clear();
	this->BackwardsInputSmoothnessPortMapping.clear();
	this->FirstUnusedSmoothnessPort = 0;

	//set the other values to defaults
	this->Hierarchy = 0;
	this->SmoothnessScalars.clear();
	this->LeafMap.clear();
	this->BranchMap.clear();

	//give default GPU selection
	this->GPUsUsed.insert(0);

}

vtkCudaHierarchicalMaxFlowSegmentation2::~vtkCudaHierarchicalMaxFlowSegmentation2(){
	if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
	this->SmoothnessScalars.clear();
	this->LeafMap.clear();
	this->InputDataPortMapping.clear();
	this->BackwardsInputDataPortMapping.clear();
	this->InputSmoothnessPortMapping.clear();
	this->BackwardsInputSmoothnessPortMapping.clear();
	this->BranchMap.clear();
	this->GPUsUsed.clear();
	this->MaxGPUUsageNonDefault.clear();
}

//------------------------------------------------------------//

void vtkCudaHierarchicalMaxFlowSegmentation2::AddDevice(int GPU){
	if( GPU >= 0 && GPU < vtkCudaDeviceManager::Singleton()->GetNumberOfDevices() )
		this->GPUsUsed.insert(GPU);
}

void vtkCudaHierarchicalMaxFlowSegmentation2::RemoveDevice(int GPU){
	if( this->GPUsUsed.find(GPU) != this->GPUsUsed.end() )
		this->GPUsUsed.erase(this->GPUsUsed.find(GPU));
}

bool vtkCudaHierarchicalMaxFlowSegmentation2::HasDevice(int GPU){
	return (this->GPUsUsed.find(GPU) != this->GPUsUsed.end());
}
void vtkCudaHierarchicalMaxFlowSegmentation2::ClearDevices(){
	this->GPUsUsed.clear();
}

void vtkCudaHierarchicalMaxFlowSegmentation2::SetMaxGPUUsage(double usage, int device){
	if( usage < 0.0 ) usage = 0.0;
	else if( usage > 1.0 ) usage = 1.0;
	if( device >= 0 && device < vtkCudaDeviceManager::Singleton()->GetNumberOfDevices() )
		this->MaxGPUUsageNonDefault[device] = usage;
}

double vtkCudaHierarchicalMaxFlowSegmentation2::GetMaxGPUUsage(int device){
	if( this->MaxGPUUsageNonDefault.find(device) != this->MaxGPUUsageNonDefault.end() )
		return this->MaxGPUUsageNonDefault[device];
	return this->MaxGPUUsage;
}

void vtkCudaHierarchicalMaxFlowSegmentation2::ClearMaxGPUUsage(){
	this->MaxGPUUsageNonDefault.clear();
}
//------------------------------------------------------------//

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
		this->SmoothnessScalars[node] = value;
		this->Modified();
	}else{
		this->SmoothnessScalars[node] = 0.0;
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

void vtkCudaHierarchicalMaxFlowSegmentation2::SetDataInput(int idx, vtkDataObject *input)
{
	//we are adding/switching an input, so no need to resort list
	if( input != NULL ){
	
		//if their is no pair in the mapping, create one
		if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() ){
			int DataPortNumber = this->FirstUnusedDataPort;
			this->FirstUnusedDataPort++;
			this->InputDataPortMapping.insert(std::pair<vtkIdType,int>(idx,DataPortNumber));
			this->BackwardsInputDataPortMapping.insert(std::pair<vtkIdType,int>(DataPortNumber,idx));
		}
		this->SetNthInputConnection(0, this->InputDataPortMapping[idx], input->GetProducerPort() );

	}else{
		//if their is no pair in the mapping, just exit, nothing to do
		if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() ) return;

		int DataPortNumber = this->InputDataPortMapping[idx];
		this->InputDataPortMapping.erase(this->InputDataPortMapping.find(idx));
		this->BackwardsInputDataPortMapping.erase(this->BackwardsInputDataPortMapping.find(DataPortNumber));

		//if we are the last input, no need to reshuffle
		if(DataPortNumber == this->FirstUnusedDataPort - 1){
			this->SetNthInputConnection(0, DataPortNumber,  0);
		
		//if we are not, move the last input into this spot
		}else{
			vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedDataPort - 1));
			this->SetNthInputConnection(0, DataPortNumber, swappedInput->GetProducerPort() );
			this->SetNthInputConnection(0, this->FirstUnusedDataPort - 1, 0 );

			//correct the mappings
			vtkIdType swappedId = this->BackwardsInputDataPortMapping[this->FirstUnusedDataPort - 1];
			this->InputDataPortMapping.erase(this->InputDataPortMapping.find(swappedId));
			this->BackwardsInputDataPortMapping.erase(this->BackwardsInputDataPortMapping.find(this->FirstUnusedDataPort - 1));
			this->InputDataPortMapping.insert(std::pair<vtkIdType,int>(swappedId,DataPortNumber) );
			this->BackwardsInputDataPortMapping.insert(std::pair<int,vtkIdType>(DataPortNumber,swappedId) );

		}

		//decrement the number of unused DataPorts
		this->FirstUnusedDataPort--;

	}
}

void vtkCudaHierarchicalMaxFlowSegmentation2::SetSmoothnessInput(int idx, vtkDataObject *input)
{
	//we are adding/switching an input, so no need to resort list
	if( input != NULL ){
	
		//if their is no pair in the mapping, create one
		if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() ){
			int SmoothnessPortNumber = this->FirstUnusedSmoothnessPort;
			this->FirstUnusedSmoothnessPort++;
			this->InputSmoothnessPortMapping.insert(std::pair<vtkIdType,int>(idx,SmoothnessPortNumber));
			this->BackwardsInputSmoothnessPortMapping.insert(std::pair<vtkIdType,int>(SmoothnessPortNumber,idx));
		}
		this->SetNthInputConnection(1, this->InputSmoothnessPortMapping[idx], input->GetProducerPort() );

	}else{
		//if their is no pair in the mapping, just exit, nothing to do
		if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() ) return;

		int SmoothnessPortNumber = this->InputSmoothnessPortMapping[idx];
		this->InputSmoothnessPortMapping.erase(this->InputSmoothnessPortMapping.find(idx));
		this->BackwardsInputSmoothnessPortMapping.erase(this->BackwardsInputSmoothnessPortMapping.find(SmoothnessPortNumber));

		//if we are the last input, no need to reshuffle
		if(SmoothnessPortNumber == this->FirstUnusedSmoothnessPort - 1){
			this->SetNthInputConnection(1, SmoothnessPortNumber,  0);
		
		//if we are not, move the last input into this spot
		}else{
			vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedSmoothnessPort - 1));
			this->SetNthInputConnection(1, SmoothnessPortNumber, swappedInput->GetProducerPort() );
			this->SetNthInputConnection(1, this->FirstUnusedSmoothnessPort - 1, 0 );

			//correct the mappings
			vtkIdType swappedId = this->BackwardsInputSmoothnessPortMapping[this->FirstUnusedSmoothnessPort - 1];
			this->InputSmoothnessPortMapping.erase(this->InputSmoothnessPortMapping.find(swappedId));
			this->BackwardsInputSmoothnessPortMapping.erase(this->BackwardsInputSmoothnessPortMapping.find(this->FirstUnusedSmoothnessPort - 1));
			this->InputSmoothnessPortMapping.insert(std::pair<vtkIdType,int>(swappedId,SmoothnessPortNumber) );
			this->BackwardsInputSmoothnessPortMapping.insert(std::pair<int,vtkIdType>(SmoothnessPortNumber,swappedId) );

		}

		//decrement the number of unused SmoothnessPorts
		this->FirstUnusedSmoothnessPort--;

	}
}

vtkDataObject *vtkCudaHierarchicalMaxFlowSegmentation2::GetDataInput(int idx)
{
	if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputDataPortMapping[idx]));
}

vtkDataObject *vtkCudaHierarchicalMaxFlowSegmentation2::GetSmoothnessInput(int idx)
{
	if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(1, this->InputSmoothnessPortMapping[idx]));
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
			if( this->InputDataPortMapping.find(node) == this->InputDataPortMapping.end() ){
				vtkErrorMacro(<<"Missing data prior for leaf node.");
				return -1;
			}
			int inputPortNumber = this->InputDataPortMapping[node];
			if( !(inputVector[0])->GetInformationObject(inputPortNumber) && (inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ){
				vtkErrorMacro(<<"Missing data prior for leaf node.");
				return -1;
			}
		}
		
		//check validity of data terms
		if( this->InputDataPortMapping.find(node) != this->InputDataPortMapping.end() ){
			int inputPortNumber = this->InputDataPortMapping[node];
			if( (inputVector[0])->GetInformationObject(inputPortNumber) &&
				(inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ){

				//check for right scalar type
				vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
				if( CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1 ){
					vtkErrorMacro(<<"Data type must be FLOAT and only have one component.");
					return -1;
				}
				if( CurrImage->GetScalarRange()[0] < 0.0 ){
					vtkErrorMacro(<<"Data prior must be non-negative.");
					return -1;
				}
			
				//check to make sure that the sizes are consistant
				if( Extent[0] == -1 ){
					CurrImage->GetExtent(Extent);
				}else{
					int CurrExtent[6];
					CurrImage->GetExtent(CurrExtent);
					if( CurrExtent[0] != Extent[0] || CurrExtent[1] != Extent[1] || CurrExtent[2] != Extent[2] ||
						CurrExtent[3] != Extent[3] || CurrExtent[4] != Extent[4] || CurrExtent[5] != Extent[5] ){
						vtkErrorMacro(<<"Inconsistant object extent.");
						return -1;
					}
				}

			}
		}

		if( this->InputSmoothnessPortMapping.find(node) != this->InputSmoothnessPortMapping.end() ){
			int inputPortNumber = this->InputSmoothnessPortMapping[node];
			if( (inputVector[1])->GetInformationObject(inputPortNumber) &&
				(inputVector[1])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ){

				//check for right scalar type
				vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
				if( CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1 ){
					vtkErrorMacro(<<"Smoothness type must be FLOAT and only have one component.");
					return -1;
				}
				if( CurrImage->GetScalarRange()[0] < 0.0 ){
					vtkErrorMacro(<<"Smoothness prior must be non-negative.");
					return -1;
				}

				//check to make sure that the sizes are consistant
				if( Extent[0] == -1 ){
					CurrImage->GetExtent(Extent);
				}else{
					int CurrExtent[6];
					CurrImage->GetExtent(CurrExtent);
					if( CurrExtent[0] != Extent[0] || CurrExtent[1] != Extent[1] || CurrExtent[2] != Extent[2] ||
						CurrExtent[3] != Extent[3] || CurrExtent[4] != Extent[4] || CurrExtent[5] != Extent[5] ){
						vtkErrorMacro(<<"Inconsistant object extent.");
						return -1;
					}
				}
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

	if( this->Debug ) vtkDebugMacro(<< "Starting input data preparation." );

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
			int inputNumber = this->InputDataPortMapping[node];
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
		if( this->InputSmoothnessPortMapping.find(node) != this->InputSmoothnessPortMapping.end() ){
			int inputNumber = this->InputSmoothnessPortMapping[node];
			CurrImage = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(inputNumber)->Get(vtkDataObject::DATA_OBJECT()));
		}
		if( this->Hierarchy->IsLeaf(node) )
			leafSmoothnessTermBuffers[this->LeafMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
		else
            branchSmoothnessTermBuffers[this->BranchMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
		
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
			if( this->SmoothnessScalars.find(node) != this->SmoothnessScalars.end() )
				leafSmoothnessConstants[this->LeafMap[node]] = this->SmoothnessScalars[node];
			else
				leafSmoothnessConstants[this->LeafMap[node]] = 1.0f;
		else
			if( this->SmoothnessScalars.find(node) != this->SmoothnessScalars.end() )
				branchSmoothnessConstants[this->BranchMap[node]] = this->SmoothnessScalars[node];
			else
				branchSmoothnessConstants[this->BranchMap[node]] = 1.0f;
	}
	iterator->Delete();
	
	//if verbose, print progress
	if( this->Debug ) vtkDebugMacro(<<"Starting CPU buffer acquisition");

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
	if( this->Debug ) vtkDebugMacro(<<"Relate parent sink with child source buffer pointers.");

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
				leafIncBuffers[this->LeafMap[node]] = sourceFlowBuffer;
			else
				leafIncBuffers[this->LeafMap[node]] = branchSinkBuffers[this->BranchMap[parent]];
		}else{
			if( parent == this->Hierarchy->GetRoot() )
				branchIncBuffers[this->BranchMap[node]] = sourceFlowBuffer;
			else
				branchIncBuffers[this->BranchMap[node]] = branchSinkBuffers[this->BranchMap[parent]];
		}
	}
	iterator->Delete();
	
	//if verbose, print progress
	if( this->Debug ) vtkDebugMacro(<<"Building workers.");
    for(std::set<int>::iterator gpuIterator = GPUsUsed.begin(); gpuIterator != GPUsUsed.end(); gpuIterator++){
		double usage = this->MaxGPUUsage;
		if( this->MaxGPUUsageNonDefault.find(*gpuIterator) != this->MaxGPUUsageNonDefault.end() )
			usage = this->MaxGPUUsageNonDefault[*gpuIterator];
        Worker* newWorker = new Worker( *gpuIterator, usage, this );
        this->Workers.insert( newWorker );
        if(newWorker->NumBuffers < 8){
            vtkErrorMacro(<<"Could not allocate sufficient GPU buffers.");
            for(std::set<Task*>::iterator taskIterator = FinishedTasks.begin(); taskIterator != FinishedTasks.end(); taskIterator++)
                delete *taskIterator;
            FinishedTasks.clear();
            for(std::set<Worker*>::iterator workerIterator = Workers.begin(); workerIterator != Workers.end(); workerIterator++)
                delete *workerIterator;
            Workers.clear();
            while( CPUBuffersAcquired.size() > 0 ){
                float* tempBuffer = CPUBuffersAcquired.front();
                delete[] tempBuffer;
                CPUBuffersAcquired.pop_front();
            }
            return -1;
        }
    }

	//if verbose, print progress
	if( this->Debug ) vtkDebugMacro(<<"Find priority structures.");

	//create LIFO priority queue (priority stack) data structure
    FigureOutBufferPriorities( this->Hierarchy->GetRoot() );
	
	//add tasks in for the normal iterations (done first for dependancy reasons)
	if( this->Debug ) vtkDebugMacro(<<"Creating tasks for normal iterations.");
	NumTasksGoingToHappen = 0;
	if( this->NumberOfIterations > 0 ){
		CreateClearWorkingBufferTasks(this->Hierarchy->GetRoot());
		CreateUpdateSpatialFlowsTasks(this->Hierarchy->GetRoot());
		CreateApplySinkPotentialBranchTasks(this->Hierarchy->GetRoot());
		CreateApplySinkPotentialLeafTasks(this->Hierarchy->GetRoot());
		CreateApplySourcePotentialTask(this->Hierarchy->GetRoot());
		CreateDivideOutWorkingBufferTask(this->Hierarchy->GetRoot());
		CreateUpdateLabelsTask(this->Hierarchy->GetRoot());
		AddIterationTaskDependencies(this->Hierarchy->GetRoot());
	}
	
	//add tasks in for the initialization (done second for dependancy reasons)
	if( this->Debug ) vtkDebugMacro(<<"Creating tasks for initialization.");
	if( this->NumberOfIterations > 0 ) CreateInitializeAllSpatialFlowsToZeroTasks(this->Hierarchy->GetRoot());
	CreateInitializeLeafSinkFlowsToCapTasks(this->Hierarchy->GetRoot());
	CreateCopyMinimalLeafSinkFlowsTasks(this->Hierarchy->GetRoot());
	CreateFindInitialLabellingAndSumTasks(this->Hierarchy->GetRoot());
	CreateClearSourceWorkingBufferTask();
	CreateDivideOutLabelsTasks(this->Hierarchy->GetRoot());
	if( this->NumberOfIterations > 0 ) CreatePropogateLabelsTasks(this->Hierarchy->GetRoot());

	if( this->Debug ) vtkDebugMacro(<<"Number of tasks to be run: " << NumTasksGoingToHappen);
	
	//if verbose, print progress
	if( this->Debug ) vtkDebugMacro(<<"Running tasks");
	NumMemCpies = 0;
	NumKernelRuns = 0;
	int NumTasksDone = 0;
	while( this->CurrentTasks.size() > 0 ){

		int MinWeight = INT_MAX;
		int MinUnConflictWeight = INT_MAX;
		std::vector<Task*> MinTasks;
		std::vector<Task*> MinUnConflictTasks;
		std::vector<Worker*> MinWorkers;
		std::vector<Worker*> MinUnConflictWorkers;
		for(std::set<Task*>::iterator taskIt = CurrentTasks.begin(); MinWeight > 0 && taskIt != CurrentTasks.end(); taskIt++){
			if( !(*taskIt)->CanDo() ) continue;
			
			//find if the task is conflicted and put in appropriate contest
			Worker* possibleWorker = 0;
			int conflictWeight = (*taskIt)->Conflicted(&possibleWorker);
			if( conflictWeight ){
				if( conflictWeight < MinUnConflictWeight ){
					MinUnConflictWeight = conflictWeight;
					MinUnConflictTasks.clear();
					MinUnConflictTasks.push_back(*taskIt);
					MinUnConflictWorkers.clear();
					MinUnConflictWorkers.push_back(possibleWorker);
				}else if(conflictWeight == MinUnConflictWeight){
					MinUnConflictTasks.push_back(*taskIt);
					MinUnConflictWorkers.push_back(possibleWorker);
				}
				continue;
			}
			
			if( possibleWorker ){ //only one worker can do this task
				int weight = (*taskIt)->CalcWeight(possibleWorker);
				if( weight < MinWeight ){
					MinWeight = weight;
					MinTasks.clear();
					MinTasks.push_back(*taskIt);
					MinWorkers.clear();
					MinWorkers.push_back(possibleWorker);
				}else if( weight == MinWeight ){
					MinTasks.push_back(*taskIt);
					MinWorkers.push_back(possibleWorker);
				}
			}else{ //all workers have a chance, find the emptiest one
				for(std::set<Worker*>::iterator workerIt = Workers.begin(); workerIt != Workers.end(); workerIt++){
					int weight = (*taskIt)->CalcWeight(*workerIt);
					if( weight < MinWeight ){
						MinWeight = weight;
						MinTasks.clear();
						MinTasks.push_back(*taskIt);
						MinWorkers.clear();
						MinWorkers.push_back(*workerIt);
					}else if( weight == MinWeight ){
						MinTasks.push_back(*taskIt);
						MinWorkers.push_back(*workerIt);
					}
				}
			}
		}
		
		//figure out if it is cheaper to run a conflicted or non-conflicted task
		if( MinUnConflictWeight >= MinWeight ){
			int taskIdx = rand() % MinTasks.size();
			MinTasks[taskIdx]->Perform(MinWorkers[taskIdx]);
		}else{
			int taskIdx = rand() % MinUnConflictTasks.size();
			MinUnConflictTasks[taskIdx]->UnConflict(MinUnConflictWorkers[taskIdx]);
			MinUnConflictTasks[taskIdx]->Perform(MinUnConflictWorkers[taskIdx]);
		}

		//if there are conflicts
		//update progress
		NumTasksDone++;
		if( this->Debug && NumTasksDone % ReportRate == 0 ){
			for(std::set<Worker*>::iterator workerIt = Workers.begin(); workerIt != Workers.end(); workerIt++)
				(*workerIt)->CallSyncThreads();
			vtkDebugMacro(<< "Finished " << NumTasksDone << " with " << NumMemCpies << " memory transfers.");
		}
		
	}
	if( this->Debug ) vtkDebugMacro(<< "Finished all " << NumTasksDone << " tasks with a total of " << NumMemCpies << " memory transfers.");
	assert( BlockedTasks.size() == 0 );
	
	//remove tasks
	if( this->Debug ) vtkDebugMacro(<< "Deallocating tasks" );
	for(std::set<Task*>::iterator taskIterator = FinishedTasks.begin(); taskIterator != FinishedTasks.end(); taskIterator++)
		delete *taskIterator;
	FinishedTasks.clear();

	//remove workers
	if( this->Debug ) vtkDebugMacro(<< "Deallocating workers" );
	for(std::set<Worker*>::iterator workerIterator = Workers.begin(); workerIterator != Workers.end(); workerIterator++)
		delete *workerIterator;
	Workers.clear();

	//deallocate CPU buffers
	if( this->Debug ) vtkDebugMacro(<< "Deallocating CPU buffers" );
	while( CPUBuffersAcquired.size() > 0 ){
		float* tempBuffer = CPUBuffersAcquired.front();
		delete[] tempBuffer;
		CPUBuffersAcquired.pop_front();
	}

	//deallocate structure that holds the pointers to the buffers
	delete[] bufferPointers;
	this->branchFlowXBuffers = 0;
	this->branchFlowYBuffers = 0;
	this->branchFlowZBuffers = 0;
	this->branchDivBuffers = 0;
	this->branchSinkBuffers = 0;
	this->branchLabelBuffers = 0;
	this->branchWorkingBuffers = 0;
	this->leafFlowXBuffers = 0;
	this->leafFlowYBuffers = 0;
	this->leafFlowZBuffers = 0;
	this->leafDivBuffers = 0;
	this->leafSinkBuffers =	 0;

	delete[] leafIncBuffers;
	delete[] branchIncBuffers;
	delete[] leafDataTermBuffers;
	delete[] leafSmoothnessTermBuffers;
	delete[] branchSmoothnessTermBuffers;
	delete[] leafLabelBuffers;
	delete[] leafSmoothnessConstants;
	delete[] branchSmoothnessConstants;
	leafIncBuffers = 0;
	branchIncBuffers = 0;
	leafDataTermBuffers = 0;
	leafSmoothnessTermBuffers = 0;
	branchSmoothnessTermBuffers = 0;
	leafLabelBuffers = 0;
	leafSmoothnessConstants = 0;
	branchSmoothnessConstants = 0;

	//clear old lists
	this->CurrentTasks.clear();
	this->BlockedTasks.clear();
	this->CPUInUse.clear();
	this->CPU2PriorityMap.clear();
	this->ReadOnly.clear();
	this->NoCopyBack.clear();
	this->ClearWorkingBufferTasks.clear();
	this->UpdateSpatialFlowsTasks.clear();
	this->ApplySinkPotentialBranchTasks.clear();
	this->ApplySinkPotentialLeafTasks.clear();
	this->ApplySourcePotentialTasks.clear();
	this->DivideOutWorkingBufferTasks.clear();
	this->UpdateLabelsTasks.clear();
	this->InitializeLeafSinkFlowsTasks.clear();
	this->MinimizeLeafSinkFlowsTasks.clear();
	this->PropogateLeafSinkFlowsTasks.clear();
	this->InitialLabellingSumTasks.clear();
	this->CorrectLabellingTasks.clear();
	this->PropogateLabellingTasks.clear();
	this->LastBufferUse.clear();
	this->Overwritten.clear();

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
		int Number = LeafMap[currNode];
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
		int Number = BranchMap[currNode];
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

void vtkCudaHierarchicalMaxFlowSegmentation2::ReturnBufferGPU2CPU(Worker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream){
	if( !CPUBuffer ) return; 
	if( ReadOnly.find(CPUBuffer) != ReadOnly.end() ) return;
	if( Overwritten[CPUBuffer] == 0 ) return;
	Overwritten[CPUBuffer] = 0;
	caller->ReserveGPU();
	LastBufferUse[CPUBuffer] = caller;
	if( NoCopyBack.find(CPUBuffer) != NoCopyBack.end() ) return;
	CUDA_CopyBufferToCPU( GPUBuffer, CPUBuffer, VolumeSize, stream);
	NumMemCpies++;
}

void vtkCudaHierarchicalMaxFlowSegmentation2::MoveBufferCPU2GPU(Worker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream){
	if( !CPUBuffer ) return; 
	caller->ReserveGPU();
	if( LastBufferUse[CPUBuffer] ) LastBufferUse[CPUBuffer]->CallSyncThreads();
	LastBufferUse[CPUBuffer] = 0;
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

//------------------------------------------------------------//
//------------------------------------------------------------//

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateClearWorkingBufferTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateClearWorkingBufferTasks( this->Hierarchy->GetChild(currNode,i) );
	if( NumKids == 0 ) return;
	
	//create the new task
	Task* newTask = new Task(this,0,1,this->NumberOfIterations,currNode,Task::ClearWorkingBufferTask);
	this->ClearWorkingBufferTasks[currNode] = newTask;
	
	//modify the task accordingly
	if(currNode == this->Hierarchy->GetRoot()){
		newTask->Active = -NumLeaves; //wait for source buffer to finish being used
		newTask->AddBuffer(sourceWorkingBuffer);
	}else{
		NoCopyBack.insert(branchWorkingBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchWorkingBuffers[BranchMap[currNode]]);
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateUpdateSpatialFlowsTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateUpdateSpatialFlowsTasks( this->Hierarchy->GetChild(currNode,i) );
	if( currNode == this->Hierarchy->GetRoot() ) return;
	
	//create the new task
	//initial Active is -(6+NumKids) if branch since 4 clear buffers, 2 init flow happen in the initialization and NumKids number of label clears
	//initial Active is -7 if leaf since 4 clear buffers, 2 init flow happen in the initialization and NumKids number of label clears
	Task* newTask = new Task(this,-(6+(NumKids?NumKids:1)),1,this->NumberOfIterations,currNode,Task::UpdateSpatialFlowsTask);
	this->UpdateSpatialFlowsTasks[currNode] = newTask;
	if(NumKids != 0){
		newTask->AddBuffer(branchSinkBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchIncBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchDivBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchLabelBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchFlowXBuffers[BranchMap[currNode]]);
        newTask->AddBuffer(branchFlowYBuffers[BranchMap[currNode]]);
        newTask->AddBuffer(branchFlowZBuffers[BranchMap[currNode]]);
        newTask->AddBuffer(branchSmoothnessTermBuffers[BranchMap[currNode]]);
	}else{
		newTask->AddBuffer(leafSinkBuffers[LeafMap[currNode]]);
		newTask->AddBuffer(leafIncBuffers[LeafMap[currNode]]);
		newTask->AddBuffer(leafDivBuffers[LeafMap[currNode]]);
		newTask->AddBuffer(leafLabelBuffers[LeafMap[currNode]]);
		newTask->AddBuffer(leafFlowXBuffers[LeafMap[currNode]]);
        newTask->AddBuffer(leafFlowYBuffers[LeafMap[currNode]]);
        newTask->AddBuffer(leafFlowZBuffers[LeafMap[currNode]]);
        newTask->AddBuffer(leafSmoothnessTermBuffers[LeafMap[currNode]]);
	}

}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateApplySinkPotentialBranchTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateApplySinkPotentialBranchTasks( this->Hierarchy->GetChild(currNode,i) );
	if( NumKids == 0 ) return;
	
	//create the new task
	if(currNode != this->Hierarchy->GetRoot()){
		Task* newTask = new Task(this,-2,2,this->NumberOfIterations,currNode,Task::ApplySinkPotentialBranchTask);
		this->ApplySinkPotentialBranchTasks[currNode] = newTask;
		newTask->AddBuffer(branchWorkingBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchIncBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchDivBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchLabelBuffers[BranchMap[currNode]]);
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateApplySinkPotentialLeafTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateApplySinkPotentialLeafTasks( this->Hierarchy->GetChild(currNode,i) );
	if( NumKids != 0 ) return;
	
	//create the new task
	Task* newTask = new Task(this,-1,1,this->NumberOfIterations,currNode,Task::ApplySinkPotentialLeafTask);
	this->ApplySinkPotentialLeafTasks[currNode] = newTask;
	newTask->AddBuffer(leafSinkBuffers[LeafMap[currNode]]);
	newTask->AddBuffer(leafIncBuffers[LeafMap[currNode]]);
	newTask->AddBuffer(leafDivBuffers[LeafMap[currNode]]);
	newTask->AddBuffer(leafLabelBuffers[LeafMap[currNode]]);
	newTask->AddBuffer(leafDataTermBuffers[LeafMap[currNode]]);
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateDivideOutWorkingBufferTask(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateDivideOutWorkingBufferTask( this->Hierarchy->GetChild(currNode,i) );
	if( NumKids == 0 ) return;
	
	//create the new task
	Task* newTask = new Task(this,-(NumKids+1),NumKids+1,this->NumberOfIterations,currNode,Task::DivideOutWorkingBufferTask);
	this->DivideOutWorkingBufferTasks[currNode] = newTask;
	if( currNode != this->Hierarchy->GetRoot() ){
		newTask->AddBuffer(branchWorkingBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchSinkBuffers[BranchMap[currNode]]);
	}else{
		newTask->AddBuffer(sourceWorkingBuffer);
		newTask->AddBuffer(sourceFlowBuffer);
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateApplySourcePotentialTask(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateApplySourcePotentialTask( this->Hierarchy->GetChild(currNode,i) );
	if( currNode == this->Hierarchy->GetRoot() ) return;
	vtkIdType parentNode = this->Hierarchy->GetParent(currNode);

	//find appropriate working buffer
	float* workingBuffer = 0;
	if( parentNode == this->Hierarchy->GetRoot() ) workingBuffer = sourceWorkingBuffer;
	else workingBuffer = branchWorkingBuffers[BranchMap[parentNode]];

	//create the new task
	Task* newTask = new Task(this,-2,2,this->NumberOfIterations,currNode,Task::ApplySourcePotentialTask);
	this->ApplySourcePotentialTasks[currNode] = newTask;
	newTask->AddBuffer(workingBuffer);
	if(NumKids != 0){
		newTask->AddBuffer(branchSinkBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchDivBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchLabelBuffers[BranchMap[currNode]]);
	}else{
		newTask->AddBuffer(leafSinkBuffers[LeafMap[currNode]]);
		newTask->AddBuffer(leafDivBuffers[LeafMap[currNode]]);
		newTask->AddBuffer(leafLabelBuffers[LeafMap[currNode]]);
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateUpdateLabelsTask(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateUpdateLabelsTask( this->Hierarchy->GetChild(currNode,i) );
	if( currNode == this->Hierarchy->GetRoot() ) return;
	
	//find appropriate number of repetitions
	int NumReps = NumKids ? this->NumberOfIterations-1: this->NumberOfIterations;

	//create the new task
	Task* newTask = new Task(this,-2,2,NumReps,currNode,Task::UpdateLabelsTask);
	this->UpdateLabelsTasks[currNode] = newTask;
	if(NumKids != 0){
		newTask->AddBuffer(branchSinkBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchIncBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchDivBuffers[BranchMap[currNode]]);
		newTask->AddBuffer(branchLabelBuffers[BranchMap[currNode]]);
	}else{
		newTask->AddBuffer(leafSinkBuffers[LeafMap[currNode]]);
		newTask->AddBuffer(leafIncBuffers[LeafMap[currNode]]);
		newTask->AddBuffer(leafDivBuffers[LeafMap[currNode]]);
		newTask->AddBuffer(leafLabelBuffers[LeafMap[currNode]]);
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation2::AddIterationTaskDependencies(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		AddIterationTaskDependencies( this->Hierarchy->GetChild(currNode,i) );

	if( NumKids == 0 ){
		vtkIdType parNode = this->Hierarchy->GetParent(currNode);
		this->UpdateSpatialFlowsTasks[currNode]->AddTaskToSignal(this->ApplySinkPotentialLeafTasks[currNode]);
		this->ApplySinkPotentialLeafTasks[currNode]->AddTaskToSignal(this->ApplySourcePotentialTasks[currNode]);
		this->ApplySourcePotentialTasks[currNode]->AddTaskToSignal(this->DivideOutWorkingBufferTasks[parNode]);
		this->ApplySourcePotentialTasks[currNode]->AddTaskToSignal(this->UpdateLabelsTasks[currNode]);
		this->UpdateLabelsTasks[currNode]->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
	}else if( currNode == this->Hierarchy->GetRoot() ){
		this->ClearWorkingBufferTasks[currNode]->AddTaskToSignal(this->DivideOutWorkingBufferTasks[currNode]);
		for(int i = 0; i < NumKids; i++)
			this->ClearWorkingBufferTasks[currNode]->AddTaskToSignal(this->ApplySourcePotentialTasks[this->Hierarchy->GetChild(currNode,i)]);
		this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->ClearWorkingBufferTasks[currNode]);
		for(int i = 0; i < NumKids; i++)
			this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->UpdateLabelsTasks[this->Hierarchy->GetChild(currNode,i)]);
	}else{
		vtkIdType parNode = this->Hierarchy->GetParent(currNode);
		this->ClearWorkingBufferTasks[currNode]->AddTaskToSignal(this->ApplySinkPotentialBranchTasks[currNode]);
		for(int i = 0; i < NumKids; i++)
			this->ClearWorkingBufferTasks[currNode]->AddTaskToSignal(this->ApplySourcePotentialTasks[this->Hierarchy->GetChild(currNode,i)]);
		this->UpdateSpatialFlowsTasks[currNode]->AddTaskToSignal(this->ApplySinkPotentialBranchTasks[currNode]);
		this->ApplySinkPotentialBranchTasks[currNode]->AddTaskToSignal(this->DivideOutWorkingBufferTasks[currNode]);
		this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->ApplySourcePotentialTasks[currNode]);
		this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->ClearWorkingBufferTasks[currNode]);
		for(int i = 0; i < NumKids; i++)
			this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->UpdateLabelsTasks[this->Hierarchy->GetChild(currNode,i)]);
		this->ApplySourcePotentialTasks[currNode]->AddTaskToSignal(this->DivideOutWorkingBufferTasks[parNode]);
		this->ApplySourcePotentialTasks[currNode]->AddTaskToSignal(this->UpdateLabelsTasks[currNode]);
		this->UpdateLabelsTasks[currNode]->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateInitializeAllSpatialFlowsToZeroTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateInitializeAllSpatialFlowsToZeroTasks( this->Hierarchy->GetChild(currNode,i) );
	
	//modify the task accordingly
	if( NumKids == 0 ){
		Task* newTask1 = new Task(this,0,1,1,currNode,Task::ClearBufferInitially);
		Task* newTask2 = new Task(this,0,1,1,currNode,Task::ClearBufferInitially);
		Task* newTask3 = new Task(this,0,1,1,currNode,Task::ClearBufferInitially);
		Task* newTask4 = new Task(this,0,1,1,currNode,Task::ClearBufferInitially);
		newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
		newTask2->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
		newTask3->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
		newTask4->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
		newTask1->AddBuffer(this->leafDivBuffers[LeafMap[currNode]]);
		newTask2->AddBuffer(this->leafFlowXBuffers[LeafMap[currNode]]);
		newTask3->AddBuffer(this->leafFlowYBuffers[LeafMap[currNode]]);
		newTask4->AddBuffer(this->leafFlowZBuffers[LeafMap[currNode]]);
	}else if(currNode != this->Hierarchy->GetRoot()){
		Task* newTask1 = new Task(this,0,1,1,currNode,Task::ClearBufferInitially);
		Task* newTask2 = new Task(this,0,1,1,currNode,Task::ClearBufferInitially);
		Task* newTask3 = new Task(this,0,1,1,currNode,Task::ClearBufferInitially);
		Task* newTask4 = new Task(this,0,1,1,currNode,Task::ClearBufferInitially);
		newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
		newTask2->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
		newTask3->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
		newTask4->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
		newTask1->AddBuffer(this->branchDivBuffers[BranchMap[currNode]]);
		newTask2->AddBuffer(this->branchFlowXBuffers[BranchMap[currNode]]);
		newTask3->AddBuffer(this->branchFlowYBuffers[BranchMap[currNode]]);
		newTask4->AddBuffer(this->branchFlowZBuffers[BranchMap[currNode]]);
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateInitializeLeafSinkFlowsToCapTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateInitializeLeafSinkFlowsToCapTasks( this->Hierarchy->GetChild(currNode,i) );
	if( NumKids > 0 ) return;
	
	if( LeafMap[currNode] != 0 ){
		Task* newTask1 = new Task(this,0,1,1,currNode,Task::InitializeLeafFlows);
		Task* newTask2 = new Task(this,-2,1,1,currNode,Task::MinimizeLeafFlows);
		InitializeLeafSinkFlowsTasks.insert(std::pair<int,Task*>(LeafMap[currNode],newTask1));
		MinimizeLeafSinkFlowsTasks.insert(std::pair<int,Task*>(LeafMap[currNode],newTask2));
		newTask1->AddBuffer(this->leafSinkBuffers[LeafMap[currNode]]);
		newTask1->AddBuffer(this->leafDataTermBuffers[LeafMap[currNode]]);
		newTask2->AddBuffer(this->leafSinkBuffers[0]);
		newTask2->AddBuffer(this->leafSinkBuffers[LeafMap[currNode]]);
		newTask1->AddTaskToSignal(newTask2);
		if( InitializeLeafSinkFlowsTasks.find(0) != InitializeLeafSinkFlowsTasks.end() )
			InitializeLeafSinkFlowsTasks[0]->AddTaskToSignal(newTask2);
	}else{
		Task* newTask1 = new Task(this,0,1,1,currNode,Task::InitializeLeafFlows);
		InitializeLeafSinkFlowsTasks.insert(std::pair<int,Task*>(0,newTask1));
		newTask1->AddBuffer(this->leafSinkBuffers[0]);
		newTask1->AddBuffer(this->leafDataTermBuffers[0]);
		for( std::map<int,Task*>::iterator it = MinimizeLeafSinkFlowsTasks.begin();
			 it != this->MinimizeLeafSinkFlowsTasks.end(); it++)
			newTask1->AddTaskToSignal(it->second);
	}

}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateCopyMinimalLeafSinkFlowsTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateCopyMinimalLeafSinkFlowsTasks( this->Hierarchy->GetChild(currNode,i) );

	Task* newTask1 = new Task(this,-((int)this->MinimizeLeafSinkFlowsTasks.size()),1,1,currNode,Task::PropogateLeafFlows);
	PropogateLeafSinkFlowsTasks.insert(std::pair<vtkIdType,Task*>(currNode,newTask1));
	if( currNode != this->Hierarchy->GetRoot() ) newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
	for(int i = 0; i < NumKids; i++)
		newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[this->Hierarchy->GetChild(currNode,i)]);
	newTask1->AddBuffer(this->leafSinkBuffers[0]);
	for( std::map<int,Task*>::iterator it = this->MinimizeLeafSinkFlowsTasks.begin(); it != this->MinimizeLeafSinkFlowsTasks.end(); it++)
		it->second->AddTaskToSignal(newTask1);

	if( this->Hierarchy->GetRoot() == currNode )
		newTask1->AddBuffer(this->sourceFlowBuffer);
	else if( NumKids > 0 )
		newTask1->AddBuffer(this->branchSinkBuffers[BranchMap[currNode]]);
	else
		newTask1->AddBuffer(this->leafSinkBuffers[LeafMap[currNode]]);

}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateFindInitialLabellingAndSumTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateFindInitialLabellingAndSumTasks( this->Hierarchy->GetChild(currNode,i) );
	if( NumKids > 0 ) return;

	Task* newTask1 = new Task(this,-1,1,1,currNode,Task::InitializeLeafLabels);
	Task* newTask2 = new Task(this,-2,1,1,currNode,Task::AccumulateLabels);
	this->PropogateLeafSinkFlowsTasks[currNode]->AddTaskToSignal(newTask1);
	newTask1->AddTaskToSignal(newTask2);
	this->InitialLabellingSumTasks.insert(std::pair<vtkIdType,Task*>(currNode,newTask2));
	newTask1->AddBuffer(this->leafSinkBuffers[LeafMap[currNode]]);
	newTask1->AddBuffer(this->leafDataTermBuffers[LeafMap[currNode]]);
	newTask1->AddBuffer(this->leafLabelBuffers[LeafMap[currNode]]);
	newTask2->AddBuffer(this->sourceWorkingBuffer);
	newTask2->AddBuffer(this->leafLabelBuffers[LeafMap[currNode]]);
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateClearSourceWorkingBufferTask(){
	Task* newTask = new Task(this,0,1,1,this->Hierarchy->GetRoot(),Task::ClearBufferInitially);
	newTask->AddBuffer(this->sourceWorkingBuffer);
	for( std::map<vtkIdType,Task*>::iterator it = InitialLabellingSumTasks.begin(); it != InitialLabellingSumTasks.end(); it++)
		newTask->AddTaskToSignal(it->second);
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreateDivideOutLabelsTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreateDivideOutLabelsTasks( this->Hierarchy->GetChild(currNode,i) );
	if( NumKids > 0 ) return;
	
	Task* newTask1 = new Task(this,-(int)InitialLabellingSumTasks.size(),1,1,currNode,Task::CorrectLabels);
	this->CorrectLabellingTasks[currNode] = newTask1;
	for(std::map<vtkIdType,Task*>::iterator taskIt = InitialLabellingSumTasks.begin(); taskIt != InitialLabellingSumTasks.end(); taskIt++)
		taskIt->second->AddTaskToSignal(newTask1);
	newTask1->AddBuffer(this->sourceWorkingBuffer);
	newTask1->AddBuffer(this->leafLabelBuffers[LeafMap[currNode]]);
	newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
	newTask1->AddTaskToSignal(this->ClearWorkingBufferTasks[this->Hierarchy->GetRoot()]);
}

void vtkCudaHierarchicalMaxFlowSegmentation2::CreatePropogateLabelsTasks(vtkIdType currNode){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++)
		CreatePropogateLabelsTasks( this->Hierarchy->GetChild(currNode,i) );
	if( currNode == this->Hierarchy->GetRoot() || NumKids == 0 ) return;
	
	//clear the current buffer
	Task* newTask1 = new Task(this,0,1,1,currNode,Task::ClearBufferInitially);
	newTask1->AddBuffer(this->branchLabelBuffers[BranchMap[currNode]]);

	//accumulate from children
	for(int i = 0; i < NumKids; i++){
		vtkIdType child = this->Hierarchy->GetChild(currNode,i);
		Task* newTask2 = new Task(this,-1,1,1,child,Task::AccumulateLabels);
		this->PropogateLabellingTasks[child] = newTask2;
		newTask1->AddTaskToSignal(newTask2);
		newTask2->AddBuffer(this->branchLabelBuffers[BranchMap[currNode]]);
		if( this->Hierarchy->IsLeaf(child) ){
			newTask2->Active--;
			newTask2->AddBuffer(this->leafLabelBuffers[LeafMap[child]]);
			this->CorrectLabellingTasks[child]->AddTaskToSignal(newTask2);
		}else{
			newTask2->AddBuffer(this->branchLabelBuffers[BranchMap[child]]);
			int NumKids2 = this->Hierarchy->GetNumberOfChildren(child);
			for(int i2 = 0; i2 < NumKids2; i2++)
				this->PropogateLabellingTasks[this->Hierarchy->GetChild(child,i2)]->AddTaskToSignal(newTask2);
		}
		newTask2->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
	}

}
