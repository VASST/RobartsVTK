#include "vtkCudaHierarchicalMaxFlowSegmentation.h"
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

#define SQR(X) X*X

vtkStandardNewMacro(vtkCudaHierarchicalMaxFlowSegmentation);

vtkCudaHierarchicalMaxFlowSegmentation::vtkCudaHierarchicalMaxFlowSegmentation(){
	
	//configure the IO ports
	this->SetNumberOfInputPorts(2);
	this->SetNumberOfOutputPorts(0);

	//set algorithm mathematical parameters to defaults
	this->NumberOfIterations = 100;
	this->StepSize = 0.1;
	this->CC = 0.25;
	this->MaxGPUUsage = 0.75;

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

}

vtkCudaHierarchicalMaxFlowSegmentation::~vtkCudaHierarchicalMaxFlowSegmentation(){
	if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
	this->SmoothnessScalars.clear();
	this->LeafMap.clear();
	this->InputDataPortMapping.clear();
	this->BackwardsInputDataPortMapping.clear();
	this->InputSmoothnessPortMapping.clear();
	this->BackwardsInputSmoothnessPortMapping.clear();
	this->BranchMap.clear();
}

void vtkCudaHierarchicalMaxFlowSegmentation::Reinitialize(int withData = 0){
	//no long-term data stored and no helper classes, so no body for this method
}

void vtkCudaHierarchicalMaxFlowSegmentation::Deinitialize(int withData = 0){
	//no long-term data stored and no helper classes, so no body for this method
}
//------------------------------------------------------------

void vtkCudaHierarchicalMaxFlowSegmentation::SetHierarchy(vtkTree* graph){
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

vtkTree* vtkCudaHierarchicalMaxFlowSegmentation::GetHierarchy(){
	return this->Hierarchy;
}

//------------------------------------------------------------

void vtkCudaHierarchicalMaxFlowSegmentation::AddSmoothnessScalar(vtkIdType node, double value){
	if( value >= 0.0 ){
		this->SmoothnessScalars.insert(std::pair<vtkIdType,double>(node,value));
		this->Modified();
	}else{
		vtkErrorMacro(<<"Cannot use a negative smoothness value.");
	}
}

//------------------------------------------------------------

int vtkCudaHierarchicalMaxFlowSegmentation::FillInputPortInformation(int i, vtkInformation* info){
	info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
	info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
	info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
	return this->Superclass::FillInputPortInformation(i,info);
}

void vtkCudaHierarchicalMaxFlowSegmentation::SetDataInput(int idx, vtkDataObject *input)
{
	//we are adding/switching an input, so no need to resort list
	if( input != NULL ){
	
		//if their is no pair in the mapping, create one
		if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() ){
			int portNumber = this->FirstUnusedDataPort;
			this->FirstUnusedDataPort++;
			this->InputDataPortMapping.insert(std::pair<vtkIdType,int>(idx,portNumber));
			this->BackwardsInputDataPortMapping.insert(std::pair<vtkIdType,int>(portNumber,idx));
		}
		this->SetNthInputConnection(0, this->InputDataPortMapping[idx], input->GetProducerPort() );

	}else{
		//if their is no pair in the mapping, just exit, nothing to do
		if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() ) return;

		int portNumber = this->InputDataPortMapping[idx];
		this->InputDataPortMapping.erase(this->InputDataPortMapping.find(idx));
		this->BackwardsInputDataPortMapping.erase(this->BackwardsInputDataPortMapping.find(portNumber));

		//if we are the last input, no need to reshuffle
		if(portNumber == this->FirstUnusedDataPort - 1){
			this->SetNthInputConnection(0, portNumber,  0);
		
		//if we are not, move the last input into this spot
		}else{
			vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedDataPort - 1));
			this->SetNthInputConnection(0, portNumber, swappedInput->GetProducerPort() );
			this->SetNthInputConnection(0, this->FirstUnusedDataPort - 1, 0 );

			//correct the mappings
			vtkIdType swappedId = this->BackwardsInputDataPortMapping[this->FirstUnusedDataPort - 1];
			this->InputDataPortMapping.erase(this->InputDataPortMapping.find(swappedId));
			this->BackwardsInputDataPortMapping.erase(this->BackwardsInputDataPortMapping.find(this->FirstUnusedDataPort - 1));
			this->InputDataPortMapping.insert(std::pair<vtkIdType,int>(swappedId,portNumber) );
			this->BackwardsInputDataPortMapping.insert(std::pair<int,vtkIdType>(portNumber,swappedId) );

		}

		//decrement the number of unused ports
		this->FirstUnusedDataPort--;

	}
}

void vtkCudaHierarchicalMaxFlowSegmentation::SetSmoothnessInput(int idx, vtkDataObject *input)
{
	//we are adding/switching an input, so no need to resort list
	if( input != NULL ){
	
		//if their is no pair in the mapping, create one
		if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() ){
			int portNumber = this->FirstUnusedSmoothnessPort;
			this->FirstUnusedSmoothnessPort++;
			this->InputSmoothnessPortMapping.insert(std::pair<vtkIdType,int>(idx,portNumber));
			this->BackwardsInputSmoothnessPortMapping.insert(std::pair<vtkIdType,int>(portNumber,idx));
		}
		this->SetNthInputConnection(1, this->InputSmoothnessPortMapping[idx], input->GetProducerPort() );

	}else{
		//if their is no pair in the mapping, just exit, nothing to do
		if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() ) return;

		int portNumber = this->InputSmoothnessPortMapping[idx];
		this->InputSmoothnessPortMapping.erase(this->InputSmoothnessPortMapping.find(idx));
		this->BackwardsInputSmoothnessPortMapping.erase(this->BackwardsInputSmoothnessPortMapping.find(portNumber));

		//if we are the last input, no need to reshuffle
		if(portNumber == this->FirstUnusedSmoothnessPort - 1){
			this->SetNthInputConnection(1, portNumber,  0);
		
		//if we are not, move the last input into this spot
		}else{
			vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedSmoothnessPort - 1));
			this->SetNthInputConnection(1, portNumber, swappedInput->GetProducerPort() );
			this->SetNthInputConnection(1, this->FirstUnusedSmoothnessPort - 1, 0 );

			//correct the mappings
			vtkIdType swappedId = this->BackwardsInputSmoothnessPortMapping[this->FirstUnusedSmoothnessPort - 1];
			this->InputSmoothnessPortMapping.erase(this->InputSmoothnessPortMapping.find(swappedId));
			this->BackwardsInputSmoothnessPortMapping.erase(this->BackwardsInputSmoothnessPortMapping.find(this->FirstUnusedSmoothnessPort - 1));
			this->InputSmoothnessPortMapping.insert(std::pair<vtkIdType,int>(swappedId,portNumber) );
			this->BackwardsInputSmoothnessPortMapping.insert(std::pair<int,vtkIdType>(portNumber,swappedId) );

		}

		//decrement the number of unused ports
		this->FirstUnusedSmoothnessPort--;

	}
}

vtkDataObject *vtkCudaHierarchicalMaxFlowSegmentation::GetDataInput(int idx)
{
	if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputDataPortMapping[idx]));
}

vtkDataObject *vtkCudaHierarchicalMaxFlowSegmentation::GetSmoothnessInput(int idx)
{
	if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(1, this->InputSmoothnessPortMapping[idx]));
}

vtkDataObject *vtkCudaHierarchicalMaxFlowSegmentation::GetOutput(int idx)
{
	//look up port in mapping
	std::map<vtkIdType,int>::iterator port = this->LeafMap.find(idx);
	if( port == this->LeafMap.end() )
		return 0;

	return vtkImageData::SafeDownCast(this->GetExecutive()->GetOutputData(port->second));
}

//----------------------------------------------------------------------------

int vtkCudaHierarchicalMaxFlowSegmentation::CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges ){
	
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

int vtkCudaHierarchicalMaxFlowSegmentation::RequestInformation(
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

int vtkCudaHierarchicalMaxFlowSegmentation::RequestUpdateExtent(
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

void vtkCudaHierarchicalMaxFlowSegmentation::PropogateLabels( vtkIdType currNode ){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	
	//clear own label buffer if not a leaf
	if( NumKids > 0 ){
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchLabelBuffers[BranchMap[currNode]]);
		this->GetGPUBuffersV2(-1);
		CUDA_zeroOutBuffer(CPU2GPUMap[branchLabelBuffers[BranchMap[currNode]]],VolumeSize,GetStream());
	}

	//update graph for all kids
	for(int kid = 0; kid < NumKids; kid++)
		PropogateLabels( this->Hierarchy->GetChild(currNode,kid) );

	//find parent index
	if( currNode == this->Hierarchy->GetRoot() ) return;
	vtkIdType parent = 	this->Hierarchy->GetParent(currNode);
	if( parent == this->Hierarchy->GetRoot() ) return;
	int parentIndex = this->BranchMap[parent];
	float* currVal =   this->Hierarchy->IsLeaf(currNode) ?
		currVal = this->leafLabelBuffers[this->LeafMap[currNode]] :
		currVal = this->branchLabelBuffers[this->BranchMap[currNode]];
	
	//sum value into parent (if parent exists and is not the root)
	this->CPUInUse.clear();
	this->CPUInUse.insert(currVal);
	this->CPUInUse.insert(branchLabelBuffers[parentIndex]);
	this->GetGPUBuffersV2(-1);
	CUDA_SumBuffer(CPU2GPUMap[branchLabelBuffers[parentIndex]],CPU2GPUMap[currVal],VolumeSize,GetStream());
	
}

void vtkCudaHierarchicalMaxFlowSegmentation::SolveMaxFlow( vtkIdType currNode, int* TimeStep ){
	
	//get number of kids
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);

	//figure out what type of node we are
	bool isRoot = (currNode == this->Hierarchy->GetRoot());
	bool isLeaf = (NumKids == 0);
	bool isBranch = (!isRoot && !isLeaf);

	//RB : clear working buffer
	if( !isLeaf ){

		//organize the GPU to obtain the buffers
		float* workingBufferUsed = isRoot ? sourceWorkingBuffer :
				branchWorkingBuffers[BranchMap[currNode]] ;
		this->CPUInUse.clear();
		this->CPUInUse.insert(workingBufferUsed);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//activate the kernel
		//std::cout << currNode << "\t Clear working buffer" << std::endl;
		NumKernelRuns++;
		if( isBranch )
			CUDA_zeroOutBuffer(CPU2GPUMap[workingBufferUsed],VolumeSize,this->GetStream());
		else
			CUDA_SetBufferToValue(CPU2GPUMap[workingBufferUsed],1.0f/CC,VolumeSize,this->GetStream());

		//remove current working buffer from the no-copy list
		this->NoCopyBack.erase( this->NoCopyBack.find(workingBufferUsed) );
	}

	// BL: Update spatial flow
	if( isLeaf ){
		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafDivBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafIncBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafSinkBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafLabelBuffers[LeafMap[currNode]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//compute the gradient step amount (store in div buffer for now)
		//std::cout << currNode << "\t Find gradient descent step size" << std::endl;
		NumKernelRuns++;
		CUDA_flowGradientStep(CPU2GPUMap[leafSinkBuffers[LeafMap[currNode]]], CPU2GPUMap[leafIncBuffers[LeafMap[currNode]]],
							  CPU2GPUMap[leafDivBuffers[LeafMap[currNode]]], CPU2GPUMap[leafLabelBuffers[LeafMap[currNode]]],
							  StepSize, CC, VolumeSize, GetStream());
		
		//re-organize the GPU for the next step
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafDivBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafFlowXBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafFlowYBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafFlowZBuffers[LeafMap[currNode]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//apply gradient descent to the flows
		//std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
		NumKernelRuns++;
		CUDA_applyStep(CPU2GPUMap[leafDivBuffers[LeafMap[currNode]]], CPU2GPUMap[leafFlowXBuffers[LeafMap[currNode]]],
					   CPU2GPUMap[leafFlowYBuffers[LeafMap[currNode]]], CPU2GPUMap[leafFlowZBuffers[LeafMap[currNode]]],
					   VX, VY, VZ, VolumeSize, GetStream() );
		
		//add the smoothness term
		if(leafSmoothnessTermBuffers[LeafMap[currNode]]){
			this->CPUInUse.insert(leafSmoothnessTermBuffers[LeafMap[currNode]]);
			//this->GetGPUBuffers();
		}
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;
		
		//run kernel on CPU
		//std::cout << currNode << "\t Find Projection multiplier" << std::endl;
		NumKernelRuns++;
		CUDA_computeFlowMag(CPU2GPUMap[leafDivBuffers[LeafMap[currNode]]], CPU2GPUMap[leafFlowXBuffers[LeafMap[currNode]]],
					   CPU2GPUMap[leafFlowYBuffers[LeafMap[currNode]]], CPU2GPUMap[leafFlowZBuffers[LeafMap[currNode]]],
					   CPU2GPUMap[leafSmoothnessTermBuffers[LeafMap[currNode]]], leafSmoothnessConstants[LeafMap[currNode]],
					   VX, VY, VZ, VolumeSize, GetStream() );
		
		//project onto set and recompute the divergence
		//std::cout << currNode << "\t Project flows into valid range and compute divergence" << std::endl;
		NumKernelRuns += 2;
		CUDA_projectOntoSet(CPU2GPUMap[leafDivBuffers[LeafMap[currNode]]], CPU2GPUMap[leafFlowXBuffers[LeafMap[currNode]]],
					   CPU2GPUMap[leafFlowYBuffers[LeafMap[currNode]]], CPU2GPUMap[leafFlowZBuffers[LeafMap[currNode]]],
					   VX, VY, VZ, VolumeSize, GetStream() );

	}else if( isBranch ){
		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchDivBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchIncBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchSinkBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchLabelBuffers[BranchMap[currNode]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;
		
		//run kernel on CPU
		//std::cout << currNode << "\t Find gradient descent step size" << std::endl;
		NumKernelRuns++;
		CUDA_flowGradientStep(CPU2GPUMap[branchSinkBuffers[BranchMap[currNode]]], CPU2GPUMap[branchIncBuffers[BranchMap[currNode]]],
							  CPU2GPUMap[branchDivBuffers[BranchMap[currNode]]], CPU2GPUMap[branchLabelBuffers[BranchMap[currNode]]],
							  StepSize, CC,VolumeSize,GetStream());

		//re-organize the GPU for the next step
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchDivBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchFlowXBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchFlowYBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchFlowZBuffers[BranchMap[currNode]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;
		
		//run kernel on CPU
		//std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
		NumKernelRuns++;
		CUDA_applyStep(CPU2GPUMap[branchDivBuffers[BranchMap[currNode]]], CPU2GPUMap[branchFlowXBuffers[BranchMap[currNode]]],
					   CPU2GPUMap[branchFlowYBuffers[BranchMap[currNode]]], CPU2GPUMap[branchFlowZBuffers[BranchMap[currNode]]],
					   VX, VY, VZ, VolumeSize, GetStream() );
		
		//add the smoothness term
		if(branchSmoothnessTermBuffers[BranchMap[currNode]]){
			this->CPUInUse.insert(branchSmoothnessTermBuffers[BranchMap[currNode]]);
			//this->GetGPUBuffers();
		}
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//compute the multiplier for projecting back onto the feasible flow set (and store in div buffer)
		//std::cout << currNode << "\t Find Projection multiplier" << std::endl;
		NumKernelRuns++;
		CUDA_computeFlowMag(CPU2GPUMap[branchDivBuffers[BranchMap[currNode]]], CPU2GPUMap[branchFlowXBuffers[BranchMap[currNode]]],
					   CPU2GPUMap[branchFlowYBuffers[BranchMap[currNode]]], CPU2GPUMap[branchFlowZBuffers[BranchMap[currNode]]],
					   CPU2GPUMap[branchSmoothnessTermBuffers[BranchMap[currNode]]], branchSmoothnessConstants[BranchMap[currNode]],
					   VX, VY, VZ, VolumeSize, GetStream() );
		
		//project onto set and recompute the divergence
		NumKernelRuns += 2;
		CUDA_projectOntoSet(CPU2GPUMap[branchDivBuffers[BranchMap[currNode]]], CPU2GPUMap[branchFlowXBuffers[BranchMap[currNode]]],
					   CPU2GPUMap[branchFlowYBuffers[BranchMap[currNode]]], CPU2GPUMap[branchFlowZBuffers[BranchMap[currNode]]],
					   VX, VY, VZ, VolumeSize, GetStream() );
	}

	//RB : Update everything for the children
	for(int kid = 0; kid < NumKids; kid++)
		SolveMaxFlow( this->Hierarchy->GetChild(currNode,kid), TimeStep );

	// B : Add sink potential to working buffer
	if( isBranch ){
		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchWorkingBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchIncBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchLabelBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchDivBuffers[BranchMap[currNode]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//activate the kernel
		//std::cout << currNode << "\t Add sink potential to working buffer" << std::endl;
		NumKernelRuns++;
		CUDA_storeSinkFlowInBuffer(CPU2GPUMap[branchWorkingBuffers[BranchMap[currNode]]], CPU2GPUMap[branchIncBuffers[BranchMap[currNode]]],
								  CPU2GPUMap[branchDivBuffers[BranchMap[currNode]]], CPU2GPUMap[branchLabelBuffers[BranchMap[currNode]]],
								  CC, VolumeSize, GetStream() );

	}

	// B : Divide working buffer by N+1 and store in sink buffer
	if( isBranch ){
		//since we are overwriting it, the current sink buffer can be considered garbage
		this->NoCopyBack.insert(branchSinkBuffers[BranchMap[currNode]]);

		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchWorkingBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchSinkBuffers[BranchMap[currNode]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//run the kernel
		//std::cout << currNode << "\t Update sink flow" << std::endl;
		NumKernelRuns++;
		CUDA_divideAndStoreBuffer(CPU2GPUMap[branchWorkingBuffers[BranchMap[currNode]]],CPU2GPUMap[branchSinkBuffers[BranchMap[currNode]]],
			(float)(NumKids+1),VolumeSize,this->GetStream());
		
		//since we are done with the working buffer, we can mark it as garbage, and we need to keep the sink value, so no longer garbage
		this->NoCopyBack.insert(branchWorkingBuffers[BranchMap[currNode]]);
		this->NoCopyBack.erase(NoCopyBack.find(branchSinkBuffers[BranchMap[currNode]]));
	}

	//R  : Divide working buffer by N and store in sink buffer
	if( isRoot ){
		//since we are overwriting it, the current sink buffer can be considered garbage
		this->NoCopyBack.insert(sourceFlowBuffer);
		
		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(sourceWorkingBuffer);
		this->CPUInUse.insert(sourceFlowBuffer);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//run the kernel
		//std::cout << currNode << "\t Update sink flow" << std::endl;
		NumKernelRuns++;
		CUDA_divideAndStoreBuffer(CPU2GPUMap[sourceWorkingBuffer],CPU2GPUMap[sourceFlowBuffer],(float)NumKids,VolumeSize,this->GetStream());

		//since we are done with the working buffer, we can mark it as garbage, and we need to keep the sink value, so no longer garbage
		this->NoCopyBack.insert(sourceWorkingBuffer);
		this->NoCopyBack.erase(NoCopyBack.find(sourceFlowBuffer));
	}

	//  L: Find sink potential and store, constrained, in sink
	if( isLeaf ){
		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafIncBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafSinkBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafLabelBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafDivBuffers[LeafMap[currNode]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;
		
		//activate the first unconstrained kernel
		//std::cout << currNode << "\t Update sink flow" << std::endl;
		NumKernelRuns++;
		CUDA_updateLeafSinkFlow(CPU2GPUMap[leafSinkBuffers[LeafMap[currNode]]], CPU2GPUMap[leafIncBuffers[LeafMap[currNode]]],
								CPU2GPUMap[leafDivBuffers[LeafMap[currNode]]], CPU2GPUMap[leafLabelBuffers[LeafMap[currNode]]],
								CC, VolumeSize, GetStream() );

		this->CPUInUse.clear();
		this->CPUInUse.insert(leafSinkBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafDataTermBuffers[LeafMap[currNode]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//activate the second constrained kernel
		NumKernelRuns++;
		CUDA_constrainLeafSinkFlow(CPU2GPUMap[leafSinkBuffers[LeafMap[currNode]]], CPU2GPUMap[leafDataTermBuffers[LeafMap[currNode]]],
									VolumeSize, GetStream() );
	}

	//RB : Update children's labels
	for(int kid = NumKids-1; kid >= 0; kid--)
		UpdateLabel( this->Hierarchy->GetChild(currNode,kid), TimeStep );

	// BL: Find source potential and store in parent's working buffer
	if( !isRoot ){
		//get parent's working buffer
		float* workingBuffer = (this->Hierarchy->GetParent(currNode) == this->Hierarchy->GetRoot()) ?
								sourceWorkingBuffer :
								branchWorkingBuffers[BranchMap[this->Hierarchy->GetParent(currNode)]];

		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(workingBuffer);
		if( isBranch ){
			this->CPUInUse.insert(branchSinkBuffers[BranchMap[currNode]]);
			this->CPUInUse.insert(branchLabelBuffers[BranchMap[currNode]]);
			this->CPUInUse.insert(branchDivBuffers[BranchMap[currNode]]);
		}else{
			this->CPUInUse.insert(leafSinkBuffers[LeafMap[currNode]]);
			this->CPUInUse.insert(leafLabelBuffers[LeafMap[currNode]]);
			this->CPUInUse.insert(leafDivBuffers[LeafMap[currNode]]);
		}
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//activate the kernel
		//std::cout << currNode << "\t Add source potential to parent working buffer" << std::endl;
		NumKernelRuns++;
		if( isBranch ){
			CUDA_storeSourceFlowInBuffer(CPU2GPUMap[workingBuffer], CPU2GPUMap[branchSinkBuffers[BranchMap[currNode]]],
									  CPU2GPUMap[branchDivBuffers[BranchMap[currNode]]], CPU2GPUMap[branchLabelBuffers[BranchMap[currNode]]],
									  CC, VolumeSize, GetStream() );
		}else{
			CUDA_storeSourceFlowInBuffer(CPU2GPUMap[workingBuffer], CPU2GPUMap[leafSinkBuffers[LeafMap[currNode]]],
									  CPU2GPUMap[leafDivBuffers[LeafMap[currNode]]], CPU2GPUMap[leafLabelBuffers[LeafMap[currNode]]],
									  CC, VolumeSize, GetStream() );
		}
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation::UpdateLabel( vtkIdType node, int* TimeStep ){
	int NumKids = this->Hierarchy->GetNumberOfChildren(node);

	if( this->Hierarchy->GetRoot() == node ) return;
	
	//std::cout << node << "\t Update labels" << std::endl;
	if( NumKids == 0 ){
		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafIncBuffers[LeafMap[node]]);
		this->CPUInUse.insert(leafSinkBuffers[LeafMap[node]]);
		this->CPUInUse.insert(leafLabelBuffers[LeafMap[node]]);
		this->CPUInUse.insert(leafDivBuffers[LeafMap[node]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//activate the first unconstrained kernel
		NumKernelRuns++;
		CUDA_updateLabel(CPU2GPUMap[leafSinkBuffers[LeafMap[node]]], CPU2GPUMap[leafIncBuffers[LeafMap[node]]],
						 CPU2GPUMap[leafDivBuffers[LeafMap[node]]], CPU2GPUMap[leafLabelBuffers[LeafMap[node]]],
						 CC, VolumeSize, GetStream() );

	}else{
		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchIncBuffers[BranchMap[node]]);
		this->CPUInUse.insert(branchSinkBuffers[BranchMap[node]]);
		this->CPUInUse.insert(branchLabelBuffers[BranchMap[node]]);
		this->CPUInUse.insert(branchDivBuffers[BranchMap[node]]);
		//this->GetGPUBuffers();
		this->GetGPUBuffersV2(*TimeStep);
		(*TimeStep)++;

		//activate the first unconstrained kernel
		NumKernelRuns++;
		CUDA_updateLabel(CPU2GPUMap[branchSinkBuffers[BranchMap[node]]], CPU2GPUMap[branchIncBuffers[BranchMap[node]]],
						 CPU2GPUMap[branchDivBuffers[BranchMap[node]]], CPU2GPUMap[branchLabelBuffers[BranchMap[node]]],
						 CC, VolumeSize, GetStream() );
	}
}

int vtkCudaHierarchicalMaxFlowSegmentation::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){
								
	//check input consistancy
	int Extent[6]; int NumNodes; int NumLeaves; int NumEdges;
	int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
	if( result || NumNodes == 0 ) return -1;
	int NumBranches = NumNodes - NumLeaves - 1;

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
	int TotalNumberOfBuffers = 0;

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
	if( this->Debug ){
		vtkDebugMacro(<<"Find priority structures.");
	}

	//add all the working buffers from the branches to the garbage (no copy necessary) list
	NoCopyBack.insert( sourceWorkingBuffer );
	for(int i = 0; i < NumBranches; i++ )
		NoCopyBack.insert( branchWorkingBuffers[i] );

	//find buffer ordering by simulating a single iteration
	int reference = 0;
	SimulateIterationForBufferOrdering( this->Hierarchy->GetRoot(), &reference );
	SimulateIterationForBufferOrderingUpdateLabelStep( this->Hierarchy->GetRoot(), &reference );

	//if verbose, print progress
	if( this->Debug ){
		vtkDebugMacro(<<"Starting GPU buffer acquisition");
	}

	//Get GPU buffers
	int BuffersAcquired = 0;
	double PercentAcquired = 0.0;
	UnusedGPUBuffers.clear();
	std::list<float*> AllGPUBufferBlocks;
	CPU2GPUMap.clear();
	GPU2CPUMap.clear();
	CPU2GPUMap.insert(std::pair<float*,float*>((float*)0,(float*)0));
	GPU2CPUMap.insert(std::pair<float*,float*>((float*)0,(float*)0));
	while(true) {
		//try acquiring some new buffers
		float* NewAcquiredBuffers = 0;
		int NewNumberAcquired = 0;
		int Pad = VX*VY;
		double NewPercentAcquired = 0;
		CUDA_GetGPUBuffers( TotalNumberOfBuffers-BuffersAcquired, this->MaxGPUUsage-PercentAcquired, &NewAcquiredBuffers, Pad, VolumeSize, &NewNumberAcquired, &NewPercentAcquired );
		BuffersAcquired += NewNumberAcquired;
		PercentAcquired += NewPercentAcquired;

		//if no new buffers were acquired, exit the loop
		if( NewNumberAcquired == 0 ) break;

		//else, load the new buffers into the list of unused buffers
		AllGPUBufferBlocks.push_back(NewAcquiredBuffers);
		NewAcquiredBuffers += Pad;
		for(int i = 0; i < NewNumberAcquired; i++){
			UnusedGPUBuffers.push_back(NewAcquiredBuffers);
			NewAcquiredBuffers += VolumeSize;
		}
	}
	
	//if verbose, print progress
	if( this->Debug ){
		vtkDebugMacro(<<"Starting initialization");
	}
	NumMemCpies = 0;

	//initialize solution
	//initalize all spatial flows and divergences to zero
	for(int i = 0; i < NumBranches; i++ ){
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchFlowXBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_zeroOutBuffer(CPU2GPUMap[branchFlowXBuffers[i]], VolumeSize, GetStream() );
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchFlowYBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_zeroOutBuffer(CPU2GPUMap[branchFlowYBuffers[i]], VolumeSize, GetStream() );
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchFlowZBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_zeroOutBuffer(CPU2GPUMap[branchFlowZBuffers[i]], VolumeSize, GetStream() );
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchDivBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_zeroOutBuffer(CPU2GPUMap[branchDivBuffers[i]], VolumeSize, GetStream() );
	}
	for(int i = 0; i < NumLeaves; i++ ){
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafFlowXBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_zeroOutBuffer(CPU2GPUMap[leafFlowXBuffers[i]], VolumeSize, GetStream() );
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafFlowYBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_zeroOutBuffer(CPU2GPUMap[leafFlowYBuffers[i]], VolumeSize, GetStream() );
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafFlowZBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_zeroOutBuffer(CPU2GPUMap[leafFlowZBuffers[i]], VolumeSize, GetStream() );
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafDivBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_zeroOutBuffer(CPU2GPUMap[leafDivBuffers[i]], VolumeSize, GetStream() );
	}

	//initialize all leak sink flows to their constraints
	for(int i = 0; i < NumLeaves; i++ ){
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafSinkBuffers[i]);
		this->CPUInUse.insert(leafDataTermBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_CopyBuffer(CPU2GPUMap[leafSinkBuffers[i]], CPU2GPUMap[leafDataTermBuffers[i]], VolumeSize, GetStream() );
	}

	//find the minimum sink flow
	for(int i = 1; i < NumLeaves; i++ ){
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafSinkBuffers[0]);
		this->CPUInUse.insert(leafSinkBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_MinBuffer(CPU2GPUMap[leafSinkBuffers[0]], CPU2GPUMap[leafSinkBuffers[i]], VolumeSize, GetStream() );
	}

	//copy minimum sink flow over all leaves and sum the resulting labels into the source working buffer
	this->CPUInUse.clear();
	this->CPUInUse.insert(sourceWorkingBuffer);
	this->CPUInUse.insert(leafSinkBuffers[0]);
	this->CPUInUse.insert(leafLabelBuffers[0]);
	this->CPUInUse.insert(leafDataTermBuffers[0]);
	this->GetGPUBuffersV2(-1);
	CUDA_zeroOutBuffer(CPU2GPUMap[sourceWorkingBuffer], VolumeSize, GetStream() );
	CUDA_LblBuffer(CPU2GPUMap[leafLabelBuffers[0]], CPU2GPUMap[leafSinkBuffers[0]], CPU2GPUMap[leafDataTermBuffers[0]], VolumeSize, GetStream() );
	CUDA_SumBuffer(CPU2GPUMap[sourceWorkingBuffer], CPU2GPUMap[leafLabelBuffers[0]], VolumeSize, GetStream() );
	for(int i = 1; i < NumLeaves; i++ ){
		this->CPUInUse.clear();
		this->CPUInUse.insert(sourceWorkingBuffer);
		this->CPUInUse.insert(leafSinkBuffers[0]);
		this->CPUInUse.insert(leafSinkBuffers[i]);
		this->CPUInUse.insert(leafLabelBuffers[i]);
		this->CPUInUse.insert(leafDataTermBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_CopyBuffer(CPU2GPUMap[leafSinkBuffers[i]], CPU2GPUMap[leafSinkBuffers[0]], VolumeSize, GetStream() );
		CUDA_LblBuffer(CPU2GPUMap[leafLabelBuffers[i]], CPU2GPUMap[leafSinkBuffers[i]], CPU2GPUMap[leafDataTermBuffers[i]], VolumeSize, GetStream() );
		CUDA_SumBuffer(CPU2GPUMap[sourceWorkingBuffer], CPU2GPUMap[leafLabelBuffers[i]], VolumeSize, GetStream() );
	}

	//divide the labels out to constrain them to validity
	for(int i = 0; i < NumLeaves; i++ ){
		this->CPUInUse.clear();
		this->CPUInUse.insert(sourceWorkingBuffer);
		this->CPUInUse.insert(leafLabelBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_DivBuffer(CPU2GPUMap[leafLabelBuffers[i]], CPU2GPUMap[sourceWorkingBuffer], VolumeSize, GetStream() );
	}

	//apply minimal sink flow over the remaining hierarchy
	for(int i = 0; i < NumBranches; i++ ){
		this->CPUInUse.clear();
		this->CPUInUse.insert(leafSinkBuffers[0]);
		this->CPUInUse.insert(branchSinkBuffers[i]);
		this->GetGPUBuffersV2(-1);
		CUDA_CopyBuffer(CPU2GPUMap[branchSinkBuffers[i]], CPU2GPUMap[leafSinkBuffers[0]], VolumeSize, GetStream() );
	}
	this->CPUInUse.clear();
	this->CPUInUse.insert(leafSinkBuffers[0]);
	this->CPUInUse.insert(sourceFlowBuffer);
	this->GetGPUBuffersV2(-1);
	CUDA_CopyBuffer(CPU2GPUMap[sourceFlowBuffer], CPU2GPUMap[leafSinkBuffers[0]], VolumeSize, GetStream() );

	//propogate labels up the hierarchy
	PropogateLabels( this->Hierarchy->GetRoot() );
	this->CallSyncThreads();

	if( this->Debug )
		vtkDebugMacro(<< "Finished initialization with a total of " << NumMemCpies << " memory transfers.");

	//Solve maximum flow problem in an iterative bottom-up manner
	if( this->Debug )
		vtkDebugMacro(<<"Starting max-flow iterations.");
	NumMemCpies = 0;
	NumKernelRuns = 0;
	this->ReserveGPU();
	for( int iteration = 0; iteration < this->NumberOfIterations; iteration++ ){
		int oldNumMemCpies = NumMemCpies;
		int TimeStep = 0;
		SolveMaxFlow( this->Hierarchy->GetRoot(), &TimeStep );
		this->CallSyncThreads();
		if( this->Debug )
			vtkDebugMacro(<< "Finished iteration " << (iteration+1) << " with " << (NumMemCpies-oldNumMemCpies) << " memory transfers.");
	}
	if( this->Debug )
		vtkDebugMacro(<< "Finished all iterations with a total of " << NumMemCpies << " memory transfers.");

	//Copy back any uncopied leaf label buffers (others don't matter anymore)
	for( int i = 0; i < NumLeaves; i++ ){
		if( CPU2GPUMap.find(leafLabelBuffers[i]) != CPU2GPUMap.end() )
			ReturnBufferGPU2CPU(leafLabelBuffers[i], CPU2GPUMap[leafLabelBuffers[i]]);
	}
	if( this->Debug )
		vtkDebugMacro(<< "Results copied back to CPU " );

	//Return all GPU buffers
	while( AllGPUBufferBlocks.size() > 0 ){
		CUDA_ReturnGPUBuffers( AllGPUBufferBlocks.front() );
		AllGPUBufferBlocks.pop_front();
	}
	
	//deallocate priority structure
	DeallocatePrioritySet();

	//deallocate CPU buffers
	while( CPUBuffersAcquired.size() > 0 ){
		float* tempBuffer = CPUBuffersAcquired.front();
		delete[] tempBuffer;
		CPUBuffersAcquired.pop_front();
	}

	//deallocate structure that holds the pointers to the buffers
	delete[] bufferPointers;

	return 1;
}

void vtkCudaHierarchicalMaxFlowSegmentation::ReturnBufferGPU2CPU(float* CPUBuffer, float* GPUBuffer){
	if( ReadOnly.find(CPUBuffer) != ReadOnly.end() ) return;
	if( NoCopyBack.find(CPUBuffer) != NoCopyBack.end() ) return;
	CUDA_CopyBufferToCPU( GPUBuffer, CPUBuffer, VolumeSize, GetStream());
	NumMemCpies++;
}

void vtkCudaHierarchicalMaxFlowSegmentation::MoveBufferCPU2GPU(float* CPUBuffer, float* GPUBuffer){
	if( NoCopyBack.find(CPUBuffer) != NoCopyBack.end() ) return;
	CUDA_CopyBufferToGPU( GPUBuffer, CPUBuffer, VolumeSize, GetStream());
	NumMemCpies++;
}

int vtkCudaHierarchicalMaxFlowSegmentation::RequestDataObject(
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

//-----------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------//

class vtkCudaHierarchicalMaxFlowSegmentation::CircListNode {
public:
	CircListNode* Prev;
	CircListNode* Next;
	
	int			Value;

	CircListNode(int val){
		Value = val;
		Next = Prev = this;
	}

	CircListNode* InsertBehind(int val){
		CircListNode* newNode = new CircListNode(val);
		newNode->Next = this;
		newNode->Prev = this->Prev;
		this->Prev = newNode;
		newNode->Prev->Next = newNode;
		return newNode;
	}

	CircListNode* InsertFront(int val){
		CircListNode* newNode = new CircListNode(val);
		newNode->Prev = this;
		newNode->Next = this->Next;
		this->Next = newNode;
		newNode->Next->Prev = newNode;
		return newNode;
	}

	bool Equals(CircListNode* other){
		return this->Value == other->Value;
	}

	bool ComesBefore(CircListNode* other, int reference){
		if( this->Value <= reference ){
			if( other->Value >= reference ) return false;
			if( other->Value <= this->Value) return false;
			return true;
		}else{
			if( other->Value <= reference ) return true;
			if( other->Value > this->Value ) return true;
			return false;
		}
	}
};

void vtkCudaHierarchicalMaxFlowSegmentation::ClearBufferOrdering( vtkIdType currNode ){
	//push down to leaves
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int i = 0; i < NumKids; i++) ClearBufferOrdering( this->Hierarchy->GetChild(currNode,i) );

	//clear all values in the Priority Set Num Uses, assuming Priority set itself is entirely empty
	if( NumKids == 0 ){
		this->PrioritySetNumUses[leafLabelBuffers[LeafMap[currNode]]] = 0;
		this->PrioritySetNumUses[leafDivBuffers[LeafMap[currNode]]] = 0;
		this->PrioritySetNumUses[leafSinkBuffers[LeafMap[currNode]]] = 0;
		this->PrioritySetNumUses[leafDataTermBuffers[LeafMap[currNode]]] = 0;
		this->PrioritySetNumUses[leafSmoothnessTermBuffers[LeafMap[currNode]]] = 0;
		this->PrioritySetNumUses[leafFlowXBuffers[LeafMap[currNode]]] = 0;
		this->PrioritySetNumUses[leafFlowYBuffers[LeafMap[currNode]]] = 0;
		this->PrioritySetNumUses[leafFlowZBuffers[LeafMap[currNode]]] = 0;

	}else if( currNode == this->Hierarchy->GetRoot() ){
		this->PrioritySetNumUses[sourceWorkingBuffer] = 0;
		this->PrioritySetNumUses[sourceFlowBuffer] = 0;

	}else{
		this->PrioritySetNumUses[branchLabelBuffers[BranchMap[currNode]]] = 0;
		this->PrioritySetNumUses[branchDivBuffers[BranchMap[currNode]]] = 0;
		this->PrioritySetNumUses[branchSinkBuffers[BranchMap[currNode]]] = 0;
		this->PrioritySetNumUses[branchWorkingBuffers[BranchMap[currNode]]] = 0;
		this->PrioritySetNumUses[branchSmoothnessTermBuffers[BranchMap[currNode]]] = 0;
		this->PrioritySetNumUses[branchFlowXBuffers[BranchMap[currNode]]] = 0;
		this->PrioritySetNumUses[branchFlowYBuffers[BranchMap[currNode]]] = 0;
		this->PrioritySetNumUses[branchFlowZBuffers[BranchMap[currNode]]] = 0;
	}

}

void vtkCudaHierarchicalMaxFlowSegmentation::UpdateBufferOrderingAt( float* buffer, int reference ){
	//if we are null, don't bother with use
	if( buffer == 0 ) return;

	//if we are new, we need an initial circular node
	if( this->PrioritySetNumUses[buffer] == 0 ){
		this->PrioritySetNumUses[buffer] = 1;
		this->PrioritySet[buffer] = new CircListNode(reference);

	//if we've already been called, ignore us
	}else if(this->PrioritySet[buffer]->Prev->Value == reference){
		return;

	//else this is a new use, so increment and insert into the circle
	}else{
		this->PrioritySetNumUses[buffer]++;
		this->PrioritySet[buffer]->InsertBehind(reference);
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation::SimulateIterationForBufferOrdering( vtkIdType currNode, int* reference ){
	//get number of kids
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);

	//figure out what type of node we are
	bool isRoot = (currNode == this->Hierarchy->GetRoot());
	bool isLeaf = (NumKids == 0);
	bool isBranch = (!isRoot && !isLeaf);

	//RB : clear working buffer
	if( !isLeaf ){
		float* workingBufferUsed = isRoot ? sourceWorkingBuffer :
				branchWorkingBuffers[BranchMap[currNode]] ;
		UpdateBufferOrderingAt(workingBufferUsed,*reference);
		(*reference)++;
	}

	// BL: Update spatial flow
	if( isLeaf ){
		//find capacity
		UpdateBufferOrderingAt(leafDivBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafIncBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafSinkBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafLabelBuffers[LeafMap[currNode]],*reference);
		(*reference)++;

		//gradient descent on spatial flow
		UpdateBufferOrderingAt(leafDivBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafFlowXBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafFlowYBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafFlowZBuffers[LeafMap[currNode]],*reference);
		(*reference)++;

		//find projection and divergence term
		UpdateBufferOrderingAt(leafDivBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafFlowXBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafFlowYBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafFlowZBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafSmoothnessTermBuffers[LeafMap[currNode]],*reference);
		(*reference)++;

	}else if( isBranch ){
		//find capacity
		UpdateBufferOrderingAt(branchDivBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchIncBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchSinkBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchLabelBuffers[BranchMap[currNode]],*reference);
		(*reference)++;

		//gradient descent on spatial flow
		UpdateBufferOrderingAt(branchDivBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchFlowXBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchFlowYBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchFlowZBuffers[BranchMap[currNode]],*reference);
		(*reference)++;

		//find projection and divergence term
		UpdateBufferOrderingAt(branchDivBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchFlowXBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchFlowYBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchFlowZBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchSmoothnessTermBuffers[BranchMap[currNode]],*reference);
		(*reference)++;

	}

	//RB : Update everything for the children
	for(int kid = 0; kid < NumKids; kid++)
		SimulateIterationForBufferOrdering( this->Hierarchy->GetChild(currNode,kid), reference );

	// B : Add sink potential to working buffer
	if( isBranch ){
		UpdateBufferOrderingAt(branchWorkingBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchIncBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchLabelBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchDivBuffers[BranchMap[currNode]],*reference);
		(*reference)++;
	}

	// B : Divide working buffer by N+1 and store in sink buffer
	if( isBranch ){
		UpdateBufferOrderingAt(branchWorkingBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchSinkBuffers[BranchMap[currNode]],*reference);
		(*reference)++;
	}

	//R  : Divide working buffer by N and store in sink buffer
	if( isRoot ){
		UpdateBufferOrderingAt(sourceWorkingBuffer,*reference);
		UpdateBufferOrderingAt(sourceFlowBuffer,*reference);
		(*reference)++;
	}

	//  L: Find sink potential and store, constrained, in sink
	if( isLeaf ){

		UpdateBufferOrderingAt(leafIncBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafSinkBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafLabelBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafDivBuffers[LeafMap[currNode]],*reference);
		
		//increment the step counter
		(*reference)++;

		UpdateBufferOrderingAt(leafSinkBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafDataTermBuffers[LeafMap[currNode]],*reference);
		
		//increment the step counter
		(*reference)++;
	}

	//RB : Update children's labels
	for(int kid = NumKids-1; kid >= 0; kid--)
		SimulateIterationForBufferOrderingUpdateLabelStep( this->Hierarchy->GetChild(currNode,kid), reference );

	// BL: Find source potential and store in parent's working buffer
	if( !isRoot ){
		//get parent's working buffer
		float* workingBuffer = (this->Hierarchy->GetParent(currNode) == this->Hierarchy->GetRoot()) ?
								sourceWorkingBuffer :
								branchWorkingBuffers[BranchMap[this->Hierarchy->GetParent(currNode)]];
		UpdateBufferOrderingAt(workingBuffer,*reference);

		//organize the GPU to obtain the buffers
		if( isBranch ){
			UpdateBufferOrderingAt(branchSinkBuffers[BranchMap[currNode]],*reference);
			UpdateBufferOrderingAt(branchLabelBuffers[BranchMap[currNode]],*reference);
			UpdateBufferOrderingAt(branchDivBuffers[BranchMap[currNode]],*reference);
		}else{
			UpdateBufferOrderingAt(leafSinkBuffers[LeafMap[currNode]],*reference);
			UpdateBufferOrderingAt(leafLabelBuffers[LeafMap[currNode]],*reference);
			UpdateBufferOrderingAt(leafDivBuffers[LeafMap[currNode]],*reference);
		}

		//increment the step counter
		(*reference)++;
	}
}

//apply the reference counting for the label step
void vtkCudaHierarchicalMaxFlowSegmentation::SimulateIterationForBufferOrderingUpdateLabelStep( vtkIdType currNode, int* reference ){
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	if( this->Hierarchy->GetRoot() == currNode ) return;
	if( NumKids == 0 ){
		UpdateBufferOrderingAt(leafIncBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafSinkBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafLabelBuffers[LeafMap[currNode]],*reference);
		UpdateBufferOrderingAt(leafDivBuffers[LeafMap[currNode]],*reference);
	}else{
		UpdateBufferOrderingAt(branchIncBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchSinkBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchLabelBuffers[BranchMap[currNode]],*reference);
		UpdateBufferOrderingAt(branchDivBuffers[BranchMap[currNode]],*reference);
	}

	//move to the next timestep
	(*reference)++;
}

void vtkCudaHierarchicalMaxFlowSegmentation::DeallocatePrioritySet(){
	for( std::map<float*,int>::iterator useIterator = this->PrioritySetNumUses.begin();
		 useIterator != this->PrioritySetNumUses.end(); useIterator++ ){
		
		//we know how many items need to be removed, so only remove that many
		CircListNode* currNode = this->PrioritySet[useIterator->first];
		for(int i = 0; i < useIterator->second; i++ ){
			CircListNode* nextNode = currNode->Next;
			delete currNode;
			currNode = nextNode;
		}
	}

	//clear the larger structures
	this->PrioritySet.clear();
	this->PrioritySetNumUses.clear();
}

void vtkCudaHierarchicalMaxFlowSegmentation::GetGPUBuffersV2(int reference){
		
	for( std::set<float*>::iterator iterator = CPUInUse.begin();
		 iterator != CPUInUse.end(); iterator++ ){

		//check if this buffer needs to be assigned
		if( CPU2GPUMap.find( *iterator ) != CPU2GPUMap.end() ) continue;

		//start assigning from the list of unused buffers
		if( UnusedGPUBuffers.size() > 0 ){
			float* NewGPUBuffer = UnusedGPUBuffers.front();
			UnusedGPUBuffers.pop_front();
			CPU2GPUMap.insert( std::pair<float*,float*>(*iterator, NewGPUBuffer) );
			GPU2CPUMap.insert( std::pair<float*,float*>(NewGPUBuffer, *iterator) );
			MoveBufferCPU2GPU(*iterator,NewGPUBuffer);
			continue;
		}
		
		//else, we have to assign a space already in use, try no-copy-backs and read onlies first
		int lastBufferTime = -1;
		float* lastBuffer = 0;
		CircListNode* lastBufferNode = 0;
		for( std::set<float*>::iterator iterator2 = NoCopyBack.begin();
			iterator2 != NoCopyBack.end(); iterator2++ ){
			
			//can't deallocate something in use, null, or not on GPU
			if( CPUInUse.find(*iterator2) != CPUInUse.end() ) continue;
			if( *iterator2 == 0 ) continue;
			if( CPU2GPUMap.find(*iterator2) == CPU2GPUMap.end() ) continue;

			//if this is the first valid one, assume we use it
			if( lastBufferTime == -1 ){
				lastBuffer = *iterator2;
				lastBufferNode = this->PrioritySet[lastBuffer];
				lastBufferTime = lastBufferNode->Value;
				continue;
			}

			//else, compare against the current minimum
			if( lastBufferNode->ComesBefore( this->PrioritySet[*iterator2], reference ) ){
				lastBuffer = *iterator2;
				lastBufferNode = this->PrioritySet[lastBuffer];
				lastBufferTime = lastBufferNode->Value;
				continue;
			}

			//if they are equal, prefer removing a no-copy one
			if( lastBufferNode->Equals( this->PrioritySet[*iterator2] ) &&
				this->NoCopyBack.find(*iterator2) != this->NoCopyBack.end() ){
				lastBuffer = *iterator2;
				lastBufferNode = this->PrioritySet[lastBuffer];
				lastBufferTime = lastBufferNode->Value;
			}
		}

		//now for read-onlies
		for( std::set<float*>::iterator iterator2 = ReadOnly.begin();
			iterator2 != ReadOnly.end(); iterator2++ ){
			
			//can't deallocate something in use, null, or not on GPU
			if( CPUInUse.find(*iterator2) != CPUInUse.end() ) continue;
			if( *iterator2 == 0 ) continue;
			if( CPU2GPUMap.find(*iterator2) == CPU2GPUMap.end() ) continue;

			//if this is the first valid one, assume we use it
			if( lastBufferTime == -1 ){
				lastBuffer = *iterator2;
				lastBufferNode = this->PrioritySet[lastBuffer];
				lastBufferTime = lastBufferNode->Value;
				continue;
			}

			//else, compare against the current minimum
			if( lastBufferNode->ComesBefore( this->PrioritySet[*iterator2], reference ) ){
				lastBuffer = *iterator2;
				lastBufferNode = this->PrioritySet[lastBuffer];
				lastBufferTime = lastBufferNode->Value;
				continue;
			}

			//if they are equal, prefer removing a no-copy one
			if( lastBufferNode->Equals( this->PrioritySet[*iterator2] ) &&
				this->NoCopyBack.find(*iterator2) != this->NoCopyBack.end() ){
				lastBuffer = *iterator2;
				lastBufferNode = this->PrioritySet[lastBuffer];
				lastBufferTime = lastBufferNode->Value;
			}
		}

		//if we have a catch, use it and swap buffers
		if( lastBuffer ){
			float* NewGPUBuffer = CPU2GPUMap[lastBuffer];
			CPU2GPUMap.erase( CPU2GPUMap.find(lastBuffer) );
			GPU2CPUMap.erase( NewGPUBuffer );
			CPU2GPUMap.insert( std::pair<float*,float*>(*iterator, NewGPUBuffer) );
			GPU2CPUMap.insert( std::pair<float*,float*>(NewGPUBuffer, *iterator) );
			ReturnBufferGPU2CPU(lastBuffer,NewGPUBuffer);
			MoveBufferCPU2GPU(*iterator,NewGPUBuffer);
			continue;
		}

		//else, we have to assign a space already in use regardless of no-copy-back status
		lastBufferTime = -1;
		lastBuffer = 0;
		lastBufferNode = 0;
		for( std::map<float*,float*>::iterator iterator2 = CPU2GPUMap.begin();
			iterator2 != CPU2GPUMap.end(); iterator2++ ){
			
			//can't deallocate something in use	
			if( CPUInUse.find(iterator2->first) != CPUInUse.end() ) continue;

			//can't deallocate the null pointer
			if( iterator2->first == 0 ) continue;

			//if this is the first valid one, assume we use it
			if( lastBufferTime == -1 ){
				lastBuffer = iterator2->first;
				lastBufferNode = this->PrioritySet[lastBuffer];
				lastBufferTime = lastBufferNode->Value;
				continue;
			}

			//else, compare against the current minimum
			if( lastBufferNode->ComesBefore( this->PrioritySet[iterator2->first], reference ) ){
				lastBuffer = iterator2->first;
				lastBufferNode = this->PrioritySet[lastBuffer];
				lastBufferTime = lastBufferNode->Value;
				continue;
			}

			//if they are equal, prefer removing a no-copy one
			if( lastBufferNode->Equals( this->PrioritySet[iterator2->first] ) &&
				this->NoCopyBack.find(iterator2->first) != this->NoCopyBack.end() ){
				lastBuffer = iterator2->first;
				lastBufferNode = this->PrioritySet[lastBuffer];
				lastBufferTime = lastBufferNode->Value;
			}

		}

		//swap buffers
		float* NewGPUBuffer = CPU2GPUMap[lastBuffer];
		CPU2GPUMap.erase( CPU2GPUMap.find(lastBuffer) );
		GPU2CPUMap.erase( NewGPUBuffer );
		CPU2GPUMap.insert( std::pair<float*,float*>(*iterator, NewGPUBuffer) );
		GPU2CPUMap.insert( std::pair<float*,float*>(NewGPUBuffer, *iterator) );
		ReturnBufferGPU2CPU(lastBuffer,NewGPUBuffer);
		MoveBufferCPU2GPU(*iterator,NewGPUBuffer);

	}
	
	//increment the next time everything on CPU is called
	if( reference == -1 ) return;
	for( std::set<float*>::iterator iterator = CPUInUse.begin();
		 iterator != CPUInUse.end(); iterator++ ){
		CircListNode* bufferNode = this->PrioritySet[*iterator];
		assert( bufferNode->Value == reference );
		this->PrioritySet[*iterator] = bufferNode->Next;
	}

}