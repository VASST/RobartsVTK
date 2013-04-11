#include "vtkCudaHierarchicalMaxFlowSegmentation.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTreeDFSIterator.h"

#include <assert.h>
#include <math.h>
#include <float.h>

#include <set>
#include <list>

#include "CUDA_hierarchicalmaxflow.h"

#define SQR(X) X*X

vtkStandardNewMacro(vtkCudaHierarchicalMaxFlowSegmentation);

vtkCudaHierarchicalMaxFlowSegmentation::vtkCudaHierarchicalMaxFlowSegmentation(){
	
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

vtkCudaHierarchicalMaxFlowSegmentation::~vtkCudaHierarchicalMaxFlowSegmentation(){
	if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
	this->SmoothnessScalars.clear();
	this->LeafMap.clear();
	this->InputPortMapping.clear();
	this->BackwardsInputPortMapping.clear();
	this->BranchMap.clear();

	//deconstruct priority stack datastructure
	std::list< std::list< float* > >::iterator stackIterator = PriorityStacks.begin();
	for( ; stackIterator != PriorityStacks.end(); stackIterator++ )
		stackIterator->clear();
	PriorityStacks.clear();
	Priority.clear();

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

void vtkCudaHierarchicalMaxFlowSegmentation::SetInput(int idx, vtkDataObject *input)
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

vtkDataObject *vtkCudaHierarchicalMaxFlowSegmentation::GetInput(int idx)
{
	if( this->InputPortMapping.find(idx) == this->InputPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputPortMapping.find(idx)->second));
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
	
	//update graph for all kids
	int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
	for(int kid = 0; kid < NumKids; kid++)
		PropogateLabels( this->Hierarchy->GetChild(currNode,kid) );

	//sum value into parent (if parent exists and is not the root)
	if( currNode == this->Hierarchy->GetRoot() ) return;
	vtkIdType parent = 	this->Hierarchy->GetParent(currNode);
	if( parent == this->Hierarchy->GetRoot() ) return;
	int parentIndex = this->BranchMap[parent];
	float* currVal =   this->Hierarchy->IsLeaf(currNode) ?
		currVal = this->leafLabelBuffers[this->LeafMap[currNode]] :
		currVal = this->branchLabelBuffers[this->BranchMap[currNode]];
	for( int x = 0; x < VolumeSize; x++ )
		(this->branchLabelBuffers[parentIndex])[x] += currVal[x];
	
}

void vtkCudaHierarchicalMaxFlowSegmentation::SolveMaxFlow( vtkIdType currNode ){
	

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
		this->GetGPUBuffers();

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
		this->GetGPUBuffers();

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
		this->GetGPUBuffers();

		//apply gradient descent to the flows
		//std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
		NumKernelRuns++;
		CUDA_applyStep(CPU2GPUMap[leafDivBuffers[LeafMap[currNode]]], CPU2GPUMap[leafFlowXBuffers[LeafMap[currNode]]],
					   CPU2GPUMap[leafFlowYBuffers[LeafMap[currNode]]], CPU2GPUMap[leafFlowZBuffers[LeafMap[currNode]]],
					   VX, VY, VZ, VolumeSize, GetStream() );
		
		//add the smoothness term
		if(leafSmoothnessTermBuffers[LeafMap[currNode]]){
			this->CPUInUse.insert(leafSmoothnessTermBuffers[LeafMap[currNode]]);
			this->GetGPUBuffers();
		}
		
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
		this->GetGPUBuffers();
		
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
		this->GetGPUBuffers();
		
		//run kernel on CPU
		//std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
		NumKernelRuns++;
		CUDA_applyStep(CPU2GPUMap[branchDivBuffers[BranchMap[currNode]]], CPU2GPUMap[branchFlowXBuffers[BranchMap[currNode]]],
					   CPU2GPUMap[branchFlowYBuffers[BranchMap[currNode]]], CPU2GPUMap[branchFlowZBuffers[BranchMap[currNode]]],
					   VX, VY, VZ, VolumeSize, GetStream() );
		
		//add the smoothness term
		if(branchSmoothnessTermBuffers[BranchMap[currNode]]){
			this->CPUInUse.insert(branchSmoothnessTermBuffers[BranchMap[currNode]]);
			this->GetGPUBuffers();
		}

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
		SolveMaxFlow( this->Hierarchy->GetChild(currNode,kid) );

	// B : Add sink potential to working buffer
	if( isBranch ){
		//organize the GPU to obtain the buffers
		this->CPUInUse.clear();
		this->CPUInUse.insert(branchWorkingBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchIncBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchLabelBuffers[BranchMap[currNode]]);
		this->CPUInUse.insert(branchDivBuffers[BranchMap[currNode]]);
		this->GetGPUBuffers();

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
		this->GetGPUBuffers();

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
		this->GetGPUBuffers();

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
		this->GetGPUBuffers();
		
		//activate the first unconstrained kernel
		//std::cout << currNode << "\t Update sink flow" << std::endl;
		NumKernelRuns++;
		CUDA_updateLeafSinkFlow(CPU2GPUMap[leafSinkBuffers[LeafMap[currNode]]], CPU2GPUMap[leafIncBuffers[LeafMap[currNode]]],
								CPU2GPUMap[leafDivBuffers[LeafMap[currNode]]], CPU2GPUMap[leafLabelBuffers[LeafMap[currNode]]],
								CC, VolumeSize, GetStream() );

		this->CPUInUse.clear();
		this->CPUInUse.insert(leafSinkBuffers[LeafMap[currNode]]);
		this->CPUInUse.insert(leafDataTermBuffers[LeafMap[currNode]]);
		this->GetGPUBuffers();

		//activate the second constrained kernel
		NumKernelRuns++;
		CUDA_constrainLeafSinkFlow(CPU2GPUMap[leafSinkBuffers[LeafMap[currNode]]], CPU2GPUMap[leafDataTermBuffers[LeafMap[currNode]]],
									VolumeSize, GetStream() );
	}

	//RB : Update children's labels
	for(int kid = 0; kid < NumKids; kid++)
		UpdateLabel( this->Hierarchy->GetChild(currNode,kid) );

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
		this->GetGPUBuffers();

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

void vtkCudaHierarchicalMaxFlowSegmentation::UpdateLabel( vtkIdType node ){
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
		this->GetGPUBuffers();

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
		this->GetGPUBuffers();

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
		vtkDebugMacro(<< "Starting input data preparation and CPU buffer allocation." );

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

	//allocate source flow buffer
	sourceFlowBuffer = new float[VolumeSize];
	TotalNumberOfBuffers++;
	for(int x = 0; x < VolumeSize; x++)
		sourceFlowBuffer[x] = 0.0f;

	//allocate source working buffer
	sourceWorkingBuffer = new float[VolumeSize];
	TotalNumberOfBuffers++;
	for(int x = 0; x < VolumeSize; x++)
		sourceWorkingBuffer[x] = 0.0f;

	//allocate required memory buffers for the brach nodes
	//		buffers needed:
	//			per brach node (not the root):
	//				1 label buffer
	//				3 spatial flow buffers
	//				1 divergence buffer
	//				1 outgoing flow buffer
	//				1 working temp buffer (ie: guk)
	TotalNumberOfBuffers += 7*NumBranches;
	float* branchNodeBuffer = new float[7*VolumeSize*NumBranches];
	this->branchFlowXBuffers = new float* [NumBranches];
	this->branchFlowYBuffers = new float* [NumBranches];
	this->branchFlowZBuffers = new float* [NumBranches];
	this->branchDivBuffers = new float* [NumBranches];
	this->branchSinkBuffers = new float* [NumBranches];
	this->branchLabelBuffers = new float* [NumBranches];
	this->branchWorkingBuffers = new float* [NumBranches];
	float* ptr = branchNodeBuffer;
	for(int i = 0; i < NumBranches; i++ ){
		branchFlowXBuffers[i] = ptr; ptr += VolumeSize;
		branchFlowYBuffers[i] = ptr; ptr += VolumeSize;
		branchFlowZBuffers[i] = ptr; ptr += VolumeSize;
		branchDivBuffers[i] = ptr; ptr += VolumeSize;
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
	TotalNumberOfBuffers += 5*NumLeaves;
	float*	leafNodeBuffer = new float[5*VolumeSize*NumLeaves];
	this->leafFlowXBuffers = new float* [NumLeaves];
	this->leafFlowYBuffers = new float* [NumLeaves];
	this->leafFlowZBuffers = new float* [NumLeaves];
	this->leafDivBuffers = new float* [NumLeaves];
	this->leafSinkBuffers = new float* [NumLeaves];
	ptr = leafNodeBuffer;
	for(int i = 0; i < NumLeaves; i++ ){
		leafFlowXBuffers[i] = ptr; ptr += VolumeSize;
		leafFlowYBuffers[i] = ptr; ptr += VolumeSize;
		leafFlowZBuffers[i] = ptr; ptr += VolumeSize;
		leafDivBuffers[i] = ptr; ptr += VolumeSize;
		leafSinkBuffers[i] = ptr; ptr += VolumeSize;
	}

	//initalize all spatial flows and divergences to zero
	for(int x = 0; x < VolumeSize; x++){
		for(int i = 0; i < NumBranches; i++ ){
			(branchFlowXBuffers[i])[x] = 0.0f;
			(branchFlowYBuffers[i])[x] = 0.0f;
			(branchFlowZBuffers[i])[x] = 0.0f;
			(branchDivBuffers[i])[x] = 0.0f;
			(branchLabelBuffers[i])[x] = 0.0f;
			(branchWorkingBuffers[i])[x] = 0.0f;
		}
		for(int i = 0; i < NumLeaves; i++ ){
			(leafFlowXBuffers[i])[x] = 0.0f;
			(leafFlowYBuffers[i])[x] = 0.0f;
			(leafFlowZBuffers[i])[x] = 0.0f;
			(leafDivBuffers[i])[x] = 0.0f;
		}
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
		sourceFlowBuffer[x] = (leafDataTermBuffers[maxProbLabel])[x];
	}
	PropogateLabels( this->Hierarchy->GetRoot() );

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

	//create LIFO priority queue (priority stack) data structure
	FigureOutBufferPriorities( this->Hierarchy->GetRoot() );
	for( std::map<float*,int>::iterator priorityIt = CPU2PriorityMap.begin(); priorityIt != CPU2PriorityMap.end(); priorityIt++ )
		BuildStackUpToPriority(priorityIt->second);

	//add all the working buffers from the branches to the garbage (no copy necessary) list
	NoCopyBack.insert( sourceWorkingBuffer );
	for(int i = 0; i < NumBranches; i++ )
		NoCopyBack.insert( branchWorkingBuffers[i] );

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
		double NewPercentAcquired = 0;
		CUDA_GetGPUBuffers( TotalNumberOfBuffers-BuffersAcquired, this->MaxGPUUsage-PercentAcquired, &NewAcquiredBuffers, VolumeSize, &NewNumberAcquired, &NewPercentAcquired );
		BuffersAcquired += NewNumberAcquired;
		PercentAcquired += NewPercentAcquired;

		//if no new buffers were acquired, exit the loop
		if( NewNumberAcquired == 0 ) break;

		//else, load the new buffers into the list of unused buffers
		AllGPUBufferBlocks.push_back(NewAcquiredBuffers);
		for(int i = 0; i < NewNumberAcquired; i++){
			UnusedGPUBuffers.push_back(NewAcquiredBuffers);
			NewAcquiredBuffers += VolumeSize;
		}
	}

	//Solve maximum flow problem in an iterative bottom-up manner
	if( this->Debug )
		vtkDebugMacro(<<"Starting max-flow iterations.");
	NumMemCpies = 0;
	NumKernelRuns = 0;
	this->ReserveGPU();
	for( int iteration = 0; iteration < this->NumberOfIterations; iteration++ ){
		int oldNumMemCpies = NumMemCpies;
		SolveMaxFlow( this->Hierarchy->GetRoot() );
		
		if( this->Debug )
			vtkDebugMacro(<< "Finished iteration " << (iteration+1) << " with " << (NumMemCpies-oldNumMemCpies) << " memory transfers.");

	}

	//Copy back any uncopied leaf label buffers (others don't matter anymore)
	for( int i = 0; i < NumLeaves; i++ ){
		if( CPU2GPUMap.find(leafLabelBuffers[i]) != CPU2GPUMap.end() )
			ReturnBufferGPU2CPU(leafLabelBuffers[i], CPU2GPUMap.find(leafLabelBuffers[i])->second);
	}
	if( this->Debug )
		vtkDebugMacro(<< "Results copied back to CPU " );

	//Return all GPU buffers
	while( AllGPUBufferBlocks.size() > 0 ){
		CUDA_ReturnGPUBuffers( AllGPUBufferBlocks.front() );
		AllGPUBufferBlocks.pop_front();
	}
	
	//deallocate branch temporary buffers
	delete[] branchNodeBuffer;
	delete[] branchFlowXBuffers;
	delete[] branchFlowYBuffers;
	delete[] branchFlowZBuffers;
	delete[] branchDivBuffers;
	delete[] branchSinkBuffers;
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
	delete[] leafDivBuffers;
	delete[] leafIncBuffers;
	delete[] leafSinkBuffers;
	delete[] leafLabelBuffers;
	delete[] leafDataTermBuffers;
	delete[] leafSmoothnessTermBuffers;
	delete[] leafSmoothnessConstants;

	//delete source node buffers
	delete[] sourceFlowBuffer;
	delete[] sourceWorkingBuffer;

	return 1;
}

void vtkCudaHierarchicalMaxFlowSegmentation::FigureOutBufferPriorities( vtkIdType currNode ){
	
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

void vtkCudaHierarchicalMaxFlowSegmentation::GetGPUBuffers(){
	
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
			
			//update the priority stacks
			AddToStack(*iterator);
			continue;
		}

		//see if there is some garbage we can deallocate first
		bool flag = false;
		for( std::set<float*>::iterator iterator2 = NoCopyBack.begin();
			 iterator2 != NoCopyBack.end(); iterator2++ ){
			if( CPUInUse.find(*iterator2) != CPUInUse.end() ) continue;
			if( CPU2GPUMap.find(*iterator2) == CPU2GPUMap.end() ) continue;
			float* NewGPUBuffer = CPU2GPUMap[*iterator2];
			CPU2GPUMap.erase( CPU2GPUMap.find(*iterator2) );
			GPU2CPUMap.erase( GPU2CPUMap.find(NewGPUBuffer) );
			CPU2GPUMap.insert( std::pair<float*,float*>(*iterator, NewGPUBuffer) );
			GPU2CPUMap.insert( std::pair<float*,float*>(NewGPUBuffer, *iterator) );
			MoveBufferCPU2GPU(*iterator,NewGPUBuffer);
			
			//update the priority stacks
			RemoveFromStack(*iterator2);
			AddToStack(*iterator);
			flag = true;
			break;
		}
		if( flag ) continue;

		//else, we have to move something in use back to the CPU
		flag = false;
		std::list< std::list< float* > >::iterator stackIterator = PriorityStacks.begin();
		for( ; stackIterator != PriorityStacks.end(); stackIterator++ ){
			for(std::list< float* >::iterator subIterator = stackIterator->begin(); subIterator != stackIterator->end(); subIterator++ ){
				
				//can't remove this one because it is in use
				if( CPUInUse.find( *subIterator ) != CPUInUse.end() ) continue;

				//else, find it and move it back to the CPU
				float* NewGPUBuffer = CPU2GPUMap.find(*subIterator)->second;
				CPU2GPUMap.erase( CPU2GPUMap.find(*subIterator) );
				GPU2CPUMap.erase( GPU2CPUMap.find(NewGPUBuffer) );
				CPU2GPUMap.insert( std::pair<float*,float*>(*iterator, NewGPUBuffer) );
				GPU2CPUMap.insert( std::pair<float*,float*>(NewGPUBuffer, *iterator) );
				ReturnBufferGPU2CPU(*subIterator,NewGPUBuffer);
				MoveBufferCPU2GPU(*iterator,NewGPUBuffer);
				
				//update the priority stack and leave immediately since our iterators
				//no longer have a valid contract (changed container)
				RemoveFromStack(*subIterator);
				AddToStack(*iterator);
				flag = true;
				break;

			}
			if( flag ) break;
		}


	}
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

void vtkCudaHierarchicalMaxFlowSegmentation::BuildStackUpToPriority( int priority ){
	while( PriorityStacks.size() < priority ){
		PriorityStacks.push_back( std::list<float*>() );
		Priority.push_back( (int) PriorityStacks.size() );
	}
}

void vtkCudaHierarchicalMaxFlowSegmentation::AddToStack( float* CPUBuffer ){
	int neededPriority = CPU2PriorityMap.find(CPUBuffer)->second;
	std::list< std::list< float* > >::iterator stackIterator = PriorityStacks.begin();
	std::list< int >::iterator priorityIterator = Priority.begin();
	for( ; *priorityIterator != neededPriority; stackIterator++, priorityIterator++);
	stackIterator->push_front(CPUBuffer);
}

void vtkCudaHierarchicalMaxFlowSegmentation::RemoveFromStack( float* CPUBuffer ){
	int neededPriority = CPU2PriorityMap.find(CPUBuffer)->second;
	std::list< std::list< float* > >::iterator stackIterator = PriorityStacks.begin();
	std::list< int >::iterator priorityIterator = Priority.begin();
	for( ; *priorityIterator != neededPriority; stackIterator++, priorityIterator++);
	for(std::list< float* >::iterator subIterator = stackIterator->begin(); subIterator != stackIterator->end(); subIterator++ ){
		if( *subIterator == CPUBuffer ){
			stackIterator->erase(subIterator);
			return;
		}
	}
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
