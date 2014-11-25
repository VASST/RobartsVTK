/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkDirectedAcyclicGraphMaxFlowSegmentation.cxx

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkDirectedAcyclicGraphMaxFlowSegmentation.cxx
 *
 *  @brief Implementation file with definitions of CPU-based solver for generalized DirectedAcyclicGraph max-flow
 *			segmentation problems.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *	
 *	@note May 20th 2014 - Documentation first compiled.
 *
 *  @note This is the base class for GPU accelerated max-flow segmentors in vtkCudaImageAnalytics
 *
 */

#include "vtkDirectedAcyclicGraphMaxFlowSegmentation.h"
#include "vtkMaxFlowSegmentationUtilities.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkRootedDirectedAcyclicGraphForwardIterator.h"
#include "vtkRootedDirectedAcyclicGraphBackwardIterator.h"
#include "vtkImageData.h"
#include "vtkFloatArray.h"
#include "vtkDataSetAttributes.h"

#include <assert.h>
#include <float.h>
#include <limits.h>

#include <set>
#include <list>
#include <vector>

#define SQR(X) X*X

vtkStandardNewMacro(vtkDirectedAcyclicGraphMaxFlowSegmentation);

vtkDirectedAcyclicGraphMaxFlowSegmentation::vtkDirectedAcyclicGraphMaxFlowSegmentation(){
	
	//configure the IO ports
	this->SetNumberOfInputPorts(2);
	this->SetNumberOfOutputPorts(1);

	//set algorithm mathematical parameters to defaults
	this->NumberOfIterations = 100;
	this->StepSize = 0.1;
	this->CC = 0.25;

	//set up the input mapping structure
	this->InputDataPortMapping.clear();
	this->BackwardsInputDataPortMapping.clear();
	this->FirstUnusedDataPort = 0;
	this->InputSmoothnessPortMapping.clear();
	this->BackwardsInputSmoothnessPortMapping.clear();
	this->FirstUnusedSmoothnessPort = 0;
	this->BranchNumParents = 0;
	this->BranchNumChildren = 0;
	this->BranchWeightedNumChildren = 0;
	this->LeafNumParents = 0;

	//set the other values to defaults
	this->Structure = 0;
	this->SmoothnessScalars.clear();
	this->LeafMap.clear();
	this->BranchMap.clear();

}

vtkDirectedAcyclicGraphMaxFlowSegmentation::~vtkDirectedAcyclicGraphMaxFlowSegmentation(){
	if( this->Structure ) this->Structure->UnRegister(this);
	this->SmoothnessScalars.clear();
	this->LeafMap.clear();
	this->InputDataPortMapping.clear();
	this->BackwardsInputDataPortMapping.clear();
	this->InputSmoothnessPortMapping.clear();
	this->BackwardsInputSmoothnessPortMapping.clear();
	this->BranchMap.clear();
}

//------------------------------------------------------------

void vtkDirectedAcyclicGraphMaxFlowSegmentation::AddSmoothnessScalar(vtkIdType node, double value){
	if( value >= 0.0 ){
		this->SmoothnessScalars.insert(std::pair<vtkIdType,double>(node,value));
		this->Modified();
	}else{
		vtkErrorMacro(<<"Cannot use a negative smoothness value.");
	}
}

//------------------------------------------------------------

void vtkDirectedAcyclicGraphMaxFlowSegmentation::Update(){
	this->SetOutputPortAmount();
	this->Superclass::Update();
}

void vtkDirectedAcyclicGraphMaxFlowSegmentation::UpdateInformation(){
	this->SetOutputPortAmount();
	this->Superclass::UpdateInformation();
}

void vtkDirectedAcyclicGraphMaxFlowSegmentation::UpdateWholeExtent(){
	this->SetOutputPortAmount();
	this->Superclass::UpdateWholeExtent();
}

void vtkDirectedAcyclicGraphMaxFlowSegmentation::SetOutputPortAmount(){
	int NumLeaves = 0;
	vtkRootedDirectedAcyclicGraphForwardIterator* iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
	iterator->SetDAG(this->Structure);
	iterator->SetRootVertex(this->Structure->GetRoot());
	while( iterator->HasNext() ){
		vtkIdType currNode = iterator->Next();
		if( this->Structure->IsLeaf(currNode) ) NumLeaves++;
	}
	iterator->Delete();
	this->SetNumberOfOutputPorts(NumLeaves);
}

//------------------------------------------------------------

int vtkDirectedAcyclicGraphMaxFlowSegmentation::FillInputPortInformation(int i, vtkInformation* info){
	info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
	info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
	info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
	return this->Superclass::FillInputPortInformation(i,info);
}

void vtkDirectedAcyclicGraphMaxFlowSegmentation::SetDataInput(int idx, vtkDataObject *input)
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

void vtkDirectedAcyclicGraphMaxFlowSegmentation::SetSmoothnessInput(int idx, vtkDataObject *input)
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

vtkDataObject *vtkDirectedAcyclicGraphMaxFlowSegmentation::GetDataInput(int idx)
{
	if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputDataPortMapping[idx]));
}

vtkDataObject *vtkDirectedAcyclicGraphMaxFlowSegmentation::GetSmoothnessInput(int idx)
{
	if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(1, this->InputSmoothnessPortMapping[idx]));
}

vtkDataObject *vtkDirectedAcyclicGraphMaxFlowSegmentation::GetOutput(int idx)
{
	//look up port in mapping
	std::map<vtkIdType,int>::iterator port = this->LeafMap.find(idx);
	if( port == this->LeafMap.end() )
		return 0;

	return vtkImageData::SafeDownCast(this->GetExecutive()->GetOutputData(port->second));
}

//----------------------------------------------------------------------------

int vtkDirectedAcyclicGraphMaxFlowSegmentation::CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges ){
	
	//check to make sure that the Structure is specified
	if( !this->Structure ){
		vtkErrorMacro(<<"Structure must be provided.");
		return -1;
	}

	this->LeafMap.clear();
	this->BranchMap.clear();

	//check to make sure that there is an image associated with each leaf node
	NumLeaves = 0;
	NumNodes = 0;
	NumEdges = 0;
	Extent[0] = -1;
	vtkRootedDirectedAcyclicGraphForwardIterator* iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
	iterator->SetDAG(this->Structure);
	iterator->SetRootVertex(this->Structure->GetRoot());
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		NumNodes++;

		NumEdges += this->Structure->GetNumberOfChildren(node);

        if( this->Structure->IsLeaf(node) ){
            int value = (int) this->LeafMap.size();
            this->LeafMap[node] = value;
        }else if(node != this->Structure->GetRoot()){
            int value = (int) this->BranchMap.size();
            this->BranchMap[node] = value;
        }

		//make sure all leaf nodes have a data term
		if( this->Structure->IsLeaf(node) ){

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

	//find edges based on \sum{degree(V)} = 2E
	NumEdges = Structure->GetNumberOfEdges();

	return 0;
}

int vtkDirectedAcyclicGraphMaxFlowSegmentation::RequestInformation(
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

int vtkDirectedAcyclicGraphMaxFlowSegmentation::RequestUpdateExtent(
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

int vtkDirectedAcyclicGraphMaxFlowSegmentation::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){
								
	//check input consistancy
	int Extent[6];
	int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
	if( result || NumNodes == 0 ) return -1;
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
	vtkRootedDirectedAcyclicGraphForwardIterator* iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
	iterator->SetDAG(this->Structure);
	iterator->SetRootVertex(this->Structure->GetRoot());
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( node == this->Structure->GetRoot() ) continue;
		if( !this->Structure->IsLeaf(node) )
			BranchMap.insert(std::pair<vtkIdType,int>(node,(int) this->BranchMap.size()));
	}
	iterator->Delete();

	//get the data term buffers
	this->leafDataTermBuffers = new float* [NumLeaves];
	iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
	iterator->SetDAG(this->Structure);
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( this->Structure->IsLeaf(node) ){
			int inputNumber = this->InputDataPortMapping[node];
			vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputNumber)->Get(vtkDataObject::DATA_OBJECT()));
			leafDataTermBuffers[this->LeafMap[node]] = (float*) CurrImage->GetScalarPointer();

			//add the data term buffer in and set it to read only
			TotalNumberOfBuffers++;

		}
	}
	iterator->Delete();
	
	//get the smoothness term buffers
	this->leafSmoothnessTermBuffers = new float* [NumLeaves];
	this->branchSmoothnessTermBuffers = new float* [NumBranches];
	iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
	iterator->SetDAG(this->Structure);
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( node == this->Structure->GetRoot() ) continue;
		vtkImageData* CurrImage = 0;
		if( this->InputSmoothnessPortMapping.find(node) != this->InputSmoothnessPortMapping.end() ){
			int inputNumber = this->InputSmoothnessPortMapping[node];
			CurrImage = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(inputNumber)->Get(vtkDataObject::DATA_OBJECT()));
		}
		if( this->Structure->IsLeaf(node) )
			leafSmoothnessTermBuffers[this->LeafMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
		else
			branchSmoothnessTermBuffers[this->BranchMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
		
		// add the smoothness buffer in as read only
		if( CurrImage ){
			TotalNumberOfBuffers++;
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
	iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
	iterator->SetDAG(this->Structure);
	leafSmoothnessConstants = new float[NumLeaves];
	branchSmoothnessConstants = new float[NumBranches];
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( node == this->Structure->GetRoot() ) continue;
		if( this->Structure->IsLeaf(node) )
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
	

	//allocate required memory buffers for the brach nodes
	int NumBuffersPerBranch = 8;
	int NumBuffersPerLeaf = 6;
	int NumBuffersSource = 2;
	//		buffers needed:
	//			per brach node (not the root):
	//				1 label buffer
	//				3 spatial flow buffers
	//				1 divergence buffer
	//				1 outgoing flow buffer
	//				1 incoming flow buffer
	//				1 working buffer
	//and for the leaf nodes
	//			per leaf node (note label buffer is in output)
	//				3 spatial flow buffers
	//				1 divergence buffer
	//				1 sink flow buffer
	//				1 incoming flow buffer
	int NumberOfAdditionalCPUBuffersNeeded = 0;
	
	//source flow and working buffers
	std::list<float**> BufferPointerLocs;
	NumberOfAdditionalCPUBuffersNeeded += NumBuffersSource;
	TotalNumberOfBuffers += NumBuffersSource;
	BufferPointerLocs.push_front(&sourceFlowBuffer);
	BufferPointerLocs.push_front(&sourceWorkingBuffer);

	//allocate those buffer pointers and put on list
	NumberOfAdditionalCPUBuffersNeeded += NumBuffersPerBranch * NumBranches;
	TotalNumberOfBuffers += NumBuffersPerBranch * NumBranches;
	NumberOfAdditionalCPUBuffersNeeded += NumBuffersPerLeaf * NumLeaves;
	TotalNumberOfBuffers += NumBuffersPerLeaf * NumLeaves;
	float** bufferPointers = new float* [NumBuffersPerBranch * NumBranches + NumBuffersPerLeaf * NumLeaves];
	float** tempPtr = bufferPointers;
	this->branchFlowXBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchFlowYBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchFlowZBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchDivBuffers =		tempPtr; tempPtr += NumBranches;
	this->branchSourceBuffers =		tempPtr; tempPtr += NumBranches;
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
		BufferPointerLocs.push_front(&(branchSourceBuffers[i]));
	for(int i = 0; i < NumBranches; i++ )
		BufferPointerLocs.push_front(&(branchLabelBuffers[i]));
	for(int i = 0; i < NumBranches; i++ )
		BufferPointerLocs.push_front(&(branchWorkingBuffers[i]));
	this->leafFlowXBuffers =		tempPtr; tempPtr += NumLeaves;
	this->leafFlowYBuffers =		tempPtr; tempPtr += NumLeaves;
	this->leafFlowZBuffers =		tempPtr; tempPtr += NumLeaves;
	this->leafDivBuffers =			tempPtr; tempPtr += NumLeaves;
	this->leafSourceBuffers =		tempPtr; tempPtr += NumLeaves;
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
	for(int i = 0; i < NumLeaves; i++ )
		BufferPointerLocs.push_front(&(leafSourceBuffers[i]));

	//try to obtain required CPU buffers
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
	if( this->Debug )
		vtkDebugMacro(<<"Relate parent sink with child source buffer pointers.");

	//figure out weightings on tree
	BranchNumParents = new float[NumBranches];
	for(int i = 0; i < NumBranches; i++) BranchNumParents[i] = 0.0f;
	BranchNumChildren = new float[NumBranches];
	for(int i = 0; i < NumBranches; i++) BranchNumChildren[i] = 0.0f;
	BranchWeightedNumChildren = new float[NumBranches];
	for(int i = 0; i < NumBranches; i++) BranchWeightedNumChildren[i] = 0.0f;
	LeafNumParents = new float[NumLeaves];
	for(int i = 0; i < NumLeaves; i++) LeafNumParents[i] = 0.0f;
	SourceNumChildren = 0.0f;
	SourceWeightedNumChildren = 0.0f;
	vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));
	vtkRootedDirectedAcyclicGraphBackwardIterator* Iterator = vtkRootedDirectedAcyclicGraphBackwardIterator::New();
	Iterator->SetDAG(this->Structure);
	Iterator->SetRootVertex(this->Structure->GetRoot());
	while(Iterator->HasNext()){
		vtkIdType CurrNode = Iterator->Next();

		//find the number of parents
		if(this->Structure->IsLeaf(CurrNode))
			for(vtkIdType i = 0; i < this->Structure->GetNumberOfParents(CurrNode); i++)
				LeafNumParents[LeafMap[CurrNode]] += Weights ? Weights->GetValue(this->Structure->GetInEdge(CurrNode,i).Id) : 1.0;
		else if(this->Structure->GetRoot() != CurrNode)
			for(vtkIdType i = 0; i < this->Structure->GetNumberOfParents(CurrNode); i++)
				BranchNumParents[BranchMap[CurrNode]] += Weights ? Weights->GetValue(this->Structure->GetInEdge(CurrNode,i).Id) : 1.0;

		//find the number of children
		if(this->Structure->GetRoot() == CurrNode)
			for(vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(CurrNode); i++)
				SourceNumChildren += Weights ? Weights->GetValue(this->Structure->GetOutEdge(CurrNode,i).Id) : 1.0;
		else
			for(vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(CurrNode); i++)
				BranchNumChildren[BranchMap[CurrNode]] += Weights ? Weights->GetValue(this->Structure->GetOutEdge(CurrNode,i).Id) : 1.0;

	}
	Iterator->Restart();
	while(Iterator->HasNext()){
		vtkIdType CurrNode = Iterator->Next();

		//find the number of children weighted
		if(this->Structure->GetRoot() == CurrNode)
			for(vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(CurrNode); i++){
				float temp = (Weights ? Weights->GetValue(this->Structure->GetOutEdge(CurrNode,i).Id) : 1.0 ) /
					(this->Structure->IsLeaf(this->Structure->GetChild(CurrNode,i)) ? LeafNumParents[LeafMap[this->Structure->GetChild(CurrNode,i)]] :
					BranchNumParents[BranchMap[this->Structure->GetChild(CurrNode,i)]] );
				SourceWeightedNumChildren += temp*temp;
			}
		else
			for(vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(CurrNode); i++){
				 float temp = (Weights ? Weights->GetValue(this->Structure->GetOutEdge(CurrNode,i).Id) : 1.0 ) /
					(this->Structure->IsLeaf(this->Structure->GetChild(CurrNode,i)) ? LeafNumParents[LeafMap[this->Structure->GetChild(CurrNode,i)]] :
					BranchNumParents[BranchMap[this->Structure->GetChild(CurrNode,i)]] );
				 BranchWeightedNumChildren[BranchMap[CurrNode]] += temp*temp;
			}

	}
	Iterator->Delete();

	//run algorithm proper
	if( this->Debug )
		vtkDebugMacro(<<"Starting initialization");
	this->InitializeAlgorithm();
	if( this->Debug )
		vtkDebugMacro(<<"Starting max-flow algorithm.");
	this->RunAlgorithm();

	//deallocate CPU buffers
	while( CPUBuffersAcquired.size() > 0 ){
		float* tempBuffer = CPUBuffersAcquired.front();
		delete[] tempBuffer;
		CPUBuffersAcquired.pop_front();
	}
	CPUBuffersAcquired.clear();
	CPUBuffersSize.clear();

	//deallocate structure that holds the pointers to the buffers
	delete[] bufferPointers;
	delete[] BranchNumParents;	BranchNumParents = 0;
	delete[] BranchNumChildren;	BranchNumChildren = 0;
	delete[] LeafNumParents;	LeafNumParents = 0;
	SourceNumChildren = 0;

	return 1;
}

int vtkDirectedAcyclicGraphMaxFlowSegmentation::RequestDataObject(
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

int vtkDirectedAcyclicGraphMaxFlowSegmentation::InitializeAlgorithm(){

	//initalize all spatial flows and divergences to zero
	for(int i = 0; i < NumBranches; i++ ){
		zeroOutBuffer(branchFlowXBuffers[i], VolumeSize);
		zeroOutBuffer(branchFlowYBuffers[i], VolumeSize);
		zeroOutBuffer(branchFlowZBuffers[i], VolumeSize);
		zeroOutBuffer(branchDivBuffers[i], VolumeSize);
	}
	for(int i = 0; i < NumLeaves; i++ ){
		zeroOutBuffer(leafFlowXBuffers[i], VolumeSize);
		zeroOutBuffer(leafFlowYBuffers[i], VolumeSize);
		zeroOutBuffer(leafFlowZBuffers[i], VolumeSize);
		zeroOutBuffer(leafDivBuffers[i], VolumeSize);
	}

	//initialize all leak sink flows to their constraints
	for(int i = 0; i < NumLeaves; i++ )
		copyBuffer(leafSinkBuffers[i], leafDataTermBuffers[i], VolumeSize);

	//find the minimum sink flow
	for(int i = 1; i < NumLeaves; i++ )
		minBuffer(leafSinkBuffers[0], leafSinkBuffers[i], VolumeSize);

	//copy minimum sink flow over all leaves and sum the resulting labels into the source flow buffer
	lblBuffer(leafLabelBuffers[0], leafSinkBuffers[0], leafDataTermBuffers[0], VolumeSize);
	copyBuffer(sourceFlowBuffer, leafLabelBuffers[0], VolumeSize);
	for(int i = 1; i < NumLeaves; i++ ){
		copyBuffer(leafSinkBuffers[i], leafSinkBuffers[0], VolumeSize);
		copyBuffer(leafSourceBuffers[i], leafSinkBuffers[0], VolumeSize);
		lblBuffer(leafLabelBuffers[i], leafSinkBuffers[i], leafDataTermBuffers[i], VolumeSize);
		sumBuffer(sourceFlowBuffer, leafLabelBuffers[i], VolumeSize);
	}

	//divide the labels out to constrain them to validity
	for(int i = 0; i < NumLeaves; i++ )
		divBuffer(leafLabelBuffers[i], sourceFlowBuffer, VolumeSize);

	//apply minimal sink flow over the remaining Structure
	for(int i = 0; i < NumBranches; i++ ){
		copyBuffer(branchSinkBuffers[i], leafSinkBuffers[0], VolumeSize);
		copyBuffer(branchSourceBuffers[i], leafSinkBuffers[0], VolumeSize);
	}
	for(int i = 0; i < NumLeaves; i++ )
		copyBuffer(leafSourceBuffers[i], leafSinkBuffers[0], VolumeSize);
	copyBuffer(sourceFlowBuffer, leafSinkBuffers[0], VolumeSize);

	//propogate labels up the Structure
	PropogateLabels( );

	return 1;
}

int vtkDirectedAcyclicGraphMaxFlowSegmentation::RunAlgorithm(){
	//Solve maximum flow problem in an iterative bottom-up manner
	for( int iteration = 0; iteration < this->NumberOfIterations; iteration++ ){
		SolveMaxFlow();
		if( this->Debug )
			vtkDebugMacro(<< "Finished iteration " << (iteration+1) << ".");
	}
	return 1;
}

void vtkDirectedAcyclicGraphMaxFlowSegmentation::PropogateLabels( ){
	
	vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

	vtkRootedDirectedAcyclicGraphBackwardIterator* Iterator = vtkRootedDirectedAcyclicGraphBackwardIterator::New();
	Iterator->SetDAG(this->Structure);
	Iterator->SetRootVertex(this->Structure->GetRoot());
	while(Iterator->HasNext()){
		vtkIdType CurrNode = Iterator->Next();
		
		//if we are a leaf or root label, we are finished and can therefore leave
		if(this->Structure->IsLeaf(CurrNode) || this->Structure->GetRoot() == CurrNode ) continue;

		//clear own label buffer
		zeroOutBuffer(branchLabelBuffers[BranchMap[CurrNode]],VolumeSize);

		//sum in weighted version of child's label
		for(vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(CurrNode); i++ ){
			float W = Weights ? Weights->GetValue(this->Structure->GetOutEdge(CurrNode,i).Id) : 1.0f;
			if(this->Structure->IsLeaf(this->Structure->GetChild(CurrNode,i)))
				sumScaledBuffer(branchLabelBuffers[BranchMap[CurrNode]],
								leafLabelBuffers[LeafMap[this->Structure->GetChild(CurrNode,i)]],
								W / LeafNumParents[LeafMap[this->Structure->GetChild(CurrNode,i)]],
								VolumeSize);
			else
				sumScaledBuffer(branchLabelBuffers[BranchMap[CurrNode]],
								branchLabelBuffers[BranchMap[this->Structure->GetChild(CurrNode,i)]],
								W / BranchNumParents[BranchMap[this->Structure->GetChild(CurrNode,i)]],
								VolumeSize);
		}

	}
	Iterator->Delete();
}

void vtkDirectedAcyclicGraphMaxFlowSegmentation::SolveMaxFlow( ){
	vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
	vtkRootedDirectedAcyclicGraphBackwardIterator* BackIterator = vtkRootedDirectedAcyclicGraphBackwardIterator::New();
	ForIterator->SetDAG(this->Structure);
	BackIterator->SetDAG(this->Structure);
	vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

	//update spatial flows (order independant)
	ForIterator->SetRootVertex(this->Structure->GetRoot());
	ForIterator->Restart();
	while(ForIterator->HasNext()){
		vtkIdType currNode = ForIterator->Next();
		if( this->Structure->IsLeaf(currNode) ){

			//compute the gradient step amount (store in div buffer for now)
			//std::cout << currNode << "\t Find gradient descent step size" << std::endl;
			dagmf_flowGradientStep(leafSinkBuffers[LeafMap[currNode]], leafSourceBuffers[LeafMap[currNode]],
									leafDivBuffers[LeafMap[currNode]], leafLabelBuffers[LeafMap[currNode]],
									StepSize, CC, VolumeSize);

			//apply gradient descent to the flows
			//std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
			dagmf_applyStep(leafDivBuffers[LeafMap[currNode]], leafFlowXBuffers[LeafMap[currNode]],
							leafFlowYBuffers[LeafMap[currNode]], leafFlowZBuffers[LeafMap[currNode]],
							VX, VY, VZ, VolumeSize);
		
			//std::cout << currNode << "\t Find Projection multiplier" << std::endl;
			dagmf_computeFlowMag(leafDivBuffers[LeafMap[currNode]], leafFlowXBuffers[LeafMap[currNode]],
							leafFlowYBuffers[LeafMap[currNode]], leafFlowZBuffers[LeafMap[currNode]],
							leafSmoothnessTermBuffers[LeafMap[currNode]], leafSmoothnessConstants[LeafMap[currNode]],
							VX, VY, VZ, VolumeSize);
		
			//project onto set and recompute the divergence
			//std::cout << currNode << "\t Project flows into valid range and compute divergence" << std::endl;
			dagmf_projectOntoSet(leafDivBuffers[LeafMap[currNode]], leafFlowXBuffers[LeafMap[currNode]],
							leafFlowYBuffers[LeafMap[currNode]], leafFlowZBuffers[LeafMap[currNode]],
							VX, VY, VZ, VolumeSize);

		}else if( currNode != this->Structure->GetRoot() ){
		
			//std::cout << currNode << "\t Find gradient descent step size" << std::endl;
			dagmf_flowGradientStep(branchSinkBuffers[BranchMap[currNode]], branchSourceBuffers[BranchMap[currNode]],
									branchDivBuffers[BranchMap[currNode]], branchLabelBuffers[BranchMap[currNode]],
									StepSize, CC,VolumeSize);
		
			//std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
			dagmf_applyStep(branchDivBuffers[BranchMap[currNode]], branchFlowXBuffers[BranchMap[currNode]],
							branchFlowYBuffers[BranchMap[currNode]], branchFlowZBuffers[BranchMap[currNode]],
							VX, VY, VZ, VolumeSize);

			//compute the multiplier for projecting back onto the feasible flow set (and store in div buffer)
			//std::cout << currNode << "\t Find Projection multiplier" << std::endl;
			dagmf_computeFlowMag(branchDivBuffers[BranchMap[currNode]], branchFlowXBuffers[BranchMap[currNode]],
							branchFlowYBuffers[BranchMap[currNode]], branchFlowZBuffers[BranchMap[currNode]],
							branchSmoothnessTermBuffers[BranchMap[currNode]], branchSmoothnessConstants[BranchMap[currNode]],
							VX, VY, VZ, VolumeSize);
		
			//project onto set and recompute the divergence
			dagmf_projectOntoSet(branchDivBuffers[BranchMap[currNode]], branchFlowXBuffers[BranchMap[currNode]],
							branchFlowYBuffers[BranchMap[currNode]], branchFlowZBuffers[BranchMap[currNode]],
							VX, VY, VZ, VolumeSize);
		}
	}
	
	//clear source buffers working down 
	ForIterator->SetRootVertex(this->Structure->GetRoot());
	ForIterator->Restart();
	while(ForIterator->HasNext()){
		vtkIdType currNode = ForIterator->Next();
		if(this->Structure->IsLeaf(currNode))
			zeroOutBuffer(leafSourceBuffers[LeafMap[currNode]],VolumeSize);
		else if(currNode != this->Structure->GetRoot() )
			zeroOutBuffer(branchSourceBuffers[BranchMap[currNode]],VolumeSize);
	}

	//populate source for root's children
	for(vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(this->Structure->GetRoot()); i++){
		vtkIdType Child = this->Structure->GetChild(this->Structure->GetRoot(),i);
		float W = Weights ? Weights->GetValue( this->Structure->GetOutEdge(this->Structure->GetRoot(),i).Id ) : 1.0f;
		if(this->Structure->IsLeaf(Child))
			sumScaledBuffer(leafSourceBuffers[LeafMap[Child]],
				sourceFlowBuffer, W/this->LeafNumParents[LeafMap[Child]], VolumeSize);
		else
			sumScaledBuffer(branchSourceBuffers[BranchMap[Child]],
				sourceFlowBuffer, W/this->BranchNumParents[BranchMap[Child]], VolumeSize);
	}
		
	//propogate source for all other children
	ForIterator->SetRootVertex(this->Structure->GetRoot());
	ForIterator->Restart();
	while(ForIterator->HasNext()){
		vtkIdType CurrNode = ForIterator->Next();
		if(CurrNode == this->Structure->GetRoot()) continue;
		for(vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(CurrNode); i++){
			vtkIdType Child = this->Structure->GetChild(CurrNode,i);
			float W = Weights ? Weights->GetValue( this->Structure->GetOutEdge(CurrNode,i).Id ) : 1.0f;
			if(this->Structure->IsLeaf(Child))
				sumScaledBuffer(leafSourceBuffers[LeafMap[Child]], branchSinkBuffers[BranchMap[CurrNode]],
					W/this->LeafNumParents[LeafMap[Child]], VolumeSize);
			else
				sumScaledBuffer(branchSourceBuffers[BranchMap[Child]], branchSinkBuffers[BranchMap[CurrNode]],
					W/this->BranchNumParents[BranchMap[Child]], VolumeSize);
		}
	}

	//clear working buffers
	translateBuffer(sourceWorkingBuffer,sourceFlowBuffer,1.0/this->CC,SourceWeightedNumChildren, VolumeSize);
	ForIterator->SetRootVertex(this->Structure->GetRoot());
	ForIterator->Restart();
	while(ForIterator->HasNext()){
		vtkIdType CurrNode = ForIterator->Next();
		if(CurrNode == this->Structure->GetRoot() || this->Structure->IsLeaf(CurrNode)) continue;
		dagmf_storeSinkFlowInBuffer(branchWorkingBuffers[BranchMap[CurrNode]], branchSourceBuffers[BranchMap[CurrNode]],
					branchDivBuffers[BranchMap[CurrNode]], branchLabelBuffers[BranchMap[CurrNode]],
					branchSinkBuffers[BranchMap[CurrNode]],this->BranchWeightedNumChildren[BranchMap[CurrNode]],
					CC, VolumeSize);
	}

	//update sink flows and labels working up
	BackIterator->SetRootVertex(this->Structure->GetRoot());
	BackIterator->Restart();
	while(BackIterator->HasNext()){
		vtkIdType CurrNode = BackIterator->Next();
		if(this->Structure->IsLeaf(CurrNode)){

			//update state at this location (source, sink, labels)
			updateLeafSinkFlow(leafSinkBuffers[LeafMap[CurrNode]], leafSourceBuffers[LeafMap[CurrNode]],
					leafDivBuffers[LeafMap[CurrNode]], leafLabelBuffers[LeafMap[CurrNode]],
					CC, VolumeSize);
			constrainBuffer(leafSinkBuffers[LeafMap[CurrNode]], leafDataTermBuffers[LeafMap[CurrNode]],
										VolumeSize);
				
			//push up sink capacities
			for(vtkIdType i = 0; i < this->Structure->GetNumberOfParents(CurrNode); i++){
				vtkIdType Parent = this->Structure->GetParent(CurrNode,i);
				float W = Weights ? Weights->GetValue(this->Structure->GetInEdge(CurrNode,i).Id) : 1.0f;
				if(Parent == this->Structure->GetRoot() )
					dagmf_storeSourceFlowInBuffer(sourceWorkingBuffer, leafSinkBuffers[LeafMap[CurrNode]],
											leafDivBuffers[LeafMap[CurrNode]], leafLabelBuffers[LeafMap[CurrNode]],
											leafSourceBuffers[LeafMap[CurrNode]],sourceFlowBuffer,
											CC, W/LeafNumParents[LeafMap[CurrNode]], VolumeSize);
				else
					dagmf_storeSourceFlowInBuffer(branchWorkingBuffers[BranchMap[Parent]], leafSinkBuffers[LeafMap[CurrNode]],
											leafDivBuffers[LeafMap[CurrNode]], leafLabelBuffers[LeafMap[CurrNode]],
											leafSourceBuffers[LeafMap[CurrNode]],branchSinkBuffers[BranchMap[Parent]],
											CC, W/LeafNumParents[LeafMap[CurrNode]], VolumeSize);
			}
			
			updateLabel(leafSinkBuffers[LeafMap[CurrNode]], leafSourceBuffers[LeafMap[CurrNode]],
						leafDivBuffers[LeafMap[CurrNode]], leafLabelBuffers[LeafMap[CurrNode]],
						CC, VolumeSize);

		}else if(CurrNode != this->Structure->GetRoot()){

			//update state at this location (source, sink, labels)
			divAndStoreBuffer(branchSinkBuffers[BranchMap[CurrNode]],branchWorkingBuffers[BranchMap[CurrNode]],
				this->BranchWeightedNumChildren[BranchMap[CurrNode]]+1.0f,VolumeSize);

			//push up sink capacities
			for(vtkIdType i = 0; i < this->Structure->GetNumberOfParents(CurrNode); i++){
				vtkIdType Parent = this->Structure->GetParent(CurrNode,i);
				float W = Weights ? Weights->GetValue(this->Structure->GetInEdge(CurrNode,i).Id) : 1.0f;
				if(Parent == this->Structure->GetRoot() )
					dagmf_storeSourceFlowInBuffer(sourceWorkingBuffer, branchSinkBuffers[BranchMap[CurrNode]],
											branchDivBuffers[BranchMap[CurrNode]], branchLabelBuffers[BranchMap[CurrNode]],
											branchSourceBuffers[BranchMap[CurrNode]],sourceFlowBuffer,
											CC, W/BranchNumParents[BranchMap[CurrNode]], VolumeSize);
				else
					dagmf_storeSourceFlowInBuffer(branchWorkingBuffers[BranchMap[Parent]], branchSinkBuffers[BranchMap[CurrNode]],
											branchDivBuffers[BranchMap[CurrNode]], branchLabelBuffers[BranchMap[CurrNode]],
											branchSourceBuffers[BranchMap[CurrNode]],branchSinkBuffers[BranchMap[Parent]],
											CC, W/BranchNumParents[BranchMap[CurrNode]], VolumeSize);
			}
			
			updateLabel(branchSinkBuffers[BranchMap[CurrNode]], branchSourceBuffers[BranchMap[CurrNode]],
						branchDivBuffers[BranchMap[CurrNode]], branchLabelBuffers[BranchMap[CurrNode]],
						CC, VolumeSize);

		}else{
			divAndStoreBuffer(sourceFlowBuffer,sourceWorkingBuffer, this->SourceWeightedNumChildren,VolumeSize);
		}

	}

	ForIterator->Delete();
	BackIterator->Delete();
}
