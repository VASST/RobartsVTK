/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkHierarchicalMaxFlowSegmentation.cxx

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkHierarchicalMaxFlowSegmentation.cxx
 *
 *  @brief Implementation file with definitions of CPU-based solver for generalized hierarchical max-flow
 *      segmentation problems.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 *  @note This is the base class for GPU accelerated max-flow segmentors in vtkCudaImageAnalytics
 *
 */

#include "vtkHierarchicalMaxFlowSegmentation.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMaxFlowSegmentationUtilities.h"
#include "vtkObjectFactory.h"
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTree.h"
#include "vtkTreeDFSIterator.h"
#include "vtkTrivialProducer.h"

#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <set>
#include <list>

#include <vtkVersion.h> //for VTK_MAJOR_VERSION

#define SQR(X) X*X

vtkStandardNewMacro(vtkHierarchicalMaxFlowSegmentation);

vtkHierarchicalMaxFlowSegmentation::vtkHierarchicalMaxFlowSegmentation()
{
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

  //set the other values to defaults
  this->Structure = 0;
  this->SmoothnessScalars.clear();
  this->LeafMap.clear();
  this->BranchMap.clear();
}

vtkHierarchicalMaxFlowSegmentation::~vtkHierarchicalMaxFlowSegmentation()
{
  if( this->Structure )
  {
    this->Structure->UnRegister(this);
  }
  this->SmoothnessScalars.clear();
  this->LeafMap.clear();
  this->InputDataPortMapping.clear();
  this->BackwardsInputDataPortMapping.clear();
  this->InputSmoothnessPortMapping.clear();
  this->BackwardsInputSmoothnessPortMapping.clear();
  this->BranchMap.clear();
}

//------------------------------------------------------------

void vtkHierarchicalMaxFlowSegmentation::AddSmoothnessScalar(vtkIdType node, double value)
{
  if( value >= 0.0 )
  {
    this->SmoothnessScalars.insert(std::pair<vtkIdType,double>(node,value));
    this->Modified();
  }
  else
  {
    vtkErrorMacro("Cannot use a negative smoothness value.");
  }
}

//------------------------------------------------------------

void vtkHierarchicalMaxFlowSegmentation::Update()
{
  this->SetOutputPortAmount();
  this->Superclass::Update();
}

void vtkHierarchicalMaxFlowSegmentation::UpdateInformation()
{
  this->SetOutputPortAmount();
  this->Superclass::UpdateInformation();
}

void vtkHierarchicalMaxFlowSegmentation::UpdateWholeExtent()
{
  this->SetOutputPortAmount();
  this->Superclass::UpdateWholeExtent();
}

void vtkHierarchicalMaxFlowSegmentation::SetOutputPortAmount()
{
  int NumLeaves = 0;
  vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
  iterator->SetTree(this->Structure);
  iterator->SetStartVertex(this->Structure->GetRoot());
  while( iterator->HasNext() )
  {
    vtkIdType currNode = iterator->Next();
    if( this->Structure->IsLeaf(currNode) )
    {
      NumLeaves++;
    }
  }
  iterator->Delete();
  this->SetNumberOfOutputPorts(NumLeaves);
}

//------------------------------------------------------------

int vtkHierarchicalMaxFlowSegmentation::FillInputPortInformation(int i, vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  return this->Superclass::FillInputPortInformation(i,info);
}

void vtkHierarchicalMaxFlowSegmentation::SetDataInputDataObject(int idx, vtkDataObject *input)
{
	//if we have no input data object, clear the corresponding input connection
	if( input == NULL )
	{
	  this->SetDataInputConnection(idx,NULL);
	  return;
	}

	//else, create a trivial producer to mimic a connection
	vtkTrivialProducer* trivProd = vtkTrivialProducer::New();
	trivProd->SetOutput(input);
	this->SetDataInputConnection(idx,trivProd->GetOutputPort());
	trivProd->Delete();
}

void vtkHierarchicalMaxFlowSegmentation::SetDataInputConnection(int idx, vtkAlgorithmOutput *input)
{

  //we are adding/switching an input, so no need to resort list
  if( input != NULL )
  {

    //if their is no pair in the mapping, create one
    if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
    {
      int portNumber = this->FirstUnusedDataPort;
      this->FirstUnusedDataPort++;
      this->InputDataPortMapping.insert(std::pair<vtkIdType,int>(idx,portNumber));
      this->BackwardsInputDataPortMapping.insert(std::pair<vtkIdType,int>(portNumber,idx));
    }
    this->SetNthInputConnection(0, this->InputDataPortMapping[idx], input );

  }
  else
  {
    //if their is no pair in the mapping, just exit, nothing to do
    if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
    {
      return;
    }

    int portNumber = this->InputDataPortMapping[idx];
    this->InputDataPortMapping.erase(this->InputDataPortMapping.find(idx));
    this->BackwardsInputDataPortMapping.erase(this->BackwardsInputDataPortMapping.find(portNumber));

    //if we are the last input, no need to reshuffle
    if(portNumber == this->FirstUnusedDataPort - 1)
    {
      this->SetNthInputConnection(0, portNumber,  0);

      //if we are not, move the last input into this spot
    }
    else
    {
	  vtkAlgorithmOutput* swappedInput = this->GetInputConnection(0, this->FirstUnusedDataPort - 1);
      this->SetNthInputConnection(0, portNumber, swappedInput );
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

void vtkHierarchicalMaxFlowSegmentation::SetSmoothnessInputDataObject(int idx, vtkDataObject *input)
{
	//if we have no input data object, clear the corresponding input connection
	if( input == NULL )
	{
	  this->SetSmoothnessInputConnection(idx,NULL);
	  return;
	}

	//else, create a trivial producer to mimic a connection
	vtkTrivialProducer* trivProd = vtkTrivialProducer::New();
	trivProd->SetOutput(input);
	this->SetSmoothnessInputConnection(idx,trivProd->GetOutputPort());
	trivProd->Delete();
}

void vtkHierarchicalMaxFlowSegmentation::SetSmoothnessInputConnection(int idx, vtkAlgorithmOutput *input)
{
  //we are adding/switching an input, so no need to resort list
  if( input != NULL )
  {

    //if their is no pair in the mapping, create one
    if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() )
    {
      int portNumber = this->FirstUnusedSmoothnessPort;
      this->FirstUnusedSmoothnessPort++;
      this->InputSmoothnessPortMapping.insert(std::pair<vtkIdType,int>(idx,portNumber));
      this->BackwardsInputSmoothnessPortMapping.insert(std::pair<vtkIdType,int>(portNumber,idx));
    }
    this->SetNthInputConnection(0, this->InputSmoothnessPortMapping[idx], input );

  }
  else
  {
    //if their is no pair in the mapping, just exit, nothing to do
    if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() )
    {
      return;
    }

    int portNumber = this->InputSmoothnessPortMapping[idx];
    this->InputSmoothnessPortMapping.erase(this->InputSmoothnessPortMapping.find(idx));
    this->BackwardsInputSmoothnessPortMapping.erase(this->BackwardsInputSmoothnessPortMapping.find(portNumber));

    //if we are the last input, no need to reshuffle
    if(portNumber == this->FirstUnusedSmoothnessPort - 1)
    {
      this->SetNthInputConnection(1, portNumber,  0);

      //if we are not, move the last input into this spot
    }
    else
    {
	  vtkAlgorithmOutput* swappedInput = this->GetInputConnection(0, this->FirstUnusedSmoothnessPort - 1);
      this->SetNthInputConnection(0, portNumber, swappedInput );
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

vtkDataObject *vtkHierarchicalMaxFlowSegmentation::GetDataInputDataObject(int idx)
{
  if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
  {
    return 0;
  }
  return this->GetExecutive()->GetInputData(0, this->InputDataPortMapping[idx]);
}

vtkAlgorithmOutput *vtkHierarchicalMaxFlowSegmentation::GetDataInputConnection(int idx)
{
  if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
  {
    return 0;
  }
  return this->GetInputConnection(0,this->InputDataPortMapping[idx]);
}

vtkDataObject *vtkHierarchicalMaxFlowSegmentation::GetSmoothnessInputDataObject(int idx)
{
  if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() )
  {
    return 0;
  }
  return this->GetExecutive()->GetInputData(1, this->InputSmoothnessPortMapping[idx]);
}

vtkAlgorithmOutput *vtkHierarchicalMaxFlowSegmentation::GetSmoothnessInputConnection(int idx)
{
  if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() )
  {
    return 0;
  }
  return this->GetInputConnection(1, this->InputSmoothnessPortMapping[idx]);
}

vtkDataObject *vtkHierarchicalMaxFlowSegmentation::GetOutputDataObject(int idx)
{
  //look up port in mapping
  std::map<vtkIdType,int>::iterator port = this->LeafMap.find(idx);
  if( port == this->LeafMap.end() )
  {
    return 0;
  }

  return this->GetExecutive()->GetOutputData(port->second);
}

vtkAlgorithmOutput *vtkHierarchicalMaxFlowSegmentation::GetOutputPort(int idx)
{
  //look up port in mapping
  std::map<vtkIdType,int>::iterator port = this->LeafMap.find(idx);
  if( port == this->LeafMap.end() )
  {
    return 0;
  }

  return this->vtkAlgorithm::GetOutputPort(port->second);
}

//----------------------------------------------------------------------------

int vtkHierarchicalMaxFlowSegmentation::CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges )
{

  //check to make sure that the hierarchy is specified
  if( !this->Structure )
  {
    vtkErrorMacro("Hierarchy must be provided.");
    return -1;
  }

  this->LeafMap.clear();
  this->BranchMap.clear();

  //check to make sure that there is an image associated with each leaf node
  NumLeaves = 0;
  NumNodes = 0;
  Extent[0] = -1;
  vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
  iterator->SetTree(this->Structure);
  iterator->SetStartVertex(this->Structure->GetRoot());
  while( iterator->HasNext() )
  {
    vtkIdType node = iterator->Next();
    NumNodes++;

    if( this->Structure->IsLeaf(node) )
    {
      int value = (int) this->LeafMap.size();
      this->LeafMap[node] = value;
    }
    else if(node != this->Structure->GetRoot())
    {
      int value = (int) this->BranchMap.size();
      this->BranchMap[node] = value;
    }



    //make sure all leaf nodes have a data term
    if( this->Structure->IsLeaf(node) )
    {

      NumLeaves++;
      if( this->InputDataPortMapping.find(node) == this->InputDataPortMapping.end() )
      {
        vtkErrorMacro("Missing data prior for leaf node.");
        return -1;
      }
      int inputPortNumber = this->InputDataPortMapping[node];
      if( !(inputVector[0])->GetInformationObject(inputPortNumber) && (inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) )
      {
        vtkErrorMacro("Missing data prior for leaf node.");
        return -1;
      }
    }

    //check validity of data terms
    if( this->InputDataPortMapping.find(node) != this->InputDataPortMapping.end() )
    {
      int inputPortNumber = this->InputDataPortMapping[node];
      if( (inputVector[0])->GetInformationObject(inputPortNumber) &&
          (inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) )
      {

        //check for right scalar type
        vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
        if( CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1 )
        {
          vtkErrorMacro("Data type must be FLOAT and only have one component.");
          return -1;
        }
        if( CurrImage->GetScalarRange()[0] < 0.0 )
        {
          vtkErrorMacro("Data prior must be non-negative.");
          return -1;
        }

        //check to make sure that the sizes are consistent
        if( Extent[0] == -1 )
        {
          CurrImage->GetExtent(Extent);
        }
        else
        {
          int CurrExtent[6];
          CurrImage->GetExtent(CurrExtent);
          if( CurrExtent[0] != Extent[0] || CurrExtent[1] != Extent[1] || CurrExtent[2] != Extent[2] ||
              CurrExtent[3] != Extent[3] || CurrExtent[4] != Extent[4] || CurrExtent[5] != Extent[5] )
          {
            vtkErrorMacro("Inconsistant object extent.");
            return -1;
          }
        }

      }
    }

    if( this->InputSmoothnessPortMapping.find(node) != this->InputSmoothnessPortMapping.end() )
    {
      int inputPortNumber = this->InputSmoothnessPortMapping[node];
      if( (inputVector[1])->GetInformationObject(inputPortNumber) &&
          (inputVector[1])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) )
      {

        //check for right scalar type
        vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
        if( CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1 )
        {
          vtkErrorMacro("Smoothness type must be FLOAT and only have one component.");
          return -1;
        }
        if( CurrImage->GetScalarRange()[0] < 0.0 )
        {
          vtkErrorMacro("Smoothness prior must be non-negative.");
          return -1;
        }

        //check to make sure that the sizes are consistent
        if( Extent[0] == -1 )
        {
          CurrImage->GetExtent(Extent);
        }
        else
        {
          int CurrExtent[6];
          CurrImage->GetExtent(CurrExtent);
          if( CurrExtent[0] != Extent[0] || CurrExtent[1] != Extent[1] || CurrExtent[2] != Extent[2] ||
              CurrExtent[3] != Extent[3] || CurrExtent[4] != Extent[4] || CurrExtent[5] != Extent[5] )
          {
            vtkErrorMacro("Inconsistant object extent.");
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

int vtkHierarchicalMaxFlowSegmentation::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  //check input for consistency
  int Extent[6];
  int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
  if( result || NumNodes == 0 )
  {
    return -1;
  }

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
  //check input for consistency
  int Extent[6];
  int NumNodes;
  int NumLeaves;
  int NumEdges;
  int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
  if( result || NumNodes == 0 )
  {
    return -1;
  }

  //set up the extents
  for(int i = 0; i < NumLeaves; i++ )
  {
    vtkInformation *outputInfo = outputVector->GetInformationObject(i);
    vtkImageData *outputBuffer = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
    outputInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),Extent,6);
    outputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),Extent,6);
  }

  return 1;
}

int vtkHierarchicalMaxFlowSegmentation::RequestData(vtkInformation *request,
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector)
{
  //check input consistency
  int Extent[6];
  int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
  if( result || NumNodes == 0 )
  {
    return -1;
  }
  int NumBranches = NumNodes - NumLeaves - 1;

  if( this->Debug )
  {
    vtkDebugMacro( "Starting input data preparation." );
  }

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
  iterator->SetTree(this->Structure);
  iterator->SetStartVertex(this->Structure->GetRoot());
  while( iterator->HasNext() )
  {
    vtkIdType node = iterator->Next();
    if( node == this->Structure->GetRoot() )
    {
      continue;
    }
    if( !this->Structure->IsLeaf(node) )
    {
      BranchMap.insert(std::pair<vtkIdType,int>(node,(int) this->BranchMap.size()));
    }
  }
  iterator->Delete();

  //get the data term buffers
  this->leafDataTermBuffers = new float* [NumLeaves];
  iterator = vtkTreeDFSIterator::New();
  iterator->SetTree(this->Structure);
  while( iterator->HasNext() )
  {
    vtkIdType node = iterator->Next();
    if( this->Structure->IsLeaf(node) )
    {
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
  iterator = vtkTreeDFSIterator::New();
  iterator->SetTree(this->Structure);
  while( iterator->HasNext() )
  {
    vtkIdType node = iterator->Next();
    if( node == this->Structure->GetRoot() )
    {
      continue;
    }
    vtkImageData* CurrImage = 0;
    if( this->InputSmoothnessPortMapping.find(node) != this->InputSmoothnessPortMapping.end() )
    {
      int inputNumber = this->InputSmoothnessPortMapping[node];
      CurrImage = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(inputNumber)->Get(vtkDataObject::DATA_OBJECT()));
    }
    if( this->Structure->IsLeaf(node) )
    {
      leafSmoothnessTermBuffers[this->LeafMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
    }
    else
    {
      branchSmoothnessTermBuffers[this->BranchMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
    }

    // add the smoothness buffer in as read only
    if( CurrImage )
    {
      TotalNumberOfBuffers++;
    }
  }
  iterator->Delete();

  //get the output buffers
  this->leafLabelBuffers = new float* [NumLeaves];
  for(int i = 0; i < NumLeaves; i++ )
  {
    vtkInformation *outputInfo = outputVector->GetInformationObject(i);
    vtkImageData *outputBuffer = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
    outputBuffer->SetExtent(Extent);
    outputBuffer->Modified();
#if (VTK_MAJOR_VERSION < 6)
    outputBuffer->AllocateScalars();
#else
    outputBuffer->AllocateScalars(outputInfo);
#endif
    leafLabelBuffers[i] = (float*) outputBuffer->GetScalarPointer();
    TotalNumberOfBuffers++;
  }

  //convert smoothness constants mapping to two mappings
  iterator = vtkTreeDFSIterator::New();
  iterator->SetTree(this->Structure);
  leafSmoothnessConstants = new float[NumLeaves];
  branchSmoothnessConstants = new float[NumBranches];
  while( iterator->HasNext() )
  {
    vtkIdType node = iterator->Next();
    if( node == this->Structure->GetRoot() )
    {
      continue;
    }
    if( this->Structure->IsLeaf(node) )
      if( this->SmoothnessScalars.find(node) != this->SmoothnessScalars.end() )
      {
        leafSmoothnessConstants[this->LeafMap[node]] = this->SmoothnessScalars[node];
      }
      else
      {
        leafSmoothnessConstants[this->LeafMap[node]] = 1.0f;
      }
    else if( this->SmoothnessScalars.find(node) != this->SmoothnessScalars.end() )
    {
      branchSmoothnessConstants[this->BranchMap[node]] = this->SmoothnessScalars[node];
    }
    else
    {
      branchSmoothnessConstants[this->BranchMap[node]] = 1.0f;
    }
  }
  iterator->Delete();

  //if verbose, print progress
  if( this->Debug )
  {
    vtkDebugMacro("Starting CPU buffer acquisition");
  }

  //source flow and working buffers
  std::list<float**> BufferPointerLocs;
  int NumberOfAdditionalCPUBuffersNeeded = 2;
  TotalNumberOfBuffers += 2;
  BufferPointerLocs.push_front(&sourceFlowBuffer);
  BufferPointerLocs.push_front(&sourceWorkingBuffer);

  //allocate required memory buffers for the brach nodes
  //    buffers needed:
  //      per brach node (not the root):
  //        1 label buffer
  //        3 spatial flow buffers
  //        1 divergence buffer
  //        1 outgoing flow buffer
  //        1 working temp buffer (ie: guk)
  //and for the leaf nodes
  //      per leaf node (note label buffer is in output)
  //        3 spatial flow buffers
  //        1 divergence buffer
  //        1 sink flow buffer
  //        1 working temp buffer (ie: guk)
  NumberOfAdditionalCPUBuffersNeeded += 7 * NumBranches;
  TotalNumberOfBuffers += 7*NumBranches;
  NumberOfAdditionalCPUBuffersNeeded += 5 * NumLeaves;
  TotalNumberOfBuffers += 5 * NumLeaves;

  //allocate those buffer pointers and put on list
  float** bufferPointers = new float* [7 * NumBranches + 5 * NumLeaves];
  float** tempPtr = bufferPointers;
  this->branchFlowXBuffers =    tempPtr;
  tempPtr += NumBranches;
  this->branchFlowYBuffers =    tempPtr;
  tempPtr += NumBranches;
  this->branchFlowZBuffers =    tempPtr;
  tempPtr += NumBranches;
  this->branchDivBuffers =    tempPtr;
  tempPtr += NumBranches;
  this->branchSinkBuffers =    tempPtr;
  tempPtr += NumBranches;
  this->branchLabelBuffers =    tempPtr;
  tempPtr += NumBranches;
  this->branchWorkingBuffers =  tempPtr;
  tempPtr += NumBranches;
  for(int i = 0; i < NumBranches; i++ )
  {
    BufferPointerLocs.push_front(&(branchFlowXBuffers[i]));
  }
  for(int i = 0; i < NumBranches; i++ )
  {
    BufferPointerLocs.push_front(&(branchFlowYBuffers[i]));
  }
  for(int i = 0; i < NumBranches; i++ )
  {
    BufferPointerLocs.push_front(&(branchFlowZBuffers[i]));
  }
  for(int i = 0; i < NumBranches; i++ )
  {
    BufferPointerLocs.push_front(&(branchDivBuffers[i]));
  }
  for(int i = 0; i < NumBranches; i++ )
  {
    BufferPointerLocs.push_front(&(branchSinkBuffers[i]));
  }
  for(int i = 0; i < NumBranches; i++ )
  {
    BufferPointerLocs.push_front(&(branchLabelBuffers[i]));
  }
  for(int i = 0; i < NumBranches; i++ )
  {
    BufferPointerLocs.push_front(&(branchWorkingBuffers[i]));
  }
  this->leafFlowXBuffers =    tempPtr;
  tempPtr += NumLeaves;
  this->leafFlowYBuffers =    tempPtr;
  tempPtr += NumLeaves;
  this->leafFlowZBuffers =    tempPtr;
  tempPtr += NumLeaves;
  this->leafDivBuffers =      tempPtr;
  tempPtr += NumLeaves;
  this->leafSinkBuffers =      tempPtr;
  tempPtr += NumLeaves;
  for(int i = 0; i < NumLeaves; i++ )
  {
    BufferPointerLocs.push_front(&(leafFlowXBuffers[i]));
  }
  for(int i = 0; i < NumLeaves; i++ )
  {
    BufferPointerLocs.push_front(&(leafFlowYBuffers[i]));
  }
  for(int i = 0; i < NumLeaves; i++ )
  {
    BufferPointerLocs.push_front(&(leafFlowZBuffers[i]));
  }
  for(int i = 0; i < NumLeaves; i++ )
  {
    BufferPointerLocs.push_front(&(leafDivBuffers[i]));
  }
  for(int i = 0; i < NumLeaves; i++ )
  {
    BufferPointerLocs.push_front(&(leafSinkBuffers[i]));
  }

  //try to obtain required CPU buffers
  while( NumberOfAdditionalCPUBuffersNeeded > 0 )
  {
    int NumBuffersAcquired = (NumberOfAdditionalCPUBuffersNeeded < INT_MAX / VolumeSize) ?
                             NumberOfAdditionalCPUBuffersNeeded : INT_MAX / VolumeSize;
    for( ; NumBuffersAcquired > 0; NumBuffersAcquired--)
    {
      try
      {
        float* NewCPUBuffer = new float[VolumeSize*NumBuffersAcquired];
        if( !NewCPUBuffer )
        {
          continue;
        }
        CPUBuffersAcquired.push_front( NewCPUBuffer );
        CPUBuffersSize.push_front( NumBuffersAcquired );
        NumberOfAdditionalCPUBuffersNeeded -= NumBuffersAcquired;
        break;
      }
      catch( ... ) { };
    }
    if( NumBuffersAcquired == 0 )
    {
      break;
    }
  }

  //if we cannot obtain all required buffers, return an error and exit
  if( NumberOfAdditionalCPUBuffersNeeded > 0 )
  {
    while( CPUBuffersAcquired.size() > 0 )
    {
      float* tempBuffer = CPUBuffersAcquired.front();
      delete[] tempBuffer;
      CPUBuffersAcquired.pop_front();
    }
    vtkErrorMacro("Not enough CPU memory. Cannot run algorithm.");
    return -1;
  }

  //put buffer pointers into given structures
  std::list<float**>::iterator bufferNameIt = BufferPointerLocs.begin();
  std::list<float*>::iterator bufferAllocIt = CPUBuffersAcquired.begin();
  std::list<int>::iterator bufferSizeIt = CPUBuffersSize.begin();
  for( ; bufferAllocIt != CPUBuffersAcquired.end(); bufferAllocIt++, bufferSizeIt++ )
  {
    for( int i = 0; i < *bufferSizeIt; i++ )
    {
      *(*bufferNameIt) = (*bufferAllocIt) + VolumeSize*i;
      bufferNameIt++;
    }
  }

  //if verbose, print progress
  if( this->Debug )
  {
    vtkDebugMacro("Relate parent sink with child source buffer pointers.");
  }

  //create pointers to parent outflow
  leafIncBuffers = new float* [NumLeaves];
  branchIncBuffers = new float* [NumBranches];
  iterator = vtkTreeDFSIterator::New();
  iterator->SetTree(this->Structure);
  iterator->Next();
  while( iterator->HasNext() )
  {
    vtkIdType node = iterator->Next();
    vtkIdType parent = this->Structure->GetParent(node);
    if( this->Structure->IsLeaf(node) )
    {
      if( parent == this->Structure->GetRoot() )
      {
        leafIncBuffers[this->LeafMap[node]] = sourceFlowBuffer;
      }
      else
      {
        leafIncBuffers[this->LeafMap[node]] = branchSinkBuffers[this->BranchMap[parent]];
      }
    }
    else
    {
      if( parent == this->Structure->GetRoot() )
      {
        branchIncBuffers[this->BranchMap[node]] = sourceFlowBuffer;
      }
      else
      {
        branchIncBuffers[this->BranchMap[node]] = branchSinkBuffers[this->BranchMap[parent]];
      }
    }
  }
  iterator->Delete();

  //run algorithm proper
  if( this->Debug )
  {
    vtkDebugMacro("Starting initialization");
  }
  this->InitializeAlgorithm();
  if( this->Debug )
  {
    vtkDebugMacro("Starting max-flow algorithm.");
  }
  this->RunAlgorithm();

  //deallocate CPU buffers
  while( CPUBuffersAcquired.size() > 0 )
  {
    float* tempBuffer = CPUBuffersAcquired.front();
    delete[] tempBuffer;
    CPUBuffersAcquired.pop_front();
  }
  CPUBuffersAcquired.clear();
  CPUBuffersSize.clear();

  //deallocate structure that holds the pointers to the buffers
  delete[] bufferPointers;

  return 1;
}

int vtkHierarchicalMaxFlowSegmentation::RequestDataObject(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector ,
  vtkInformationVector* outputVector)
{

  vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
  if (!inInfo)
  {
    return 0;
  }
  vtkImageData *input = vtkImageData::SafeDownCast(inInfo->Get(vtkImageData::DATA_OBJECT()));

  if (input)
  {
    for(int i=0; i < outputVector->GetNumberOfInformationObjects(); ++i)
    {
      vtkInformation* info = outputVector->GetInformationObject(0);
      vtkDataSet *output = vtkDataSet::SafeDownCast(
                             info->Get(vtkDataObject::DATA_OBJECT()));

      if (!output || !output->IsA(input->GetClassName()))
      {
        vtkImageData* newOutput = input->NewInstance();
#if (VTK_MAJOR_VERSION < 6)
        newOutput->SetPipelineInformation(info);
#else
        newOutput->CopyInformationFromPipeline(info);
#endif
        newOutput->Delete();
      }
      return 1;
    }
  }
  return 0;
}


//----------------------------------------------------------------------------
// CPU VERSION OF THE ALGORITHM
//----------------------------------------------------------------------------

int vtkHierarchicalMaxFlowSegmentation::InitializeAlgorithm()
{
  //initalize all spatial flows and divergences to zero
  for(int i = 0; i < NumBranches; i++ )
  {
    zeroOutBuffer(branchFlowXBuffers[i], VolumeSize);
    zeroOutBuffer(branchFlowYBuffers[i], VolumeSize);
    zeroOutBuffer(branchFlowZBuffers[i], VolumeSize);
    zeroOutBuffer(branchDivBuffers[i], VolumeSize);
  }
  for(int i = 0; i < NumLeaves; i++ )
  {
    zeroOutBuffer(leafFlowXBuffers[i], VolumeSize);
    zeroOutBuffer(leafFlowYBuffers[i], VolumeSize);
    zeroOutBuffer(leafFlowZBuffers[i], VolumeSize);
    zeroOutBuffer(leafDivBuffers[i], VolumeSize);
  }

  //initialize all leak sink flows to their constraints
  for(int i = 0; i < NumLeaves; i++ )
  {
    copyBuffer(leafSinkBuffers[i], leafDataTermBuffers[i], VolumeSize);
  }

  //find the minimum sink flow
  for(int i = 1; i < NumLeaves; i++ )
  {
    minBuffer(leafSinkBuffers[0], leafSinkBuffers[i], VolumeSize);
  }

  //copy minimum sink flow over all leaves and sum the resulting labels into the source working buffer
  zeroOutBuffer(sourceWorkingBuffer, VolumeSize);
  lblBuffer(leafLabelBuffers[0], leafSinkBuffers[0], leafDataTermBuffers[0], VolumeSize);
  sumBuffer(sourceWorkingBuffer, leafLabelBuffers[0], VolumeSize);
  for(int i = 1; i < NumLeaves; i++ )
  {
    copyBuffer(leafSinkBuffers[i], leafSinkBuffers[0], VolumeSize);
    lblBuffer(leafLabelBuffers[i], leafSinkBuffers[i], leafDataTermBuffers[i], VolumeSize);
    sumBuffer(sourceWorkingBuffer, leafLabelBuffers[i], VolumeSize);
  }

  //divide the labels out to constrain them to validity
  for(int i = 0; i < NumLeaves; i++ )
  {
    divBuffer(leafLabelBuffers[i], sourceWorkingBuffer, VolumeSize);
  }

  //apply minimal sink flow over the remaining hierarchy
  for(int i = 0; i < NumBranches; i++ )
  {
    copyBuffer(branchSinkBuffers[i], leafSinkBuffers[0], VolumeSize);
  }
  copyBuffer(sourceFlowBuffer, leafSinkBuffers[0], VolumeSize);

  //propogate labels up the hierarchy
  PropogateLabels( this->Structure->GetRoot() );

  return 1;
}

int vtkHierarchicalMaxFlowSegmentation::RunAlgorithm()
{
  //Solve maximum flow problem in an iterative bottom-up manner
  for( int iteration = 0; iteration < this->NumberOfIterations; iteration++ )
  {
    SolveMaxFlow( this->Structure->GetRoot() );
    if( this->Debug )
    {
      vtkDebugMacro( "Finished iteration " << (iteration+1) << ".");
    }
  }
  return 1;
}

void vtkHierarchicalMaxFlowSegmentation::PropogateLabels( vtkIdType currNode )
{
  int NumKids = this->Structure->GetNumberOfChildren(currNode);

  //clear own label buffer if not a leaf
  if( NumKids > 0 )
  {
    zeroOutBuffer(branchLabelBuffers[BranchMap[currNode]],VolumeSize);
  }

  //update graph for all kids
  for(int kid = 0; kid < NumKids; kid++)
  {
    PropogateLabels( this->Structure->GetChild(currNode,kid) );
  }

  //find parent index
  if( currNode == this->Structure->GetRoot() )
  {
    return;
  }
  vtkIdType parent =   this->Structure->GetParent(currNode);
  if( parent == this->Structure->GetRoot() )
  {
    return;
  }
  int parentIndex = this->BranchMap[parent];
  float* currVal =   this->Structure->IsLeaf(currNode) ?
                     currVal = this->leafLabelBuffers[this->LeafMap[currNode]] :
                               currVal = this->branchLabelBuffers[this->BranchMap[currNode]];

  //sum value into parent (if parent exists and is not the root)
  sumBuffer(branchLabelBuffers[parentIndex],currVal,VolumeSize);

}

void vtkHierarchicalMaxFlowSegmentation::SolveMaxFlow( vtkIdType currNode )
{

  //get number of kids
  int NumKids = this->Structure->GetNumberOfChildren(currNode);

  //figure out what type of node we are
  bool isRoot = (currNode == this->Structure->GetRoot());
  bool isLeaf = (NumKids == 0);
  bool isBranch = (!isRoot && !isLeaf);

  //RB : clear working buffer
  if( !isLeaf )
  {

    float* workingBufferUsed = isRoot ? sourceWorkingBuffer :
                               branchWorkingBuffers[BranchMap[currNode]] ;

    //std::cout << currNode << "\t Clear working buffer" << std::endl;
    if( isBranch )
    {
      zeroOutBuffer(workingBufferUsed,VolumeSize);
    }
    else
    {
      setBufferToValue(workingBufferUsed,1.0f/CC,VolumeSize);
    }

  }

  // BL: Update spatial flow
  if( isLeaf )
  {

    //compute the gradient step amount (store in div buffer for now)
    //std::cout << currNode << "\t Find gradient descent step size" << std::endl;
    ghmf_flowGradientStep(leafSinkBuffers[LeafMap[currNode]], leafIncBuffers[LeafMap[currNode]],
                          leafDivBuffers[LeafMap[currNode]], leafLabelBuffers[LeafMap[currNode]],
                          StepSize, CC, VolumeSize);

    //apply gradient descent to the flows
    //std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
    ghmf_applyStep(leafDivBuffers[LeafMap[currNode]], leafFlowXBuffers[LeafMap[currNode]],
                   leafFlowYBuffers[LeafMap[currNode]], leafFlowZBuffers[LeafMap[currNode]],
                   VX, VY, VZ, VolumeSize);

    //std::cout << currNode << "\t Find Projection multiplier" << std::endl;
    ghmf_computeFlowMag(leafDivBuffers[LeafMap[currNode]], leafFlowXBuffers[LeafMap[currNode]],
                        leafFlowYBuffers[LeafMap[currNode]], leafFlowZBuffers[LeafMap[currNode]],
                        leafSmoothnessTermBuffers[LeafMap[currNode]], leafSmoothnessConstants[LeafMap[currNode]],
                        VX, VY, VZ, VolumeSize);

    //project onto set and recompute the divergence
    //std::cout << currNode << "\t Project flows into valid range and compute divergence" << std::endl;
    ghmf_projectOntoSet(leafDivBuffers[LeafMap[currNode]], leafFlowXBuffers[LeafMap[currNode]],
                        leafFlowYBuffers[LeafMap[currNode]], leafFlowZBuffers[LeafMap[currNode]],
                        VX, VY, VZ, VolumeSize);

  }
  else if( isBranch )
  {

    //std::cout << currNode << "\t Find gradient descent step size" << std::endl;
    ghmf_flowGradientStep(branchSinkBuffers[BranchMap[currNode]], branchIncBuffers[BranchMap[currNode]],
                          branchDivBuffers[BranchMap[currNode]], branchLabelBuffers[BranchMap[currNode]],
                          StepSize, CC,VolumeSize);

    //std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
    ghmf_applyStep(branchDivBuffers[BranchMap[currNode]], branchFlowXBuffers[BranchMap[currNode]],
                   branchFlowYBuffers[BranchMap[currNode]], branchFlowZBuffers[BranchMap[currNode]],
                   VX, VY, VZ, VolumeSize);

    //compute the multiplier for projecting back onto the feasible flow set (and store in div buffer)
    //std::cout << currNode << "\t Find Projection multiplier" << std::endl;
    ghmf_computeFlowMag(branchDivBuffers[BranchMap[currNode]], branchFlowXBuffers[BranchMap[currNode]],
                        branchFlowYBuffers[BranchMap[currNode]], branchFlowZBuffers[BranchMap[currNode]],
                        branchSmoothnessTermBuffers[BranchMap[currNode]], branchSmoothnessConstants[BranchMap[currNode]],
                        VX, VY, VZ, VolumeSize);

    //project onto set and recompute the divergence
    ghmf_projectOntoSet(branchDivBuffers[BranchMap[currNode]], branchFlowXBuffers[BranchMap[currNode]],
                        branchFlowYBuffers[BranchMap[currNode]], branchFlowZBuffers[BranchMap[currNode]],
                        VX, VY, VZ, VolumeSize);
  }

  //RB : Update everything for the children
  for(int kid = 0; kid < NumKids; kid++)
  {
    SolveMaxFlow( this->Structure->GetChild(currNode,kid) );
  }

  // B : Add sink potential to working buffer
  if( isBranch )
  {
    //std::cout << currNode << "\t Add sink potential to working buffer" << std::endl;
    storeSinkFlowInBuffer(branchWorkingBuffers[BranchMap[currNode]], branchIncBuffers[BranchMap[currNode]],
                          branchDivBuffers[BranchMap[currNode]], branchLabelBuffers[BranchMap[currNode]],
                          CC, VolumeSize);

  }

  // B : Divide working buffer by N+1 and store in sink buffer
  if( isBranch )
  {
    //std::cout << currNode << "\t Update sink flow" << std::endl;
    divAndStoreBuffer(branchWorkingBuffers[BranchMap[currNode]],branchSinkBuffers[BranchMap[currNode]],
                      (float)(NumKids+1),VolumeSize);

  }

  //R  : Divide working buffer by N and store in sink buffer
  if( isRoot )
  {
    //std::cout << currNode << "\t Update sink flow" << std::endl;
    divAndStoreBuffer(sourceWorkingBuffer,sourceFlowBuffer,(float)NumKids,VolumeSize);

  }

  //  L: Find sink potential and store, constrained, in sink
  if( isLeaf )
  {
    //std::cout << currNode << "\t Update sink flow" << std::endl;
    updateLeafSinkFlow(leafSinkBuffers[LeafMap[currNode]], leafIncBuffers[LeafMap[currNode]],
                       leafDivBuffers[LeafMap[currNode]], leafLabelBuffers[LeafMap[currNode]],
                       CC, VolumeSize);
    constrainBuffer(leafSinkBuffers[LeafMap[currNode]], leafDataTermBuffers[LeafMap[currNode]],
                    VolumeSize);
  }

  //RB : Update children's labels
  for(int kid = NumKids-1; kid >= 0; kid--)
  {
    UpdateLabel( this->Structure->GetChild(currNode,kid) );
  }

  // BL: Find source potential and store in parent's working buffer
  if( !isRoot )
  {
    //get parent's working buffer
    float* workingBuffer = (this->Structure->GetParent(currNode) == this->Structure->GetRoot()) ?
                           sourceWorkingBuffer :
                           branchWorkingBuffers[BranchMap[this->Structure->GetParent(currNode)]];

    //std::cout << currNode << "\t Add source potential to parent working buffer" << std::endl;
    if( isBranch )
    {
      storeSourceFlowInBuffer(workingBuffer, branchSinkBuffers[BranchMap[currNode]],
                              branchDivBuffers[BranchMap[currNode]], branchLabelBuffers[BranchMap[currNode]],
                              CC, VolumeSize);
    }
    else
    {
      storeSourceFlowInBuffer(workingBuffer, leafSinkBuffers[LeafMap[currNode]],
                              leafDivBuffers[LeafMap[currNode]], leafLabelBuffers[LeafMap[currNode]],
                              CC, VolumeSize);
    }
  }
}

void vtkHierarchicalMaxFlowSegmentation::UpdateLabel( vtkIdType node )
{
  int NumKids = this->Structure->GetNumberOfChildren(node);
  if( this->Structure->GetRoot() == node )
  {
    return;
  }

  //std::cout << node << "\t Update labels" << std::endl;
  if( NumKids == 0 )
    updateLabel(leafSinkBuffers[LeafMap[node]], leafIncBuffers[LeafMap[node]],
                leafDivBuffers[LeafMap[node]], leafLabelBuffers[LeafMap[node]],
                CC, VolumeSize);
  else
    updateLabel(branchSinkBuffers[BranchMap[node]], branchIncBuffers[BranchMap[node]],
                branchDivBuffers[BranchMap[node]], branchLabelBuffers[BranchMap[node]],
                CC, VolumeSize);
}
