/*=========================================================================

Robarts Visualization Toolkit

Copyright (c) 2016 Virtual Augmentation and Simulation for Surgery and Therapy, Robarts Research Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

=========================================================================*/

/** @file vtkDirectedAcyclicGraphMaxFlowSegmentation.cxx
 *
 *  @brief Implementation file with definitions of CPU-based solver for generalized DirectedAcyclicGraph max-flow
 *      segmentation problems.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note May 20th 2014 - Documentation first compiled.
 *
 *  @note This is the base class for GPU accelerated max-flow segmentors in vtkCudaImageAnalytics
 *
 */

#include "vtkDataSetAttributes.h"
#include "vtkDirectedAcyclicGraphMaxFlowSegmentation.h"
#include "vtkFloatArray.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMaxFlowSegmentationUtilities.h"
#include "vtkObjectFactory.h"
#include "vtkRootedDirectedAcyclicGraphBackwardIterator.h"
#include "vtkRootedDirectedAcyclicGraphForwardIterator.h"
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTrivialProducer.h"
#include "vtkSetGet.h"

#include <assert.h>
#include <float.h>
#include <limits.h>

#include <set>
#include <list>
#include <vector>

#define SQR(X) X*X

//----------------------------------------------------------------------------

vtkStandardNewMacro(vtkDirectedAcyclicGraphMaxFlowSegmentation);

//----------------------------------------------------------------------------
vtkDirectedAcyclicGraphMaxFlowSegmentation::vtkDirectedAcyclicGraphMaxFlowSegmentation()
{
  //configure the IO ports
  this->SetNumberOfInputPorts(2);
  this->SetNumberOfOutputPorts(1);

  //set algorithm mathematical parameters to defaults
  this->NumberOfIterations = 100;
  this->StepSize = 0.1;
  this->CC = 0.25;

  //set up the input mapping structure
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

//----------------------------------------------------------------------------
vtkDirectedAcyclicGraphMaxFlowSegmentation::~vtkDirectedAcyclicGraphMaxFlowSegmentation()
{
  if (this->Structure)
  {
    this->Structure->UnRegister(this);
  }
  this->SmoothnessScalars.clear();
  this->LeafMap.clear();
  this->InputSmoothnessPortMapping.clear();
  this->BackwardsInputSmoothnessPortMapping.clear();
  this->BranchMap.clear();
}

//----------------------------------------------------------------------------
void vtkDirectedAcyclicGraphMaxFlowSegmentation::SetStructure(vtkRootedDirectedAcyclicGraph* t) {
	vtkSetObjectBodyMacro(Structure, vtkRootedDirectedAcyclicGraph, t);
	this->SetOutputPortAmount();
}

//----------------------------------------------------------------------------
void vtkDirectedAcyclicGraphMaxFlowSegmentation::AddSmoothnessScalar(vtkIdType node, double value)
{
  if (value >= 0.0)
  {
    this->SmoothnessScalars.insert(std::pair<vtkIdType, double>(node, value));
    this->Modified();
  }
  else
  {
    vtkErrorMacro("Cannot use a negative smoothness value.");
  }
}
//----------------------------------------------------------------------------
void vtkDirectedAcyclicGraphMaxFlowSegmentation::SetOutputPortAmount()
{
  //populate the leaf and branch maps so output ports are accessible
  NumLeaves = 0;
  NumBranches = 0;
  this->LeafMap.clear();
  this->BackwardsLeafMap.clear();
  this->BranchMap.clear();
  vtkRootedDirectedAcyclicGraphForwardIterator* iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  iterator->SetDAG(this->Structure);
  iterator->SetRootVertex(this->Structure->GetRoot());
  while (iterator->HasNext())
  {
    vtkIdType currNode = iterator->Next();
	if (this->Structure->IsLeaf(currNode))
	{
		this->LeafMap[currNode] = NumLeaves;
		this->BackwardsLeafMap[NumLeaves] = currNode;
		NumLeaves++;
		
	}
	else if (currNode != this->Structure->GetRoot())
	{
		this->BranchMap[currNode] = NumBranches;
		NumBranches++;
	}
  }
  iterator->Delete();
  this->SetNumberOfOutputPorts(NumLeaves);
  NumNodes = NumLeaves + NumBranches + 1;
}


//----------------------------------------------------------------------------
int vtkDirectedAcyclicGraphMaxFlowSegmentation::FillInputPortInformation(int i, vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  return this->Superclass::FillInputPortInformation(i, info);
}

//----------------------------------------------------------------------------
void vtkDirectedAcyclicGraphMaxFlowSegmentation::SetDataInputDataObject(int idx, vtkDataObject* input)
{
  //if we have no input data object, clear the corresponding input connection
  if (input == NULL)
  {
    this->SetDataInputConnection(idx, NULL);
    return;
  }

  //else, create a trivial producer to mimic a connection
  vtkTrivialProducer* trivProd = vtkTrivialProducer::New();
  trivProd->SetOutput(input);
  this->SetDataInputConnection(idx, trivProd->GetOutputPort());
  trivProd->Delete();
}

//----------------------------------------------------------------------------
void vtkDirectedAcyclicGraphMaxFlowSegmentation::SetDataInputConnection(int idx, vtkAlgorithmOutput* input)
{
  //we are adding/switching an input, so no need to resort list
  if (input != NULL)
  {
    //if their is no pair in the mapping, the data term is not a leaf
    if (this->LeafMap.find(idx) == this->LeafMap.end())
    {
		vtkErrorMacro("Only leaf nodes have data terms");
		return;
    }
    this->SetNthInputConnection(0, this->LeafMap[idx], input);
  }
  else
  {
    //if their is no pair in the mapping, just exit, nothing to do
    if (this->LeafMap.find(idx) == this->LeafMap.end())
    {
      return;
    }

    int portNumber = this->LeafMap[idx];
	this->SetNthInputConnection(0, portNumber, 0);
  }
}

//----------------------------------------------------------------------------
void vtkDirectedAcyclicGraphMaxFlowSegmentation::SetSmoothnessInputDataObject(int idx, vtkDataObject* input)
{
  //if we have no input data object, clear the corresponding input connection
  if (input == NULL)
  {
    this->SetSmoothnessInputConnection(idx, NULL);
    return;
  }

  //else, create a trivial producer to mimic a connection
  vtkTrivialProducer* trivProd = vtkTrivialProducer::New();
  trivProd->SetOutput(input);
  this->SetSmoothnessInputConnection(idx, trivProd->GetOutputPort());
  trivProd->Delete();
}

//----------------------------------------------------------------------------
void vtkDirectedAcyclicGraphMaxFlowSegmentation::SetSmoothnessInputConnection(int idx, vtkAlgorithmOutput* input)
{
  //we are adding/switching an input, so no need to resort list
  if (input != NULL)
  {
    //if their is no pair in the mapping, create one
    if (this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end())
    {
      int portNumber = this->FirstUnusedSmoothnessPort;
      this->FirstUnusedSmoothnessPort++;
      this->InputSmoothnessPortMapping.insert(std::pair<vtkIdType, int>(idx, portNumber));
      this->BackwardsInputSmoothnessPortMapping.insert(std::pair<vtkIdType, int>(portNumber, idx));
    }
    this->SetNthInputConnection(1, this->InputSmoothnessPortMapping[idx], input);

  }
  else
  {
    //if their is no pair in the mapping, just exit, nothing to do
    if (this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end())
    {
      return;
    }

    int portNumber = this->InputSmoothnessPortMapping[idx];
    this->InputSmoothnessPortMapping.erase(this->InputSmoothnessPortMapping.find(idx));
    this->BackwardsInputSmoothnessPortMapping.erase(this->BackwardsInputSmoothnessPortMapping.find(portNumber));

    //if we are the last input, no need to reshuffle
    if (portNumber == this->FirstUnusedSmoothnessPort - 1)
    {
      this->SetNthInputConnection(1, portNumber,  0);

      //if we are not, move the last input into this spot
    }
    else
    {
      vtkAlgorithmOutput* swappedInput = this->GetInputConnection(0, this->FirstUnusedSmoothnessPort - 1);
      this->SetNthInputConnection(0, portNumber, swappedInput);
      this->SetNthInputConnection(1, this->FirstUnusedSmoothnessPort - 1, 0);

      //correct the mappings
      vtkIdType swappedId = this->BackwardsInputSmoothnessPortMapping[this->FirstUnusedSmoothnessPort - 1];
      this->InputSmoothnessPortMapping.erase(this->InputSmoothnessPortMapping.find(swappedId));
      this->BackwardsInputSmoothnessPortMapping.erase(this->BackwardsInputSmoothnessPortMapping.find(this->FirstUnusedSmoothnessPort - 1));
      this->InputSmoothnessPortMapping.insert(std::pair<vtkIdType, int>(swappedId, portNumber));
      this->BackwardsInputSmoothnessPortMapping.insert(std::pair<int, vtkIdType>(portNumber, swappedId));

    }

    //decrement the number of unused ports
    this->FirstUnusedSmoothnessPort--;
  }
}

//----------------------------------------------------------------------------
vtkAlgorithmOutput* vtkDirectedAcyclicGraphMaxFlowSegmentation::GetDataInputConnection(int idx)
{
  if (this->LeafMap.find(idx) == this->LeafMap.end())
  {
    return 0;
  }
  return this->GetInputConnection(0, this->LeafMap[idx]);
}

//----------------------------------------------------------------------------
vtkDataObject* vtkDirectedAcyclicGraphMaxFlowSegmentation::GetDataInputDataObject(int idx)
{
  if (this->LeafMap.find(idx) == this->LeafMap.end())
  {
    return 0;
  }
  return this->GetExecutive()->GetInputData(0, this->LeafMap[idx]);
}

//----------------------------------------------------------------------------
vtkAlgorithmOutput* vtkDirectedAcyclicGraphMaxFlowSegmentation::GetSmoothnessInputConnection(int idx)
{
  if (this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end())
  {
    return 0;
  }
  return this->GetInputConnection(1, this->InputSmoothnessPortMapping[idx]);
}

//----------------------------------------------------------------------------
vtkDataObject* vtkDirectedAcyclicGraphMaxFlowSegmentation::GetSmoothnessInputDataObject(int idx)
{
  if (this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end())
  {
    return 0;
  }
  return this->GetExecutive()->GetInputData(1, this->InputSmoothnessPortMapping[idx]);
}

//----------------------------------------------------------------------------
vtkDataObject* vtkDirectedAcyclicGraphMaxFlowSegmentation::GetOutputDataObject(int idx)
{
  //look up port in mapping
  std::map<vtkIdType, int>::iterator port = this->LeafMap.find(idx);
  if (port == this->LeafMap.end())
  {
    return 0;
  }

  return this->GetExecutive()->GetOutputData(port->second);
}

//----------------------------------------------------------------------------
vtkAlgorithmOutput* vtkDirectedAcyclicGraphMaxFlowSegmentation::GetOutputPort(int idx)
{
  //look up port in mapping
  std::map<vtkIdType, int>::iterator port = this->LeafMap.find(idx);
  if (port == this->LeafMap.end())
  {
    return 0;
  }
  vtkAlgorithmOutput* retVal = this->vtkAlgorithm::GetOutputPort(port->second);
  return retVal;
}


//----------------------------------------------------------------------------
int vtkDirectedAcyclicGraphMaxFlowSegmentation::CheckInputConsistancy(vtkInformationVector** inputVector, int* extent, int& numNodes, int& numLeaves, int& numEdges)
{
  //check to make sure that the Structure is specified
  if (!this->Structure)
  {
    vtkErrorMacro("Structure must be provided.");
    return -1;
  }

  //this->LeafMap.clear();
  //this->BranchMap.clear();

  //check to make sure that there is an image associated with each leaf node
  numLeaves = 0;
  numNodes = 0;
  numEdges = 0;
  extent[0] = -1;
  vtkRootedDirectedAcyclicGraphForwardIterator* iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  iterator->SetDAG(this->Structure);
  iterator->SetRootVertex(this->Structure->GetRoot());
  while (iterator->HasNext())
  {
    vtkIdType node = iterator->Next();
    numNodes++;

    numEdges += this->Structure->GetNumberOfChildren(node);



    //make sure all leaf nodes have a data term
    if (this->Structure->IsLeaf(node))
    {
      numLeaves++;
      int inputPortNumber = this->LeafMap[node];
      if (!(inputVector[0])->GetInformationObject(inputPortNumber) && (inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()))
      {
        vtkErrorMacro("Missing data prior for leaf node.");
        return -1;
      }
    }

    //check validity of data terms
    int inputPortNumber = this->LeafMap[node];
    if ((inputVector[0])->GetInformationObject(inputPortNumber) &&
        (inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()))
    {
      //check for right scalar type
      vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
      if (CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1)
      {
        vtkErrorMacro("Data type must be FLOAT and only have one component.");
        return -1;
      }
      if (CurrImage->GetScalarRange()[0] < 0.0)
      {
        vtkErrorMacro("Data prior must be non-negative.");
        return -1;
      }

	  //check to make sure that the sizes are consistent
      if (extent[0] == -1)
      {
        CurrImage->GetExtent(extent);
      }
      else
      {
        int CurrExtent[6];
        CurrImage->GetExtent(CurrExtent);
        if (CurrExtent[0] != extent[0] || CurrExtent[1] != extent[1] || CurrExtent[2] != extent[2] ||
            CurrExtent[3] != extent[3] || CurrExtent[4] != extent[4] || CurrExtent[5] != extent[5])
        {
          vtkErrorMacro("Inconsistant object extent.");
          return -1;
        }
      }

    }

    if (this->InputSmoothnessPortMapping.find(node) != this->InputSmoothnessPortMapping.end())
    {
      int inputPortNumber = this->InputSmoothnessPortMapping[node];
      if ((inputVector[1])->GetInformationObject(inputPortNumber) &&
          (inputVector[1])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()))
      {
        //check for right scalar type
        vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
        if (CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1)
        {
          vtkErrorMacro("Smoothness type must be FLOAT and only have one component.");
          return -1;
        }
        if (CurrImage->GetScalarRange()[0] < 0.0)
        {
          vtkErrorMacro("Smoothness prior must be non-negative.");
          return -1;
        }

        //check to make sure that the sizes are consistent
        if (extent[0] == -1)
        {
          CurrImage->GetExtent(extent);
        }
        else
        {
          int CurrExtent[6];
          CurrImage->GetExtent(CurrExtent);
          if (CurrExtent[0] != extent[0] || CurrExtent[1] != extent[1] || CurrExtent[2] != extent[2] ||
              CurrExtent[3] != extent[3] || CurrExtent[4] != extent[4] || CurrExtent[5] != extent[5])
          {
            vtkErrorMacro("Inconsistant object extent.");
            return -1;
          }
        }
      }
    }
  }
  iterator->Delete();

  //find edges based on \sum{degree(V)} = 2E
  numEdges = Structure->GetNumberOfEdges();

  return 0;
}

//----------------------------------------------------------------------------
int vtkDirectedAcyclicGraphMaxFlowSegmentation::RequestInformation(vtkInformation* request, vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  int oldVal = this->vtkImageAlgorithm::RequestInformation(request, inputVector, outputVector);

  //set the number of output ports
  for (int i = 0; i < outputVector->GetNumberOfInformationObjects(); i++)
	  vtkDataObject::SetPointDataActiveScalarInfo(outputVector->GetInformationObject(i), VTK_FLOAT, -1);

  return oldVal;
}

//----------------------------------------------------------------------------
int vtkDirectedAcyclicGraphMaxFlowSegmentation::RequestData(vtkInformation* request, vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  //check input consistency
  int extent[6];
  int result = CheckInputConsistancy(inputVector, extent, NumNodes, NumLeaves, NumEdges);
  if (result || NumNodes == 0)
  {
    return -1;
  }
  NumBranches = NumNodes - NumLeaves - 1;

  if (this->Debug)
  {
    vtkDebugMacro("Starting input data preparation.");
  }

  //set the number of output ports
  outputVector->SetNumberOfInformationObjects(NumLeaves);
  this->SetNumberOfOutputPorts(NumLeaves);

  //find the size of the volume
  VX = (extent[1] - extent[0] + 1);
  VY = (extent[3] - extent[2] + 1);
  VZ = (extent[5] - extent[4] + 1);
  VolumeSize = VX * VY * VZ;

  //make a container for the total number of memory buffers
  TotalNumberOfBuffers = 0;

  //create relevant node identifier to buffer mappings
  this->BranchMap.clear();
  vtkRootedDirectedAcyclicGraphForwardIterator* forward_iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forward_iterator->SetDAG(this->Structure);
  forward_iterator->SetRootVertex(this->Structure->GetRoot());
  while (forward_iterator->HasNext())
  {
    vtkIdType node = forward_iterator->Next();
    if (node == this->Structure->GetRoot())
    {
      continue;
    }
    if (!this->Structure->IsLeaf(node))
    {
      BranchMap.insert(std::pair<vtkIdType, int>(node, (int) this->BranchMap.size()));
    }
  }
  forward_iterator->Delete();

  //get the data term buffers
  this->LeafDataTermBuffers = new float* [NumLeaves];
  forward_iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forward_iterator->SetDAG(this->Structure);
  while (forward_iterator->HasNext())
  {
    vtkIdType node = forward_iterator->Next();
    if (this->Structure->IsLeaf(node))
    {
      int inputNumber = this->LeafMap[node];
      vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputNumber)->Get(vtkDataObject::DATA_OBJECT()));
      LeafDataTermBuffers[this->LeafMap[node]] = (float*) CurrImage->GetScalarPointer();

      //add the data term buffer in and set it to read only
      TotalNumberOfBuffers++;
    }
  }
  forward_iterator->Delete();

  //get the smoothness term buffers
  this->LeafSmoothnessTermBuffers = new float* [NumLeaves];
  this->BranchSmoothnessTermBuffers = new float* [NumBranches];
  forward_iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forward_iterator->SetDAG(this->Structure);
  while (forward_iterator->HasNext())
  {
    vtkIdType node = forward_iterator->Next();
    if (node == this->Structure->GetRoot())
    {
      continue;
    }
    vtkImageData* CurrImage = 0;
    if (this->InputSmoothnessPortMapping.find(node) != this->InputSmoothnessPortMapping.end())
    {
      int inputNumber = this->InputSmoothnessPortMapping[node];
      CurrImage = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(inputNumber)->Get(vtkDataObject::DATA_OBJECT()));
    }
    if (this->Structure->IsLeaf(node))
    {
      LeafSmoothnessTermBuffers[this->LeafMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
    }
    else
    {
      BranchSmoothnessTermBuffers[this->BranchMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
    }

    // add the smoothness buffer in as read only
    if (CurrImage)
    {
      TotalNumberOfBuffers++;
    }
  }
  forward_iterator->Delete();

  //get the output buffers
  this->LeafLabelBuffers = new float* [NumLeaves];
  for (int i = 0; i < NumLeaves; i++)
  {
    vtkInformation* outputInfo = outputVector->GetInformationObject(i);
    vtkImageData* outputBuffer = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
    outputBuffer->SetExtent(extent);
    outputBuffer->Modified();
    outputBuffer->AllocateScalars(outputInfo);
    LeafLabelBuffers[i] = (float*) outputBuffer->GetScalarPointer();
    TotalNumberOfBuffers++;
  }

  //convert smoothness constants mapping to two mappings
  forward_iterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forward_iterator->SetDAG(this->Structure);
  LeafSmoothnessConstants = new float[NumLeaves];
  BranchSmoothnessConstants = new float[NumBranches];
  while (forward_iterator->HasNext())
  {
    vtkIdType node = forward_iterator->Next();
    if (node == this->Structure->GetRoot())
    {
      continue;
    }
    if (this->Structure->IsLeaf(node))
      if (this->SmoothnessScalars.find(node) != this->SmoothnessScalars.end())
      {
        LeafSmoothnessConstants[this->LeafMap[node]] = this->SmoothnessScalars[node];
      }
      else
      {
        LeafSmoothnessConstants[this->LeafMap[node]] = 1.0f;
      }
    else if (this->SmoothnessScalars.find(node) != this->SmoothnessScalars.end())
    {
      BranchSmoothnessConstants[this->BranchMap[node]] = this->SmoothnessScalars[node];
    }
    else
    {
      BranchSmoothnessConstants[this->BranchMap[node]] = 1.0f;
    }
  }
  forward_iterator->Delete();

  //if verbose, print progress
  if (this->Debug)
  {
    vtkDebugMacro("Starting CPU buffer acquisition");
  }

  //allocate required memory buffers for the branch nodes
  int numBuffersPerBranch = 8;
  int numBuffersPerLeaf = 6;
  int numBuffersSource = 2;
  //    buffers needed:
  //      per branch node (not the root):
  //        1 label buffer
  //        3 spatial flow buffers
  //        1 divergence buffer
  //        1 outgoing flow buffer
  //        1 incoming flow buffer
  //        1 working buffer
  //and for the leaf nodes
  //      per leaf node (note label buffer is in output)
  //        3 spatial flow buffers
  //        1 divergence buffer
  //        1 sink flow buffer
  //        1 incoming flow buffer
  int numberOfAdditionalCPUBuffersNeeded = 0;

  //source flow and working buffers
  std::list<float**> BufferPointerLocs;
  numberOfAdditionalCPUBuffersNeeded += numBuffersSource;
  TotalNumberOfBuffers += numBuffersSource;
  BufferPointerLocs.push_front(&SourceFlowBuffer);
  BufferPointerLocs.push_front(&SourceWorkingBuffer);

  //allocate those buffer pointers and put on list
  numberOfAdditionalCPUBuffersNeeded += numBuffersPerBranch * NumBranches;
  TotalNumberOfBuffers += numBuffersPerBranch * NumBranches;
  numberOfAdditionalCPUBuffersNeeded += numBuffersPerLeaf * NumLeaves;
  TotalNumberOfBuffers += numBuffersPerLeaf * NumLeaves;
  float** bufferPointers = new float* [numBuffersPerBranch * NumBranches + numBuffersPerLeaf * NumLeaves];
  float** tempPtr = bufferPointers;
  this->BranchFlowXBuffers = tempPtr;
  tempPtr += NumBranches;
  this->BranchFlowYBuffers = tempPtr;
  tempPtr += NumBranches;
  this->BranchFlowZBuffers = tempPtr;
  tempPtr += NumBranches;
  this->BranchDivBuffers = tempPtr;
  tempPtr += NumBranches;
  this->BranchSourceBuffers = tempPtr;
  tempPtr += NumBranches;
  this->BranchSinkBuffers = tempPtr;
  tempPtr += NumBranches;
  this->BranchLabelBuffers = tempPtr;
  tempPtr += NumBranches;
  this->BranchWorkingBuffers = tempPtr;
  tempPtr += NumBranches;
  for (int i = 0; i < NumBranches; i++)
  {
    BufferPointerLocs.push_front(&(BranchFlowXBuffers[i]));
  }
  for (int i = 0; i < NumBranches; i++)
  {
    BufferPointerLocs.push_front(&(BranchFlowYBuffers[i]));
  }
  for (int i = 0; i < NumBranches; i++)
  {
    BufferPointerLocs.push_front(&(BranchFlowZBuffers[i]));
  }
  for (int i = 0; i < NumBranches; i++)
  {
    BufferPointerLocs.push_front(&(BranchDivBuffers[i]));
  }
  for (int i = 0; i < NumBranches; i++)
  {
    BufferPointerLocs.push_front(&(BranchSinkBuffers[i]));
  }
  for (int i = 0; i < NumBranches; i++)
  {
    BufferPointerLocs.push_front(&(BranchSourceBuffers[i]));
  }
  for (int i = 0; i < NumBranches; i++)
  {
    BufferPointerLocs.push_front(&(BranchLabelBuffers[i]));
  }
  for (int i = 0; i < NumBranches; i++)
  {
    BufferPointerLocs.push_front(&(BranchWorkingBuffers[i]));
  }
  this->LeafFlowXBuffers = tempPtr;
  tempPtr += NumLeaves;
  this->LeafFlowYBuffers = tempPtr;
  tempPtr += NumLeaves;
  this->LeafFlowZBuffers = tempPtr;
  tempPtr += NumLeaves;
  this->LeafDivBuffers = tempPtr;
  tempPtr += NumLeaves;
  this->LeafSourceBuffers = tempPtr;
  tempPtr += NumLeaves;
  this->LeafSinkBuffers = tempPtr;
  tempPtr += NumLeaves;
  for (int i = 0; i < NumLeaves; i++)
  {
    BufferPointerLocs.push_front(&(LeafFlowXBuffers[i]));
  }
  for (int i = 0; i < NumLeaves; i++)
  {
    BufferPointerLocs.push_front(&(LeafFlowYBuffers[i]));
  }
  for (int i = 0; i < NumLeaves; i++)
  {
    BufferPointerLocs.push_front(&(LeafFlowZBuffers[i]));
  }
  for (int i = 0; i < NumLeaves; i++)
  {
    BufferPointerLocs.push_front(&(LeafDivBuffers[i]));
  }
  for (int i = 0; i < NumLeaves; i++)
  {
    BufferPointerLocs.push_front(&(LeafSinkBuffers[i]));
  }
  for (int i = 0; i < NumLeaves; i++)
  {
    BufferPointerLocs.push_front(&(LeafSourceBuffers[i]));
  }

  //try to obtain required CPU buffers
  while (numberOfAdditionalCPUBuffersNeeded > 0)
  {
    int numBuffersAcquired = (numberOfAdditionalCPUBuffersNeeded < INT_MAX / VolumeSize) ?
                             numberOfAdditionalCPUBuffersNeeded : INT_MAX / VolumeSize;
    for (; numBuffersAcquired > 0; numBuffersAcquired--)
    {
      try
      {
        float* newCPUBuffer = new float[VolumeSize * numBuffersAcquired];
        if (!newCPUBuffer)
        {
          continue;
        }
        CPUBuffersAcquired.push_front(newCPUBuffer);
        CPUBuffersSize.push_front(numBuffersAcquired);
        numberOfAdditionalCPUBuffersNeeded -= numBuffersAcquired;
        break;
      }
      catch (...) { };
    }
    if (numBuffersAcquired == 0)
    {
      break;
    }
  }

  //if we cannot obtain all required buffers, return an error and exit
  if (numberOfAdditionalCPUBuffersNeeded > 0)
  {
    while (CPUBuffersAcquired.size() > 0)
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
  for (; bufferAllocIt != CPUBuffersAcquired.end(); bufferAllocIt++, bufferSizeIt++)
  {
    for (int i = 0; i < *bufferSizeIt; i++)
    {
      *(*bufferNameIt) = (*bufferAllocIt) + VolumeSize * i;
      bufferNameIt++;
    }
  }

  //if verbose, print progress
  if (this->Debug)
  {
    vtkDebugMacro("Relate parent sink with child source buffer pointers.");
  }

  //figure out weightings on tree
  BranchNumParents = new float[NumBranches];
  for (int i = 0; i < NumBranches; i++)
  {
    BranchNumParents[i] = 0.0f;
  }
  BranchNumChildren = new float[NumBranches];
  for (int i = 0; i < NumBranches; i++)
  {
    BranchNumChildren[i] = 0.0f;
  }
  BranchWeightedNumChildren = new float[NumBranches];
  for (int i = 0; i < NumBranches; i++)
  {
    BranchWeightedNumChildren[i] = 0.0f;
  }
  LeafNumParents = new float[NumLeaves];
  for (int i = 0; i < NumLeaves; i++)
  {
    LeafNumParents[i] = 0.0f;
  }

  SourceNumChildren = 0.0f;
  SourceWeightedNumChildren = 0.0f;
  vtkFloatArray* weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));
  vtkRootedDirectedAcyclicGraphBackwardIterator* back_iterator = vtkRootedDirectedAcyclicGraphBackwardIterator::New();
  back_iterator->SetDAG(this->Structure);
  back_iterator->SetRootVertex(this->Structure->GetRoot());
  while (back_iterator->HasNext())
  {
    vtkIdType currNode = back_iterator->Next();

    //find the number of parents
    if (this->Structure->IsLeaf(currNode))
    {
      for (vtkIdType i = 0; i < this->Structure->GetNumberOfParents(currNode); i++)
      {
        LeafNumParents[LeafMap[currNode]] += weights ? weights->GetValue(this->Structure->GetInEdge(currNode, i).Id) : 1.0;
      }
    }
    else if (this->Structure->GetRoot() != currNode)
    {
      for (vtkIdType i = 0; i < this->Structure->GetNumberOfParents(currNode); i++)
      {
        BranchNumParents[BranchMap[currNode]] += weights ? weights->GetValue(this->Structure->GetInEdge(currNode, i).Id) : 1.0;
      }
    }

    //find the number of children
    if (this->Structure->GetRoot() == currNode)
    {
      for (vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(currNode); i++)
      {
        SourceNumChildren += weights ? weights->GetValue(this->Structure->GetOutEdge(currNode, i).Id) : 1.0;
      }
    }
    else
    {
      for (vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(currNode); i++)
      {
        BranchNumChildren[BranchMap[currNode]] += weights ? weights->GetValue(this->Structure->GetOutEdge(currNode, i).Id) : 1.0;
      }
    }
  }
  back_iterator->Restart();
  while (back_iterator->HasNext())
  {
    vtkIdType currNode = back_iterator->Next();

    //find the number of children weighted
    if (this->Structure->GetRoot() == currNode)
    {
      for (vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(currNode); i++)
      {
        float temp = (weights ? weights->GetValue(this->Structure->GetOutEdge(currNode, i).Id) : 1.0) /
                     (this->Structure->IsLeaf(this->Structure->GetChild(currNode, i)) ? LeafNumParents[LeafMap[this->Structure->GetChild(currNode, i)]] :
                      BranchNumParents[BranchMap[this->Structure->GetChild(currNode, i)]]);
        SourceWeightedNumChildren += temp * temp;
      }
    }
    else
    {
      for (vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(currNode); i++)
      {
        float temp = (weights ? weights->GetValue(this->Structure->GetOutEdge(currNode, i).Id) : 1.0) /
                     (this->Structure->IsLeaf(this->Structure->GetChild(currNode, i)) ? LeafNumParents[LeafMap[this->Structure->GetChild(currNode, i)]] :
                      BranchNumParents[BranchMap[this->Structure->GetChild(currNode, i)]]);
        BranchWeightedNumChildren[BranchMap[currNode]] += temp * temp;
      }
    }
  }
  back_iterator->Delete();

  //run algorithm proper
  if (this->Debug)
  {
    vtkDebugMacro("Starting initialization");
  }
  this->InitializeAlgorithm();
  if (this->Debug)
  {
    vtkDebugMacro("Starting max-flow algorithm.");
  }
  this->RunAlgorithm();

  //deallocate CPU buffers
  while (CPUBuffersAcquired.size() > 0)
  {
    float* tempBuffer = CPUBuffersAcquired.front();
    delete[] tempBuffer;
    CPUBuffersAcquired.pop_front();
  }
  CPUBuffersAcquired.clear();
  CPUBuffersSize.clear();

  //deallocate structure that holds the pointers to the buffers
  delete[] bufferPointers;
  delete[] BranchNumParents;
  BranchNumParents = 0;
  delete[] BranchNumChildren;
  BranchNumChildren = 0;
  delete[] LeafNumParents;
  LeafNumParents = 0;
  SourceNumChildren = 0;

  return 1;
}

//----------------------------------------------------------------------------
int vtkDirectedAcyclicGraphMaxFlowSegmentation::RequestDataObject(vtkInformation* vtkNotUsed(request), vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
  if (!inInfo)
  {
    return 0;
  }
  vtkImageData* input = vtkImageData::SafeDownCast(inInfo->Get(vtkImageData::DATA_OBJECT()));

  if (input)
  {
    for (int i = 0; i < outputVector->GetNumberOfInformationObjects(); ++i)
    {
      vtkInformation* info = outputVector->GetInformationObject(0);
      vtkDataSet* output = vtkDataSet::SafeDownCast(
                             info->Get(vtkDataObject::DATA_OBJECT()));

      return 1;
    }
  }
  return 0;
}

//----------------------------------------------------------------------------
int vtkDirectedAcyclicGraphMaxFlowSegmentation::InitializeAlgorithm()
{
  //initialize all spatial flows and divergences to zero
  for (int i = 0; i < NumBranches; i++)
  {
    zeroOutBuffer(BranchFlowXBuffers[i], VolumeSize);
    zeroOutBuffer(BranchFlowYBuffers[i], VolumeSize);
    zeroOutBuffer(BranchFlowZBuffers[i], VolumeSize);
    zeroOutBuffer(BranchDivBuffers[i], VolumeSize);
  }
  for (int i = 0; i < NumLeaves; i++)
  {
    zeroOutBuffer(LeafFlowXBuffers[i], VolumeSize);
    zeroOutBuffer(LeafFlowYBuffers[i], VolumeSize);
    zeroOutBuffer(LeafFlowZBuffers[i], VolumeSize);
    zeroOutBuffer(LeafDivBuffers[i], VolumeSize);
  }

  //initialize all leak sink flows to their constraints
  for (int i = 0; i < NumLeaves; i++)
  {
    copyBuffer(LeafSinkBuffers[i], LeafDataTermBuffers[i], VolumeSize);
  }

  //find the minimum sink flow
  for (int i = 1; i < NumLeaves; i++)
  {
    minBuffer(LeafSinkBuffers[0], LeafSinkBuffers[i], VolumeSize);
  }

  //copy minimum sink flow over all leaves and sum the resulting labels into the source flow buffer
  lblBuffer(LeafLabelBuffers[0], LeafSinkBuffers[0], LeafDataTermBuffers[0], VolumeSize);
  copyBuffer(SourceFlowBuffer, LeafLabelBuffers[0], VolumeSize);
  for (int i = 1; i < NumLeaves; i++)
  {
    copyBuffer(LeafSinkBuffers[i], LeafSinkBuffers[0], VolumeSize);
    copyBuffer(LeafSourceBuffers[i], LeafSinkBuffers[0], VolumeSize);
    lblBuffer(LeafLabelBuffers[i], LeafSinkBuffers[i], LeafDataTermBuffers[i], VolumeSize);
    sumBuffer(SourceFlowBuffer, LeafLabelBuffers[i], VolumeSize);
  }

  //divide the labels out to constrain them to validity
  for (int i = 0; i < NumLeaves; i++)
  {
    divBuffer(LeafLabelBuffers[i], SourceFlowBuffer, VolumeSize);
  }

  //apply minimal sink flow over the remaining Structure
  for (int i = 0; i < NumBranches; i++)
  {
    copyBuffer(BranchSinkBuffers[i], LeafSinkBuffers[0], VolumeSize);
    copyBuffer(BranchSourceBuffers[i], LeafSinkBuffers[0], VolumeSize);
  }
  for (int i = 0; i < NumLeaves; i++)
  {
    copyBuffer(LeafSourceBuffers[i], LeafSinkBuffers[0], VolumeSize);
  }
  copyBuffer(SourceFlowBuffer, LeafSinkBuffers[0], VolumeSize);

  //propagate labels up the Structure
  PropogateLabels();

  return 1;
}

//----------------------------------------------------------------------------
int vtkDirectedAcyclicGraphMaxFlowSegmentation::RunAlgorithm()
{
  //Solve maximum flow problem in an iterative bottom-up manner
  for (int iteration = 0; iteration < this->NumberOfIterations; iteration++)
  {
    SolveMaxFlow();
    if (this->Debug)
    {
      vtkDebugMacro("Finished iteration " << (iteration + 1) << ".");
    }
  }
  return 1;
}

//----------------------------------------------------------------------------
void vtkDirectedAcyclicGraphMaxFlowSegmentation::PropogateLabels()
{
  vtkFloatArray* weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  vtkRootedDirectedAcyclicGraphBackwardIterator* iterator = vtkRootedDirectedAcyclicGraphBackwardIterator::New();
  iterator->SetDAG(this->Structure);
  iterator->SetRootVertex(this->Structure->GetRoot());
  while (iterator->HasNext())
  {
    vtkIdType currNode = iterator->Next();

    //if we are a leaf or root label, we are finished and can therefore leave
    if (this->Structure->IsLeaf(currNode) || this->Structure->GetRoot() == currNode)
    {
      continue;
    }

    //clear own label buffer
    zeroOutBuffer(BranchLabelBuffers[BranchMap[currNode]], VolumeSize);

    //sum in weighted version of child's label
    for (vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(currNode); i++)
    {
      float W = weights ? weights->GetValue(this->Structure->GetOutEdge(currNode, i).Id) : 1.0f;
      if (this->Structure->IsLeaf(this->Structure->GetChild(currNode, i)))
      {
        sumScaledBuffer(BranchLabelBuffers[BranchMap[currNode]],
                        LeafLabelBuffers[LeafMap[this->Structure->GetChild(currNode, i)]],
                        W / LeafNumParents[LeafMap[this->Structure->GetChild(currNode, i)]],
                        VolumeSize);
      }
      else
      {
        sumScaledBuffer(BranchLabelBuffers[BranchMap[currNode]],
                        BranchLabelBuffers[BranchMap[this->Structure->GetChild(currNode, i)]],
                        W / BranchNumParents[BranchMap[this->Structure->GetChild(currNode, i)]],
                        VolumeSize);
      }
    }

  }
  iterator->Delete();
}

//----------------------------------------------------------------------------
void vtkDirectedAcyclicGraphMaxFlowSegmentation::SolveMaxFlow()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  vtkRootedDirectedAcyclicGraphBackwardIterator* backIterator = vtkRootedDirectedAcyclicGraphBackwardIterator::New();
  forIterator->SetDAG(this->Structure);
  backIterator->SetDAG(this->Structure);
  vtkFloatArray* weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  //update spatial flows (order independent)
  forIterator->SetRootVertex(this->Structure->GetRoot());
  forIterator->Restart();
  while (forIterator->HasNext())
  {
    vtkIdType currNode = forIterator->Next();
    if (this->Structure->IsLeaf(currNode))
    {
      //compute the gradient step amount (store in div buffer for now)
      //std::cout << currNode << "\t Find gradient descent step size" << std::endl;
      dagmf_flowGradientStep(LeafSinkBuffers[LeafMap[currNode]], LeafSourceBuffers[LeafMap[currNode]],
                             LeafDivBuffers[LeafMap[currNode]], LeafLabelBuffers[LeafMap[currNode]],
                             StepSize, CC, VolumeSize);

      //apply gradient descent to the flows
      //std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
      dagmf_applyStep(LeafDivBuffers[LeafMap[currNode]], LeafFlowXBuffers[LeafMap[currNode]],
                      LeafFlowYBuffers[LeafMap[currNode]], LeafFlowZBuffers[LeafMap[currNode]],
                      VX, VY, VZ, VolumeSize);

      //std::cout << currNode << "\t Find Projection multiplier" << std::endl;
      dagmf_computeFlowMag(LeafDivBuffers[LeafMap[currNode]], LeafFlowXBuffers[LeafMap[currNode]],
                           LeafFlowYBuffers[LeafMap[currNode]], LeafFlowZBuffers[LeafMap[currNode]],
                           LeafSmoothnessTermBuffers[LeafMap[currNode]], LeafSmoothnessConstants[LeafMap[currNode]],
                           VX, VY, VZ, VolumeSize);

      //project onto set and recompute the divergence
      //std::cout << currNode << "\t Project flows into valid range and compute divergence" << std::endl;
      dagmf_projectOntoSet(LeafDivBuffers[LeafMap[currNode]], LeafFlowXBuffers[LeafMap[currNode]],
                           LeafFlowYBuffers[LeafMap[currNode]], LeafFlowZBuffers[LeafMap[currNode]],
                           VX, VY, VZ, VolumeSize);
    }
    else if (currNode != this->Structure->GetRoot())
    {
      //std::cout << currNode << "\t Find gradient descent step size" << std::endl;
      dagmf_flowGradientStep(BranchSinkBuffers[BranchMap[currNode]], BranchSourceBuffers[BranchMap[currNode]],
                             BranchDivBuffers[BranchMap[currNode]], BranchLabelBuffers[BranchMap[currNode]],
                             StepSize, CC, VolumeSize);

      //std::cout << currNode << "\t Update spatial flows part 1" << std::endl;
      dagmf_applyStep(BranchDivBuffers[BranchMap[currNode]], BranchFlowXBuffers[BranchMap[currNode]],
                      BranchFlowYBuffers[BranchMap[currNode]], BranchFlowZBuffers[BranchMap[currNode]],
                      VX, VY, VZ, VolumeSize);

      //compute the multiplier for projecting back onto the feasible flow set (and store in div buffer)
      //std::cout << currNode << "\t Find Projection multiplier" << std::endl;
      dagmf_computeFlowMag(BranchDivBuffers[BranchMap[currNode]], BranchFlowXBuffers[BranchMap[currNode]],
                           BranchFlowYBuffers[BranchMap[currNode]], BranchFlowZBuffers[BranchMap[currNode]],
                           BranchSmoothnessTermBuffers[BranchMap[currNode]], BranchSmoothnessConstants[BranchMap[currNode]],
                           VX, VY, VZ, VolumeSize);

      //project onto set and recompute the divergence
      dagmf_projectOntoSet(BranchDivBuffers[BranchMap[currNode]], BranchFlowXBuffers[BranchMap[currNode]],
                           BranchFlowYBuffers[BranchMap[currNode]], BranchFlowZBuffers[BranchMap[currNode]],
                           VX, VY, VZ, VolumeSize);
    }
  }

  //clear source buffers working down
  forIterator->SetRootVertex(this->Structure->GetRoot());
  forIterator->Restart();
  while (forIterator->HasNext())
  {
    vtkIdType currNode = forIterator->Next();
    if (this->Structure->IsLeaf(currNode))
    {
      zeroOutBuffer(LeafSourceBuffers[LeafMap[currNode]], VolumeSize);
    }
    else if (currNode != this->Structure->GetRoot())
    {
      zeroOutBuffer(BranchSourceBuffers[BranchMap[currNode]], VolumeSize);
    }
  }

  //populate source for root's children
  for (vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(this->Structure->GetRoot()); i++)
  {
    vtkIdType Child = this->Structure->GetChild(this->Structure->GetRoot(), i);
    float W = weights ? weights->GetValue(this->Structure->GetOutEdge(this->Structure->GetRoot(), i).Id) : 1.0f;
    if (this->Structure->IsLeaf(Child))
      sumScaledBuffer(LeafSourceBuffers[LeafMap[Child]],
                      SourceFlowBuffer, W / this->LeafNumParents[LeafMap[Child]], VolumeSize);
    else
      sumScaledBuffer(BranchSourceBuffers[BranchMap[Child]],
                      SourceFlowBuffer, W / this->BranchNumParents[BranchMap[Child]], VolumeSize);
  }

  //propagate source for all other children
  forIterator->SetRootVertex(this->Structure->GetRoot());
  forIterator->Restart();
  while (forIterator->HasNext())
  {
    vtkIdType currNode = forIterator->Next();
    if (currNode == this->Structure->GetRoot())
    {
      continue;
    }
    for (vtkIdType i = 0; i < this->Structure->GetNumberOfChildren(currNode); i++)
    {
      vtkIdType Child = this->Structure->GetChild(currNode, i);
      float W = weights ? weights->GetValue(this->Structure->GetOutEdge(currNode, i).Id) : 1.0f;
      if (this->Structure->IsLeaf(Child))
      {
        sumScaledBuffer(LeafSourceBuffers[LeafMap[Child]], BranchSinkBuffers[BranchMap[currNode]],
                        W / this->LeafNumParents[LeafMap[Child]], VolumeSize);
      }
      else
      {
        sumScaledBuffer(BranchSourceBuffers[BranchMap[Child]], BranchSinkBuffers[BranchMap[currNode]],
                        W / this->BranchNumParents[BranchMap[Child]], VolumeSize);
      }
    }
  }

  //clear working buffers
  translateBuffer(SourceWorkingBuffer, SourceFlowBuffer, 1.0 / this->CC, SourceWeightedNumChildren, VolumeSize);
  forIterator->SetRootVertex(this->Structure->GetRoot());
  forIterator->Restart();
  while (forIterator->HasNext())
  {
    vtkIdType currNode = forIterator->Next();
    if (currNode == this->Structure->GetRoot() || this->Structure->IsLeaf(currNode))
    {
      continue;
    }
    dagmf_storeSinkFlowInBuffer(BranchWorkingBuffers[BranchMap[currNode]], BranchSourceBuffers[BranchMap[currNode]],
                                BranchDivBuffers[BranchMap[currNode]], BranchLabelBuffers[BranchMap[currNode]],
                                BranchSinkBuffers[BranchMap[currNode]], this->BranchWeightedNumChildren[BranchMap[currNode]],
                                CC, VolumeSize);
  }

  //update sink flows and labels working up
  backIterator->SetRootVertex(this->Structure->GetRoot());
  backIterator->Restart();
  while (backIterator->HasNext())
  {
    vtkIdType currNode = backIterator->Next();
    if (this->Structure->IsLeaf(currNode))
    {
      //update state at this location (source, sink, labels)
      updateLeafSinkFlow(LeafSinkBuffers[LeafMap[currNode]], LeafSourceBuffers[LeafMap[currNode]],
                         LeafDivBuffers[LeafMap[currNode]], LeafLabelBuffers[LeafMap[currNode]], CC, VolumeSize);
      constrainBuffer(LeafSinkBuffers[LeafMap[currNode]], LeafDataTermBuffers[LeafMap[currNode]], VolumeSize);

      //push up sink capacities
      for (vtkIdType i = 0; i < this->Structure->GetNumberOfParents(currNode); i++)
      {
        vtkIdType Parent = this->Structure->GetParent(currNode, i);
        float W = weights ? weights->GetValue(this->Structure->GetInEdge(currNode, i).Id) : 1.0f;
        if (Parent == this->Structure->GetRoot())
        {
          dagmf_storeSourceFlowInBuffer(SourceWorkingBuffer, LeafSinkBuffers[LeafMap[currNode]],
                                        LeafDivBuffers[LeafMap[currNode]], LeafLabelBuffers[LeafMap[currNode]],
                                        LeafSourceBuffers[LeafMap[currNode]], SourceFlowBuffer,
                                        CC, W / LeafNumParents[LeafMap[currNode]], VolumeSize);
        }
        else
        {
          dagmf_storeSourceFlowInBuffer(BranchWorkingBuffers[BranchMap[Parent]], LeafSinkBuffers[LeafMap[currNode]],
                                        LeafDivBuffers[LeafMap[currNode]], LeafLabelBuffers[LeafMap[currNode]],
                                        LeafSourceBuffers[LeafMap[currNode]], BranchSinkBuffers[BranchMap[Parent]],
                                        CC, W / LeafNumParents[LeafMap[currNode]], VolumeSize);
        }
      }

      updateLabel(LeafSinkBuffers[LeafMap[currNode]], LeafSourceBuffers[LeafMap[currNode]],
                  LeafDivBuffers[LeafMap[currNode]], LeafLabelBuffers[LeafMap[currNode]], CC, VolumeSize);

    }
    else if (currNode != this->Structure->GetRoot())
    {
      //update state at this location (source, sink, labels)
      divAndStoreBuffer(BranchSinkBuffers[BranchMap[currNode]], BranchWorkingBuffers[BranchMap[currNode]],
                        this->BranchWeightedNumChildren[BranchMap[currNode]] + 1.0f, VolumeSize);

      //push up sink capacities
      for (vtkIdType i = 0; i < this->Structure->GetNumberOfParents(currNode); i++)
      {
        vtkIdType parent = this->Structure->GetParent(currNode, i);
        float W = weights ? weights->GetValue(this->Structure->GetInEdge(currNode, i).Id) : 1.0f;
        if (parent == this->Structure->GetRoot())
        {
          dagmf_storeSourceFlowInBuffer(SourceWorkingBuffer, BranchSinkBuffers[BranchMap[currNode]],
                                        BranchDivBuffers[BranchMap[currNode]], BranchLabelBuffers[BranchMap[currNode]],
                                        BranchSourceBuffers[BranchMap[currNode]], SourceFlowBuffer,
                                        CC, W / BranchNumParents[BranchMap[currNode]], VolumeSize);
        }
        else
        {
          dagmf_storeSourceFlowInBuffer(BranchWorkingBuffers[BranchMap[parent]], BranchSinkBuffers[BranchMap[currNode]],
                                        BranchDivBuffers[BranchMap[currNode]], BranchLabelBuffers[BranchMap[currNode]],
                                        BranchSourceBuffers[BranchMap[currNode]], BranchSinkBuffers[BranchMap[parent]],
                                        CC, W / BranchNumParents[BranchMap[currNode]], VolumeSize);
        }
      }

      updateLabel(BranchSinkBuffers[BranchMap[currNode]], BranchSourceBuffers[BranchMap[currNode]],
                  BranchDivBuffers[BranchMap[currNode]], BranchLabelBuffers[BranchMap[currNode]], CC, VolumeSize);

    }
    else
    {
      divAndStoreBuffer(SourceFlowBuffer, SourceWorkingBuffer, this->SourceWeightedNumChildren, VolumeSize);
    }

  }

  forIterator->Delete();
  backIterator->Delete();
}