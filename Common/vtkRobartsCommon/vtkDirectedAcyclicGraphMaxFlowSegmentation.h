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

/** @file vtkDirectedAcyclicGraphMaxFlowSegmentation.h
 *
 *  @brief Header file with definitions of CPU-based solver for generalized DirectedAcyclicGraph max-flow
 *      segmentation problems.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note May 20th 2014 - Documentation first compiled.
 *
 *  @note This is the base class for GPU accelerated max-flow segmentors in vtkCudaImageAnalytics
 *
 */

#ifndef __vtkDirectedAcyclicGraphMaxFlowSegmentation_h__
#define __vtkDirectedAcyclicGraphMaxFlowSegmentation_h__

#include "vtkRobartsCommonExport.h"

#include "vtkImageAlgorithm.h"
#include "vtkRootedDirectedAcyclicGraph.h"

class vtkInformation;
class vtkInformationVector;

#include <map>
#include <list>
#include <limits.h>
#include <float.h>

class vtkRobartsCommonExport vtkDirectedAcyclicGraphMaxFlowSegmentation : public vtkImageAlgorithm
{
public:
  vtkTypeMacro(vtkDirectedAcyclicGraphMaxFlowSegmentation, vtkImageAlgorithm);
  static vtkDirectedAcyclicGraphMaxFlowSegmentation* New();

  // Description:
  // Set the DirectedAcyclicGraph model used in the segmentation. Leaves in the tree correspond
  // to disjoint labels in the output image. Branches correspond to super-objects,
  // collections of these labels
  virtual void SetStructure(vtkRootedDirectedAcyclicGraph*);
  vtkGetObjectMacro(Structure, vtkRootedDirectedAcyclicGraph)

  // Description:
  // Weight the smoothness term. If no scalar is provided, it is assumed to be 1. If
  // no smoothness term is provided, it is assumed to be the unit function.
  void AddSmoothnessScalar(vtkIdType node, double alpha);

  // Description:
  // Get and Set the number of iterations used by the algorithm (computing convergence term
  // is too slow).
  vtkSetClampMacro(NumberOfIterations, int, 0, INT_MAX);
  vtkGetMacro(NumberOfIterations, int);

  // Description:
  // Get and Set the labeling constant, CC, of the algorithm. The default value is 0.25
  // and is unlikely to require modification.
  vtkSetClampMacro(CC, float, 0.0f, FLT_MAX);
  vtkGetMacro(CC, float);

  // Description:
  // Get and Set the step size of the algorithm for updating spatial flows. The default
  // value is 0.1 and is unlikely to require modification.
  vtkSetClampMacro(StepSize, float, 0.0f, 1.0f);
  vtkGetMacro(StepSize, float);

  // Description:
  // Get and Set the data cost for the objects. The algorithm only uses those which
  // correspond to leaf nodes due to the data term pushdown theorem. These must be
  // supplied for every leaf node for the algorithm to run.
  // The Structure must be supplied prior to invoking these methods.
  vtkDataObject* GetDataInputDataObject(int idx);
  void SetDataInputDataObject(int idx, vtkDataObject* input);
  vtkAlgorithmOutput* GetDataInputConnection(int idx);
  void SetDataInputConnection(int idx, vtkAlgorithmOutput* input);

  // Description:
  // Get and Set the smoothness for the objects. The algorithm only uses those which
  // correspond to non-root nodes. If not supplied, they are assumed to be the unit
  // field (ie: S(x)=1 for all voxels x).
  // The Structure must be supplied prior to invoking these methods.
  vtkDataObject* GetSmoothnessInputDataObject(int idx);
  void SetSmoothnessInputDataObject(int idx, vtkDataObject* input);
  vtkAlgorithmOutput* GetSmoothnessInputConnection(int idx);
  void SetSmoothnessInputConnection(int idx, vtkAlgorithmOutput* input);

  // Description:
  // Get the final probabilistic labeling, assuming idx refers to a leaf
  // node in the supplied Structure
  vtkDataObject* GetOutputDataObject(int idx);
  vtkAlgorithmOutput* GetOutputPort(int idx);

  // Description:
  // If the subclass does not define an Execute method, then the task
  // will be broken up, multiple threads will be spawned, and each thread
  // will call this method. It is public so that the thread functions
  // can call this method.
  virtual int RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);
  virtual int RequestInformation(vtkInformation* request,
                                 vtkInformationVector** inputVector,
                                 vtkInformationVector* outputVector);
  virtual int RequestDataObject(vtkInformation* request,
                                vtkInformationVector** inputVector,
                                vtkInformationVector* outputVector);
  virtual int FillInputPortInformation(int i, vtkInformation* info);

protected:
  void SetOutputPortAmount();
  int CheckInputConsistancy(vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges);

  virtual int InitializeAlgorithm();
  virtual int RunAlgorithm();

  void PropogateLabels();
  void SolveMaxFlow();

  vtkRootedDirectedAcyclicGraph*  Structure;
  std::map<vtkIdType, double>     SmoothnessScalars;
  std::map<vtkIdType, int>        LeafMap;
  std::map<int, vtkIdType>        BackwardsLeafMap;
  std::map<vtkIdType, int>        BranchMap;
  int                             NumLeaves;
  int                             NumBranches;
  int                             NumNodes;
  int                             NumEdges;

  int                             NumberOfIterations;
  float                           CC;
  float                           StepSize;
  int                             VolumeSize;
  int                             VX, VY, VZ;

  int                             FirstUnusedDataPort;
  std::map<vtkIdType, int>        InputSmoothnessPortMapping;
  std::map<int, vtkIdType>        BackwardsInputSmoothnessPortMapping;
  int                             FirstUnusedSmoothnessPort;

  //pointers to variable structures, easier to keep as part of the class definition
  std::list<float*>               CPUBuffersAcquired;
  std::list<int>                  CPUBuffersSize;
  int                             TotalNumberOfBuffers;
  float**                         BranchFlowXBuffers;
  float**                         BranchFlowYBuffers;
  float**                         BranchFlowZBuffers;
  float**                         BranchDivBuffers;
  float**                         BranchSourceBuffers;
  float**                         BranchSinkBuffers;
  float**                         BranchLabelBuffers;
  float**                         BranchWorkingBuffers;
  float**                         BranchSmoothnessTermBuffers;
  float*                          BranchSmoothnessConstants;

  float**                         LeafFlowXBuffers;
  float**                         LeafFlowYBuffers;
  float**                         LeafFlowZBuffers;
  float**                         LeafDivBuffers;
  float**                         LeafSourceBuffers;
  float**                         LeafSinkBuffers;
  float**                         LeafLabelBuffers;
  float**                         LeafDataTermBuffers;
  float**                         LeafSmoothnessTermBuffers;
  float*                          LeafSmoothnessConstants;

  float*                          SourceFlowBuffer;
  float*                          SourceWorkingBuffer;

  float*                          LeafNumParents;
  float*                          BranchNumParents;
  float*                          BranchNumChildren;
  float                           SourceNumChildren;
  float*                          BranchWeightedNumChildren;
  float                           SourceWeightedNumChildren;

protected:
  vtkDirectedAcyclicGraphMaxFlowSegmentation();
  virtual ~vtkDirectedAcyclicGraphMaxFlowSegmentation();

private:
  vtkDirectedAcyclicGraphMaxFlowSegmentation operator=(const vtkDirectedAcyclicGraphMaxFlowSegmentation&);
  vtkDirectedAcyclicGraphMaxFlowSegmentation(const vtkDirectedAcyclicGraphMaxFlowSegmentation&);
};

#endif // __vtkDirectedAcyclicGraphMaxFlowSegmentation_h__
