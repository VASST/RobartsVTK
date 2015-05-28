/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkDirectedAcyclicGraphMaxFlowSegmentation.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

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

#ifndef __VTKDIRECTEDACYCLICGRAPHMAXFLOWSEGMENTATION_H__
#define __VTKDIRECTEDACYCLICGRAPHMAXFLOWSEGMENTATION_H__

#include "vtkImageAlgorithm.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkRootedDirectedAcyclicGraph.h"

#include <map>
#include <list>
#include <limits.h>
#include <float.h>

class vtkDirectedAcyclicGraphMaxFlowSegmentation : public vtkImageAlgorithm
{
public:
  vtkTypeMacro( vtkDirectedAcyclicGraphMaxFlowSegmentation, vtkImageAlgorithm );
  static vtkDirectedAcyclicGraphMaxFlowSegmentation *New();
  
  // Description:
  // Set the DirectedAcyclicGraph model used in the segmentation. Leaves in the tree correspond
  // to disjoint labels in the output image. Branches correspond to super-objects,
  // collections of these labels
  vtkSetObjectMacro(Structure,vtkRootedDirectedAcyclicGraph)
  vtkGetObjectMacro(Structure,vtkRootedDirectedAcyclicGraph)
  
  // Description:
  // Weight the smoothness term. If no scalar is provided, it is assumed to be 1. If
  // no smoothness term is provided, it is assumed to be the unit function.
  void AddSmoothnessScalar( vtkIdType node, double alpha );
  
  // Description:
  // Get and Set the number of iterations used by the algorithm (computing convergence term
  // is too slow).
  vtkSetClampMacro(NumberOfIterations,int,0,INT_MAX);
  vtkGetMacro(NumberOfIterations,int);
  
  // Description:
  // Get and Set the labeling constant, CC, of the algorithm. The default value is 0.25
  // and is unlikely to require modification.
  vtkSetClampMacro(CC,float,0.0f,FLT_MAX);
  vtkGetMacro(CC,float);

  // Description:
  // Get and Set the step size of the algorithm for updating spatial flows. The default
  // value is 0.1 and is unlikely to require modification.
  vtkSetClampMacro(StepSize,float,0.0f,1.0f);
  vtkGetMacro(StepSize,float);
  
  // Description:
  // Get and Set the data cost for the objects. The algorithm only uses those which
  // correspond to leaf nodes due to the data term pushdown theorem. These must be
  // supplied for every leaf node for the algorithm to run.
  // The Structure must be supplied prior to invoking these methods.
  vtkDataObject* GetDataInput(int idx);
  void SetDataInput(int idx, vtkDataObject *input);
  
  // Description:
  // Get and Set the smoothness for the objects. The algorithm only uses those which
  // correspond to non-root nodes. If not supplied, they are assumed to be the unit
  // field (ie: S(x)=1 for all voxels x).
  // The Structure must be supplied prior to invoking these methods.
  vtkDataObject* GetSmoothnessInput(int idx);
  void SetSmoothnessInput(int idx, vtkDataObject *input);
  
  // Description:
  // Get the final probabilistic labelling, assuming idx refers to a leaf
  // node in the supplied Structure
  vtkDataObject* GetOutput(int idx);
  
  // Description:
  // If the subclass does not define an Execute method, then the task
  // will be broken up, multiple threads will be spawned, and each thread
  // will call this method. It is public so that the thread functions
  // can call this method.
  virtual int RequestData(vtkInformation *request, 
               vtkInformationVector **inputVector, 
               vtkInformationVector *outputVector);
  virtual int RequestInformation( vtkInformation* request,
               vtkInformationVector** inputVector,
               vtkInformationVector* outputVector);
  virtual int RequestUpdateExtent( vtkInformation* request,
               vtkInformationVector** inputVector,
               vtkInformationVector* outputVector);
  virtual int RequestDataObject( vtkInformation* request,
               vtkInformationVector** inputVector,
               vtkInformationVector* outputVector);
  virtual int FillInputPortInformation(int i, vtkInformation* info);

  // Description:
  // Bring this algorithm's outputs up-to-date.
  virtual void Update();

  // Description:
  // Backward compatibility method to invoke UpdateInformation on executive.
  virtual void UpdateInformation();

  // Description:
  // Bring this algorithm's outputs up-to-date.
  virtual void UpdateWholeExtent();

protected:
  vtkDirectedAcyclicGraphMaxFlowSegmentation();
  virtual ~vtkDirectedAcyclicGraphMaxFlowSegmentation();

  void SetOutputPortAmount();
  int CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges );
  
  virtual int InitializeAlgorithm();
  virtual int RunAlgorithm();

  void PropogateLabels( );
  void SolveMaxFlow( );
  
  vtkRootedDirectedAcyclicGraph* Structure;
  std::map<vtkIdType,double> SmoothnessScalars;
  std::map<vtkIdType,int> LeafMap;
  std::map<vtkIdType,int> BranchMap;
  int    NumLeaves;
  int    NumBranches;
  int    NumNodes;
  int    NumEdges;

  int NumberOfIterations;
  float CC;
  float StepSize;
  int VolumeSize;
  int VX, VY, VZ;
  
  std::map<vtkIdType,int> InputDataPortMapping;
  std::map<int,vtkIdType> BackwardsInputDataPortMapping;
  int FirstUnusedDataPort;
  std::map<vtkIdType,int> InputSmoothnessPortMapping;
  std::map<int,vtkIdType> BackwardsInputSmoothnessPortMapping;
  int FirstUnusedSmoothnessPort;

  //pointers to variable structures, easier to keep as part of the class definition
  std::list<float*> CPUBuffersAcquired;
  std::list<int> CPUBuffersSize;
  int TotalNumberOfBuffers;
  float**  branchFlowXBuffers;
  float**  branchFlowYBuffers;
  float**  branchFlowZBuffers;
  float**  branchDivBuffers;
  float**  branchSourceBuffers;
  float**  branchSinkBuffers;
  float**  branchLabelBuffers;
  float**  branchWorkingBuffers;
  float**  branchSmoothnessTermBuffers;
  float*  branchSmoothnessConstants;

  float**  leafFlowXBuffers;
  float**  leafFlowYBuffers;
  float**  leafFlowZBuffers;
  float**  leafDivBuffers;
  float**  leafSourceBuffers;
  float**  leafSinkBuffers;
  float**  leafLabelBuffers;
  float**  leafDataTermBuffers;
  float**  leafSmoothnessTermBuffers;
  float*  leafSmoothnessConstants;
  
  float*  sourceFlowBuffer;
  float*  sourceWorkingBuffer;
  
  float*  LeafNumParents;
  float*  BranchNumParents;
  float*  BranchNumChildren;
  float  SourceNumChildren;
  float*  BranchWeightedNumChildren;
  float  SourceWeightedNumChildren;

private:
  vtkDirectedAcyclicGraphMaxFlowSegmentation operator=(const vtkDirectedAcyclicGraphMaxFlowSegmentation&){} //not implemented
  vtkDirectedAcyclicGraphMaxFlowSegmentation(const vtkDirectedAcyclicGraphMaxFlowSegmentation&){} //not implemented

};

#endif
