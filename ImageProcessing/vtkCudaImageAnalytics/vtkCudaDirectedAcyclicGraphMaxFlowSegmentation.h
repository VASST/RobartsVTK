/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaDirectedAcyclicGraphMaxFlowSegmentation.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkHierarchicalMaxFlowSegmentation2.h
 *
 *  @brief Header file with definitions of GPU-based solver for DAG-based max-flow
 *      segmentation problems with greedy scheduling over multiple GPUs. See
 *      vtkDirectedAcyclicGraphMaxFlowSegmentation.h for most of the interface documentation.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note June 22nd 2014 - Documentation first compiled.
 *
 */

#ifndef __VTKCUDADIRECTEDACYCLICGRAPHMAXFLOWSEGMENTATION_H__
#define __VTKCUDADIRECTEDACYCLICGRAPHMAXFLOWSEGMENTATION_H__

#include "vtkCudaImageAnalyticsModule.h"

#include "vtkDirectedAcyclicGraphMaxFlowSegmentation.h"
#include <map>
#include <set>

class CudaObject;
class vtkCudaMaxFlowSegmentationScheduler;
class vtkCudaMaxFlowSegmentationTask;
class vtkCudaMaxFlowSegmentationWorker;

class VTKCUDAIMAGEANALYTICS_EXPORT vtkCudaDirectedAcyclicGraphMaxFlowSegmentation : public vtkDirectedAcyclicGraphMaxFlowSegmentation
{
public:
  vtkTypeMacro( vtkCudaDirectedAcyclicGraphMaxFlowSegmentation, vtkDirectedAcyclicGraphMaxFlowSegmentation );
  static vtkCudaDirectedAcyclicGraphMaxFlowSegmentation *New();

  // Description:
  // Insert, remove, and verify a given GPU into the set of GPUs usable by the algorithm. This
  // set defaults to {GPU0} and must be non-empty when the update is invoked.
  void AddDevice(int GPU);
  void RemoveDevice(int GPU);
  bool HasDevice(int GPU);

  // Description:
  // Clears the set of GPUs usable by the algorithm
  void ClearDevices();

  // Description:
  // Set the class to use a single GPU, the one provided.
  void SetDevice(int GPU);

  // Description:
  // Get and Set the maximum percent of GPU memory usable by the algorithm.
  // Recommended to keep below 98% on compute-only cards, and 90% on cards
  // used for running the monitors. The number provided will act as a de
  // facto value for all cards. (Default is 90%.)
  vtkSetClampMacro(MaxGPUUsage,double,0.0,1.0);
  vtkGetMacro(MaxGPUUsage,double);

  // Description:
  // Get, Set, and Clear exceptions, allowing for a particular card to have its
  // memory consumption managed separately.
  void SetMaxGPUUsage(double usage, int device);
  double GetMaxGPUUsage(int device);
  void ClearMaxGPUUsage();

  // Description:
  // Get and Set how often the algorithm should report if in Debug mode. If set
  // to 0, the algorithm doesn't report task completions. Default is 100 tasks.
  vtkSetClampMacro(ReportRate,int,0,INT_MAX);
  vtkGetMacro(ReportRate,int);

protected:
  vtkCudaDirectedAcyclicGraphMaxFlowSegmentation();
  virtual ~vtkCudaDirectedAcyclicGraphMaxFlowSegmentation();

  vtkCudaMaxFlowSegmentationScheduler* Scheduler;

  std::set<int> GPUsUsed;

  double          MaxGPUUsage;
  std::map<int,double>  MaxGPUUsageNonDefault;
  int            ReportRate;

  virtual int InitializeAlgorithm();
  virtual int RunAlgorithm();

  void FigureOutBufferPriorities( vtkIdType currNode );
  void PropogateLabels( vtkIdType currNode );
  void SolveMaxFlow( vtkIdType currNode, int* timeStep );
  void UpdateLabel( vtkIdType node, int* timeStep );

  std::map<vtkIdType,vtkCudaMaxFlowSegmentationTask*> UpdateSpatialFlowsTasks;
  void CreateUpdateSpatialFlowsTasks();
  std::map<vtkIdType,vtkCudaMaxFlowSegmentationTask*> ResetSinkFlowTasks;
  void CreateResetSinkFlowRootTasks();
  void CreateResetSinkFlowBranchTasks();
  std::map<vtkIdType,vtkCudaMaxFlowSegmentationTask*> ApplySinkPotentialLeafTasks;
  void CreateApplySinkPotentialLeafTasks();
  std::map<vtkIdType,vtkCudaMaxFlowSegmentationTask*> PushUpSourceFlowsTasks;
  void CreatePushUpSourceFlowsLeafTasks();
  void CreatePushUpSourceFlowsBranchTasks();
  std::map<vtkIdType,vtkCudaMaxFlowSegmentationTask*> PushDownSinkFlowsTasks;
  void CreatePushDownSinkFlowsRootTasks();
  void CreatePushDownSinkFlowsBranchTasks();
  std::map<vtkIdType,vtkCudaMaxFlowSegmentationTask*> UpdateLabelsTasks;
  void CreateUpdateLabelsTasks();
  std::map<vtkIdType,vtkCudaMaxFlowSegmentationTask*> ClearSourceBufferTasks;
  void CreateClearSourceBufferTasks();
  void AssociateFinishSignals();

  void InitializeSpatialFlowsTasks();
  void InitializeSinkFlowsTasks();

private:
  vtkCudaDirectedAcyclicGraphMaxFlowSegmentation operator=(const vtkCudaDirectedAcyclicGraphMaxFlowSegmentation&);
  vtkCudaDirectedAcyclicGraphMaxFlowSegmentation(const vtkCudaDirectedAcyclicGraphMaxFlowSegmentation&);
};

#endif
