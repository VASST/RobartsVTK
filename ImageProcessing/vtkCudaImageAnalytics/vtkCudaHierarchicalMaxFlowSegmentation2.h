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

/** @file vtkHierarchicalMaxFlowSegmentation2.h
 *
 *  @brief Header file with definitions of GPU-based solver for generalized hierarchical max-flow
 *      segmentation problems with greedy scheduling over multiple GPUs. See
 *      vtkHierarchicalMaxFlowSegmentation.h for most of the interface documentation.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#ifndef __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2_H__
#define __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2_H__

#include "vtkCudaImageAnalyticsExport.h"

#include "vtkHierarchicalMaxFlowSegmentation.h"
#include <map>
#include <set>

class CudaObject;
class vtkCudaMaxFlowSegmentationScheduler;
class vtkCudaMaxFlowSegmentationTask;
class vtkCudaMaxFlowSegmentationWorker;

class vtkCudaImageAnalyticsExport vtkCudaHierarchicalMaxFlowSegmentation2 : public vtkHierarchicalMaxFlowSegmentation
{
public:
  vtkTypeMacro(vtkCudaHierarchicalMaxFlowSegmentation2, vtkHierarchicalMaxFlowSegmentation);
  static vtkCudaHierarchicalMaxFlowSegmentation2* New();

  // Description:
  // Insert, remove, and verify a given GPU into the set of GPUs usable by the algorithm. This
  // set defaults to {GPU0} and must be non-empty when the update is invoked.
  void AddDevice(int GPU);
  void RemoveDevice(int GPU);
  bool HasDevice(int GPU);

  // Description:
  // Clears the set of GPUs usable by the algorithm,
  void ClearDevices();

  // Description:
  // Set the class to use a single GPU, the one provided.
  void SetDevice(int GPU) { this->ClearDevices(); this->AddDevice(GPU); }

  // Description:
  // Get and Set the maximum percent of GPU memory usable by the algorithm.
  // Recommended to keep below 98% on compute-only cards, and 90% on cards
  // used for running the monitors. The number provided will act as a de
  // facto value for all cards. (Default is 90%.)
  vtkSetClampMacro(MaxGPUUsage, double, 0.0, 1.0);
  vtkGetMacro(MaxGPUUsage, double);

  // Description:
  // Get, Set, and Clear exceptions, allowing for a particular card to have its
  // memory consumption managed separately.
  void SetMaxGPUUsage(double usage, int device);
  double GetMaxGPUUsage(int device);
  void ClearMaxGPUUsage();

  // Description:
  // Get and Set how often the algorithm should report if in Debug mode. If set
  // to 0, the algorithm doesn't report task completions. Default is 100 tasks.
  vtkSetClampMacro(ReportRate, int, 0, INT_MAX);
  vtkGetMacro(ReportRate, int);

protected:
  vtkCudaHierarchicalMaxFlowSegmentation2();
  virtual ~vtkCudaHierarchicalMaxFlowSegmentation2();

  vtkCudaMaxFlowSegmentationScheduler* Scheduler;

  std::set<int> GPUsUsed;

  double          MaxGPUUsage;
  std::map<int, double>  MaxGPUUsageNonDefault;
  int            ReportRate;

  virtual int InitializeAlgorithm();
  virtual int RunAlgorithm();

  void FigureOutBufferPriorities(vtkIdType currNode);
  void PropogateLabels(vtkIdType currNode);
  void SolveMaxFlow(vtkIdType currNode, int* timeStep);
  void UpdateLabel(vtkIdType node, int* timeStep);

  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> ClearWorkingBufferTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> UpdateSpatialFlowsTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> ApplySinkPotentialBranchTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> ApplySinkPotentialLeafTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> ApplySourcePotentialTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> DivideOutWorkingBufferTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> UpdateLabelsTasks;

  void CreateClearWorkingBufferTasks(vtkIdType currNode);
  void CreateUpdateSpatialFlowsTasks(vtkIdType currNode);
  void CreateApplySinkPotentialBranchTasks(vtkIdType currNode);
  void CreateApplySinkPotentialLeafTasks(vtkIdType currNode);
  void CreateApplySourcePotentialTask(vtkIdType currNode);
  void CreateDivideOutWorkingBufferTask(vtkIdType currNode);
  void CreateUpdateLabelsTask(vtkIdType currNode);
  void AddIterationTaskDependencies(vtkIdType currNode);

  std::map<int, vtkCudaMaxFlowSegmentationTask*> InitializeLeafSinkFlowsTasks;
  std::map<int, vtkCudaMaxFlowSegmentationTask*> MinimizeLeafSinkFlowsTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> PropogateLeafSinkFlowsTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> InitialLabellingSumTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> CorrectLabellingTasks;
  std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*> PropogateLabellingTasks;

  void CreateInitializeAllSpatialFlowsToZeroTasks(vtkIdType currNode);
  void CreateInitializeLeafSinkFlowsToCapTasks(vtkIdType currNode);
  void CreateCopyMinimalLeafSinkFlowsTasks(vtkIdType currNode);
  void CreateFindInitialLabellingAndSumTasks(vtkIdType currNode);
  void CreateClearSourceWorkingBufferTask();
  void CreateDivideOutLabelsTasks(vtkIdType currNode);
  void CreatePropogateLabelsTasks(vtkIdType currNode);

private:
  vtkCudaHierarchicalMaxFlowSegmentation2 operator=(const vtkCudaHierarchicalMaxFlowSegmentation2&);
  vtkCudaHierarchicalMaxFlowSegmentation2(const vtkCudaHierarchicalMaxFlowSegmentation2&);
};

#endif
