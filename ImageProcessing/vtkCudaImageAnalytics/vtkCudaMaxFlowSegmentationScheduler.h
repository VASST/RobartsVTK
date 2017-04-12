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

/** @file vtkCudaMaxFlowSegmentation.h
 *
 *  @brief Header file with definitions of individual chunks of GPU based code which can be
 *      handled semi-synchronously.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 *  @note This is not a front-end class.
 *
 */

#ifndef __VTKCUDAMAXFLOWSEGMENTATIONSCHEDULER_H__
#define __VTKCUDAMAXFLOWSEGMENTATIONSCHEDULER_H__

#include "vtkCudaImageAnalyticsExport.h"

#include "CUDA_hierarchicalmaxflow.h"
#include <map>
#include <set>

class vtkCudaMaxFlowSegmentationTask;
class vtkCudaMaxFlowSegmentationWorker;

class vtkCudaImageAnalyticsExport vtkCudaMaxFlowSegmentationScheduler
{
private:
  vtkCudaMaxFlowSegmentationScheduler();
  ~vtkCudaMaxFlowSegmentationScheduler();

  friend class vtkCudaHierarchicalMaxFlowSegmentation2;
  friend class vtkCudaDirectedAcyclicGraphMaxFlowSegmentation;
  friend class vtkCudaMaxFlowSegmentationTask;
  friend class vtkCudaMaxFlowSegmentationWorker;

  void Clear();
  int RunAlgorithmIteration();
  bool CanRunAlgorithmIteration();

  int CreateWorker(int GPU, double MaxUsage);
  void SyncWorkers();
  void ReturnLeaves();

  //Mappings for CPU-GPU buffer sharing
  void ReturnBufferGPU2CPU(vtkCudaMaxFlowSegmentationWorker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream);
  void MoveBufferCPU2GPU(vtkCudaMaxFlowSegmentationWorker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream);

private:
  std::set<vtkCudaMaxFlowSegmentationWorker*>         Workers;

  std::map<float*, vtkCudaMaxFlowSegmentationWorker*> LastBufferUse;
  std::map<float*, int>                               Overwritten;

  std::set<float*>                                    CPUInUse;
  std::map<float*, int>                               CPU2PriorityMap;

  int                                                 TotalNumberOfBuffers;
  int                                                 NumLeaves;

  //Mappings for task management
  std::set<vtkCudaMaxFlowSegmentationTask*>           CurrentTasks;
  std::set<vtkCudaMaxFlowSegmentationTask*>           BlockedTasks;
  std::set<vtkCudaMaxFlowSegmentationTask*>           FinishedTasks;

  std::set<float*>                                    ReadOnly;
  std::set<float*>                                    NoCopyBack;

  int                                                 NumMemCpies;
  int                                                 NumKernelRuns;
  int                                                 NumTasksGoingToHappen;

  int                                                 VolumeSize;
  int                                                 VX;
  int                                                 VY;
  int                                                 VZ;

  float**                                             LeafLabelBuffers;

  float                                               CC;
  float                                               StepSize;
};

#endif