/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkHierarchicalMaxFlowSegmentation2Task.cxx

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

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
  void Clear();
  int RunAlgorithmIteration();
  bool CanRunAlgorithmIteration();

  friend class vtkCudaMaxFlowSegmentationWorker;
  int CreateWorker(int GPU, double MaxUsage);
  void SyncWorkers();
  void ReturnLeaves();
  std::set<vtkCudaMaxFlowSegmentationWorker*> Workers;

  //Mappings for CPU-GPU buffer sharing
  void ReturnBufferGPU2CPU(vtkCudaMaxFlowSegmentationWorker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream);
  void MoveBufferCPU2GPU(vtkCudaMaxFlowSegmentationWorker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream);
  std::map<float*,vtkCudaMaxFlowSegmentationWorker*> LastBufferUse;
  std::map<float*,int> Overwritten;

  std::set<float*> CPUInUse;
  std::map<float*,int> CPU2PriorityMap;

  int TotalNumberOfBuffers;
  int NumLeaves;

  //Mappings for task management
  friend class vtkCudaMaxFlowSegmentationTask;
  std::set<vtkCudaMaxFlowSegmentationTask*> CurrentTasks;
  std::set<vtkCudaMaxFlowSegmentationTask*> BlockedTasks;
  std::set<vtkCudaMaxFlowSegmentationTask*> FinishedTasks;

  std::set< float* > ReadOnly;
  std::set< float* > NoCopyBack;

  int    NumMemCpies;
  int    NumKernelRuns;
  int    NumTasksGoingToHappen;

  int    VolumeSize;
  int    VX;
  int    VY;
  int    VZ;

  float** leafLabelBuffers;

  float CC;
  float StepSize;
};

#endif