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

/** @file vtkHierarchicalMaxFlowSegmentation2.cxx
 *
 *  @brief Implementation file with definitions of GPU-based solver for generalized hierarchical max-flow
 *      segmentation problems with greedy scheduling over multiple GPUs.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#include "CudaObject.h"
#include "vtkCudaHierarchicalMaxFlowSegmentation2.h"
#include "vtkCudaMaxFlowSegmentationScheduler.h"
#include "vtkCudaMaxFlowSegmentationTask.h"
#include "vtkCudaMaxFlowSegmentationWorker.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTreeDFSIterator.h"
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <list>
#include <math.h>
#include <vector>

//----------------------------------------------------------------------------

vtkStandardNewMacro(vtkCudaHierarchicalMaxFlowSegmentation2);

//----------------------------------------------------------------------------
vtkCudaHierarchicalMaxFlowSegmentation2::vtkCudaHierarchicalMaxFlowSegmentation2()
{
  //set algorithm mathematical parameters to defaults
  this->MaxGPUUsage = 0.90;
  this->ReportRate = 100;

  //give default GPU selection
  this->GPUsUsed.insert(0);

  //create scheduler
  this->Scheduler = new vtkCudaMaxFlowSegmentationScheduler();
}

//----------------------------------------------------------------------------
vtkCudaHierarchicalMaxFlowSegmentation2::~vtkCudaHierarchicalMaxFlowSegmentation2()
{
  this->GPUsUsed.clear();
  this->MaxGPUUsageNonDefault.clear();
  delete this->Scheduler;
}


//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::AddDevice(int GPU)
{
  if (GPU >= 0 && GPU < vtkCudaDeviceManager::Singleton()->GetNumberOfDevices())
  {
    this->GPUsUsed.insert(GPU);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::RemoveDevice(int GPU)
{
  if (this->GPUsUsed.find(GPU) != this->GPUsUsed.end())
  {
    this->GPUsUsed.erase(this->GPUsUsed.find(GPU));
  }
}

//----------------------------------------------------------------------------
bool vtkCudaHierarchicalMaxFlowSegmentation2::HasDevice(int GPU)
{
  return (this->GPUsUsed.find(GPU) != this->GPUsUsed.end());
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::ClearDevices()
{
  this->GPUsUsed.clear();
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::SetMaxGPUUsage(double usage, int device)
{
  if (usage < 0.0)
  {
    usage = 0.0;
  }
  else if (usage > 1.0)
  {
    usage = 1.0;
  }
  if (device >= 0 && device < vtkCudaDeviceManager::Singleton()->GetNumberOfDevices())
  {
    this->MaxGPUUsageNonDefault[device] = usage;
  }
}

//----------------------------------------------------------------------------
double vtkCudaHierarchicalMaxFlowSegmentation2::GetMaxGPUUsage(int device)
{
  if (this->MaxGPUUsageNonDefault.find(device) != this->MaxGPUUsageNonDefault.end())
  {
    return this->MaxGPUUsageNonDefault[device];
  }
  return this->MaxGPUUsage;
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::ClearMaxGPUUsage()
{
  this->MaxGPUUsageNonDefault.clear();
}

//----------------------------------------------------------------------------
int vtkCudaHierarchicalMaxFlowSegmentation2::InitializeAlgorithm()
{
  //if verbose, print progress
  this->Scheduler->Clear();
  this->Scheduler->Clear();
  this->Scheduler->TotalNumberOfBuffers = this->TotalNumberOfBuffers;
  this->Scheduler->VolumeSize = this->VolumeSize;
  this->Scheduler->VX = this->VX;
  this->Scheduler->VY = this->VY;
  this->Scheduler->VZ = this->VZ;
  this->Scheduler->CC = this->CC;
  this->Scheduler->StepSize = this->StepSize;

  if (this->Debug)
  {
    vtkDebugMacro("Building workers.");
  }

  for (std::set<int>::iterator gpuIterator = GPUsUsed.begin(); gpuIterator != GPUsUsed.end(); gpuIterator++)
  {
    double usage = this->MaxGPUUsage;
    if (this->MaxGPUUsageNonDefault.find(*gpuIterator) != this->MaxGPUUsageNonDefault.end())
    {
      usage = this->MaxGPUUsageNonDefault[*gpuIterator];
    }
    if (this->Scheduler->CreateWorker(*gpuIterator, usage))
    {
      vtkErrorMacro("Could not allocate sufficient GPU buffers.");
      Scheduler->Clear();
      while (CPUBuffersAcquired.size() > 0)
      {
        float* tempBuffer = CPUBuffersAcquired.front();
        delete[] tempBuffer;
        CPUBuffersAcquired.pop_front();
      }
    }
  }

  //if verbose, print progress
  if (this->Debug)
  {
    vtkDebugMacro("Find priority structures.");
  }

  //create LIFO priority queue (priority stack) data structure
  FigureOutBufferPriorities(this->Structure->GetRoot());

  //add tasks in for the normal iterations (done first for dependency reasons)
  if (this->Debug)
  {
    vtkDebugMacro("Creating tasks for normal iterations.");
  }
  if (this->NumberOfIterations > 0)
  {
    CreateClearWorkingBufferTasks(this->Structure->GetRoot());
    CreateUpdateSpatialFlowsTasks(this->Structure->GetRoot());
    CreateApplySinkPotentialBranchTasks(this->Structure->GetRoot());
    CreateApplySinkPotentialLeafTasks(this->Structure->GetRoot());
    CreateApplySourcePotentialTask(this->Structure->GetRoot());
    CreateDivideOutWorkingBufferTask(this->Structure->GetRoot());
    CreateUpdateLabelsTask(this->Structure->GetRoot());
    AddIterationTaskDependencies(this->Structure->GetRoot());
  }

  //add tasks in for the initialization (done second for dependency reasons)
  if (this->Debug)
  {
    vtkDebugMacro("Creating tasks for initialization.");
  }
  if (this->NumberOfIterations > 0)
  {
    CreateInitializeAllSpatialFlowsToZeroTasks(this->Structure->GetRoot());
  }
  CreateInitializeLeafSinkFlowsToCapTasks(this->Structure->GetRoot());
  CreateCopyMinimalLeafSinkFlowsTasks(this->Structure->GetRoot());
  CreateFindInitialLabellingAndSumTasks(this->Structure->GetRoot());
  CreateClearSourceWorkingBufferTask();
  CreateDivideOutLabelsTasks(this->Structure->GetRoot());
  if (this->NumberOfIterations > 0)
  {
    CreatePropogateLabelsTasks(this->Structure->GetRoot());
  }

  if (this->Debug)
  {
    vtkDebugMacro("Number of tasks to be run: " << Scheduler->NumTasksGoingToHappen);
  }

  return 1;
}

//----------------------------------------------------------------------------
int vtkCudaHierarchicalMaxFlowSegmentation2::RunAlgorithm()
{
  //connect sink flows
  Scheduler->LeafLabelBuffers = this->leafLabelBuffers;
  Scheduler->NumLeaves = this->NumLeaves;

  //if verbose, print progress
  if (this->Debug)
  {
    vtkDebugMacro("Running tasks");
  }
  int NumTasksDone = 0;
  while (Scheduler->CanRunAlgorithmIteration())
  {
    Scheduler->RunAlgorithmIteration();

    //if there are conflicts
    //update progress
    NumTasksDone++;
    if (this->Debug && ReportRate > 0 && NumTasksDone % ReportRate == 0)
    {
      Scheduler->SyncWorkers();
      vtkDebugMacro("Finished " << NumTasksDone << " with " << Scheduler->NumMemCpies << " memory transfers.");
    }
  }
  Scheduler->ReturnLeaves();

  if (this->Debug)
  {
    vtkDebugMacro("Finished all " << NumTasksDone << " tasks with a total of " << Scheduler->NumMemCpies << " memory transfers.");
  }
  assert(Scheduler->BlockedTasks.size() == 0);

  Scheduler->Clear();

  Scheduler->Clear();
  this->ClearWorkingBufferTasks.clear();
  this->UpdateSpatialFlowsTasks.clear();
  this->ApplySinkPotentialBranchTasks.clear();
  this->ApplySinkPotentialLeafTasks.clear();
  this->ApplySourcePotentialTasks.clear();
  this->DivideOutWorkingBufferTasks.clear();
  this->UpdateLabelsTasks.clear();
  this->InitializeLeafSinkFlowsTasks.clear();
  this->MinimizeLeafSinkFlowsTasks.clear();
  this->PropogateLeafSinkFlowsTasks.clear();
  this->InitialLabellingSumTasks.clear();
  this->CorrectLabellingTasks.clear();
  this->PropogateLabellingTasks.clear();

  return 1;
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::FigureOutBufferPriorities(vtkIdType currNode)
{
  //Propagate down the tree
  int NumKids = this->Structure->GetNumberOfChildren(currNode);
  for (int kid = 0; kid < NumKids; kid++)
  {
    FigureOutBufferPriorities(this->Structure->GetChild(currNode, kid));
  }

  //if we are the root, figure out the buffers
  if (this->Structure->GetRoot() == currNode)
  {
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(sourceFlowBuffer, NumKids + 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(sourceWorkingBuffer, NumKids + 3));

    //if we are a leaf, handle separately
  }
  else if (NumKids == 0)
  {
    int Number = LeafMap[currNode];
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(leafDivBuffers[Number], 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(leafFlowXBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(leafFlowYBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(leafFlowZBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(leafSinkBuffers[Number], 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(leafDataTermBuffers[Number], 1));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(leafLabelBuffers[Number], 3));
    if (leafSmoothnessTermBuffers[Number])
    {
      this->Scheduler->CPU2PriorityMap[leafSmoothnessTermBuffers[Number]]++;
    }

    //else, we are a branch
  }
  else
  {
    int Number = BranchMap[currNode];
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(branchDivBuffers[Number], 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(branchFlowXBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(branchFlowYBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(branchFlowZBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(branchSinkBuffers[Number], NumKids + 4));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(branchLabelBuffers[Number], 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(branchWorkingBuffers[Number], NumKids + 3));
    if (branchSmoothnessTermBuffers[Number])
    {
      this->Scheduler->CPU2PriorityMap[branchSmoothnessTermBuffers[Number]]++;
    }
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateClearWorkingBufferTasks(vtkIdType currNode)
{
  int NumKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < NumKids; i++)
  {
    CreateClearWorkingBufferTasks(this->Structure->GetChild(currNode, i));
  }
  if (NumKids == 0)
  {
    return;
  }

  //create the new task
  vtkCudaMaxFlowSegmentationTask* newTask = 0;
  if (currNode == this->Structure->GetRoot())
  {
    newTask = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -NumLeaves, 1, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::ClearWorkingBufferTask);
  }
  else
  {
    newTask = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::ClearWorkingBufferTask);
  }
  this->ClearWorkingBufferTasks[currNode] = newTask;

  //modify the task accordingly
  if (currNode == this->Structure->GetRoot())
  {
    newTask->AddBuffer(sourceWorkingBuffer);
  }
  else
  {
    Scheduler->NoCopyBack.insert(branchWorkingBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchWorkingBuffers[BranchMap[currNode]]);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateUpdateSpatialFlowsTasks(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateUpdateSpatialFlowsTasks(this->Structure->GetChild(currNode, i));
  }
  if (currNode == this->Structure->GetRoot())
  {
    return;
  }

  //create the new task
  //initial Active is -(6+NumKids) if branch since 4 clear buffers, 2 init flow happen in the initialization and NumKids number of label clears
  //initial Active is -7 if leaf since 4 clear buffers, 2 init flow happen in the initialization and NumKids number of label clears
  vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -(6 + (numKids ? numKids : 1)), 1, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::UpdateSpatialFlowsTask);
  newTask->SetConstant1(this->SmoothnessScalars[currNode]);
  this->UpdateSpatialFlowsTasks[currNode] = newTask;
  if (numKids != 0)
  {
    newTask->AddBuffer(branchSinkBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchIncBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchDivBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchLabelBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchFlowXBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchFlowYBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchFlowZBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchSmoothnessTermBuffers[BranchMap[currNode]]);
  }
  else
  {
    newTask->AddBuffer(leafSinkBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafIncBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafDivBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafLabelBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafFlowXBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafFlowYBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafFlowZBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafSmoothnessTermBuffers[LeafMap[currNode]]);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateApplySinkPotentialBranchTasks(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateApplySinkPotentialBranchTasks(this->Structure->GetChild(currNode, i));
  }
  if (numKids == 0)
  {
    return;
  }

  //create the new task
  if (currNode != this->Structure->GetRoot())
  {
    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -2, 2, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::ApplySinkPotentialBranchTask);
    this->ApplySinkPotentialBranchTasks[currNode] = newTask;
    newTask->AddBuffer(branchWorkingBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchIncBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchDivBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchLabelBuffers[BranchMap[currNode]]);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateApplySinkPotentialLeafTasks(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateApplySinkPotentialLeafTasks(this->Structure->GetChild(currNode, i));
  }
  if (numKids != 0)
  {
    return;
  }

  //create the new task
  vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -1, 1, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::ApplySinkPotentialLeafTask);
  this->ApplySinkPotentialLeafTasks[currNode] = newTask;
  newTask->AddBuffer(leafSinkBuffers[LeafMap[currNode]]);
  newTask->AddBuffer(leafIncBuffers[LeafMap[currNode]]);
  newTask->AddBuffer(leafDivBuffers[LeafMap[currNode]]);
  newTask->AddBuffer(leafLabelBuffers[LeafMap[currNode]]);
  newTask->AddBuffer(leafDataTermBuffers[LeafMap[currNode]]);
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateDivideOutWorkingBufferTask(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateDivideOutWorkingBufferTask(this->Structure->GetChild(currNode, i));
  }
  if (numKids == 0)
  {
    return;
  }

  //create the new task
  vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -(numKids + 1), numKids + 1, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::DivideOutWorkingBufferTask);
  newTask->SetConstant1(numKids + (currNode == Structure->GetRoot() ? 0 : 1));
  this->DivideOutWorkingBufferTasks[currNode] = newTask;
  if (currNode != this->Structure->GetRoot())
  {
    newTask->AddBuffer(branchWorkingBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchSinkBuffers[BranchMap[currNode]]);
  }
  else
  {
    newTask->AddBuffer(sourceWorkingBuffer);
    newTask->AddBuffer(sourceFlowBuffer);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateApplySourcePotentialTask(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateApplySourcePotentialTask(this->Structure->GetChild(currNode, i));
  }
  if (currNode == this->Structure->GetRoot())
  {
    return;
  }
  vtkIdType parentNode = this->Structure->GetParent(currNode);

  //find appropriate working buffer
  float* workingBuffer = 0;
  if (parentNode == this->Structure->GetRoot())
  {
    workingBuffer = sourceWorkingBuffer;
  }
  else
  {
    workingBuffer = branchWorkingBuffers[BranchMap[parentNode]];
  }

  //create the new task
  vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -2, 2, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::ApplySourcePotentialTask);
  this->ApplySourcePotentialTasks[currNode] = newTask;
  newTask->AddBuffer(workingBuffer);
  if (numKids != 0)
  {
    newTask->AddBuffer(branchSinkBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchDivBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchLabelBuffers[BranchMap[currNode]]);
  }
  else
  {
    newTask->AddBuffer(leafSinkBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafDivBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafLabelBuffers[LeafMap[currNode]]);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateUpdateLabelsTask(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateUpdateLabelsTask(this->Structure->GetChild(currNode, i));
  }
  if (currNode == this->Structure->GetRoot())
  {
    return;
  }

  //find appropriate number of repetitions
  int NumReps = numKids ? this->NumberOfIterations - 1 : this->NumberOfIterations;

  //create the new task
  vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -2, 2, NumReps, vtkCudaMaxFlowSegmentationTask::UpdateLabelsTask);
  this->UpdateLabelsTasks[currNode] = newTask;
  if (numKids != 0)
  {
    newTask->AddBuffer(branchSinkBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchIncBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchDivBuffers[BranchMap[currNode]]);
    newTask->AddBuffer(branchLabelBuffers[BranchMap[currNode]]);
  }
  else
  {
    newTask->AddBuffer(leafSinkBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafIncBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafDivBuffers[LeafMap[currNode]]);
    newTask->AddBuffer(leafLabelBuffers[LeafMap[currNode]]);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::AddIterationTaskDependencies(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    AddIterationTaskDependencies(this->Structure->GetChild(currNode, i));
  }

  if (numKids == 0)
  {
    vtkIdType parNode = this->Structure->GetParent(currNode);
    this->UpdateSpatialFlowsTasks[currNode]->AddTaskToSignal(this->ApplySinkPotentialLeafTasks[currNode]);
    this->ApplySinkPotentialLeafTasks[currNode]->AddTaskToSignal(this->ApplySourcePotentialTasks[currNode]);
    this->ApplySourcePotentialTasks[currNode]->AddTaskToSignal(this->DivideOutWorkingBufferTasks[parNode]);
    this->ApplySourcePotentialTasks[currNode]->AddTaskToSignal(this->UpdateLabelsTasks[currNode]);
    this->UpdateLabelsTasks[currNode]->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
  }
  else if (currNode == this->Structure->GetRoot())
  {
    this->ClearWorkingBufferTasks[currNode]->AddTaskToSignal(this->DivideOutWorkingBufferTasks[currNode]);
    for (int i = 0; i < numKids; i++)
    {
      this->ClearWorkingBufferTasks[currNode]->AddTaskToSignal(this->ApplySourcePotentialTasks[this->Structure->GetChild(currNode, i)]);
    }
    this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->ClearWorkingBufferTasks[currNode]);
    for (int i = 0; i < numKids; i++)
    {
      this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->UpdateLabelsTasks[this->Structure->GetChild(currNode, i)]);
    }
  }
  else
  {
    vtkIdType parNode = this->Structure->GetParent(currNode);
    this->ClearWorkingBufferTasks[currNode]->AddTaskToSignal(this->ApplySinkPotentialBranchTasks[currNode]);
    for (int i = 0; i < numKids; i++)
    {
      this->ClearWorkingBufferTasks[currNode]->AddTaskToSignal(this->ApplySourcePotentialTasks[this->Structure->GetChild(currNode, i)]);
    }
    this->UpdateSpatialFlowsTasks[currNode]->AddTaskToSignal(this->ApplySinkPotentialBranchTasks[currNode]);
    this->ApplySinkPotentialBranchTasks[currNode]->AddTaskToSignal(this->DivideOutWorkingBufferTasks[currNode]);
    this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->ApplySourcePotentialTasks[currNode]);
    this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->ClearWorkingBufferTasks[currNode]);
    for (int i = 0; i < numKids; i++)
    {
      this->DivideOutWorkingBufferTasks[currNode]->AddTaskToSignal(this->UpdateLabelsTasks[this->Structure->GetChild(currNode, i)]);
    }
    this->ApplySourcePotentialTasks[currNode]->AddTaskToSignal(this->DivideOutWorkingBufferTasks[parNode]);
    this->ApplySourcePotentialTasks[currNode]->AddTaskToSignal(this->UpdateLabelsTasks[currNode]);
    this->UpdateLabelsTasks[currNode]->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateInitializeAllSpatialFlowsToZeroTasks(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateInitializeAllSpatialFlowsToZeroTasks(this->Structure->GetChild(currNode, i));
  }

  //modify the task accordingly
  if (numKids == 0)
  {
    vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask2 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask3 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask4 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
    newTask2->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
    newTask3->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
    newTask4->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
    newTask1->AddBuffer(this->leafDivBuffers[LeafMap[currNode]]);
    newTask2->AddBuffer(this->leafFlowXBuffers[LeafMap[currNode]]);
    newTask3->AddBuffer(this->leafFlowYBuffers[LeafMap[currNode]]);
    newTask4->AddBuffer(this->leafFlowZBuffers[LeafMap[currNode]]);
  }
  else if (currNode != this->Structure->GetRoot())
  {
    vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask2 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask3 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask4 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
    newTask2->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
    newTask3->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
    newTask4->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
    newTask1->AddBuffer(this->branchDivBuffers[BranchMap[currNode]]);
    newTask2->AddBuffer(this->branchFlowXBuffers[BranchMap[currNode]]);
    newTask3->AddBuffer(this->branchFlowYBuffers[BranchMap[currNode]]);
    newTask4->AddBuffer(this->branchFlowZBuffers[BranchMap[currNode]]);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateInitializeLeafSinkFlowsToCapTasks(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateInitializeLeafSinkFlowsToCapTasks(this->Structure->GetChild(currNode, i));
  }
  if (numKids > 0)
  {
    return;
  }

  if (LeafMap[currNode] != 0)
  {
    vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::InitializeLeafFlows);
    vtkCudaMaxFlowSegmentationTask* newTask2 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -2, 1, 1, vtkCudaMaxFlowSegmentationTask::MinimizeLeafFlows);
    InitializeLeafSinkFlowsTasks.insert(std::pair<int, vtkCudaMaxFlowSegmentationTask*>(LeafMap[currNode], newTask1));
    MinimizeLeafSinkFlowsTasks.insert(std::pair<int, vtkCudaMaxFlowSegmentationTask*>(LeafMap[currNode], newTask2));
    newTask1->AddBuffer(this->leafSinkBuffers[LeafMap[currNode]]);
    newTask1->AddBuffer(this->leafDataTermBuffers[LeafMap[currNode]]);
    newTask2->AddBuffer(this->leafSinkBuffers[0]);
    newTask2->AddBuffer(this->leafSinkBuffers[LeafMap[currNode]]);
    newTask1->AddTaskToSignal(newTask2);
    if (InitializeLeafSinkFlowsTasks.find(0) != InitializeLeafSinkFlowsTasks.end())
    {
      InitializeLeafSinkFlowsTasks[0]->AddTaskToSignal(newTask2);
    }
  }
  else
  {
    vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::InitializeLeafFlows);
    InitializeLeafSinkFlowsTasks.insert(std::pair<int, vtkCudaMaxFlowSegmentationTask*>(0, newTask1));
    newTask1->AddBuffer(this->leafSinkBuffers[0]);
    newTask1->AddBuffer(this->leafDataTermBuffers[0]);
    for (std::map<int, vtkCudaMaxFlowSegmentationTask*>::iterator it = MinimizeLeafSinkFlowsTasks.begin();
         it != this->MinimizeLeafSinkFlowsTasks.end(); it++)
    {
      newTask1->AddTaskToSignal(it->second);
    }
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateCopyMinimalLeafSinkFlowsTasks(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateCopyMinimalLeafSinkFlowsTasks(this->Structure->GetChild(currNode, i));
  }

  vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -((int)this->MinimizeLeafSinkFlowsTasks.size()), 1, 1, vtkCudaMaxFlowSegmentationTask::PropogateLeafFlows);
  PropogateLeafSinkFlowsTasks.insert(std::pair<vtkIdType, vtkCudaMaxFlowSegmentationTask*>(currNode, newTask1));
  if (currNode != this->Structure->GetRoot())
  {
    newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
  }
  for (int i = 0; i < numKids; i++)
  {
    newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[this->Structure->GetChild(currNode, i)]);
  }
  newTask1->AddBuffer(this->leafSinkBuffers[0]);
  for (std::map<int, vtkCudaMaxFlowSegmentationTask*>::iterator it = this->MinimizeLeafSinkFlowsTasks.begin(); it != this->MinimizeLeafSinkFlowsTasks.end(); it++)
  {
    it->second->AddTaskToSignal(newTask1);
  }

  if (this->Structure->GetRoot() == currNode)
  {
    newTask1->AddBuffer(this->sourceFlowBuffer);
  }
  else if (numKids > 0)
  {
    newTask1->AddBuffer(this->branchSinkBuffers[BranchMap[currNode]]);
  }
  else
  {
    newTask1->AddBuffer(this->leafSinkBuffers[LeafMap[currNode]]);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateFindInitialLabellingAndSumTasks(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateFindInitialLabellingAndSumTasks(this->Structure->GetChild(currNode, i));
  }
  if (numKids > 0)
  {
    return;
  }

  vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -1, 1, 1, vtkCudaMaxFlowSegmentationTask::InitializeLeafLabels);
  vtkCudaMaxFlowSegmentationTask* newTask2 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -2, 1, 1, vtkCudaMaxFlowSegmentationTask::AccumulateLabels);
  this->PropogateLeafSinkFlowsTasks[currNode]->AddTaskToSignal(newTask1);
  newTask1->AddTaskToSignal(newTask2);
  this->InitialLabellingSumTasks.insert(std::pair<vtkIdType, vtkCudaMaxFlowSegmentationTask*>(currNode, newTask2));
  newTask1->AddBuffer(this->leafSinkBuffers[LeafMap[currNode]]);
  newTask1->AddBuffer(this->leafDataTermBuffers[LeafMap[currNode]]);
  newTask1->AddBuffer(this->leafLabelBuffers[LeafMap[currNode]]);
  newTask2->AddBuffer(this->sourceWorkingBuffer);
  newTask2->AddBuffer(this->leafLabelBuffers[LeafMap[currNode]]);
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateClearSourceWorkingBufferTask()
{
  vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
  newTask->AddBuffer(this->sourceWorkingBuffer);
  for (std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*>::iterator it = InitialLabellingSumTasks.begin(); it != InitialLabellingSumTasks.end(); it++)
  {
    newTask->AddTaskToSignal(it->second);
  }
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreateDivideOutLabelsTasks(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreateDivideOutLabelsTasks(this->Structure->GetChild(currNode, i));
  }
  if (numKids > 0)
  {
    return;
  }

  vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -(int)InitialLabellingSumTasks.size(), 1, 1, vtkCudaMaxFlowSegmentationTask::CorrectLabels);
  this->CorrectLabellingTasks[currNode] = newTask1;
  for (std::map<vtkIdType, vtkCudaMaxFlowSegmentationTask*>::iterator taskIt = InitialLabellingSumTasks.begin(); taskIt != InitialLabellingSumTasks.end(); taskIt++)
  {
    taskIt->second->AddTaskToSignal(newTask1);
  }
  newTask1->AddBuffer(this->sourceWorkingBuffer);
  newTask1->AddBuffer(this->leafLabelBuffers[LeafMap[currNode]]);
  newTask1->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
  newTask1->AddTaskToSignal(this->ClearWorkingBufferTasks[this->Structure->GetRoot()]);
}

//----------------------------------------------------------------------------
void vtkCudaHierarchicalMaxFlowSegmentation2::CreatePropogateLabelsTasks(vtkIdType currNode)
{
  int numKids = this->Structure->GetNumberOfChildren(currNode);
  for (int i = 0; i < numKids; i++)
  {
    CreatePropogateLabelsTasks(this->Structure->GetChild(currNode, i));
  }
  if (currNode == this->Structure->GetRoot() || numKids == 0)
  {
    return;
  }

  //clear the current buffer
  vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
  newTask1->AddBuffer(this->branchLabelBuffers[BranchMap[currNode]]);

  //accumulate from children
  for (int i = 0; i < numKids; i++)
  {
    vtkIdType child = this->Structure->GetChild(currNode, i);
    vtkCudaMaxFlowSegmentationTask* newTask2 = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, -1, 1, 1, vtkCudaMaxFlowSegmentationTask::AccumulateLabels);
    this->PropogateLabellingTasks[child] = newTask2;
    newTask1->AddTaskToSignal(newTask2);
    newTask2->AddBuffer(this->branchLabelBuffers[BranchMap[currNode]]);
    if (this->Structure->IsLeaf(child))
    {
      newTask2->DecrementActivity();
      newTask2->AddBuffer(this->leafLabelBuffers[LeafMap[child]]);
      this->CorrectLabellingTasks[child]->AddTaskToSignal(newTask2);
    }
    else
    {
      newTask2->AddBuffer(this->branchLabelBuffers[BranchMap[child]]);
      int NumKids2 = this->Structure->GetNumberOfChildren(child);
      for (int i2 = 0; i2 < NumKids2; i2++)
      {
        this->PropogateLabellingTasks[this->Structure->GetChild(child, i2)]->AddTaskToSignal(newTask2);
      }
    }
    newTask2->AddTaskToSignal(this->UpdateSpatialFlowsTasks[currNode]);
  }
}