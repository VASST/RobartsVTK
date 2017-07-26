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

/** @file vtkCudaDirectedAcyclicGraphMaxFlowSegmentation.cxx
 *
 *  @brief Implementation file with definitions of GPU-based solver for DAG-based max-flow
 *      segmentation problems with greedy scheduling over multiple GPUs.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note June 22nd 2014 - Documentation first compiled.
 *
 */

#include "CudaObject.h"
#include "vtkCudaDirectedAcyclicGraphMaxFlowSegmentation.h"
#include "vtkCudaMaxFlowSegmentationScheduler.h"
#include "vtkCudaMaxFlowSegmentationTask.h"
#include "vtkCudaMaxFlowSegmentationTask.h"
#include "vtkCudaMaxFlowSegmentationWorker.h"
#include "vtkCudaMaxFlowSegmentationWorker.h"
#include "vtkDataSetAttributes.h"
#include "vtkFloatArray.h"
#include "vtkObjectFactory.h"
#include "vtkRootedDirectedAcyclicGraphBackwardIterator.h"
#include "vtkRootedDirectedAcyclicGraphForwardIterator.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <list>
#include <math.h>
#include <vector>

//----------------------------------------------------------------------------

vtkStandardNewMacro(vtkCudaDirectedAcyclicGraphMaxFlowSegmentation);

//----------------------------------------------------------------------------
vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::vtkCudaDirectedAcyclicGraphMaxFlowSegmentation()
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
vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::~vtkCudaDirectedAcyclicGraphMaxFlowSegmentation()
{
  this->GPUsUsed.clear();
  this->MaxGPUUsageNonDefault.clear();
  delete this->Scheduler;
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::AddDevice(int GPU)
{
  if (GPU >= 0 && GPU < vtkCudaDeviceManager::Singleton()->GetNumberOfDevices())
  {
    this->GPUsUsed.insert(GPU);
  }
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::RemoveDevice(int GPU)
{
  if (this->GPUsUsed.find(GPU) != this->GPUsUsed.end())
  {
    this->GPUsUsed.erase(this->GPUsUsed.find(GPU));
  }
}

//----------------------------------------------------------------------------
bool vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::HasDevice(int GPU)
{
  return (this->GPUsUsed.find(GPU) != this->GPUsUsed.end());
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::ClearDevices()
{
  this->GPUsUsed.clear();
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::SetDevice(int GPU)
{
  this->ClearDevices();
  this->AddDevice(GPU);
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::SetMaxGPUUsage(double usage, int device)
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
double vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::GetMaxGPUUsage(int device)
{
  if (this->MaxGPUUsageNonDefault.find(device) != this->MaxGPUUsageNonDefault.end())
  {
    return this->MaxGPUUsageNonDefault[device];
  }
  return this->MaxGPUUsage;
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::ClearMaxGPUUsage()
{
  this->MaxGPUUsageNonDefault.clear();
}

//----------------------------------------------------------------------------
int vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::InitializeAlgorithm()
{
  //if verbose, print progress
  Scheduler->Clear();
  Scheduler->TotalNumberOfBuffers = this->TotalNumberOfBuffers;
  Scheduler->VolumeSize = this->VolumeSize;
  Scheduler->VX = this->VX;
  Scheduler->VY = this->VY;
  Scheduler->VZ = this->VZ;
  Scheduler->CC = this->CC;
  Scheduler->StepSize = this->StepSize;
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
  UpdateSpatialFlowsTasks.clear();
  ResetSinkFlowTasks.clear();
  ApplySinkPotentialLeafTasks.clear();
  PushUpSourceFlowsTasks.clear();
  PushDownSinkFlowsTasks.clear();
  UpdateLabelsTasks.clear();
  ClearSourceBufferTasks.clear();
  if (this->Debug)
  {
    vtkDebugMacro("Creating tasks for normal iterations.");
  }
  if (this->NumberOfIterations > 0)
  {
    CreateUpdateSpatialFlowsTasks();
    CreateResetSinkFlowRootTasks();
    CreateResetSinkFlowBranchTasks();
    CreateApplySinkPotentialLeafTasks();
    CreatePushUpSourceFlowsLeafTasks();
    CreatePushUpSourceFlowsBranchTasks();
    CreatePushDownSinkFlowsRootTasks();
    CreatePushDownSinkFlowsBranchTasks();
    CreateUpdateLabelsTasks();
    CreateClearSourceBufferTasks();
    AssociateFinishSignals();
  }

  //add tasks in for the initialization (done second for dependency reasons)
  if (this->Debug)
  {
    vtkDebugMacro("Creating tasks for initialization.");
  }
  if (this->NumberOfIterations > 0)
  {
    InitializeSpatialFlowsTasks();
  }
  InitializeSinkFlowsTasks();

  if (this->Debug)
  {
    vtkDebugMacro("Number of tasks to be run: " << Scheduler->NumTasksGoingToHappen);
  }

  return 1;
}

//----------------------------------------------------------------------------
int vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::RunAlgorithm()
{
  //connect sink flows
  Scheduler->LeafLabelBuffers = this->LeafLabelBuffers;
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

  UpdateSpatialFlowsTasks.clear();
  ResetSinkFlowTasks.clear();
  ApplySinkPotentialLeafTasks.clear();
  PushUpSourceFlowsTasks.clear();
  PushDownSinkFlowsTasks.clear();
  UpdateLabelsTasks.clear();
  ClearSourceBufferTasks.clear();

  return 1;
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::FigureOutBufferPriorities(vtkIdType currNode)
{
  //Propagate down the tree
  int NumKids = this->Structure->GetNumberOfChildren(currNode);
  int NumPars = this->Structure->GetNumberOfParents(currNode);
  for (int kid = 0; kid < NumKids; kid++)
  {
    FigureOutBufferPriorities(this->Structure->GetChild(currNode, kid));
  }

  //if we are the root, figure out the buffers
  if (this->Structure->GetRoot() == currNode)
  {
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(SourceFlowBuffer, NumKids + 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(SourceWorkingBuffer, NumKids + 3));

    //if we are a leaf, handle separately
  }
  else if (NumKids == 0)
  {
    int Number = LeafMap[currNode];
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(LeafDivBuffers[Number], 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(LeafFlowXBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(LeafFlowYBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(LeafFlowZBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(LeafSinkBuffers[Number], 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(LeafSourceBuffers[Number], NumPars + 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(LeafDataTermBuffers[Number], 1));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(LeafLabelBuffers[Number], 3));
    if (LeafSmoothnessTermBuffers[Number])
    {
      this->Scheduler->CPU2PriorityMap[LeafSmoothnessTermBuffers[Number]]++;
    }

    //else, we are a branch
  }
  else
  {
    int Number = BranchMap[currNode];
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(BranchDivBuffers[Number], 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(BranchFlowXBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(BranchFlowYBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(BranchFlowZBuffers[Number], 2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(BranchSinkBuffers[Number], NumKids + 4));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(BranchSourceBuffers[Number], NumPars + 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(BranchLabelBuffers[Number], 3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*, int>(BranchWorkingBuffers[Number], NumKids + 3));
    if (BranchSmoothnessTermBuffers[Number])
    {
      this->Scheduler->CPU2PriorityMap[BranchSmoothnessTermBuffers[Number]]++;
    }
  }
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateUpdateSpatialFlowsTasks()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType currNode = forIterator->Next();

    int NumKids = this->Structure->GetNumberOfChildren(currNode);
    int NumParents = this->Structure->GetNumberOfParents(currNode);
    if (currNode == this->Structure->GetRoot())
    {
      continue;
    }

    int StartValue = 4 + NumKids + NumParents;
    StartValue -= (Structure->GetDownLevel(currNode) == 1 ? 1 : 0);
    StartValue += (Structure->IsLeaf(currNode) == 1 ? 1 : 0);

    //create the new task
    //initial Active is -7 (4 clear buffers, 2 set source/sink, 1 set label)
    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(currNode, currNode, Scheduler, -StartValue, NumParents + 1, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::UpdateSpatialFlowsTask);
    newTask->SetConstant1(this->SmoothnessScalars[currNode]);
    this->UpdateSpatialFlowsTasks[currNode] = newTask;
    if (NumKids != 0)
    {
      newTask->AddBuffer(BranchSinkBuffers[BranchMap[currNode]]);
      newTask->AddBuffer(BranchSourceBuffers[BranchMap[currNode]]);
      newTask->AddBuffer(BranchDivBuffers[BranchMap[currNode]]);
      newTask->AddBuffer(BranchLabelBuffers[BranchMap[currNode]]);
      newTask->AddBuffer(BranchFlowXBuffers[BranchMap[currNode]]);
      newTask->AddBuffer(BranchFlowYBuffers[BranchMap[currNode]]);
      newTask->AddBuffer(BranchFlowZBuffers[BranchMap[currNode]]);
      newTask->AddBuffer(BranchSmoothnessTermBuffers[BranchMap[currNode]]);
    }
    else
    {
      newTask->AddBuffer(LeafSinkBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(LeafSourceBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(LeafDivBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(LeafLabelBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(LeafFlowXBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(LeafFlowYBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(LeafFlowZBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(LeafSmoothnessTermBuffers[LeafMap[currNode]]);
    }
  }
  forIterator->Delete();
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateResetSinkFlowRootTasks()
{
  vtkIdType Node = Structure->GetRoot();
  int NumKids = Structure->GetNumberOfChildren(Node);
  vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, -NumLeaves - NumBranches, NumKids, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::ResetSinkFlowRoot);
  ResetSinkFlowTasks[Node] = newTask;
  newTask->SetConstant1(1.0 / (this->CC * this->SourceWeightedNumChildren));
  newTask->AddBuffer(SourceFlowBuffer);
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateResetSinkFlowBranchTasks()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType Node = forIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    if (NumKids == 0 || Node == Structure->GetRoot())
    {
      continue;
    }

    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, -1, 1, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::ResetSinkFlowBranch);
    ResetSinkFlowTasks[Node] = newTask;

    float W = 1.0 / (this->BranchWeightedNumChildren[BranchMap[Node]] + 1.0);
    newTask->SetConstant1(W);
    newTask->SetConstant2(1 - W);
    newTask->AddBuffer(BranchSinkBuffers[BranchMap[Node]]);
    newTask->AddBuffer(BranchSourceBuffers[BranchMap[Node]]);
    newTask->AddBuffer(BranchDivBuffers[BranchMap[Node]]);
    newTask->AddBuffer(BranchLabelBuffers[BranchMap[Node]]);
  }
  forIterator->Delete();

}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateApplySinkPotentialLeafTasks()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType Node = forIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    if (NumKids != 0 || Node == Structure->GetRoot())
    {
      continue;
    }

    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, -1, 1, this->NumberOfIterations, vtkCudaMaxFlowSegmentationTask::ApplySinkPotentialLeafTask);
    ApplySinkPotentialLeafTasks[Node] = newTask;

    newTask->AddBuffer(LeafSinkBuffers[LeafMap[Node]]);
    newTask->AddBuffer(LeafSourceBuffers[LeafMap[Node]]);
    newTask->AddBuffer(LeafDivBuffers[LeafMap[Node]]);
    newTask->AddBuffer(LeafLabelBuffers[LeafMap[Node]]);
    newTask->AddBuffer(LeafDataTermBuffers[LeafMap[Node]]);

  }
  forIterator->Delete();
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreatePushUpSourceFlowsLeafTasks()
{
  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType Node = forIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);
    if (NumKids != 0 || Node == Structure->GetRoot())
    {
      continue;
    }

    for (int i = 0; i < NumParents; i++)
    {
      vtkIdType Edge = Structure->GetInEdge(Node, i).Id;
      vtkIdType Parent = Structure->GetParent(Node, i);

      vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node, Parent, Scheduler, -2, 2, this->NumberOfIterations - 1, vtkCudaMaxFlowSegmentationTask::PushUpSourceFlows);
      PushUpSourceFlowsTasks[Edge] = newTask;

      float W = Weights ? Weights->GetValue(Edge) : 1.0 / (float) Structure->GetNumberOfParents(Node);
      if (Parent == Structure->GetRoot())
      {
        newTask->AddBuffer(SourceFlowBuffer);
        W = W / this->SourceWeightedNumChildren ;
      }
      else
      {
        newTask->AddBuffer(BranchSinkBuffers[BranchMap[Parent]]);
        W = W / (this->BranchWeightedNumChildren[BranchMap[Parent]] + 1);
      }
      newTask->SetConstant1(W);
      newTask->AddBuffer(this->LeafSinkBuffers[LeafMap[Node]]);
      newTask->AddBuffer(this->LeafSourceBuffers[LeafMap[Node]]);
      newTask->AddBuffer(this->LeafDivBuffers[LeafMap[Node]]);
      newTask->AddBuffer(this->LeafLabelBuffers[LeafMap[Node]]);
    }

  }
  forIterator->Delete();
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreatePushUpSourceFlowsBranchTasks()
{
  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType Node = forIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);
    if (NumKids == 0 || Node == Structure->GetRoot())
    {
      continue;
    }

    for (int i = 0; i < NumParents; i++)
    {
      vtkIdType Edge = Structure->GetInEdge(Node, i).Id;
      vtkIdType Parent = Structure->GetParent(Node, i);

      vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node, Parent, Scheduler, -2 - NumKids, 2 + NumKids, this->NumberOfIterations - 1, vtkCudaMaxFlowSegmentationTask::PushUpSourceFlows);
      PushUpSourceFlowsTasks[Edge] = newTask;

      float W = Weights ? Weights->GetValue(Edge) : 1.0 / (float) Structure->GetNumberOfParents(Node);
      if (Parent == Structure->GetRoot())
      {
        newTask->AddBuffer(SourceFlowBuffer);
        W = W / this->SourceWeightedNumChildren ;
      }
      else
      {
        newTask->AddBuffer(BranchSinkBuffers[BranchMap[Parent]]);
        W = W / (this->BranchWeightedNumChildren[BranchMap[Parent]] + 1) ;
      }
      newTask->SetConstant1(W);
      newTask->AddBuffer(this->BranchSinkBuffers[BranchMap[Node]]);
      newTask->AddBuffer(this->BranchSourceBuffers[BranchMap[Node]]);
      newTask->AddBuffer(this->BranchDivBuffers[BranchMap[Node]]);
      newTask->AddBuffer(this->BranchLabelBuffers[BranchMap[Node]]);
    }

  }
  forIterator->Delete();
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreatePushDownSinkFlowsRootTasks()
{
  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));
  vtkIdType Node = Structure->GetRoot();
  int NumKids = Structure->GetNumberOfChildren(Node);

  for (int i = 0; i < NumKids; i++)
  {
    vtkIdType Edge = Structure->GetOutEdge(Node, i).Id;
    vtkIdType Child = Structure->GetChild(Node, i);
    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node, Child, Scheduler, -2 - NumKids, 2 + NumKids, this->NumberOfIterations - 1, vtkCudaMaxFlowSegmentationTask::PushDownSinkFlows);
    PushDownSinkFlowsTasks[Edge] = newTask;
    float W = Weights ? Weights->GetValue(Edge) : 1.0 / (double)Structure->GetNumberOfParents(Child);
    newTask->SetConstant1(W);
    newTask->AddBuffer(SourceFlowBuffer);
    if (Structure->IsLeaf(Child))
    {
      newTask->AddBuffer(LeafSourceBuffers[LeafMap[Child]]);
    }
    else
    {
      newTask->AddBuffer(BranchSourceBuffers[BranchMap[Child]]);
    }
  }
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreatePushDownSinkFlowsBranchTasks()
{
  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType Node = forIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);
    if (NumKids == 0 || Node == Structure->GetRoot())
    {
      continue;
    }

    for (int i = 0; i < NumKids; i++)
    {
      vtkIdType Edge = Structure->GetOutEdge(Node, i).Id;
      vtkIdType Child = Structure->GetChild(Node, i);
      vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node, Child, Scheduler, -2 - NumKids, 2 + NumKids, this->NumberOfIterations - 1, vtkCudaMaxFlowSegmentationTask::PushDownSinkFlows);
      PushDownSinkFlowsTasks[Edge] = newTask;
      float W = Weights ? Weights->GetValue(Edge) : 1.0 / (double)Structure->GetNumberOfParents(Child);
      newTask->SetConstant1(W);
      newTask->AddBuffer(BranchSinkBuffers[BranchMap[Node]]);
      if (Structure->IsLeaf(Child))
      {
        newTask->AddBuffer(LeafSourceBuffers[LeafMap[Child]]);
      }
      else
      {
        newTask->AddBuffer(BranchSourceBuffers[BranchMap[Child]]);
      }
    }

  }
  forIterator->Delete();

}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateUpdateLabelsTasks()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType Node = forIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    if (Node == Structure->GetRoot())
    {
      continue;
    }

    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, -1 - NumKids, 1 + NumKids, this->NumberOfIterations - (NumKids ? 1 : 0), vtkCudaMaxFlowSegmentationTask::UpdateLabelsTask);
    UpdateLabelsTasks[Node] = newTask;
    if (Structure->IsLeaf(Node))
    {
      newTask->AddBuffer(LeafSinkBuffers[LeafMap[Node]]);
      newTask->AddBuffer(LeafSourceBuffers[LeafMap[Node]]);
      newTask->AddBuffer(LeafDivBuffers[LeafMap[Node]]);
      newTask->AddBuffer(LeafLabelBuffers[LeafMap[Node]]);
    }
    else
    {
      newTask->AddBuffer(BranchSinkBuffers[BranchMap[Node]]);
      newTask->AddBuffer(BranchSourceBuffers[BranchMap[Node]]);
      newTask->AddBuffer(BranchDivBuffers[BranchMap[Node]]);
      newTask->AddBuffer(BranchLabelBuffers[BranchMap[Node]]);
    }
  }
  forIterator->Delete();
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateClearSourceBufferTasks()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType Node = forIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);
    if (Node == Structure->GetRoot())
    {
      continue;
    }

    int NumRequired = (NumKids) ? 1 + NumParents + NumKids : NumParents;

    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, -NumRequired, NumRequired, this->NumberOfIterations - 1, vtkCudaMaxFlowSegmentationTask::ClearSourceBuffer);
    ClearSourceBufferTasks[Node] = newTask;
    if (Structure->IsLeaf(Node))
    {
      newTask->AddBuffer(LeafSourceBuffers[LeafMap[Node]]);
    }
    else
    {
      newTask->AddBuffer(BranchSourceBuffers[BranchMap[Node]]);
    }
  }
  forIterator->Delete();
}

//Index
// (1) Update spatial flows
// (2) Reset sink flows
// (3) Push up source flows
// (4) Push down sink flows
// (5) Update labels
// (6) Clear source flows

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::AssociateFinishSignals()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType Node = forIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);

    bool isLeaf = Structure->IsLeaf(Node);
    bool isRoot = (Structure->GetRoot() == Node);
    bool isBranch = !isLeaf && !isRoot;

    //link (1) to (2) in B and L
    if (!isRoot)
    {
      UpdateSpatialFlowsTasks[Node]->AddTaskToSignal(isLeaf ? ApplySinkPotentialLeafTasks[Node] : ResetSinkFlowTasks[Node]);
    }

    //link (2) to (5) in L and B
    if (isLeaf)
    {
      ApplySinkPotentialLeafTasks[Node]->AddTaskToSignal(UpdateLabelsTasks[Node]);
    }
    if (isBranch)
    {
      ResetSinkFlowTasks[Node]->AddTaskToSignal(UpdateLabelsTasks[Node]);
    }

    //link (2) to Child(3) for B and R
    if (!isLeaf)
    {
      for (int i = 0; i < NumKids; i++)
      {
        ResetSinkFlowTasks[Node]->AddTaskToSignal(PushUpSourceFlowsTasks[Structure->GetOutEdge(Node, i).Id]);
      }
    }

    //link (2) to (3) for B
    if (isBranch)
    {
      for (int i = 0; i < NumParents; i++)
      {
        ResetSinkFlowTasks[Node]->AddTaskToSignal(PushUpSourceFlowsTasks[Structure->GetInEdge(Node, i).Id]);
      }
    }

    //link (2) to (4) for B and R
    if (!isLeaf)
    {
      for (int i = 0; i < NumKids; i++)
      {
        ResetSinkFlowTasks[Node]->AddTaskToSignal(PushDownSinkFlowsTasks[Structure->GetOutEdge(Node, i).Id]);
      }
    }

    //link (3) to (6) for L and B
    if (!isRoot)
    {
      for (int i = 0; i < NumParents; i++)
      {
        PushUpSourceFlowsTasks[Structure->GetInEdge(Node, i).Id]->AddTaskToSignal(ClearSourceBufferTasks[Node]);
      }
    }

    //link (3) to Parent(3)(4)(5) for L and B
    if (!isRoot)
    {
      for (int i = 0; i < NumParents; i++)
      {
        vtkIdType Parent = Structure->GetParent(Node, i);
        if (Parent != Structure->GetRoot())
        {
          PushUpSourceFlowsTasks[Structure->GetInEdge(Node, i).Id]->AddTaskToSignal(UpdateLabelsTasks[Parent]);
        }
        for (int j = 0; j < Structure->GetNumberOfParents(Parent); j++)
        {
          if (Parent != Structure->GetRoot())
          {
            PushUpSourceFlowsTasks[Structure->GetInEdge(Node, i).Id]->AddTaskToSignal(PushUpSourceFlowsTasks[Structure->GetInEdge(Parent, j).Id]);
          }
        }
        for (int j = 0; j < Structure->GetNumberOfChildren(Parent); j++)
        {
          PushUpSourceFlowsTasks[Structure->GetInEdge(Node, i).Id]->AddTaskToSignal(PushDownSinkFlowsTasks[Structure->GetOutEdge(Parent, j).Id]);
        }
      }
    }

    //link (4) to (2) for R
    if (isRoot)
    {
      for (int i = 0; i < NumKids; i++)
      {
        PushDownSinkFlowsTasks[Structure->GetOutEdge(Node, i).Id]->AddTaskToSignal(ResetSinkFlowTasks[Node]);
      }
    }

    //link (4) to (6) for B
    if (isBranch)
    {
      for (int i = 0; i < NumKids; i++)
      {
        PushDownSinkFlowsTasks[Structure->GetOutEdge(Node, i).Id]->AddTaskToSignal(ClearSourceBufferTasks[Node]);
      }
    }

    //link (4) to Child(1) for B and R
    if (!isLeaf)
    {
      for (int i = 0; i < NumKids; i++)
      {
        PushDownSinkFlowsTasks[Structure->GetOutEdge(Node, i).Id]->AddTaskToSignal(UpdateSpatialFlowsTasks[Structure->GetChild(Node, i)]);
      }
    }

    //link (5) to (3) for L
    if (isLeaf)
    {
      for (int i = 0; i < NumParents; i++)
      {
        UpdateLabelsTasks[Node]->AddTaskToSignal(PushUpSourceFlowsTasks[Structure->GetInEdge(Node, i).Id]);
      }
    }

    //link (5) to (6) for B
    if (isBranch)
    {
      UpdateLabelsTasks[Node]->AddTaskToSignal(ClearSourceBufferTasks[Node]);
    }

    //link (6) to (1) for B and L
    if (!isRoot)
    {
      ClearSourceBufferTasks[Node]->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
    }

    //link (6) to Parent(4) for B and L
    if (!isRoot)
    {
      for (int i = 0; i < NumParents; i++)
      {
        ClearSourceBufferTasks[Node]->AddTaskToSignal(PushDownSinkFlowsTasks[Structure->GetInEdge(Node, i).Id]);
      }
    }
  }
  forIterator->Delete();
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::InitializeSpatialFlowsTasks()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* forIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  forIterator->SetDAG(this->Structure);
  while (forIterator->HasNext())
  {
    vtkIdType Node = forIterator->Next();
    if (Node == this->Structure->GetRoot())
    {
      continue;
    }

    //create the new task
    //initial Active is -7 (4 clear buffers, 2 set source/sink, 1 set label)
    vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask2 = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask3 = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask4 = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    if (Structure->IsLeaf(Node))
    {
      newTask1->AddBuffer(LeafDivBuffers[LeafMap[Node]]);
      newTask2->AddBuffer(LeafFlowXBuffers[LeafMap[Node]]);
      newTask3->AddBuffer(LeafFlowYBuffers[LeafMap[Node]]);
      newTask4->AddBuffer(LeafFlowZBuffers[LeafMap[Node]]);
    }
    else
    {
      newTask1->AddBuffer(BranchDivBuffers[BranchMap[Node]]);
      newTask2->AddBuffer(BranchFlowXBuffers[BranchMap[Node]]);
      newTask3->AddBuffer(BranchFlowYBuffers[BranchMap[Node]]);
      newTask4->AddBuffer(BranchFlowZBuffers[BranchMap[Node]]);
    }
    newTask1->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
    newTask2->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
    newTask3->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
    newTask4->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
  }
  forIterator->Delete();
}

//----------------------------------------------------------------------------
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::InitializeSinkFlowsTasks()
{
  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  //find minimum sink flow
  vtkCudaMaxFlowSegmentationTask* initialCopy = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::InitializeLeafFlows);
  initialCopy->AddBuffer(SourceFlowBuffer);
  initialCopy->AddBuffer(LeafDataTermBuffers[0]);
  vtkCudaMaxFlowSegmentationTask** findMin = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for (int i = 1; i < this->NumLeaves; i++)
  {
    findMin[i] = new vtkCudaMaxFlowSegmentationTask(i, i, Scheduler, -1, 1, 1, vtkCudaMaxFlowSegmentationTask::MinimizeLeafFlows);
    findMin[i]->AddBuffer(SourceFlowBuffer);
    findMin[i]->AddBuffer(LeafDataTermBuffers[i]);
    initialCopy->AddTaskToSignal(findMin[i]);
  }

  //apply min to all leaves
  vtkCudaMaxFlowSegmentationTask** propogate = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for (int i = 0; i < this->NumLeaves; i++)
  {
    propogate[i] = new vtkCudaMaxFlowSegmentationTask(i, i, Scheduler, -NumLeaves + 1, 1, 1, vtkCudaMaxFlowSegmentationTask::PropogateLeafFlowsInc);
    propogate[i]->AddBuffer(SourceFlowBuffer);
    propogate[i]->AddBuffer(LeafSinkBuffers[i]);
    propogate[i]->AddBuffer(LeafSourceBuffers[i]);
    for (int j = 1; j < NumLeaves; j++)
    {
      findMin[j]->AddTaskToSignal(propogate[i]);
    }
  }

  //find a=0 labeling (not normalized to be valid)
  vtkCudaMaxFlowSegmentationTask** indicate = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for (int i = 0; i < this->NumLeaves; i++)
  {
    indicate[i] = new vtkCudaMaxFlowSegmentationTask(i, i, Scheduler, -1, 1, 1, vtkCudaMaxFlowSegmentationTask::InitializeLeafLabels);
    indicate[i]->AddBuffer(LeafSinkBuffers[i]);
    indicate[i]->AddBuffer(LeafDataTermBuffers[i]);
    indicate[i]->AddBuffer(LeafLabelBuffers[i]);
    indicate[i]->AddBuffer(LeafLabelBuffers[i]);
    indicate[i]->AddTaskToSignal(ResetSinkFlowTasks[this->Structure->GetRoot()]);
    propogate[i]->AddTaskToSignal(indicate[i]);
  }

  //accumulate labels
  vtkCudaMaxFlowSegmentationTask* clearAccumulator = new vtkCudaMaxFlowSegmentationTask(0, 0, Scheduler, 0, 1, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
  clearAccumulator->AddBuffer(SourceWorkingBuffer);
  vtkCudaMaxFlowSegmentationTask** accumulate = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for (int i = 0; i < this->NumLeaves; i++)
  {
    accumulate[i] = new vtkCudaMaxFlowSegmentationTask(i, i, Scheduler, -2, 1, 1, vtkCudaMaxFlowSegmentationTask::AccumulateLabels);
    accumulate[i]->AddBuffer(SourceWorkingBuffer);
    accumulate[i]->AddBuffer(LeafLabelBuffers[i]);
    clearAccumulator->AddTaskToSignal(accumulate[i]);
    indicate[i]->AddTaskToSignal(accumulate[i]);
  }

  //validate a=0 labeling
  vtkCudaMaxFlowSegmentationTask** divide = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for (int i = 0; i < this->NumLeaves; i++)
  {
    divide[i] = new vtkCudaMaxFlowSegmentationTask(i, i, Scheduler, -NumLeaves, 1, 1, vtkCudaMaxFlowSegmentationTask::CorrectLabels);
    divide[i]->AddBuffer(SourceWorkingBuffer);
    divide[i]->AddBuffer(LeafLabelBuffers[i]);
    for (int j = 0; j < NumLeaves; j++)
    {
      accumulate[j]->AddTaskToSignal(divide[i]);
    }
  }
  vtkRootedDirectedAcyclicGraphBackwardIterator* backIterator = vtkRootedDirectedAcyclicGraphBackwardIterator::New();
  backIterator->SetDAG(this->Structure);
  while (backIterator->HasNext())
  {
    vtkIdType Node = backIterator->Next();
    int NumKids = this->Structure->GetNumberOfChildren(Node);
    if (NumKids != 0)
    {
      continue;
    }
    divide[LeafMap[Node]]->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
  }

  //propogate sink flows upwards
  vtkCudaMaxFlowSegmentationTask** propogateB = new vtkCudaMaxFlowSegmentationTask* [NumBranches];
  backIterator->Restart();
  while (backIterator->HasNext())
  {
    vtkIdType Node = backIterator->Next();
    int NumKids = this->Structure->GetNumberOfChildren(Node);
    if (NumKids == 0 || Node == this->Structure->GetRoot())
    {
      continue;
    }
    propogateB[BranchMap[Node]] = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, -NumLeaves + 1, 1, 1, vtkCudaMaxFlowSegmentationTask::PropogateLeafFlowsInc);
    propogateB[BranchMap[Node]]->AddBuffer(SourceFlowBuffer);
    propogateB[BranchMap[Node]]->AddBuffer(BranchSinkBuffers[BranchMap[Node]]);
    propogateB[BranchMap[Node]]->AddBuffer(BranchSourceBuffers[BranchMap[Node]]);
    propogateB[BranchMap[Node]]->AddTaskToSignal(ResetSinkFlowTasks[this->Structure->GetRoot()]);
    for (int j = 1; j < NumLeaves; j++)
    {
      findMin[j]->AddTaskToSignal(propogateB[BranchMap[Node]]);
    }
  }

  //propagate labels up
  vtkCudaMaxFlowSegmentationTask** accumLabels = new vtkCudaMaxFlowSegmentationTask* [NumEdges];
  backIterator->Restart();
  while (backIterator->HasNext())
  {
    vtkIdType Node = backIterator->Next();
    if (Node == this->Structure->GetRoot())
    {
      continue;
    }
    if (this->Structure->IsLeaf(Node))
    {
      continue;
    }

    //clear buffer
    int numKids = Structure->GetNumberOfChildren(Node);
    vtkCudaMaxFlowSegmentationTask* clear = new vtkCudaMaxFlowSegmentationTask(Node, Node, Scheduler, -numKids, numKids, 1, vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    clear->AddBuffer(BranchLabelBuffers[BranchMap[Node]]);
    for (int i = 0; i < numKids; i++)
    {
      vtkIdType Child = this->Structure->GetChild(Node, i);
      if (this->Structure->IsLeaf(Child))
      {
        divide[LeafMap[Child]]->AddTaskToSignal(clear);
      }
      else
        for (int j = 0; j < this->Structure->GetNumberOfChildren(Child); j++)
        {
          accumLabels[this->Structure->GetOutEdge(Child, j).Id]->AddTaskToSignal(clear);
        }
    }

    //create accumulation tasks
    for (int i = 0; i < numKids; i++)
    {
      vtkIdType Child = this->Structure->GetChild(Node, i);
      accumLabels[this->Structure->GetOutEdge(Node, i).Id] = new vtkCudaMaxFlowSegmentationTask(Node, Child, Scheduler, -1, 1, 1, vtkCudaMaxFlowSegmentationTask::AccumulateLabelsWeighted);
      accumLabels[this->Structure->GetOutEdge(Node, i).Id]->SetConstant1(Weights ? Weights->GetValue(this->Structure->GetOutEdge(Node, i).Id) : 1.0 / (float)this->Structure->GetNumberOfParents(Child));
      accumLabels[this->Structure->GetOutEdge(Node, i).Id]->AddBuffer(BranchLabelBuffers[BranchMap[Node]]);
      accumLabels[this->Structure->GetOutEdge(Node, i).Id]->AddBuffer(this->Structure->IsLeaf(Child) ?
          LeafLabelBuffers[LeafMap[Child]] : BranchLabelBuffers[BranchMap[Child]]);
      clear->AddTaskToSignal(accumLabels[this->Structure->GetOutEdge(Node, i).Id]);
      accumLabels[this->Structure->GetOutEdge(Node, i).Id]->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
      accumLabels[this->Structure->GetOutEdge(Node, i).Id]->AddTaskToSignal(UpdateSpatialFlowsTasks[Child]);
    }

  }
  backIterator->Delete();

  delete[] findMin;
  delete[] indicate;
  delete[] accumulate;
  delete[] divide;
  delete[] propogate;
  delete[] propogateB;
  delete[] accumLabels;
}