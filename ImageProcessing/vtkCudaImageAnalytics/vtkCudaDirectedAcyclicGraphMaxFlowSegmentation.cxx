/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkHierarchicalMaxFlowSegmentation2.cxx

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

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

vtkStandardNewMacro(vtkCudaDirectedAcyclicGraphMaxFlowSegmentation);

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

vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::~vtkCudaDirectedAcyclicGraphMaxFlowSegmentation()
{
  this->GPUsUsed.clear();
  this->MaxGPUUsageNonDefault.clear();
  delete this->Scheduler;
}

//------------------------------------------------------------//

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::AddDevice(int GPU)
{
  if( GPU >= 0 && GPU < vtkCudaDeviceManager::Singleton()->GetNumberOfDevices() )
  {
    this->GPUsUsed.insert(GPU);
  }
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::RemoveDevice(int GPU)
{
  if( this->GPUsUsed.find(GPU) != this->GPUsUsed.end() )
  {
    this->GPUsUsed.erase(this->GPUsUsed.find(GPU));
  }
}

bool vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::HasDevice(int GPU)
{
  return (this->GPUsUsed.find(GPU) != this->GPUsUsed.end());
}
void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::ClearDevices()
{
  this->GPUsUsed.clear();
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::SetDevice(int GPU)
{
  this->ClearDevices();
  this->AddDevice(GPU);
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::SetMaxGPUUsage(double usage, int device)
{
  if( usage < 0.0 )
  {
    usage = 0.0;
  }
  else if( usage > 1.0 )
  {
    usage = 1.0;
  }
  if( device >= 0 && device < vtkCudaDeviceManager::Singleton()->GetNumberOfDevices() )
  {
    this->MaxGPUUsageNonDefault[device] = usage;
  }
}

double vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::GetMaxGPUUsage(int device)
{
  if( this->MaxGPUUsageNonDefault.find(device) != this->MaxGPUUsageNonDefault.end() )
  {
    return this->MaxGPUUsageNonDefault[device];
  }
  return this->MaxGPUUsage;
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::ClearMaxGPUUsage()
{
  this->MaxGPUUsageNonDefault.clear();
}

//-----------------------------------------------------------------------------------------------//
//-----------------------------------------------------------------------------------------------//

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
  if( this->Debug )
  {
    vtkDebugMacro("Building workers.");
  }
  for(std::set<int>::iterator gpuIterator = GPUsUsed.begin(); gpuIterator != GPUsUsed.end(); gpuIterator++)
  {
    double usage = this->MaxGPUUsage;
    if( this->MaxGPUUsageNonDefault.find(*gpuIterator) != this->MaxGPUUsageNonDefault.end() )
    {
      usage = this->MaxGPUUsageNonDefault[*gpuIterator];
    }
    if( this->Scheduler->CreateWorker(*gpuIterator,usage) )
    {
      vtkErrorMacro("Could not allocate sufficient GPU buffers.");
      Scheduler->Clear();
      while( CPUBuffersAcquired.size() > 0 )
      {
        float* tempBuffer = CPUBuffersAcquired.front();
        delete[] tempBuffer;
        CPUBuffersAcquired.pop_front();
      }
    }
  }

  //if verbose, print progress
  if( this->Debug )
  {
    vtkDebugMacro("Find priority structures.");
  }

  //create LIFO priority queue (priority stack) data structure
  FigureOutBufferPriorities( this->Structure->GetRoot() );

  //add tasks in for the normal iterations (done first for dependancy reasons)
  UpdateSpatialFlowsTasks.clear();
  ResetSinkFlowTasks.clear();
  ApplySinkPotentialLeafTasks.clear();
  PushUpSourceFlowsTasks.clear();
  PushDownSinkFlowsTasks.clear();
  UpdateLabelsTasks.clear();
  ClearSourceBufferTasks.clear();
  if( this->Debug )
  {
    vtkDebugMacro("Creating tasks for normal iterations.");
  }
  if( this->NumberOfIterations > 0 )
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

  //add tasks in for the initialization (done second for dependancy reasons)
  if( this->Debug )
  {
    vtkDebugMacro("Creating tasks for initialization.");
  }
  if( this->NumberOfIterations > 0)
  {
    InitializeSpatialFlowsTasks();
  }
  InitializeSinkFlowsTasks();

  if( this->Debug )
  {
    vtkDebugMacro("Number of tasks to be run: " << Scheduler->NumTasksGoingToHappen);
  }

  return 1;
}

int vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::RunAlgorithm()
{

  //connect sink flows
  Scheduler->leafLabelBuffers = this->leafLabelBuffers;
  Scheduler->NumLeaves = this->NumLeaves;

  //if verbose, print progress
  if( this->Debug )
  {
    vtkDebugMacro("Running tasks");
  }
  int NumTasksDone = 0;
  while( Scheduler->CanRunAlgorithmIteration() )
  {
    Scheduler->RunAlgorithmIteration();

    //if there are conflicts
    //update progress
    NumTasksDone++;
    if( this->Debug && ReportRate > 0 && NumTasksDone % ReportRate == 0 )
    {
      Scheduler->SyncWorkers();
      vtkDebugMacro( "Finished " << NumTasksDone << " with " << Scheduler->NumMemCpies << " memory transfers.");
    }

  }
  Scheduler->ReturnLeaves();
  if( this->Debug )
  {
    vtkDebugMacro( "Finished all " << NumTasksDone << " tasks with a total of " << Scheduler->NumMemCpies << " memory transfers.");
  }
  assert( Scheduler->BlockedTasks.size() == 0 );

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

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::FigureOutBufferPriorities( vtkIdType currNode )
{

  //Propogate down the tree
  int NumKids = this->Structure->GetNumberOfChildren(currNode);
  int NumPars = this->Structure->GetNumberOfParents(currNode);
  for(int kid = 0; kid < NumKids; kid++)
  {
    FigureOutBufferPriorities( this->Structure->GetChild(currNode,kid) );
  }

  //if we are the root, figure out the buffers
  if( this->Structure->GetRoot() == currNode )
  {
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(sourceFlowBuffer,NumKids+2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(sourceWorkingBuffer,NumKids+3));

    //if we are a leaf, handle separately
  }
  else if( NumKids == 0 )
  {
    int Number = LeafMap[currNode];
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(leafDivBuffers[Number],3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(leafFlowXBuffers[Number],2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(leafFlowYBuffers[Number],2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(leafFlowZBuffers[Number],2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(leafSinkBuffers[Number],3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(leafSourceBuffers[Number],NumPars+3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(leafDataTermBuffers[Number],1));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(leafLabelBuffers[Number],3));
    if( leafSmoothnessTermBuffers[Number] )
    {
      this->Scheduler->CPU2PriorityMap[leafSmoothnessTermBuffers[Number]]++;
    }

    //else, we are a branch
  }
  else
  {
    int Number = BranchMap[currNode];
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(branchDivBuffers[Number],3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(branchFlowXBuffers[Number],2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(branchFlowYBuffers[Number],2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(branchFlowZBuffers[Number],2));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(branchSinkBuffers[Number],NumKids+4));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(branchSourceBuffers[Number],NumPars+3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(branchLabelBuffers[Number],3));
    this->Scheduler->CPU2PriorityMap.insert(std::pair<float*,int>(branchWorkingBuffers[Number],NumKids+3));
    if( branchSmoothnessTermBuffers[Number] )
    {
      this->Scheduler->CPU2PriorityMap[branchSmoothnessTermBuffers[Number]]++;
    }
  }
}



//------------------------------------------------------------//
//------------------------------------------------------------//

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateUpdateSpatialFlowsTasks()
{

  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType currNode = ForIterator->Next();

    int NumKids = this->Structure->GetNumberOfChildren(currNode);
    int NumParents = this->Structure->GetNumberOfParents(currNode);
    if( currNode == this->Structure->GetRoot() )
    {
      continue;
    }

    int StartValue = 4 + NumKids + NumParents;
    StartValue -= (Structure->GetDownLevel(currNode) == 1 ? 1 : 0);
    StartValue += (Structure->IsLeaf(currNode) == 1 ? 1 : 0);

    //create the new task
    //initial Active is -7 (4 clear buffers, 2 set source/sink, 1 set label)
    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(currNode, currNode, Scheduler, -StartValue, NumParents+1, this->NumberOfIterations,vtkCudaMaxFlowSegmentationTask::UpdateSpatialFlowsTask);
    newTask->SetConstant1( this->SmoothnessScalars[currNode] );
    this->UpdateSpatialFlowsTasks[currNode] = newTask;
    if(NumKids != 0)
    {
      newTask->AddBuffer(branchSinkBuffers[BranchMap[currNode]]);
      newTask->AddBuffer(branchSourceBuffers[BranchMap[currNode]]);
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
      newTask->AddBuffer(leafSourceBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(leafDivBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(leafLabelBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(leafFlowXBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(leafFlowYBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(leafFlowZBuffers[LeafMap[currNode]]);
      newTask->AddBuffer(leafSmoothnessTermBuffers[LeafMap[currNode]]);
    }
  }
  ForIterator->Delete();
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateResetSinkFlowRootTasks()
{
  vtkIdType Node = Structure->GetRoot();
  int NumKids = Structure->GetNumberOfChildren(Node);
  vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, -NumLeaves-NumBranches, NumKids, this->NumberOfIterations,vtkCudaMaxFlowSegmentationTask::ResetSinkFlowRoot);
  ResetSinkFlowTasks[Node] = newTask;
  newTask->SetConstant1( 1.0 / (this->CC * this->SourceWeightedNumChildren) );
  newTask->AddBuffer( sourceFlowBuffer );
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateResetSinkFlowBranchTasks()
{

  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType Node = ForIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    if( NumKids == 0 || Node == Structure->GetRoot() )
    {
      continue;
    }

    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, -1, 1, this->NumberOfIterations,vtkCudaMaxFlowSegmentationTask::ResetSinkFlowBranch);
    ResetSinkFlowTasks[Node] = newTask;

    float W = 1.0 / (this->BranchWeightedNumChildren[BranchMap[Node]]+1.0);
    newTask->SetConstant1( W );
    newTask->SetConstant2( 1-W );
    newTask->AddBuffer( branchSinkBuffers[BranchMap[Node]] );
    newTask->AddBuffer( branchSourceBuffers[BranchMap[Node]] );
    newTask->AddBuffer( branchDivBuffers[BranchMap[Node]] );
    newTask->AddBuffer( branchLabelBuffers[BranchMap[Node]] );

  }
  ForIterator->Delete();

}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateApplySinkPotentialLeafTasks()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType Node = ForIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    if( NumKids != 0 || Node == Structure->GetRoot() )
    {
      continue;
    }

    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, -1, 1, this->NumberOfIterations,vtkCudaMaxFlowSegmentationTask::ApplySinkPotentialLeafTask);
    ApplySinkPotentialLeafTasks[Node] = newTask;

    newTask->AddBuffer( leafSinkBuffers[LeafMap[Node]] );
    newTask->AddBuffer( leafSourceBuffers[LeafMap[Node]] );
    newTask->AddBuffer( leafDivBuffers[LeafMap[Node]] );
    newTask->AddBuffer( leafLabelBuffers[LeafMap[Node]] );
    newTask->AddBuffer( leafDataTermBuffers[LeafMap[Node]] );

  }
  ForIterator->Delete();
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreatePushUpSourceFlowsLeafTasks()
{

  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType Node = ForIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);
    if( NumKids != 0 || Node == Structure->GetRoot() )
    {
      continue;
    }

    for(int i = 0; i < NumParents; i++)
    {
      vtkIdType Edge = Structure->GetInEdge(Node,i).Id;
      vtkIdType Parent = Structure->GetParent(Node,i);

      vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node,Parent,Scheduler, -2, 2, this->NumberOfIterations-1,vtkCudaMaxFlowSegmentationTask::PushUpSourceFlows);
      PushUpSourceFlowsTasks[Edge] = newTask;

      float W = Weights ? Weights->GetValue(Edge) : 1.0 / (float) Structure->GetNumberOfParents(Node);
      if(Parent == Structure->GetRoot())
      {
        newTask->AddBuffer(sourceFlowBuffer);
        W = W / this->SourceWeightedNumChildren ;
      }
      else
      {
        newTask->AddBuffer( branchSinkBuffers[BranchMap[Parent]]);
        W = W / (this->BranchWeightedNumChildren[BranchMap[Parent]]+1);
      }
      newTask->SetConstant1(W);
      newTask->AddBuffer( this->leafSinkBuffers[LeafMap[Node]]);
      newTask->AddBuffer( this->leafSourceBuffers[LeafMap[Node]]);
      newTask->AddBuffer( this->leafDivBuffers[LeafMap[Node]]);
      newTask->AddBuffer( this->leafLabelBuffers[LeafMap[Node]]);
    }

  }
  ForIterator->Delete();
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreatePushUpSourceFlowsBranchTasks()
{

  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType Node = ForIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);
    if( NumKids == 0 || Node == Structure->GetRoot() )
    {
      continue;
    }

    for(int i = 0; i < NumParents; i++)
    {
      vtkIdType Edge = Structure->GetInEdge(Node,i).Id;
      vtkIdType Parent = Structure->GetParent(Node,i);

      vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node,Parent,Scheduler, -2-NumKids, 2+NumKids, this->NumberOfIterations-1, vtkCudaMaxFlowSegmentationTask::PushUpSourceFlows);
      PushUpSourceFlowsTasks[Edge] = newTask;

      float W = Weights ? Weights->GetValue(Edge) : 1.0 / (float) Structure->GetNumberOfParents(Node);
      if(Parent == Structure->GetRoot())
      {
        newTask->AddBuffer(sourceFlowBuffer);
        W = W / this->SourceWeightedNumChildren ;
      }
      else
      {
        newTask->AddBuffer( branchSinkBuffers[BranchMap[Parent]]);
        W = W / (this->BranchWeightedNumChildren[BranchMap[Parent]]+1) ;
      }
      newTask->SetConstant1(W);
      newTask->AddBuffer( this->branchSinkBuffers[BranchMap[Node]]);
      newTask->AddBuffer( this->branchSourceBuffers[BranchMap[Node]]);
      newTask->AddBuffer( this->branchDivBuffers[BranchMap[Node]]);
      newTask->AddBuffer( this->branchLabelBuffers[BranchMap[Node]]);
    }

  }
  ForIterator->Delete();
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreatePushDownSinkFlowsRootTasks()
{
  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));
  vtkIdType Node = Structure->GetRoot();
  int NumKids = Structure->GetNumberOfChildren(Node);

  for(int i = 0; i < NumKids; i++)
  {
    vtkIdType Edge = Structure->GetOutEdge(Node,i).Id;
    vtkIdType Child = Structure->GetChild(Node,i);
    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node,Child,Scheduler, -2-NumKids, 2+NumKids, this->NumberOfIterations-1,vtkCudaMaxFlowSegmentationTask::PushDownSinkFlows);
    PushDownSinkFlowsTasks[Edge] = newTask;
    float W = Weights ? Weights->GetValue(Edge) : 1.0/(double)Structure->GetNumberOfParents(Child);
    newTask->SetConstant1( W );
    newTask->AddBuffer(sourceFlowBuffer);
    if(Structure->IsLeaf(Child))
    {
      newTask->AddBuffer(leafSourceBuffers[LeafMap[Child]]);
    }
    else
    {
      newTask->AddBuffer(branchSourceBuffers[BranchMap[Child]]);
    }
  }
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreatePushDownSinkFlowsBranchTasks()
{
  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType Node = ForIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);
    if( NumKids == 0 || Node == Structure->GetRoot() )
    {
      continue;
    }

    for(int i = 0; i < NumKids; i++)
    {
      vtkIdType Edge = Structure->GetOutEdge(Node,i).Id;
      vtkIdType Child = Structure->GetChild(Node,i);
      vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node,Child,Scheduler, -2-NumKids, 2+NumKids, this->NumberOfIterations-1,vtkCudaMaxFlowSegmentationTask::PushDownSinkFlows);
      PushDownSinkFlowsTasks[Edge] = newTask;
      float W = Weights ? Weights->GetValue(Edge) : 1.0/(double)Structure->GetNumberOfParents(Child);
      newTask->SetConstant1( W );
      newTask->AddBuffer(branchSinkBuffers[BranchMap[Node]]);
      if(Structure->IsLeaf(Child))
      {
        newTask->AddBuffer(leafSourceBuffers[LeafMap[Child]]);
      }
      else
      {
        newTask->AddBuffer(branchSourceBuffers[BranchMap[Child]]);
      }
    }

  }
  ForIterator->Delete();

}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateUpdateLabelsTasks()
{

  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType Node = ForIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    if( Node == Structure->GetRoot() )
    {
      continue;
    }

    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, -1-NumKids, 1+NumKids, this->NumberOfIterations - ( NumKids ? 1 : 0 ),vtkCudaMaxFlowSegmentationTask::UpdateLabelsTask);
    UpdateLabelsTasks[Node] = newTask;
    if(Structure->IsLeaf(Node))
    {
      newTask->AddBuffer(leafSinkBuffers[LeafMap[Node]]);
      newTask->AddBuffer(leafSourceBuffers[LeafMap[Node]]);
      newTask->AddBuffer(leafDivBuffers[LeafMap[Node]]);
      newTask->AddBuffer(leafLabelBuffers[LeafMap[Node]]);
    }
    else
    {
      newTask->AddBuffer(branchSinkBuffers[BranchMap[Node]]);
      newTask->AddBuffer(branchSourceBuffers[BranchMap[Node]]);
      newTask->AddBuffer(branchDivBuffers[BranchMap[Node]]);
      newTask->AddBuffer(branchLabelBuffers[BranchMap[Node]]);
    }
  }
}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::CreateClearSourceBufferTasks()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType Node = ForIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);
    if( Node == Structure->GetRoot() )
    {
      continue;
    }

    int NumRequired = (NumKids) ? 1 + NumParents + NumKids : NumParents;

    vtkCudaMaxFlowSegmentationTask* newTask = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, -NumRequired, NumRequired, this->NumberOfIterations-1,vtkCudaMaxFlowSegmentationTask::ClearSourceBuffer);
    ClearSourceBufferTasks[Node] = newTask;
    if(Structure->IsLeaf(Node))
    {
      newTask->AddBuffer(leafSourceBuffers[LeafMap[Node]]);
    }
    else
    {
      newTask->AddBuffer(branchSourceBuffers[BranchMap[Node]]);
    }
  }
}

//Index
// (1) Update spatial flows
// (2) Reset sink flows
// (3) Push up source flows
// (4) Push down sink flows
// (5) Update labels
// (6) Clear source flows

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::AssociateFinishSignals()
{
  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType Node = ForIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    int NumParents = Structure->GetNumberOfParents(Node);

    bool isLeaf = Structure->IsLeaf(Node);
    bool isRoot = (Structure->GetRoot() == Node);
    bool isBranch = !isLeaf && !isRoot;

    //link (1) to (2) in B and L
    if( !isRoot )
      UpdateSpatialFlowsTasks[Node]->AddTaskToSignal( isLeaf ?
          ApplySinkPotentialLeafTasks[Node] : ResetSinkFlowTasks[Node] );

    //link (2) to (5) in L and B
    if( isLeaf )
    {
      ApplySinkPotentialLeafTasks[Node]->AddTaskToSignal(UpdateLabelsTasks[Node]);
    }
    if( isBranch )
    {
      ResetSinkFlowTasks[Node]->AddTaskToSignal(UpdateLabelsTasks[Node]);
    }

    //link (2) to Child(3) for B and R
    if( !isLeaf )
      for(int i = 0; i < NumKids; i++)
      {
        ResetSinkFlowTasks[Node]->AddTaskToSignal(PushUpSourceFlowsTasks[Structure->GetOutEdge(Node,i).Id]);
      }

    //link (2) to (3) for B
    if( isBranch )
      for( int i = 0; i < NumParents; i++ )
      {
        ResetSinkFlowTasks[Node]->AddTaskToSignal(PushUpSourceFlowsTasks[Structure->GetInEdge(Node,i).Id]);
      }

    //link (2) to (4) for B and R
    if( !isLeaf )
      for( int i = 0; i < NumKids; i++ )
      {
        ResetSinkFlowTasks[Node]->AddTaskToSignal(PushDownSinkFlowsTasks[Structure->GetOutEdge(Node,i).Id]);
      }

    //link (3) to (6) for L and B
    if( !isRoot )
      for( int i = 0; i < NumParents; i++ )
      {
        PushUpSourceFlowsTasks[Structure->GetInEdge(Node,i).Id]->AddTaskToSignal(ClearSourceBufferTasks[Node]);
      }

    //link (3) to Parent(3)(4)(5) for L and B
    if( !isRoot )
      for( int i = 0; i < NumParents; i++ )
      {
        vtkIdType Parent = Structure->GetParent(Node,i);
        if(Parent != Structure->GetRoot())
        {
          PushUpSourceFlowsTasks[Structure->GetInEdge(Node,i).Id]->AddTaskToSignal(UpdateLabelsTasks[Parent]);
        }
        for(int j = 0; j < Structure->GetNumberOfParents(Parent); j++)
          if( Parent != Structure->GetRoot() )
            PushUpSourceFlowsTasks[Structure->GetInEdge(Node,i).Id]->AddTaskToSignal(
              PushUpSourceFlowsTasks[Structure->GetInEdge(Parent,j).Id]);
        for(int j = 0; j < Structure->GetNumberOfChildren(Parent); j++)
          PushUpSourceFlowsTasks[Structure->GetInEdge(Node,i).Id]->AddTaskToSignal(
            PushDownSinkFlowsTasks[Structure->GetOutEdge(Parent,j).Id]);
      }

    //link (4) to (2) for R
    if( isRoot )
      for( int i = 0; i < NumKids; i++ )
      {
        PushDownSinkFlowsTasks[Structure->GetOutEdge(Node,i).Id]->AddTaskToSignal(ResetSinkFlowTasks[Node]);
      }

    //link (4) to (6) for B
    if( isBranch )
      for( int i = 0; i < NumKids; i++ )
      {
        PushDownSinkFlowsTasks[Structure->GetOutEdge(Node,i).Id]->AddTaskToSignal(ClearSourceBufferTasks[Node]);
      }

    //link (4) to Child(1) for B and R
    if( !isLeaf )
      for( int i = 0; i < NumKids; i++ )
      {
        PushDownSinkFlowsTasks[Structure->GetOutEdge(Node,i).Id]->AddTaskToSignal(UpdateSpatialFlowsTasks[Structure->GetChild(Node,i)]);
      }

    //link (5) to (3) for L
    if(isLeaf)
      for( int i = 0; i < NumParents; i++ )
      {
        UpdateLabelsTasks[Node]->AddTaskToSignal(PushUpSourceFlowsTasks[Structure->GetInEdge(Node,i).Id]);
      }

    //link (5) to (6) for B
    if(isBranch)
    {
      UpdateLabelsTasks[Node]->AddTaskToSignal(ClearSourceBufferTasks[Node]);
    }

    //link (6) to (1) for B and L
    if(!isRoot)
    {
      ClearSourceBufferTasks[Node]->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
    }

    //link (6) to Parent(4) for B and L
    if(!isRoot)
      for( int i = 0; i < NumParents; i++ )
      {
        ClearSourceBufferTasks[Node]->AddTaskToSignal(PushDownSinkFlowsTasks[Structure->GetInEdge(Node,i).Id]);
      }

  }

}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::InitializeSpatialFlowsTasks()
{

  vtkRootedDirectedAcyclicGraphForwardIterator* ForIterator = vtkRootedDirectedAcyclicGraphForwardIterator::New();
  ForIterator->SetDAG(this->Structure);
  while(ForIterator->HasNext())
  {
    vtkIdType Node = ForIterator->Next();
    if( Node == this->Structure->GetRoot() )
    {
      continue;
    }

    //create the new task
    //initial Active is -7 (4 clear buffers, 2 set source/sink, 1 set label)
    vtkCudaMaxFlowSegmentationTask* newTask1 = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, 0, 1, 1,vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask2 = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, 0, 1, 1,vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask3 = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, 0, 1, 1,vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    vtkCudaMaxFlowSegmentationTask* newTask4 = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, 0, 1, 1,vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    if(Structure->IsLeaf(Node))
    {
      newTask1->AddBuffer(leafDivBuffers[LeafMap[Node]]);
      newTask2->AddBuffer(leafFlowXBuffers[LeafMap[Node]]);
      newTask3->AddBuffer(leafFlowYBuffers[LeafMap[Node]]);
      newTask4->AddBuffer(leafFlowZBuffers[LeafMap[Node]]);
    }
    else
    {
      newTask1->AddBuffer(branchDivBuffers[BranchMap[Node]]);
      newTask2->AddBuffer(branchFlowXBuffers[BranchMap[Node]]);
      newTask3->AddBuffer(branchFlowYBuffers[BranchMap[Node]]);
      newTask4->AddBuffer(branchFlowZBuffers[BranchMap[Node]]);
    }
    newTask1->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
    newTask2->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
    newTask3->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
    newTask4->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
  }
  ForIterator->Delete();

}

void vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::InitializeSinkFlowsTasks()
{
  vtkFloatArray* Weights = vtkFloatArray::SafeDownCast(this->Structure->GetEdgeData()->GetArray("Weights"));

  //find minimum sink flow
  vtkCudaMaxFlowSegmentationTask* InitialCopy = new vtkCudaMaxFlowSegmentationTask(0,0,Scheduler, 0, 1, 1,vtkCudaMaxFlowSegmentationTask::InitializeLeafFlows);
  InitialCopy->AddBuffer(sourceFlowBuffer);
  InitialCopy->AddBuffer(leafDataTermBuffers[0]);
  vtkCudaMaxFlowSegmentationTask** FindMin = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for(int i = 1; i < this->NumLeaves; i++)
  {
    FindMin[i] = new vtkCudaMaxFlowSegmentationTask(i,i,Scheduler, -1, 1, 1,vtkCudaMaxFlowSegmentationTask::MinimizeLeafFlows);
    FindMin[i]->AddBuffer(sourceFlowBuffer);
    FindMin[i]->AddBuffer(leafDataTermBuffers[i]);
    InitialCopy->AddTaskToSignal(FindMin[i]);
  }

  //apply min to all leaves
  vtkCudaMaxFlowSegmentationTask** Propogate = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for(int i = 0; i < this->NumLeaves; i++)
  {
    Propogate[i] = new vtkCudaMaxFlowSegmentationTask(i,i,Scheduler, -NumLeaves+1, 1, 1,vtkCudaMaxFlowSegmentationTask::PropogateLeafFlowsInc);
    Propogate[i]->AddBuffer(sourceFlowBuffer);
    Propogate[i]->AddBuffer(leafSinkBuffers[i]);
    Propogate[i]->AddBuffer(leafSourceBuffers[i]);
    for(int j = 1; j < NumLeaves; j++)
    {
      FindMin[j]->AddTaskToSignal(Propogate[i]);
    }
  }

  //find a=0 labeling (not normalized to be valid)
  vtkCudaMaxFlowSegmentationTask** Indicate = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for(int i = 0; i < this->NumLeaves; i++)
  {
    Indicate[i] = new vtkCudaMaxFlowSegmentationTask(i,i,Scheduler, -1, 1, 1,vtkCudaMaxFlowSegmentationTask::InitializeLeafLabels);
    Indicate[i]->AddBuffer(leafSinkBuffers[i]);
    Indicate[i]->AddBuffer(leafDataTermBuffers[i]);
    Indicate[i]->AddBuffer(leafLabelBuffers[i]);
    Indicate[i]->AddBuffer(leafLabelBuffers[i]);
    Indicate[i]->AddTaskToSignal(ResetSinkFlowTasks[Structure->GetRoot()]);
    Propogate[i]->AddTaskToSignal(Indicate[i]);
  }

  //accumulate labels
  vtkCudaMaxFlowSegmentationTask* ClearAccumulator = new vtkCudaMaxFlowSegmentationTask(0,0,Scheduler, 0, 1, 1,vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
  ClearAccumulator->AddBuffer(sourceWorkingBuffer);
  vtkCudaMaxFlowSegmentationTask** Accumulate = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for(int i = 0; i < this->NumLeaves; i++)
  {
    Accumulate[i] = new vtkCudaMaxFlowSegmentationTask(i,i,Scheduler, -2, 1, 1,vtkCudaMaxFlowSegmentationTask::AccumulateLabels);
    Accumulate[i]->AddBuffer(sourceWorkingBuffer);
    Accumulate[i]->AddBuffer(leafLabelBuffers[i]);
    ClearAccumulator->AddTaskToSignal(Accumulate[i]);
    Indicate[i]->AddTaskToSignal(Accumulate[i]);
  }

  //validate a=0 labeling
  vtkCudaMaxFlowSegmentationTask** Divide = new vtkCudaMaxFlowSegmentationTask* [NumLeaves];
  for(int i = 0; i < this->NumLeaves; i++)
  {
    Divide[i] = new vtkCudaMaxFlowSegmentationTask(i,i,Scheduler, -NumLeaves, 1, 1,vtkCudaMaxFlowSegmentationTask::CorrectLabels);
    Divide[i]->AddBuffer(sourceWorkingBuffer);
    Divide[i]->AddBuffer(leafLabelBuffers[i]);
    for(int j = 0; j < NumLeaves; j++)
    {
      Accumulate[j]->AddTaskToSignal(Divide[i]);
    }
  }
  vtkRootedDirectedAcyclicGraphBackwardIterator* BackIterator = vtkRootedDirectedAcyclicGraphBackwardIterator::New();
  BackIterator->SetDAG(this->Structure);
  while(BackIterator->HasNext())
  {
    vtkIdType Node = BackIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    if( NumKids != 0 )
    {
      continue;
    }
    Divide[LeafMap[Node]]->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
  }

  //propogate sink flows upwards
  vtkCudaMaxFlowSegmentationTask** PropogateB = new vtkCudaMaxFlowSegmentationTask* [NumBranches];
  BackIterator->Restart();
  while(BackIterator->HasNext())
  {
    vtkIdType Node = BackIterator->Next();
    int NumKids = Structure->GetNumberOfChildren(Node);
    if(NumKids == 0 || Node == this->Structure->GetRoot())
    {
      continue;
    }
    PropogateB[BranchMap[Node]] = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, -NumLeaves+1, 1, 1,vtkCudaMaxFlowSegmentationTask::PropogateLeafFlowsInc);
    PropogateB[BranchMap[Node]]->AddBuffer(sourceFlowBuffer);
    PropogateB[BranchMap[Node]]->AddBuffer(branchSinkBuffers[BranchMap[Node]]);
    PropogateB[BranchMap[Node]]->AddBuffer(branchSourceBuffers[BranchMap[Node]]);
    PropogateB[BranchMap[Node]]->AddTaskToSignal(ResetSinkFlowTasks[Structure->GetRoot()]);
    for(int j = 1; j < NumLeaves; j++)
    {
      FindMin[j]->AddTaskToSignal(PropogateB[BranchMap[Node]]);
    }
  }

  //propagate labels up
  vtkCudaMaxFlowSegmentationTask** AccumLabels = new vtkCudaMaxFlowSegmentationTask* [NumEdges];
  BackIterator->Restart();
  while(BackIterator->HasNext())
  {
    vtkIdType Node = BackIterator->Next();
    if (Node == Structure->GetRoot() )
    {
      continue;
    }
    if ( Structure->IsLeaf(Node) )
    {
      continue;
    }

    //clear buffer
    int NumKids = Structure->GetNumberOfChildren(Node);
    vtkCudaMaxFlowSegmentationTask* clear = new vtkCudaMaxFlowSegmentationTask(Node,Node,Scheduler, -NumKids, NumKids, 1,vtkCudaMaxFlowSegmentationTask::ClearBufferInitially);
    clear->AddBuffer(branchLabelBuffers[BranchMap[Node]]);
    for(int i = 0; i < NumKids; i++)
    {
      vtkIdType Child = Structure->GetChild(Node,i);
      if( Structure->IsLeaf(Child) )
      {
        Divide[LeafMap[Child]]->AddTaskToSignal(clear);
      }
      else
        for(int j = 0; j < Structure->GetNumberOfChildren(Child); j++)
        {
          AccumLabels[Structure->GetOutEdge(Child,j).Id]->AddTaskToSignal(clear);
        }
    }

    //create accumulation tasks
    for(int i = 0; i < NumKids; i++)
    {
      vtkIdType Child = Structure->GetChild(Node,i);
      AccumLabels[Structure->GetOutEdge(Node,i).Id] = new vtkCudaMaxFlowSegmentationTask(Node,Child,Scheduler, -1, 1, 1,vtkCudaMaxFlowSegmentationTask::AccumulateLabelsWeighted);
      AccumLabels[Structure->GetOutEdge(Node,i).Id]->SetConstant1(Weights ? Weights->GetValue(Structure->GetOutEdge(Node,i).Id) : 1.0 / (float)Structure->GetNumberOfParents(Child));
      AccumLabels[Structure->GetOutEdge(Node,i).Id]->AddBuffer(branchLabelBuffers[BranchMap[Node]]);
      AccumLabels[Structure->GetOutEdge(Node,i).Id]->AddBuffer( Structure->IsLeaf(Child) ?
          leafLabelBuffers[LeafMap[Child]] : branchLabelBuffers[BranchMap[Child]] );
      clear->AddTaskToSignal(AccumLabels[Structure->GetOutEdge(Node,i).Id]);
      AccumLabels[Structure->GetOutEdge(Node,i).Id]->AddTaskToSignal(UpdateSpatialFlowsTasks[Node]);
      AccumLabels[Structure->GetOutEdge(Node,i).Id]->AddTaskToSignal(UpdateSpatialFlowsTasks[Child]);
    }

  }
  BackIterator->Delete();

  delete[] FindMin;
  delete[] Indicate;
  delete[] Accumulate;
  delete[] Divide;
  delete[] Propogate;
  delete[] PropogateB;
  delete[] AccumLabels;


}