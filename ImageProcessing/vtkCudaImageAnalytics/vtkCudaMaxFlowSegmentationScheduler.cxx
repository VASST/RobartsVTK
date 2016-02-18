#include "vtkCudaMaxFlowSegmentationScheduler.h"
#include "vtkCudaMaxFlowSegmentationTask.h"
#include "vtkCudaMaxFlowSegmentationWorker.h"

vtkCudaMaxFlowSegmentationScheduler::vtkCudaMaxFlowSegmentationScheduler()
{
  Clear();
}

vtkCudaMaxFlowSegmentationScheduler::~vtkCudaMaxFlowSegmentationScheduler()
{
  Clear();
}

void vtkCudaMaxFlowSegmentationScheduler::Clear()
{
  //clear variables
  NumTasksGoingToHappen = 0;
  NumMemCpies = 0;
  NumKernelRuns = 0;

  //clear old lists
  this->CurrentTasks.clear();
  this->BlockedTasks.clear();
  this->CPUInUse.clear();
  this->CPU2PriorityMap.clear();
  this->ReadOnly.clear();
  this->NoCopyBack.clear();
  this->LastBufferUse.clear();
  this->Overwritten.clear();

  //clear workers
  for(std::set<vtkCudaMaxFlowSegmentationTask*>::iterator taskIterator = FinishedTasks.begin(); taskIterator != FinishedTasks.end(); taskIterator++)
  {
    delete *taskIterator;
  }

  //cleat tasks
  FinishedTasks.clear();
  for(std::set<vtkCudaMaxFlowSegmentationWorker*>::iterator workerIterator = Workers.begin(); workerIterator != Workers.end(); workerIterator++)
  {
    delete *workerIterator;
  }
  Workers.clear();

}

int vtkCudaMaxFlowSegmentationScheduler::CreateWorker(int GPU, double usage)
{
  vtkCudaMaxFlowSegmentationWorker* newWorker = new vtkCudaMaxFlowSegmentationWorker( GPU, usage, this );
  this->Workers.insert( newWorker );
  if(newWorker->NumBuffers < 8)
  {
    return -1;
  }
  return 0;
}

void vtkCudaMaxFlowSegmentationScheduler::ReturnLeaves()
{
  SyncWorkers();
  for(std::set<vtkCudaMaxFlowSegmentationWorker*>::iterator workerIterator = Workers.begin(); workerIterator != Workers.end(); workerIterator++)
  {
    (*workerIterator)->ReturnLeafLabels();
  }
}

void vtkCudaMaxFlowSegmentationScheduler::ReturnBufferGPU2CPU(vtkCudaMaxFlowSegmentationWorker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream)
{
  if( !CPUBuffer )
  {
    return;
  }
  if( ReadOnly.find(CPUBuffer) != ReadOnly.end() )
  {
    return;
  }
  if( Overwritten[CPUBuffer] == 0 )
  {
    return;
  }
  Overwritten[CPUBuffer] = 0;
  caller->ReserveGPU();
  LastBufferUse[CPUBuffer] = caller;
  if( NoCopyBack.find(CPUBuffer) != NoCopyBack.end() )
  {
    return;
  }
  CUDA_CopyBufferToCPU( GPUBuffer, CPUBuffer, VolumeSize, stream);
  NumMemCpies++;
}

void vtkCudaMaxFlowSegmentationScheduler::MoveBufferCPU2GPU(vtkCudaMaxFlowSegmentationWorker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream)
{
  if( !CPUBuffer )
  {
    return;
  }
  caller->ReserveGPU();
  if( LastBufferUse[CPUBuffer] )
  {
    LastBufferUse[CPUBuffer]->CallSyncThreads();
  }
  LastBufferUse[CPUBuffer] = 0;
  if( NoCopyBack.find(CPUBuffer) != NoCopyBack.end() )
  {
    return;
  }
  CUDA_CopyBufferToGPU( GPUBuffer, CPUBuffer, VolumeSize, stream);
  NumMemCpies++;
}

void vtkCudaMaxFlowSegmentationScheduler::SyncWorkers()
{
  for(std::set<vtkCudaMaxFlowSegmentationWorker*>::iterator workerIt = Workers.begin(); workerIt != Workers.end(); workerIt++)
  {
    (*workerIt)->CallSyncThreads();
  }
}

bool vtkCudaMaxFlowSegmentationScheduler::CanRunAlgorithmIteration()
{
  return (this->CurrentTasks.size() > 0) ;
}

int vtkCudaMaxFlowSegmentationScheduler::RunAlgorithmIteration()
{

  int MinWeight = INT_MAX;
  int MinUnConflictWeight = INT_MAX;
  std::vector<vtkCudaMaxFlowSegmentationTask*> MinTasks;
  std::vector<vtkCudaMaxFlowSegmentationTask*> MinUnConflictTasks;
  std::vector<vtkCudaMaxFlowSegmentationWorker*> MinWorkers;
  std::vector<vtkCudaMaxFlowSegmentationWorker*> MinUnConflictWorkers;
  for(std::set<vtkCudaMaxFlowSegmentationTask*>::iterator taskIt = CurrentTasks.begin(); MinWeight > 0 && taskIt != CurrentTasks.end(); taskIt++)
  {
    if( !(*taskIt)->CanDo() )
    {
      continue;
    }

    //find if the task is conflicted and put in appropriate contest
    vtkCudaMaxFlowSegmentationWorker* possibleWorker = 0;
    int conflictWeight = (*taskIt)->Conflicted(&possibleWorker);
    if( conflictWeight )
    {
      if( conflictWeight < MinUnConflictWeight )
      {
        MinUnConflictWeight = conflictWeight;
        MinUnConflictTasks.clear();
        MinUnConflictTasks.push_back(*taskIt);
        MinUnConflictWorkers.clear();
        MinUnConflictWorkers.push_back(possibleWorker);
      }
      else if(conflictWeight == MinUnConflictWeight)
      {
        MinUnConflictTasks.push_back(*taskIt);
        MinUnConflictWorkers.push_back(possibleWorker);
      }
      continue;
    }

    if( possibleWorker )  //only one worker can do this task
    {
      int weight = (*taskIt)->CalcWeight(possibleWorker);
      if( weight < MinWeight )
      {
        MinWeight = weight;
        MinTasks.clear();
        MinTasks.push_back(*taskIt);
        MinWorkers.clear();
        MinWorkers.push_back(possibleWorker);
      }
      else if( weight == MinWeight )
      {
        MinTasks.push_back(*taskIt);
        MinWorkers.push_back(possibleWorker);
      }
    }
    else   //all workers have a chance, find the emptiest one
    {
      for(std::set<vtkCudaMaxFlowSegmentationWorker*>::iterator workerIt = Workers.begin(); workerIt != Workers.end(); workerIt++)
      {
        int weight = (*taskIt)->CalcWeight(*workerIt);
        if( weight < MinWeight )
        {
          MinWeight = weight;
          MinTasks.clear();
          MinTasks.push_back(*taskIt);
          MinWorkers.clear();
          MinWorkers.push_back(*workerIt);
        }
        else if( weight == MinWeight )
        {
          MinTasks.push_back(*taskIt);
          MinWorkers.push_back(*workerIt);
        }
      }
    }
  }

  //figure out if it is cheaper to run a conflicted or non-conflicted task
  if( MinUnConflictWeight >= MinWeight )
  {
    int taskIdx = rand() % MinTasks.size();
    MinTasks[taskIdx]->Perform(MinWorkers[taskIdx]);
  }
  else
  {
    int taskIdx = rand() % MinUnConflictTasks.size();
    MinUnConflictTasks[taskIdx]->UnConflict(MinUnConflictWorkers[taskIdx]);
    MinUnConflictTasks[taskIdx]->Perform(MinUnConflictWorkers[taskIdx]);
  }

  return 0;
}