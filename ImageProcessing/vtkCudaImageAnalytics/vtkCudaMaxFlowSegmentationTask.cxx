/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkHierarchicalMaxFlowSegmentation2Task.cxx

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkHierarchicalMaxFlowSegmentation2Task.cxx
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

#include "vtkCudaMaxFlowSegmentationTask.h"

#include "CUDA_hierarchicalmaxflow.h"
#include "vtkCudaDeviceManager.h"
#include "CudaObject.h"

#define SQR(X) X*X

//------------------------------------------------------------------------------------------//

//Fill in all non-transient information
vtkCudaMaxFlowSegmentationTask::vtkCudaMaxFlowSegmentationTask( vtkIdType n1, vtkIdType n2, vtkCudaMaxFlowSegmentationScheduler* parent, int a, int ra, int numToDeath, TaskType t )
  : Parent(parent), Active(a), FinishDecreaseInActive(ra), Type(t), Node1(n1), Node2(n2) {
  NumToDeath = numToDeath;
  NumTimesCalled = 0;
    
  if( NumToDeath <= 0 ){
    Parent->FinishedTasks.insert(this);
    return;
  }

  Parent->NumTasksGoingToHappen += numToDeath;
  if(Active >= 0) Parent->CurrentTasks.insert(this);
  else  Parent->BlockedTasks.insert(this);

}

vtkCudaMaxFlowSegmentationTask::~vtkCudaMaxFlowSegmentationTask(){
  this->FinishedSignals.clear();
  this->RequiredCPUBuffers.clear();
}
  
void vtkCudaMaxFlowSegmentationTask::SetConstant1(float f){
  constant1 = f;
}
  
void vtkCudaMaxFlowSegmentationTask::SetConstant2(float f){
  constant2 = f;
}
//------------------------------------------------------------------------------------------//

void vtkCudaMaxFlowSegmentationTask::Signal(){
  this->Active++;
  if( this->Active == 0 ){

    if( Parent->BlockedTasks.find(this) != Parent->BlockedTasks.end() ){
      Parent->BlockedTasks.erase(this);
      Parent->CurrentTasks.insert(this);
    }

    if( Type == ClearWorkingBufferTask )
      Parent->NoCopyBack.insert( RequiredCPUBuffers[0] );
    //else if( Type == DivideOutWorkingBufferTask )
      //Parent->NoCopyBack.insert( RequiredCPUBuffers[1] );
    //else if( Type == ApplySinkPotentialLeafTask )
      //Parent->NoCopyBack.insert( RequiredCPUBuffers[0] );
  }
}

//------------------------------------------------------------------------------------------//

//manage signals for when this task is finished
//ie: allow us to signal the next task in the loop as well as
//    any parent or child node tasks as necessary
void vtkCudaMaxFlowSegmentationTask::AddTaskToSignal(vtkCudaMaxFlowSegmentationTask* t){
  FinishedSignals.push_back(t);
}
void vtkCudaMaxFlowSegmentationTask::FinishedSignal(){
  std::vector<vtkCudaMaxFlowSegmentationTask*>::iterator it = FinishedSignals.begin();
  for(; it != FinishedSignals.end(); it++ )
    if(*it) (*it)->Signal();
}
void vtkCudaMaxFlowSegmentationTask::DecrementActivity(){
  Active--;
}
//------------------------------------------------------------------------------------------//

void vtkCudaMaxFlowSegmentationTask::AddBuffer(float* b){
  RequiredCPUBuffers.push_back(b);
}

//------------------------------------------------------------------------------------------//

//Find out if we have a conflict on this task (ie: not all buffers are available on CPU or single GPU)
//returning an unconflicted device if false (null if conflict or any worker will suffice)
int vtkCudaMaxFlowSegmentationTask::Conflicted(vtkCudaMaxFlowSegmentationWorker** w){
    
  //find the GPU with most of the buffers
  int retVal = 0;
  int maxBuffersGot = 0;
  vtkCudaMaxFlowSegmentationWorker* maxGPU = 0;
  for(std::set<vtkCudaMaxFlowSegmentationWorker*>::iterator wit = Parent->Workers.begin(); wit != Parent->Workers.end(); wit++){
    int buffersGot = 0;
    for(std::vector<float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++){
      if( (*wit)->CPU2GPUMap.find(*it) != (*wit)->CPU2GPUMap.end() ) buffersGot++;
    }
    if( buffersGot > maxBuffersGot ){
      maxBuffersGot = buffersGot;
      maxGPU = *wit;
    }
  }

  //return everything that is not on that GPU
  for(std::set<vtkCudaMaxFlowSegmentationWorker*>::iterator wit = Parent->Workers.begin(); wit != Parent->Workers.end(); wit++){
    if( *wit == maxGPU ) continue;
    for(std::vector<float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++){
      if( (*wit)->CPU2GPUMap.find(*it) != (*wit)->CPU2GPUMap.end() ){
        if( Parent->ReadOnly.find(*it) != Parent->ReadOnly.end() ||
          Parent->NoCopyBack.find(*it) != Parent->NoCopyBack.end() )
          retVal += 1;
        else retVal += 3;
      }
    }
  }

  *w = maxGPU;
  return retVal;
}
  
//------------------------------------------------------------------------------------------//

//find the GPU with most of the buffers and return all claimed buffers not on that GPU
void vtkCudaMaxFlowSegmentationTask::UnConflict(vtkCudaMaxFlowSegmentationWorker* maxGPU){

  //return everything that is not on that GPU
  for(std::set<vtkCudaMaxFlowSegmentationWorker*>::iterator wit = Parent->Workers.begin(); wit != Parent->Workers.end(); wit++){
    if( *wit == maxGPU ) continue;
    bool flag = false;
    for(std::vector<float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++){
      if( (*wit)->CPU2GPUMap.find(*it) != (*wit)->CPU2GPUMap.end() ){
        (*wit)->ReturnBuffer(*it);
        flag = true;
      }
    }
    if( flag ) (*wit)->CallSyncThreads();
  }
  maxGPU->CallSyncThreads();

}
  
//------------------------------------------------------------------------------------------//

//Calculate the weight provided that there is no conflict
int vtkCudaMaxFlowSegmentationTask::CalcWeight(vtkCudaMaxFlowSegmentationWorker* w){
  int retWeight = 0;
  int numUnused = (int) w->UnusedGPUBuffers.size();
  for(std::vector<float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++){
    if( w->CPU2GPUMap.find(*it) == w->CPU2GPUMap.end() ){
      if( numUnused ){numUnused--; retWeight++;
      }else retWeight += 2;
    }
  }
  return retWeight;
}
  
//------------------------------------------------------------------------------------------//

//Perform the task at hand
void vtkCudaMaxFlowSegmentationTask::Perform(vtkCudaMaxFlowSegmentationWorker* w){
  if( !CanDo() ) return;
  w->ReserveGPU();

  //load anything that will be overwritten onto the no copy back list
  switch(Type){
  case(ClearWorkingBufferTask):      //0 - Working
  case(ApplySinkPotentialLeafTask):    //0 - Sink,    1 - Inc,  2 - Div,  3 - Label,  4 - Data
  case(ClearBufferInitially):        //0 - Any
  case(ClearSourceBuffer):        //0 - Any
  case(InitializeLeafFlows):        //0 - Sink,    1 - Data
    Parent->NoCopyBack.insert(RequiredCPUBuffers[0]);
    break;
  case(DivideOutWorkingBufferTask):    //0 - Working,  1 - Sink
  case(PropogateLeafFlows):        //0 - SinkMin,  1 - SinkElse
    Parent->NoCopyBack.insert(RequiredCPUBuffers[1]);
    break;
  case(InitializeLeafLabels):        //0 - Sink,    1 - Data,  2 - Label
    Parent->NoCopyBack.insert(RequiredCPUBuffers[2]);
    break;
  case(PropogateLeafFlowsInc):
    Parent->NoCopyBack.insert(RequiredCPUBuffers[1]);
    Parent->NoCopyBack.insert(RequiredCPUBuffers[2]);
    break;
  }

  //load required buffers onto the GPU
  w->CPUInUse.clear();
  for(std::vector<float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++)
    if(*it) w->CPUInUse.insert(*it);
  w->UpdateBuffersInUse();
    
    for(std::vector<float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++)
        if(w->CPU2GPUMap.find(*it) == w->CPU2GPUMap.end()){
            std::cout << "Problem: " << *it << std::endl;
        }


    assert(w->CPU2GPUMap.size() == w->GPU2CPUMap.size());

  //run the kernels
  float smoothnessConstant = this->constant1;
  switch(Type){
  case(ClearWorkingBufferTask):      //0 - Working
    //std::cout << Node1 << "\t" << Node2 << "\t" << "ClearWorkingBufferTask" << std::endl;
    if( !isRoot ) CUDA_zeroOutBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], Parent->VolumeSize, w->GetStream());
    else CUDA_SetBufferToValue(w->CPU2GPUMap[RequiredCPUBuffers[0]], 1.0f/Parent->CC, Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;
      
  case(UpdateSpatialFlowsTask):      //0 - Sink,    1 - Inc,  2 - Div,  3 - Label,  4 - FlowX,  5 - FlowY,  6 - FlowZ,  7 - Smoothness
    //std::cout << Node1 << "\t" << Node2 << "\t" << "UpdateSpatialFlowsTask" << std::endl;
    CUDA_flowGradientStep(w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]],
      w->CPU2GPUMap[RequiredCPUBuffers[2]], w->CPU2GPUMap[RequiredCPUBuffers[3]], Parent->StepSize, Parent->CC,
      Parent->VolumeSize, w->GetStream() );
    CUDA_applyStep(w->CPU2GPUMap[RequiredCPUBuffers[2]], w->CPU2GPUMap[RequiredCPUBuffers[4]],
      w->CPU2GPUMap[RequiredCPUBuffers[5]], w->CPU2GPUMap[RequiredCPUBuffers[6]], Parent->VX, Parent->VY, Parent->VZ, Parent->VolumeSize, w->GetStream() );
    CUDA_computeFlowMag(w->CPU2GPUMap[RequiredCPUBuffers[2]], w->CPU2GPUMap[RequiredCPUBuffers[4]],
      w->CPU2GPUMap[RequiredCPUBuffers[5]], w->CPU2GPUMap[RequiredCPUBuffers[6]],
      w->CPU2GPUMap[RequiredCPUBuffers[7]], smoothnessConstant, Parent->VX, Parent->VY, Parent->VZ, Parent->VolumeSize, w->GetStream() );
    CUDA_projectOntoSet(w->CPU2GPUMap[RequiredCPUBuffers[2]], w->CPU2GPUMap[RequiredCPUBuffers[4]],
      w->CPU2GPUMap[RequiredCPUBuffers[5]], w->CPU2GPUMap[RequiredCPUBuffers[6]], Parent->VX, Parent->VY, Parent->VZ, Parent->VolumeSize, w->GetStream() );
    Parent->Overwritten[RequiredCPUBuffers[2]] = 1;
    Parent->Overwritten[RequiredCPUBuffers[4]] = 1;
    Parent->Overwritten[RequiredCPUBuffers[5]] = 1;
    Parent->Overwritten[RequiredCPUBuffers[6]] = 1;
    Parent->NumKernelRuns += 4;
    break;

  case(ApplySinkPotentialLeafTask):    //0 - Sink,    1 - Inc,  2 - Div,  3 - Label,  4 - Data
    //std::cout << Node1 << "\t" << Node2 << "\t" << "ApplySinkPotentialLeafTask " << w->CPU2GPUMap[RequiredCPUBuffers[0]] << std::endl;
    CUDA_updateLeafSinkFlow(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
      w->CPU2GPUMap[RequiredCPUBuffers[2]],w->CPU2GPUMap[RequiredCPUBuffers[3]],Parent->CC, Parent->VolumeSize, w->GetStream());
    CUDA_constrainLeafSinkFlow(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[4]], Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 2;
    break;

  case(ApplySinkPotentialBranchTask):    //0 - Working,  1 - Inc,  2 - Div,  3 - Label
    //std::cout << Node1 << "\t" << Node2 << "\t" << "ApplySinkPotentialBranchTask" << std::endl;
    CUDA_storeSinkFlowInBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
      w->CPU2GPUMap[RequiredCPUBuffers[2]],w->CPU2GPUMap[RequiredCPUBuffers[3]], Parent->CC, Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;
      
  case(ApplySourcePotentialTask):      //0 - Working,  1 - Sink,  2 - Div,  3 - Label
    //std::cout << Node1 << "\t" << Node2 << "\t" << "ApplySourcePotentialTask" << std::endl;
    CUDA_storeSourceFlowInBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
      w->CPU2GPUMap[RequiredCPUBuffers[2]],w->CPU2GPUMap[RequiredCPUBuffers[3]], Parent->CC, Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(DivideOutWorkingBufferTask):    //0 - Working,  1 - Sink
    //std::cout << Node1 << "\t" << Node2 << "\t" << "DivideOutWorkingBufferTask" << std::endl;
    CUDA_divideAndStoreBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
      constant1,  Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[1]] = 1;
    Parent->NumKernelRuns += 1;
    break;
      
  case(UpdateLabelsTask):          //0 - Sink,    1 - Inc,  2 - Div,  3 - Label
    //std::cout << Node1 << "\t" << Node2 << "\t" << "UpdateLabelsTask" << std::endl;
    CUDA_updateLabel(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]], w->CPU2GPUMap[RequiredCPUBuffers[2]],
      w->CPU2GPUMap[RequiredCPUBuffers[3]], Parent->CC, Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[3]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(ClearBufferInitially):        //0 - Any
    //std::cout << Node1 << "\t" << Node2 << "\t" << "ClearBufferInitially" << std::endl;
    CUDA_zeroOutBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(InitializeLeafFlows):        //0 - Sink,    1 - Data
    //std::cout << Node1 << "\t" << Node2 << "\t" << "InitializeLeafFlows" << std::endl;
    CUDA_CopyBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]], Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(MinimizeLeafFlows):        //0 - Sink1,  1 - Sink2
    //std::cout << Node1 << "\t" << Node2 << "\t" << "MinimizeLeafFlows" << std::endl;
    CUDA_MinBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]], Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(PropogateLeafFlows):        //0 - SinkMin,  1 - SinkElse
    //std::cout << Node1 << "\t" << Node2 << "\t" << "PropogateLeafFlows" << std::endl;
    CUDA_CopyBuffer(w->CPU2GPUMap[RequiredCPUBuffers[1]], w->CPU2GPUMap[RequiredCPUBuffers[0]], Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[1]] = 1;
    Parent->NumKernelRuns += 1;
    break;
      
  case(InitializeLeafLabels):        //0 - Sink,    1 - Data,  2 - Label
    //std::cout << Node1 << "\t" << Node2 << "\t" << "InitializeLeafLabels" << std::endl;
    CUDA_LblBuffer(w->CPU2GPUMap[RequiredCPUBuffers[2]], w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]],
      Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[2]] = 1;
    Parent->NumKernelRuns += 1;
    break;
      
  case(AccumulateLabels):          //0 - Accum,  1 - Label
    //std::cout << Node1 << "\t" << Node2 << "\t" << "AccumulateLabels" << std::endl;
    CUDA_SumBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]], Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(CorrectLabels):          //0 - Factor,  1 - Label
    //std::cout << Node1 << "\t" << Node2 << "\t" << "CorrectLabels" << std::endl;
    CUDA_DivBuffer(w->CPU2GPUMap[RequiredCPUBuffers[1]], w->CPU2GPUMap[RequiredCPUBuffers[0]], Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[1]] = 1;
    Parent->NumKernelRuns += 1;
    break;
  
  case(AccumulateLabelsWeighted):      //0 - Accum,  1 - Label
    //std::cout << Node1 << "\t" << Node2 << "\t" << "AccumulateLabelsWeighted" << "\t" << constant1 << std::endl;
    CUDA_SumScaledBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]], constant1, Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(ResetSinkFlowRoot):        //0 - Sink
    //std::cout << Node1 << "\t" << Node2 << "\t" << "ResetSinkFlowRoot" << "\t" << constant1 << std::endl;
    CUDA_ShiftBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], constant1, Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(ResetSinkFlowBranch):        //0 - Sink,    1 - Inc,  2 - Div,  3 - Label
    //std::cout << Node1 << "\t" << Node2 << "\t" << "ResetSinkFlowBranch" << "\t" << constant1 << std::endl;
    CUDA_ResetSinkBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]], w->CPU2GPUMap[RequiredCPUBuffers[2]],
      w->CPU2GPUMap[RequiredCPUBuffers[3]], constant1, 1.0/Parent->CC, Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(PushUpSourceFlows):        //0 - PSink,  1 - Sink,  2 - Inc,  3 - Div,  4 - Label
    //std::cout << Node1 << "\t" << Node2 << "\t" << "PushUpSourceFlows" << "\t" << constant1 << std::endl;
    CUDA_PushUpSourceFlows(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
      w->CPU2GPUMap[RequiredCPUBuffers[2]], w->CPU2GPUMap[RequiredCPUBuffers[3]],
      w->CPU2GPUMap[RequiredCPUBuffers[4]], constant1, 1.0/Parent->CC,Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(ClearSourceBuffer):        //0 - Inc
    //std::cout << Node1 << "\t" << Node2 << "\t" << "ClearSourceBuffer " << w->CPU2GPUMap[RequiredCPUBuffers[0]] << std::endl;
    CUDA_zeroOutBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], Parent->VolumeSize, w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[0]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(PushDownSinkFlows):        //0 - Sink,    1 - CInc
    //std::cout << Node1 << "\t" << Node2 << "\t" << "PushDownSinkFlows " << w->CPU2GPUMap[RequiredCPUBuffers[0]] << " to " << w->CPU2GPUMap[RequiredCPUBuffers[1]] << "\t" << constant1 << std::endl;
    CUDA_SumScaledBuffer(w->CPU2GPUMap[RequiredCPUBuffers[1]],w->CPU2GPUMap[RequiredCPUBuffers[0]],constant1,Parent->VolumeSize,w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[1]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  case(PropogateLeafFlowsInc):      //0 - FlowIn,  1 - FlowO,  2 - FlowO
    //std::cout << Node1 << "\t" << Node2 << "\t" << "PropogateLeafFlowsInc" << std::endl;
    CUDA_Copy2Buffers(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
      w->CPU2GPUMap[RequiredCPUBuffers[2]],Parent->VolumeSize,w->GetStream());
    Parent->Overwritten[RequiredCPUBuffers[1]] = 1;
    Parent->Overwritten[RequiredCPUBuffers[2]] = 1;
    Parent->NumKernelRuns += 1;
    break;

  default:
    break;
  }

  //  if(w->CPU2GPUMap.size() != w->GPU2CPUMap.size()){
  //      std::cout << RequiredCPUBuffers.size() << std::endl;
    //int c = 0;
  //      for(std::vector<float*>::iterator i = RequiredCPUBuffers.begin(); i != RequiredCPUBuffers.end(); i++){
  //          std::cout << c << " " << *i << std::endl;
    //  c++;
  //      }
  //      std::cout << w->CPUInUse.size() << std::endl;
  //      for(std::set<float*>::iterator i = w->CPUInUse.begin(); i != w->CPUInUse.end();i++){
  //          std::cout << *i << std::endl;
  //      }
  //      std::cout << w->CPU2GPUMap.size() << std::endl;
  //      for(std::map<float*,float*>::iterator i = w->CPU2GPUMap.begin(); i != w->CPU2GPUMap.end(); i++){
  //          std::cout << i->first << " " << i->second << std::endl;
  //      }
  //      std::cout << w->GPU2CPUMap.size() << std::endl;
  //      for(std::map<float*,float*>::iterator i = w->GPU2CPUMap.begin(); i != w->GPU2CPUMap.end();i++){
  //          std::cout << i->first << " " << i->second << std::endl;
  //      }
  //      assert(false);
  //  }

  //take them off the no-copy-back list now
  switch(Type){
  case(ClearWorkingBufferTask):      //0 - Working
  case(ApplySinkPotentialLeafTask):    //0 - Sink,    1 - Inc,  2 - Div,  3 - Label,  4 - Data
  case(ClearBufferInitially):        //0 - Any
  case(ClearSourceBuffer):        //0 - Inc
  case(InitializeLeafFlows):        //0 - Sink,    1 - Data
    Parent->NoCopyBack.erase(RequiredCPUBuffers[0]);
    break;
  case(DivideOutWorkingBufferTask):    //0 - Working,  1 - Sink
  case(PropogateLeafFlows):        //0 - SinkMin,  1 - SinkElse
    Parent->NoCopyBack.erase(RequiredCPUBuffers[1]);
    break;
  case(InitializeLeafLabels):        //0 - Sink,    1 - Data,  2 - Label
    Parent->NoCopyBack.erase(RequiredCPUBuffers[2]);
    break;
  case(PropogateLeafFlowsInc):
    Parent->NoCopyBack.erase(RequiredCPUBuffers[1]);
    Parent->NoCopyBack.erase(RequiredCPUBuffers[2]);
    break;
  }

  //send off appropriate finishing signals and return to original state
  this->FinishedSignal();
  Active -= FinishDecreaseInActive;

  //if we are at the end of our lifetime, leave
  NumTimesCalled++;
  if( Active < 0 && NumTimesCalled < NumToDeath ){
    Parent->CurrentTasks.erase(this);
    Parent->BlockedTasks.insert(this);
  }
  if( NumTimesCalled >= NumToDeath ){
    Parent->CurrentTasks.erase( this );
    Parent->FinishedTasks.insert( this );
  }
}
