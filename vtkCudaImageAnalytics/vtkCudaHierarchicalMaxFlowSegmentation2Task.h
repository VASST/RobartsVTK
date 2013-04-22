#ifndef __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2TASK_H__
#define __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2TASK_H__

#include "vtkCudaHierarchicalMaxFlowSegmentation2.h"

#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include <set>
#include <list>

#include "CUDA_hierarchicalmaxflow.h"
#include "vtkCudaDeviceManager.h"
#include "vtkCudaObject.h"

#define SQR(X) X*X

class vtkCudaHierarchicalMaxFlowSegmentation2::Task {
public:

	enum TaskType{
		//iteration components
		ClearWorkingBufferTask,			//0 - Working
		UpdateSpatialFlowsTask,			//0 - Sink,		1 - Inc,	2 - Div,	3 - Label,	4 - FlowX,	5 - FlowY,	6 - FlowZ,	7 - Smoothness
		ApplySinkPotentialBranchTask,	//0 - Working,	1 - Inc,	2 - Div,	3 - Label
		ApplySinkPotentialLeafTask,		//0 - Sink,		1 - Inc,	2 - Div,	3 - Label,	4 - Data
		ApplySourcePotentialTask,		//0 - Working,	1 - Sink,	2 - Div,	3 - Label
		DivideOutWorkingBufferTask,		//0 - Working,	1 - Sink
		UpdateLabelsTask,				//0 - Sink,		1 - Inc,	2 - Div,	3 - Label

		//initialization components
		ClearBufferInitially,			//0 - Any
		InitializeLeafFlows,			//0 - Sink,		1 - Data
		MinimizeLeafFlows,				//0 - Sink1,	1 - Sink2
		PropogateLeafFlows,				//0 - SinkMin,	1 - SinkElse
		InitializeLeafLabels,			//0 - Sink,		1 - Data,	2 - Label
		AccumulateLabels,				//0 - Accum,	1 - Label
		CorrectLabels,					//0 - Factor,	1 - Label
	};

	TaskType Type;
	vtkIdType Node;
	int NodeIndex;
	int Active;
	int FinishDecreaseInActive;
	int NumKids;
	int NumTimesCalled;
	int NumToDeath;
	bool isRoot;
	vtkCudaHierarchicalMaxFlowSegmentation2* const Parent;

	//Fill in all non-transient information
	Task( vtkCudaHierarchicalMaxFlowSegmentation2* parent, int a, int ra, int numToDeath, vtkIdType n, TaskType t )
		: Parent(parent), Active(a), FinishDecreaseInActive(ra), Node(n), Type(t) {
		NumKids = Parent->Hierarchy->GetNumberOfChildren(Node);
		isRoot = (Parent->Hierarchy->GetRoot() == Node);
		NumToDeath = numToDeath;
		NumTimesCalled = 0;
		NodeIndex = -1;
		if( NumKids == 0 ) NodeIndex = Parent->LeafMap[n];
		else if( !isRoot ) NodeIndex = Parent->BranchMap[n];
		
		if( NumToDeath <= 0 ){
			Parent->FinishedTasks.insert(this);
			return;
		}

		Parent->NumTasksGoingToHappen += numToDeath;
		if(Active >= 0) Parent->CurrentTasks.insert(this);
		else  Parent->BlockedTasks.insert(this);

	}
	
	void Signal(){
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
	bool CanDo(){ return this->Active >= 0; }

	//manage signals for when this task is finished
	//ie: allow us to signal the next task in the loop as well as
	//    any parent or child node tasks as necessary
	std::set<Task*> FinishedSignals;
	void AddTaskToSignal(Task* t){
		FinishedSignals.insert(t);
	}
	void FinishedSignal(){
		std::set<Task*>::iterator it = FinishedSignals.begin();
		for(; it != FinishedSignals.end(); it++ )
			if(*it) (*it)->Signal();
	}

	std::map<int,float*> RequiredCPUBuffers;
	void AddBuffer(float* b){
		RequiredCPUBuffers.insert(std::pair<int, float*>((int)RequiredCPUBuffers.size(),b) );
	}

	//Find out if we have a conflict on this task (ie: not all buffers are available on CPU or single GPU)
	//returning an unconflicted device if false (null if conflict or any worker will suffice)
	int Conflicted(vtkCudaHierarchicalMaxFlowSegmentation2::Worker** w){
		
		//find the GPU with most of the buffers
		int retVal = 0;
		int maxBuffersGot = 0;
		Worker* maxGPU = 0;
		for(std::set<Worker*>::iterator wit = Parent->Workers.begin(); wit != Parent->Workers.end(); wit++){
			int buffersGot = 0;
			for(std::map<int,float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++){
				if( (*wit)->CPU2GPUMap.find(it->second) != (*wit)->CPU2GPUMap.end() ) buffersGot++;
			}
			if( buffersGot > maxBuffersGot ){
				maxBuffersGot = buffersGot;
				maxGPU = *wit;
			}
		}

		//return everything that is not on that GPU
		for(std::set<Worker*>::iterator wit = Parent->Workers.begin(); wit != Parent->Workers.end(); wit++){
			if( *wit == maxGPU ) continue;
			for(std::map<int,float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++){
				if( (*wit)->CPU2GPUMap.find(it->second) != (*wit)->CPU2GPUMap.end() ){
					if( Parent->ReadOnly.find(it->second) != Parent->ReadOnly.end() ||
						Parent->NoCopyBack.find(it->second) != Parent->NoCopyBack.end() )
						retVal += 1;
					else retVal += 3;
				}
			}
		}

		*w = maxGPU;
		return retVal;
	}

	//find the GPU with most of the buffers and return all claimed buffers not on that GPU
	void UnConflict(vtkCudaHierarchicalMaxFlowSegmentation2::Worker* maxGPU){

		//return everything that is not on that GPU
		for(std::set<Worker*>::iterator wit = Parent->Workers.begin(); wit != Parent->Workers.end(); wit++){
			if( *wit == maxGPU ) continue;
			for(std::map<int,float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++){
				if( (*wit)->CPU2GPUMap.find(it->second) != (*wit)->CPU2GPUMap.end() ){
					(*wit)->ReturnBuffer(it->second);
					(*wit)->CallSyncThreads();
				}
			}
		}

	}

	//Calculate the weight provided that there is no conflict
	int CalcWeight(vtkCudaHierarchicalMaxFlowSegmentation2::Worker* w){
		int retWeight = 0;
		int numUnused = (int) w->UnusedGPUBuffers.size();
		for(std::map<int,float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++){
			if( w->CPU2GPUMap.find(it->second) == w->CPU2GPUMap.end() ){
				if( numUnused ){numUnused--; retWeight++;
				}else retWeight += 2;
			}
		}
		return retWeight;
	}

	//Perform the task at hand
	void Perform(vtkCudaHierarchicalMaxFlowSegmentation2::Worker* w){
		if( !CanDo() ) return;

		//load anything that will be overwritten onto the no copy back list
		switch(Type){
		case(ClearWorkingBufferTask):			//0 - Working
			Parent->NoCopyBack.insert(RequiredCPUBuffers[0]);
			break;
		case(ApplySinkPotentialLeafTask):		//0 - Sink,		1 - Inc,	2 - Div,	3 - Label,	4 - Data
			Parent->NoCopyBack.insert(RequiredCPUBuffers[0]);
			break;
		case(DivideOutWorkingBufferTask):		//0 - Working,	1 - Sink
			Parent->NoCopyBack.insert(RequiredCPUBuffers[1]);
			break;
		case(ClearBufferInitially):				//0 - Any
			Parent->NoCopyBack.insert(RequiredCPUBuffers[0]);
			break;
		}

		//load required buffers onto the GPU
		w->CPUInUse.clear();
		for(std::map<int,float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++)
			if(it->second) w->CPUInUse.insert(it->second);
		w->UpdateBuffersInUse();
		
        for(std::map<int,float*>::iterator it = RequiredCPUBuffers.begin(); it != RequiredCPUBuffers.end(); it++)
            if(w->CPU2GPUMap.find(it->second) == w->CPU2GPUMap.end()){
                std::cout << "Problem: " << it->second << std::endl;
            }


        assert(w->CPU2GPUMap.size() == w->GPU2CPUMap.size());

		//run the kernels
		float smoothnessConstant = 1.0f;
		switch(Type){
		case(ClearWorkingBufferTask):			//0 - Working
			if( !isRoot ) CUDA_zeroOutBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], Parent->VolumeSize, w->GetStream());
			else CUDA_SetBufferToValue(w->CPU2GPUMap[RequiredCPUBuffers[0]], 1.0f/Parent->CC, Parent->VolumeSize, w->GetStream());
			Parent->NoCopyBack.erase( RequiredCPUBuffers[0] );
			Parent->NumKernelRuns += 1;
			break;
			
		case(UpdateSpatialFlowsTask):			//0 - Sink,		1 - Inc,	2 - Div,	3 - Label,	4 - FlowX,	5 - FlowY,	6 - FlowZ,	7 - Smoothness
			smoothnessConstant = (NumKids == 0) ? Parent->leafSmoothnessConstants[NodeIndex]: Parent->branchSmoothnessConstants[NodeIndex];
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
			Parent->NumKernelRuns += 4;
			break;

		case(ApplySinkPotentialLeafTask):		//0 - Sink,		1 - Inc,	2 - Div,	3 - Label,	4 - Data
			CUDA_updateLeafSinkFlow(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
				w->CPU2GPUMap[RequiredCPUBuffers[2]],w->CPU2GPUMap[RequiredCPUBuffers[3]],Parent->CC, Parent->VolumeSize, w->GetStream());
			CUDA_constrainLeafSinkFlow(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[4]], Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 2;
			break;

		case(ApplySinkPotentialBranchTask):		//0 - Working,	1 - Inc,	2 - Div,	3 - Label
			CUDA_storeSinkFlowInBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
				w->CPU2GPUMap[RequiredCPUBuffers[2]],w->CPU2GPUMap[RequiredCPUBuffers[3]], Parent->CC, Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;
			
		case(ApplySourcePotentialTask):			//0 - Working,	1 - Sink,	2 - Div,	3 - Label
			CUDA_storeSourceFlowInBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
				w->CPU2GPUMap[RequiredCPUBuffers[2]],w->CPU2GPUMap[RequiredCPUBuffers[3]], Parent->CC, Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;

		case(DivideOutWorkingBufferTask):		//0 - Working,	1 - Sink
			CUDA_divideAndStoreBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]],
				NumKids + (isRoot ? 0 : 1), Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;
			
		case(UpdateLabelsTask):					//0 - Sink,		1 - Inc,	2 - Div,	3 - Label
			CUDA_updateLabel(w->CPU2GPUMap[RequiredCPUBuffers[0]],w->CPU2GPUMap[RequiredCPUBuffers[1]], w->CPU2GPUMap[RequiredCPUBuffers[2]],
				w->CPU2GPUMap[RequiredCPUBuffers[3]], Parent->CC, Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;

		case(ClearBufferInitially):				//0 - Any
			CUDA_zeroOutBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;

		case(InitializeLeafFlows):				//0 - Sink,		1 - Data
			CUDA_CopyBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]], Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;

		case(MinimizeLeafFlows):				//0 - Sink1,	1 - Sink2
			CUDA_MinBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]], Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;

		case(PropogateLeafFlows):				//0 - SinkMin,	1 - SinkElse
			CUDA_CopyBuffer(w->CPU2GPUMap[RequiredCPUBuffers[1]], w->CPU2GPUMap[RequiredCPUBuffers[0]], Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;
			
		case(InitializeLeafLabels):				//0 - Sink,		1 - Data,	2 - Label
			CUDA_LblBuffer(w->CPU2GPUMap[RequiredCPUBuffers[2]], w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]],
				Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;
			
		case(AccumulateLabels):					//0 - Accum,	1 - Label
			CUDA_SumBuffer(w->CPU2GPUMap[RequiredCPUBuffers[0]], w->CPU2GPUMap[RequiredCPUBuffers[1]], Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;
		
		case(CorrectLabels):					//0 - Factor,	1 - Label
			CUDA_DivBuffer(w->CPU2GPUMap[RequiredCPUBuffers[1]], w->CPU2GPUMap[RequiredCPUBuffers[0]], Parent->VolumeSize, w->GetStream());
			Parent->NumKernelRuns += 1;
			break;

		default:
			vtkErrorWithObjectMacro(Parent,<<"Invalid task type" << Type <<".");
		}

        if(w->CPU2GPUMap.size() != w->GPU2CPUMap.size()){
            std::cout << RequiredCPUBuffers.size() << std::endl;
            for(std::map<int,float*>::iterator i = RequiredCPUBuffers.begin(); i != RequiredCPUBuffers.end();i++){
                std::cout << i->first << " " << i->second << std::endl;
            }
            std::cout << w->CPUInUse.size() << std::endl;
            for(std::set<float*>::iterator i = w->CPUInUse.begin(); i != w->CPUInUse.end();i++){
                std::cout << *i << std::endl;
            }
            std::cout << w->CPU2GPUMap.size() << std::endl;
            for(std::map<float*,float*>::iterator i = w->CPU2GPUMap.begin(); i != w->CPU2GPUMap.end();i++){
                std::cout << i->first << " " << i->second << std::endl;
            }
            std::cout << w->GPU2CPUMap.size() << std::endl;
            for(std::map<float*,float*>::iterator i = w->GPU2CPUMap.begin(); i != w->GPU2CPUMap.end();i++){
                std::cout << i->first << " " << i->second << std::endl;
            }
            assert(false);
        }

		//take them off the no-copy-back list now
		switch(Type){
		case(ClearWorkingBufferTask):			//0 - Working
			Parent->NoCopyBack.erase(RequiredCPUBuffers[0]);
			break;
		case(ApplySinkPotentialLeafTask):		//0 - Sink,		1 - Inc,	2 - Div,	3 - Label,	4 - Data
			Parent->NoCopyBack.erase(RequiredCPUBuffers[0]);
			break;
		case(DivideOutWorkingBufferTask):		//0 - Working,	1 - Sink
			Parent->NoCopyBack.erase(RequiredCPUBuffers[1]);
			break;
		case(ClearBufferInitially):				//0 - Any
			Parent->NoCopyBack.erase(RequiredCPUBuffers[0]);
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


};


#endif //__VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2TASK_H__
