/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkHierarchicalMaxFlowSegmentation2Task.cxx

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkHierarchicalMaxFlowSegmentation2Task.h
 *
 *  @brief Header file with definitions of individual chunks of GPU based code which can be
 *			handled semi-synchronously. 
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *	
 *	@note August 27th 2013 - Documentation first compiled.
 *
 *  @note This is not a front-end class.
 *
 */

#ifndef __VTKCUDAMAXFLOWSEGMENTATIONTASK_H__
#define __VTKCUDAMAXFLOWSEGMENTATIONTASK_H__

#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include <set>
#include <vector>

class vtkCudaMaxFlowSegmentationScheduler;
#include "vtkCudaMaxFlowSegmentationScheduler.h"

#include "vtkCudaMaxFlowSegmentationWorker.h"

#define SQR(X) X*X

class vtkCudaMaxFlowSegmentationTask {
public:

	enum TaskType{
		//iteration components - GHMF
		ClearWorkingBufferTask,			//0 - Working
		UpdateSpatialFlowsTask,			//0 - Sink,		1 - Inc,	2 - Div,	3 - Label,	4 - FlowX,	5 - FlowY,	6 - FlowZ,	7 - Smoothness
		ApplySinkPotentialBranchTask,	//0 - Working,	1 - Inc,	2 - Div,	3 - Label
		ApplySinkPotentialLeafTask,		//0 - Sink,		1 - Inc,	2 - Div,	3 - Label,	4 - Data
		ApplySourcePotentialTask,		//0 - Working,	1 - Sink,	2 - Div,	3 - Label
		DivideOutWorkingBufferTask,		//0 - Working,	1 - Sink
		UpdateLabelsTask,				//0 - Sink,		1 - Inc,	2 - Div,	3 - Label

		//iteration components - DAGMF
		ResetSinkFlowRoot,				//0 - Sink
		ResetSinkFlowBranch,			//0 - Sink,		1 - Inc,	2 - Div,	3 - Label
		PushUpSourceFlows,				//0 - PSink,	1 - Sink,	2 - Inc,	3 - Div,	4 - Label
		PushDownSinkFlows,				//0 - Sink,		1 - CInc
		ClearSourceBuffer,				//0 - Inc

		//initialization components
		ClearBufferInitially,			//0 - Any
		InitializeLeafFlows,			//0 - Sink,		1 - Data
		MinimizeLeafFlows,				//0 - Sink1,	1 - Sink2
		PropogateLeafFlows,				//0 - SinkMin,	1 - SinkElse
		PropogateLeafFlowsInc,			//0 - FlowIn,	1 - FlowO,	2 - FlowO
		InitializeLeafLabels,			//0 - Sink,		1 - Data,	2 - Label
		AccumulateLabels,				//0 - Accum,	1 - Label
		CorrectLabels,					//0 - Factor,	1 - Label

		AccumulateLabelsWeighted		//0 - Accum,	1 - Label
	};

	TaskType Type;

//------------------------------------------------------------------------------------------//

	//Fill in all non-transient information
	vtkCudaMaxFlowSegmentationTask( vtkIdType n1, vtkIdType n2, vtkCudaMaxFlowSegmentationScheduler* parent, int a, int ra, int numToDeath, TaskType t);

	~vtkCudaMaxFlowSegmentationTask();
	
//------------------------------------------------------------------------------------------//

	void Signal();
	bool CanDo(){ return this->Active >= 0; }

//------------------------------------------------------------------------------------------//

	//manage signals for when this task is finished
	//ie: allow us to signal the next task in the loop as well as
	//    any parent or child node tasks as necessary
	void AddTaskToSignal(vtkCudaMaxFlowSegmentationTask* t);
	void FinishedSignal();
	void AddBuffer(float* b);

	//Find out if we have a conflict on this task (ie: not all buffers are available on CPU or single GPU)
	//returning an unconflicted device if false (null if conflict or any worker will suffice)
	int Conflicted(vtkCudaMaxFlowSegmentationWorker** w);

	//find the GPU with most of the buffers and return all claimed buffers not on that GPU
	void UnConflict(vtkCudaMaxFlowSegmentationWorker* maxGPU);

	//Calculate the weight provided that there is no conflict
	int CalcWeight(vtkCudaMaxFlowSegmentationWorker* w);

	//Perform the task at hand
	void Perform(vtkCudaMaxFlowSegmentationWorker* w);
	
	void SetConstant1(float c);
	void SetConstant2(float c);

	void DecrementActivity();

private:
	
	friend class vtkCudaMaxFlowSegmentationWorker;
	friend class vtkCudaMaxFlowSegmentationScheduler;

	int Active;
	int FinishDecreaseInActive;
	int NumTimesCalled;
	int NumToDeath;
	bool isRoot;
	vtkCudaMaxFlowSegmentationScheduler* const Parent;
	
	float constant1;
	float constant2;

	std::vector<vtkCudaMaxFlowSegmentationTask*> FinishedSignals;
	std::vector<float*> RequiredCPUBuffers;

	vtkIdType Node1;
	vtkIdType Node2;

};


#endif //__VTKCUDAMAXFLOWSEGMENTATIONTASK_H__
