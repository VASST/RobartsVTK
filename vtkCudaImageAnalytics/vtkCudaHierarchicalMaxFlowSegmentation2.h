/** @file vtkHierarchicalMaxFlowSegmentation2.h
 *
 *  @brief Header file with definitions of GPU-based solver for generalized hierarchical max-flow
 *			segmentation problems with greedy scheduling over multiple GPUs. See
 *			vtkHierarchicalMaxFlowSegmentation.h for most of the interface documentation.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *	
 *	@note August 27th 2013 - Documentation first compiled.
 *
 */

#ifndef __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2_H__
#define __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2_H__

#include "vtkHierarchicalMaxFlowSegmentation.h"
#include "vtkCudaObject.h"

#include <map>
#include <list>
#include <set>
#include <vector>

#include <limits.h>
#include <float.h>

class vtkCudaHierarchicalMaxFlowSegmentation2 : public vtkHierarchicalMaxFlowSegmentation
{
public:
	vtkTypeMacro( vtkCudaHierarchicalMaxFlowSegmentation2, vtkHierarchicalMaxFlowSegmentation );
	static vtkCudaHierarchicalMaxFlowSegmentation2 *New();

	// Description:
	// Insert, remove, and verify a given GPU into the set of GPUs usable by the algorithm. This
	// set defaults to {GPU0} and must be non-empty when the update is invoked.
	void AddDevice(int GPU);
	void RemoveDevice(int GPU);
	bool HasDevice(int GPU);

	// Description:
	// Clears the set of GPUs usable by the algorith,
	void ClearDevices();

	// Description:
	// Set the class to use a single GPU, the one provided.
	void SetDevice(int GPU){ this->ClearDevices(); this->AddDevice(GPU); }
	
	// Description:
	// Get and Set the maximum percent of GPU memory usable by the algorithm.
	// Recommended to keep below 98% on compute-only cards, and 90% on cards
	// used for running the monitors. The number provided will act as a de
	// facto value for all cards. (Default is 90%.)
	vtkSetClampMacro(MaxGPUUsage,double,0.0,1.0);
	vtkGetMacro(MaxGPUUsage,double);
	
	// Description:
	// Get, Set, and Clear exceptions, allowing for a particular card to have its
	// memory consumption managed separately. 
	void SetMaxGPUUsage(double usage, int device);
	double GetMaxGPUUsage(int device);
	void ClearMaxGPUUsage();
	
	// Description:
	// Get and Set how often the algorithm should report if in Debug mode. If set
	// to 0, the algorithm doesn't report task completions. Default is 100 tasks.
	vtkSetClampMacro(ReportRate,int,0,INT_MAX);
	vtkGetMacro(ReportRate,int);

protected:
	vtkCudaHierarchicalMaxFlowSegmentation2();
	virtual ~vtkCudaHierarchicalMaxFlowSegmentation2();

	std::set<int> GPUsUsed;

	double					MaxGPUUsage;
	std::map<int,double>	MaxGPUUsageNonDefault;
	int						ReportRate;
	
	virtual int InitializeAlgorithm();
	virtual int RunAlgorithm();

	void PropogateLabels( vtkIdType currNode );
	void SolveMaxFlow( vtkIdType currNode, int* timeStep );
	void UpdateLabel( vtkIdType node, int* timeStep );

	class Worker : public vtkCudaObject {
	public:
		vtkCudaHierarchicalMaxFlowSegmentation2* const Parent;
		const int GPU;
        int NumBuffers;
		std::map<float*,float*> CPU2GPUMap;
		std::map<float*,float*> GPU2CPUMap;
		std::set<float*> CPUInUse;
		std::list<float*> UnusedGPUBuffers;
		std::list<float*> AllGPUBufferBlocks;
		std::vector< std::list< float* > > PriorityStacks;
		Worker(int g, double usage, vtkCudaHierarchicalMaxFlowSegmentation2* p );
		~Worker();
		void UpdateBuffersInUse();
		void AddToStack( float* CPUBuffer );
		void RemoveFromStack( float* CPUBuffer );
		void BuildStackUpToPriority( int priority );
		void TakeDownPriorityStacks();
		int LowestBufferShift(int n);
		void ReturnLeafLabels();
		void ReturnBuffer(float* CPUBuffer);
		void Reinitialize(int withData){} // not used
		void Deinitialize(int withData){} // not used
	};
	friend class Worker;
	std::set<Worker*> Workers;
	
	//Mappings for CPU-GPU buffer sharing
	void ReturnBufferGPU2CPU(Worker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream);
	void MoveBufferCPU2GPU(Worker* caller, float* CPUBuffer, float* GPUBuffer, cudaStream_t* stream);
	void FigureOutBufferPriorities( vtkIdType currNode );
	std::map<float*,Worker*> LastBufferUse;
	std::map<float*,int> Overwritten;

	class Task;
	friend class Task;
	std::set<Task*> CurrentTasks;
	std::set<Task*> BlockedTasks;
	std::set<Task*> FinishedTasks;

	std::set<float*> CPUInUse;
	std::map<float*,int> CPU2PriorityMap;

	std::set< float* > ReadOnly;
	std::set< float* > NoCopyBack;

	int		NumMemCpies;
	int		NumKernelRuns;
	int		NumTasksGoingToHappen;

	std::map<vtkIdType,Task*> ClearWorkingBufferTasks;
	std::map<vtkIdType,Task*> UpdateSpatialFlowsTasks;
	std::map<vtkIdType,Task*> ApplySinkPotentialBranchTasks;
	std::map<vtkIdType,Task*> ApplySinkPotentialLeafTasks;
	std::map<vtkIdType,Task*> ApplySourcePotentialTasks;
	std::map<vtkIdType,Task*> DivideOutWorkingBufferTasks;
	std::map<vtkIdType,Task*> UpdateLabelsTasks;

	void CreateClearWorkingBufferTasks(vtkIdType currNode);
	void CreateUpdateSpatialFlowsTasks(vtkIdType currNode);
	void CreateApplySinkPotentialBranchTasks(vtkIdType currNode);
	void CreateApplySinkPotentialLeafTasks(vtkIdType currNode);
	void CreateApplySourcePotentialTask(vtkIdType currNode);
	void CreateDivideOutWorkingBufferTask(vtkIdType currNode);
	void CreateUpdateLabelsTask(vtkIdType currNode);
	void AddIterationTaskDependencies(vtkIdType currNode);
	
	std::map<int,Task*> InitializeLeafSinkFlowsTasks;
	std::map<int,Task*> MinimizeLeafSinkFlowsTasks;
	std::map<vtkIdType,Task*> PropogateLeafSinkFlowsTasks;
	std::map<vtkIdType,Task*> InitialLabellingSumTasks;
	std::map<vtkIdType,Task*> CorrectLabellingTasks;
	std::map<vtkIdType,Task*> PropogateLabellingTasks;

	void CreateInitializeAllSpatialFlowsToZeroTasks(vtkIdType currNode);
	void CreateInitializeLeafSinkFlowsToCapTasks(vtkIdType currNode);
	void CreateCopyMinimalLeafSinkFlowsTasks(vtkIdType currNode);
	void CreateFindInitialLabellingAndSumTasks(vtkIdType currNode);
	void CreateClearSourceWorkingBufferTask();
	void CreateDivideOutLabelsTasks(vtkIdType currNode);
	void CreatePropogateLabelsTasks(vtkIdType currNode);
	

private:
	vtkCudaHierarchicalMaxFlowSegmentation2 operator=(const vtkCudaHierarchicalMaxFlowSegmentation2&){} //not implemented
	vtkCudaHierarchicalMaxFlowSegmentation2(const vtkCudaHierarchicalMaxFlowSegmentation2&){} //not implemented
};

#endif
