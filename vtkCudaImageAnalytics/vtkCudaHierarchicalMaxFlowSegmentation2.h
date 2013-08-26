#ifndef __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2_H__
#define __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2_H__

#include "vtkHierarchicalMaxFlowSegmentation.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"
#include "vtkDirectedGraph.h"
#include "vtkTree.h"
#include <map>
#include <list>
#include <set>
#include <vector>

#include <limits.h>
#include <float.h>

#include "vtkCudaObject.h"

//INPUT PORT DESCRIPTION

//OUTPUT PORT DESCRIPTION

class vtkCudaHierarchicalMaxFlowSegmentation2 : public vtkHierarchicalMaxFlowSegmentation
{
public:
	vtkTypeMacro( vtkCudaHierarchicalMaxFlowSegmentation2, vtkHierarchicalMaxFlowSegmentation );

	static vtkCudaHierarchicalMaxFlowSegmentation2 *New();

	void AddDevice(int GPU);
	void RemoveDevice(int GPU);
	bool HasDevice(int GPU);
	void ClearDevices();
	void SetDevice(int GPU){ this->ClearDevices(); this->AddDevice(GPU); }
	
	//Get and Set the maximum 
	vtkSetClampMacro(MaxGPUUsage,double,0.0,1.0);
	vtkGetMacro(MaxGPUUsage,double);
	void SetMaxGPUUsage(double usage, int device);
	double GetMaxGPUUsage(int device);
	void ClearMaxGPUUsage();

	//Get and Set the verbose flag
	vtkSetClampMacro(ReportRate,int,0,INT_MAX);
	vtkGetMacro(ReportRate,int);
	
	// Description:
	// If the subclass does not define an Execute method, then the task
	// will be broken up, multiple threads will be spawned, and each thread
	// will call this method. It is public so that the thread functions
	// can call this method.
	virtual int RequestData(vtkInformation *request, 
							 vtkInformationVector **inputVector, 
							 vtkInformationVector *outputVector);

protected:
	vtkCudaHierarchicalMaxFlowSegmentation2();
	virtual ~vtkCudaHierarchicalMaxFlowSegmentation2();

private:
	vtkCudaHierarchicalMaxFlowSegmentation2 operator=(const vtkCudaHierarchicalMaxFlowSegmentation2&){}
	vtkCudaHierarchicalMaxFlowSegmentation2(const vtkCudaHierarchicalMaxFlowSegmentation2&){}

	std::set<int> GPUsUsed;

	double					MaxGPUUsage;
	std::map<int,double>	MaxGPUUsageNonDefault;
	int						ReportRate;

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
	
	int		TotalNumberOfBuffers;
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
};

#endif
