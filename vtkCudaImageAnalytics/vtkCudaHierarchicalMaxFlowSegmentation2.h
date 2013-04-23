#ifndef __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2_H__
#define __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2_H__

#include "vtkAlgorithm.h"
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

#include <limits.h>
#include <float.h>

#include "vtkCudaObject.h"

//INPUT PORT DESCRIPTION

//OUTPUT PORT DESCRIPTION

class vtkCudaHierarchicalMaxFlowSegmentation2 : public vtkImageAlgorithm
{
public:
	vtkTypeMacro( vtkCudaHierarchicalMaxFlowSegmentation2, vtkImageAlgorithm );

	static vtkCudaHierarchicalMaxFlowSegmentation2 *New();

	//Set the hierarchical model used in the segmentation, note that this has to be a 
	// tree.
	void SetHierarchy(vtkTree* graph);
	vtkTree* GetHierarchy();

	void AddDevice(int GPU);
	void RemoveDevice(int GPU);
	bool HasDevice(int GPU);
	void ClearDevices();
	void SetDevice(int GPU){ this->ClearDevices(); this->AddDevice(GPU); }

	//Weight the smoothness term. If no scalar is provided, it is assumed to be 1. If
	//no smoothness term is provided, it is assumed to be the unit function.
	void AddSmoothnessScalar( vtkIdType node, double alpha );
	
	//Get and Set the number of iterations used by the algorithm (computing convergence term
	//is too slow.)
	vtkSetClampMacro(NumberOfIterations,int,0,INT_MAX);
	vtkGetMacro(NumberOfIterations,int);
	
	//Get and Set the maximum 
	vtkSetClampMacro(MaxGPUUsage,double,0.0,1.0);
	vtkGetMacro(MaxGPUUsage,double);
	
	//Get and Set the labeling constant, CC, of the algorithm
	vtkSetClampMacro(CC,float,0.0f,1.0f);
	vtkGetMacro(CC,float);

	//Get and Set the step size of the algorithm for updating spatial flows
	vtkSetClampMacro(StepSize,float,0.0f,1.0f);
	vtkGetMacro(StepSize,float);

	//Get and Set the verbose flag
	vtkSetClampMacro(ReportRate,int,0,INT_MAX);
	vtkGetMacro(ReportRate,int);

	vtkDataObject* GetInput(int idx);
	void SetInput(int idx, vtkDataObject *input);
	vtkDataObject* GetOutput(int idx);
	
	// Description:
	// If the subclass does not define an Execute method, then the task
	// will be broken up, multiple threads will be spawned, and each thread
	// will call this method. It is public so that the thread functions
	// can call this method.
	virtual int RequestData(vtkInformation *request, 
							 vtkInformationVector **inputVector, 
							 vtkInformationVector *outputVector);
	virtual int RequestInformation( vtkInformation* request,
							 vtkInformationVector** inputVector,
							 vtkInformationVector* outputVector);
	virtual int RequestUpdateExtent( vtkInformation* request,
							 vtkInformationVector** inputVector,
							 vtkInformationVector* outputVector);
	virtual int RequestDataObject( vtkInformation* request,
							 vtkInformationVector** inputVector,
							 vtkInformationVector* outputVector);
	virtual int FillInputPortInformation(int i, vtkInformation* info);

protected:
	vtkCudaHierarchicalMaxFlowSegmentation2();
	virtual ~vtkCudaHierarchicalMaxFlowSegmentation2();

private:
	vtkCudaHierarchicalMaxFlowSegmentation2 operator=(const vtkCudaHierarchicalMaxFlowSegmentation2&){}
	vtkCudaHierarchicalMaxFlowSegmentation2(const vtkCudaHierarchicalMaxFlowSegmentation2&){}

	std::set<int> GPUsUsed;

	double	MaxGPUUsage;
	int		ReportRate;

	int CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges );
	void PropogateLabels( vtkIdType currNode );
	void SolveMaxFlow( vtkIdType currNode, int* timeStep );
	void UpdateLabel( vtkIdType node, int* timeStep );
	
	vtkTree* Hierarchy;
	std::map<vtkIdType,double> SmoothnessScalars;
	std::map<vtkIdType,int> LeafMap;
	std::map<vtkIdType,int> BranchMap;

	int NumberOfIterations;
	float CC;
	float StepSize;
	int VolumeSize;
	int VX, VY, VZ;
	
	std::map<vtkIdType,int> InputPortMapping;
	std::map<int,vtkIdType> BackwardsInputPortMapping;
	int FirstUnusedPort;

	//pointers to variable structures, easier to keep as part of the class definition
	float**	branchFlowXBuffers;
	float**	branchFlowYBuffers;
	float**	branchFlowZBuffers;
	float**	branchDivBuffers;
	float**	branchSinkBuffers;
	float**	branchIncBuffers;
	float**	branchLabelBuffers;
	float**	branchSmoothnessTermBuffers;
	float**	branchWorkingBuffers;
	float*	branchSmoothnessConstants;

	float**	leafFlowXBuffers;
	float**	leafFlowYBuffers;
	float**	leafFlowZBuffers;
	float**	leafDivBuffers;
	float**	leafSinkBuffers;
	float**	leafIncBuffers;
	float**	leafLabelBuffers;
	float**	leafDataTermBuffers;
	float**	leafSmoothnessTermBuffers;
	float*	leafSmoothnessConstants;

	float*	sourceFlowBuffer;
	float*	sourceWorkingBuffer;

	
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
		std::list< std::list<float*> > PriorityStacks;
		Worker(int g, vtkCudaHierarchicalMaxFlowSegmentation2* p );
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
	int		NumLeaves;
	int		NumBranches;
	int		NumNodes;
	int		NumEdges;
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
