#ifndef __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION_H__
#define __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION_H__

#include "vtkCudaObject.h"

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

//INPUT PORT DESCRIPTION

//OUTPUT PORT DESCRIPTION

class vtkCudaHierarchicalMaxFlowSegmentation : public vtkImageAlgorithm, public vtkCudaObject
{
public:
	vtkTypeMacro( vtkCudaHierarchicalMaxFlowSegmentation, vtkImageAlgorithm );

	static vtkCudaHierarchicalMaxFlowSegmentation *New();

	//Set the hierarchical model used in the segmentation, note that this has to be a 
	// tree.
	void SetHierarchy(vtkTree* graph);
	vtkTree* GetHierarchy();

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
	vtkSetMacro(Verbose,bool);
	vtkGetMacro(Verbose,bool);

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
	vtkCudaHierarchicalMaxFlowSegmentation();
	virtual ~vtkCudaHierarchicalMaxFlowSegmentation();

private:
	vtkCudaHierarchicalMaxFlowSegmentation operator=(const vtkCudaHierarchicalMaxFlowSegmentation&){}
	vtkCudaHierarchicalMaxFlowSegmentation(const vtkCudaHierarchicalMaxFlowSegmentation&){}

	void Reinitialize(int withData);
	void Deinitialize(int withData);

	double	MaxGPUUsage;
	bool	Verbose;

	int CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges );
	void PropogateLabels( vtkIdType currNode );
	void SolveMaxFlow( vtkIdType currNode );
	void UpdateLabel( vtkIdType node );
	
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

	//Mappings for CPU-GPU buffer sharing
	void GetGPUBuffers();
	void ReturnBufferGPU2CPU(float* CPUBuffer, float* GPUBuffer);
	void MoveBufferCPU2GPU(float* CPUBuffer, float* GPUBuffer);
	void AddToStack( float* CPUBuffer );
	void RemoveFromStack( float* CPUBuffer );
	void BuildStackUpToPriority( int priority );
	void FigureOutBufferPriorities( vtkIdType currNode );

	std::map<float*,float*> CPU2GPUMap;
	std::map<float*,float*> GPU2CPUMap;
	std::map<float*,int> CPU2PriorityMap;

	std::set<float*> CPUInUse;
	
	std::list< std::list<float*> > PriorityStacks;
	std::list< int > Priority;

	std::list< float* > UnusedGPUBuffers;
	std::set< float* > ReadOnly;
	std::set< float* > NoCopyBack;
	
	int		NumMemCpies;
	int		NumKernelRuns;
};

#endif
