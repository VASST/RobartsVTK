#ifndef __VTKHIERARCHICALMAXFLOWSEGMENTATION_H__
#define __VTKHIERARCHICALMAXFLOWSEGMENTATION_H__

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

class vtkHierarchicalMaxFlowSegmentation : public vtkImageAlgorithm
{
public:
	vtkTypeMacro( vtkHierarchicalMaxFlowSegmentation, vtkImageAlgorithm );

	static vtkHierarchicalMaxFlowSegmentation *New();

	//Set the hierarchical model used in the segmentation, note that this has to be a 
	// tree.
	vtkSetObjectMacro(Hierarchy,vtkTree)
	vtkGetObjectMacro(Hierarchy,vtkTree)

	//Weight the smoothness term. If no scalar is provided, it is assumed to be 1. If
	//no smoothness term is provided, it is assumed to be the unit function.
	void AddSmoothnessScalar( vtkIdType node, double alpha );
	
	//Get and Set the number of iterations used by the algorithm (computing convergence term
	//is too slow.)
	vtkSetClampMacro(NumberOfIterations,int,0,INT_MAX);
	vtkGetMacro(NumberOfIterations,int);
	
	//Get and Set the labeling constant, CC, of the algorithm
	vtkSetClampMacro(CC,float,0.0f,1.0f);
	vtkGetMacro(CC,float);

	//Get and Set the step size of the algorithm for updating spatial flows
	vtkSetClampMacro(StepSize,float,0.0f,1.0f);
	vtkGetMacro(StepSize,float);

	vtkDataObject* GetDataInput(int idx);
	void SetDataInput(int idx, vtkDataObject *input);
	vtkDataObject* GetSmoothnessInput(int idx);
	void SetSmoothnessInput(int idx, vtkDataObject *input);
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
	vtkHierarchicalMaxFlowSegmentation();
	virtual ~vtkHierarchicalMaxFlowSegmentation();

	int CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges );
	
	virtual int InitializeAlgorithm();
	virtual int RunAlgorithm();

	void PropogateLabels( vtkIdType currNode );
	void SolveMaxFlow( vtkIdType currNode );
	void UpdateLabel( vtkIdType node );
	
	vtkTree* Hierarchy;
	std::map<vtkIdType,double> SmoothnessScalars;
	std::map<vtkIdType,int> LeafMap;
	std::map<vtkIdType,int> BranchMap;
	int		NumLeaves;
	int		NumBranches;
	int		NumNodes;
	int		NumEdges;

	int NumberOfIterations;
	float CC;
	float StepSize;
	int VolumeSize;
	int VX, VY, VZ;
	
	std::map<vtkIdType,int> InputDataPortMapping;
	std::map<int,vtkIdType> BackwardsInputDataPortMapping;
	int FirstUnusedDataPort;
	std::map<vtkIdType,int> InputSmoothnessPortMapping;
	std::map<int,vtkIdType> BackwardsInputSmoothnessPortMapping;
	int FirstUnusedSmoothnessPort;

	//pointers to variable structures, easier to keep as part of the class definition
	std::list<float*> CPUBuffersAcquired;
	std::list<int> CPUBuffersSize;
	int TotalNumberOfBuffers;
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

private:
	vtkHierarchicalMaxFlowSegmentation operator=(const vtkHierarchicalMaxFlowSegmentation&){}
	vtkHierarchicalMaxFlowSegmentation(const vtkHierarchicalMaxFlowSegmentation&){}

};

#endif
