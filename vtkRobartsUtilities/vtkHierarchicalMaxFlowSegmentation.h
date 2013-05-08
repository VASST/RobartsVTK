/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkHierarchicalMaxFlowSegmentation.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

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

#include <limits.h>

//INPUT PORT DESCRIPTION

//OUTPUT PORT DESCRIPTION

class vtkHierarchicalMaxFlowSegmentation : public vtkImageAlgorithm 
{
public:
	vtkTypeMacro( vtkHierarchicalMaxFlowSegmentation, vtkImageAlgorithm );

	static vtkHierarchicalMaxFlowSegmentation *New();

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
	
	//Get and Set the labeling constant, CC, of the algorithm
	vtkSetClampMacro(CC,float,0.0f,1.0f);
	vtkGetMacro(CC,float);

	//Get and Set the step size of the algorithm for updating spatial flows
	vtkSetClampMacro(StepSize,float,0.0f,1.0f);
	vtkGetMacro(StepSize,float);

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
	vtkHierarchicalMaxFlowSegmentation();
	virtual ~vtkHierarchicalMaxFlowSegmentation();

private:
	vtkHierarchicalMaxFlowSegmentation operator=(const vtkHierarchicalMaxFlowSegmentation&){}
	vtkHierarchicalMaxFlowSegmentation(const vtkHierarchicalMaxFlowSegmentation&){}

	int CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges );
	void PropogateLabels( vtkIdType currNode, float** branchLabels, float** leafLabels, int size );
	void PropogateFlows( vtkIdType currNode, float* sourceSinkFlow, float** branchSinkFlows, float** leafSinkFlows,
											 float** branchIncFlows,
											 float** branchDivFlows, float** leafDivFlows,
											 float** branchLabels, float** leafLabels, int size );
	
	vtkTree* Hierarchy;
	std::map<vtkIdType,double> SmoothnessScalars;
	std::map<vtkIdType,int> OutputPortMapping;
	std::map<vtkIdType,int> IntermediateBufferMapping;

	int NumberOfIterations;
	float CC;
	float StepSize;
	
	std::map<vtkIdType,int> InputPortMapping;
	std::map<int,vtkIdType> BackwardsInputPortMapping;
	int FirstUnusedPort;

};

#endif
