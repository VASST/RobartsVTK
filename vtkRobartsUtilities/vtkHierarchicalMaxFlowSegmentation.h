#ifndef __VTKHIERARCHICALMAXFLOWSEGMENTATION_H__
#define __VTKHIERARCHICALMAXFLOWSEGMENTATION_H__

#include "vtkHierarchicalMaxFlowSegmentation.h"
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
	
	vtkDataObject* GetInput(int idx);
	void SetInput(int idx, vtkDataObject *input);

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
	virtual int FillInputPortInformation(int i, vtkInformation* info);

protected:
	vtkHierarchicalMaxFlowSegmentation();
	virtual ~vtkHierarchicalMaxFlowSegmentation();

private:
	vtkHierarchicalMaxFlowSegmentation operator=(const vtkHierarchicalMaxFlowSegmentation&){}
	vtkHierarchicalMaxFlowSegmentation(const vtkHierarchicalMaxFlowSegmentation&){}
	
	vtkTree* Hierarchy;
	std::map<vtkIdType,double> SmoothnessScalars;
};

#endif