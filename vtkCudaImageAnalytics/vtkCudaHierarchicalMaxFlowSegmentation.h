#ifndef __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION_H__
#define __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION_H__

#include "vtkCudaObject.h"

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

#include <limits.h>
#include <float.h>

//INPUT PORT DESCRIPTION

//OUTPUT PORT DESCRIPTION

class vtkCudaHierarchicalMaxFlowSegmentation : public vtkHierarchicalMaxFlowSegmentation, public vtkCudaObject
{
public:
	vtkTypeMacro( vtkCudaHierarchicalMaxFlowSegmentation, vtkHierarchicalMaxFlowSegmentation );

	static vtkCudaHierarchicalMaxFlowSegmentation *New();
	
	// Description:
	// If the subclass does not define an Execute method, then the task
	// will be broken up, multiple threads will be spawned, and each thread
	// will call this method. It is public so that the thread functions
	// can call this method.
	virtual int RequestData(vtkInformation *request, 
							 vtkInformationVector **inputVector, 
							 vtkInformationVector *outputVector);

protected:
	vtkCudaHierarchicalMaxFlowSegmentation();
	virtual ~vtkCudaHierarchicalMaxFlowSegmentation();

private:
	vtkCudaHierarchicalMaxFlowSegmentation operator=(const vtkCudaHierarchicalMaxFlowSegmentation&){}
	vtkCudaHierarchicalMaxFlowSegmentation(const vtkCudaHierarchicalMaxFlowSegmentation&){}

	void Reinitialize(int withData);
	void Deinitialize(int withData);

	double	MaxGPUUsage;
	void PropogateLabels( vtkIdType currNode );
	void SolveMaxFlow( vtkIdType currNode, int* timeStep );
	void UpdateLabel( vtkIdType node, int* timeStep );

	//Mappings for CPU-GPU buffer sharing
	void ReturnBufferGPU2CPU(float* CPUBuffer, float* GPUBuffer);
	void MoveBufferCPU2GPU(float* CPUBuffer, float* GPUBuffer);
	void GetGPUBuffersV2(int reference);
	std::map<float*,float*> CPU2GPUMap;
	std::map<float*,float*> GPU2CPUMap;
	std::set<float*> CPUInUse;
	std::list<float*> UnusedGPUBuffers;
	std::set<float*> ReadOnly;
	std::set<float*> NoCopyBack;

	//Prioirty structure
	class CircListNode;
	std::map< float*, CircListNode* > PrioritySet;
	std::map< float*, int > PrioritySetNumUses;
	void ClearBufferOrdering( vtkIdType currNode );
	void SimulateIterationForBufferOrdering( vtkIdType currNode, int* reference );
	void SimulateIterationForBufferOrderingUpdateLabelStep( vtkIdType currNode, int* reference );
	void UpdateBufferOrderingAt( float* buffer, int reference );
	void DeallocatePrioritySet();

	int		NumMemCpies;
	int		NumKernelRuns;
};

#endif
