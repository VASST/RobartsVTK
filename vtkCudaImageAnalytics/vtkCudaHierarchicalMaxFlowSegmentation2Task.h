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
	vtkCudaHierarchicalMaxFlowSegmentation2* const Parent;
	Task( vtkCudaHierarchicalMaxFlowSegmentation2* parent )
		: Parent(parent) {}
	virtual void Perform() = 0;
	virtual int CalcWeight(vtkCudaHierarchicalMaxFlowSegmentation2::Worker* w) = 0;
	virtual bool CanDo() = 0;
	virtual bool Conflicted() = 0;
};


#endif //__VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION2TASK_H__