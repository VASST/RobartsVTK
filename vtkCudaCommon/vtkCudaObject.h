#ifndef __VTKCUDAOBJECT_H__
#define __VTKCUDAOBJECT_H__

#include "vtkObject.h"
#include "vtkCudaDeviceManager.h"
#include "vector_types.h"
#include "cuda.h"

typedef void (* kernelFunction ) ( void* );

class vtkCudaObject : public vtkObject
{
public:
	static vtkCudaObject* New();

	void SetDevice( int d );
	int GetDevice(){ return this->DeviceNumber; };
	
	void CallKernel( kernelFunction* k, dim3 grid, dim3 threads, bool synchronized );
	void CallSyncThreads( );

protected:
	vtkCudaObject();
	~vtkCudaObject();

private:
	vtkCudaObject operator=(const vtkCudaObject&); /**< not implemented */
	vtkCudaObject(const vtkCudaObject&); /**< not implemented */

	int DeviceNumber;
	cudaStream_t DeviceStream;

	vtkCudaDeviceManager* DeviceManager;

};
#endif /* __VTKCUDAOBJECT_H__ */
