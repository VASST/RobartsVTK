#ifndef __VTKCUDAOBJECT_H__
#define __VTKCUDAOBJECT_H__

#include "vtkObject.h"
#include "vtkCudaDeviceManager.h"
#include "vector_types.h"
#include "cuda.h"

class vtkCudaObject
{
public:
	static vtkCudaObject* New();

	void SetDevice( int d );
	int GetDevice(){ return this->DeviceNumber; };
	
	void ReserveGPU( );
	void CallSyncThreads( );
	cudaStream_t* vtkCudaObject::GetStream( );

	void ReplicateObject( vtkCudaObject* object );

protected:
	vtkCudaObject();
	~vtkCudaObject();
	
	virtual void Reinitialize(int withData = 0) = 0;
	virtual void Deinitialize(int withData = 0) = 0;

private:

	int DeviceNumber;
	cudaStream_t* DeviceStream;

	vtkCudaDeviceManager* DeviceManager;

};
#endif /* __VTKCUDAOBJECT_H__ */
