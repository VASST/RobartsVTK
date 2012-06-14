/** @file vtkCudaObject.h
 *
 *  @brief Header file defining an abstract class which uses CUDA
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on June 12, 2012
 *
 *  @note Interacts primarily with the vtkCudaDeviceManager
 */

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

	void SetDevice( int d, int withData = 0 );
	int GetDevice(){ return this->DeviceNumber; };
	
	void ReserveGPU( );
	void CallSyncThreads( );
	cudaStream_t* vtkCudaObject::GetStream( );

	void ReplicateObject( vtkCudaObject* object, int withData = 0  );

protected:
	vtkCudaObject();
	~vtkCudaObject();
	
	virtual void Reinitialize(int withData = 0) = 0;
	virtual void Deinitialize(int withData = 0) = 0;

private:

	int DeviceNumber;
	cudaStream_t* DeviceStream;

	vtkCudaDeviceManager* DeviceManager;

	int withDataStatus;

};
#endif /* __VTKCUDAOBJECT_H__ */
