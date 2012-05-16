#ifndef __VTKCUDADEVICEMANAGER_H__
#define __VTKCUDADEVICEMANAGER_H__

#include "vtkObject.h"
#include "vtkMutexLock.h"
#include "vector_types.h"
#include <map>

typedef void (* kernelFunction ) ( void* );

class vtkCudaObject;

class vtkCudaDeviceManager : public vtkObject
{
public:
	static vtkCudaDeviceManager* Singleton(){ return &(vtkCudaDeviceManager::singletonManager); };

	int GetNumberOfDevices();
	bool GetDevice(vtkCudaObject* caller, int device);
	bool ReturnDevice(vtkCudaObject* caller, int device);
	bool GetStream(vtkCudaObject* caller, cudaStream_t* stream, int device);
	bool ReturnStream(vtkCudaObject* caller, cudaStream_t* stream, int device);

	bool SynchronizeStream( cudaStream_t* stream );
	bool ReserveGPU( cudaStream_t* stream );
	
	int QueryDeviceForObject( vtkCudaObject* object );
	int QueryDeviceForStream( cudaStream_t* stream );

protected:

private:
	vtkCudaDeviceManager();
	~vtkCudaDeviceManager();
	vtkCudaDeviceManager operator=(const vtkCudaDeviceManager&); /**< not implemented */
	vtkCudaDeviceManager(const vtkCudaDeviceManager&); /**< not implemented */
	
	std::map<cudaStream_t*,int> StreamToDeviceMap;
	std::multimap<vtkCudaObject*,int> ObjectToDeviceMap;

	static vtkCudaDeviceManager singletonManager;

	vtkMutexLock* regularLock;

};
#endif /* __VTKCUDADEVICEMANAGER_H__ */
