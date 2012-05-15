#include "vtkCudaDeviceManager.h"
#include "vtkObjectFactory.h"
#include "cuda_runtime_api.h"
#include "vtkCudaObject.h"

vtkCudaDeviceManager::vtkCudaDeviceManager(){

	//create the locks
	this->regularLock = vtkMutexLock::New();
	int n = this->GetNumberOfDevices();
	for(int i = 0; i < n; i++ )
		this->deviceLocks.push_back( vtkMutexLock::New() );

}

vtkCudaDeviceManager::~vtkCudaDeviceManager(){
	//synchronize and end all streams

	//decommission the devices

	//clean up variables
	this->regularLock->Delete();
	for(int i = 0; i < this->deviceLocks.size(); i++ )
		this->deviceLocks.at(i)->Delete();
}

int vtkCudaDeviceManager::GetNumberOfDevices(){
	
	int numberOfDevices = 0;
	cudaError_t result = cudaGetDeviceCount (&numberOfDevices);
	
	if( result != 0 ){
		vtkErrorMacro(<<"Catostrophic CUDA error - cannot count number of devices.");
		return -1;
	}
	return numberOfDevices;

}

bool vtkCudaDeviceManager::GetDevice(vtkCudaObject* caller, int device){
	return true;
}

bool vtkCudaDeviceManager::ReturnDevice(vtkCudaObject* caller, int device){
	return true;

}

bool vtkCudaDeviceManager::GetStream(vtkCudaObject* caller, cudaStream_t* stream, int device){
	return true;

}

bool vtkCudaDeviceManager::ReturnStream(vtkCudaObject* caller, cudaStream_t* stream, int device){
	return true;

}

bool vtkCudaDeviceManager::SynchronizeStream( cudaStream_t* stream ){
	
	//find mapped result and device
	this->regularLock->Lock();
	if( this->StreamToDeviceMap.count(stream) != 1 ){
		vtkErrorMacro(<<"Cannot synchronize unused stream.");
		this->regularLock->Unlock();
		return true;
	}
	int device = this->StreamToDeviceMap.at(stream);
	this->regularLock->Unlock();

	//synchronize the stream and return the success value
	cudaSetDevice( device );
	cudaStreamSynchronize( *stream );
	return cudaGetLastError() != cudaSuccess;

}