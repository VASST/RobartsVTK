#include "vtkCudaDeviceManager.h"
#include "vtkObjectFactory.h"
#include "cuda_runtime_api.h"
#include "vtkCudaObject.h"
#include <set>

vtkCudaDeviceManager::vtkCudaDeviceManager(){

	//create the locks
	this->regularLock = vtkMutexLock::New();
	int n = this->GetNumberOfDevices();

}

vtkCudaDeviceManager::~vtkCudaDeviceManager(){

	//define a list to collect the used device ID's in
	std::set<int> devicesInUse;
	this->regularLock->Lock();


	//synchronize and end all streams
	for( std::map<cudaStream_t*,int>::iterator it = this->StreamToDeviceMap.begin();
		 it != this->StreamToDeviceMap.end(); it++ ){
		this->SynchronizeStream( (*it).first );
		cudaStreamDestroy( *(it->first) );
		devicesInUse.insert( it->second );
	}
	this->StreamToDeviceMap.clear();

	//decommission the devices
	for( std::set<int>::iterator it = devicesInUse.begin();
		 it != devicesInUse.end(); it++ ){
		cudaSetDevice( *it );
		cudaDeviceReset( );
	}

	//clean up variables
	this->regularLock->Unlock();
	this->regularLock->Delete();

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
	if( device < 0 || device >= this->GetNumberOfDevices() ){
		vtkErrorMacro(<<"Invalid device identifier.");
		return true;
	}
	
	//remove that part of the mapping
	this->regularLock->Lock();
	this->ObjectToDeviceMap.insert( std::pair<vtkCudaObject*,int>(caller, device) );
	this->regularLock->Unlock();
	return false;
}

bool vtkCudaDeviceManager::ReturnDevice(vtkCudaObject* caller, int device){
	this->regularLock->Lock();

	//find if that is a valid mapping
	bool found = false;
	std::multimap<vtkCudaObject*,int>::iterator it = this->ObjectToDeviceMap.begin();
	for( ; it != this->ObjectToDeviceMap.end(); it++ ){
		if( it->first == caller && it->second == device ){
			found = true;
			break;
		}
	}
	if( !found ){
		vtkErrorMacro(<<"Could not locate supplied caller-device pair.");
		this->regularLock->Unlock();
		return true;
	}

	//remove that part of the mapping
	this->ObjectToDeviceMap.erase(it);
	this->regularLock->Unlock();
	return false;
}

bool vtkCudaDeviceManager::GetStream(vtkCudaObject* caller, cudaStream_t* stream, int device){
	if( device < 0 || device >= this->GetNumberOfDevices() ){
		vtkErrorMacro(<<"Invalid device identifier.");
		return true;
	}

	//create the new stream and mapping
	this->regularLock->Lock();
	cudaSetDevice(device);
	cudaStreamCreate( stream );
	this->StreamToDeviceMap.insert( std::pair<cudaStream_t*,int>(stream,device) );
	this->regularLock->Unlock();

	return false;

}

bool vtkCudaDeviceManager::ReturnStream(vtkCudaObject* caller, cudaStream_t* stream, int device){
	this->regularLock->Lock();

	//find if that is a valid mapping
	bool found = false;
	std::map<cudaStream_t*,int>::iterator it = this->StreamToDeviceMap.begin();
	for( ; it != this->StreamToDeviceMap.end(); it++ ){
		if( it->first == stream && it->second == device ){
			found = true;
			break;
		}
	}
	if( !found ){
		vtkErrorMacro(<<"Could not locate supplied caller-device pair.");
		this->regularLock->Unlock();
		return true;
	}
	
	//remove that part of the mapping
	this->StreamToDeviceMap.erase(it);
	this->regularLock->Unlock();
	return false;

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
	int oldDevice = -1;
	cudaGetDevice( &oldDevice );
	cudaSetDevice( device );
	cudaStreamSynchronize( *stream );
	cudaSetDevice( oldDevice );
	return cudaGetLastError() != cudaSuccess;

}

bool vtkCudaDeviceManager::ReserveGPU( cudaStream_t* stream ){
	
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
	return cudaGetLastError() != cudaSuccess;

}

int vtkCudaDeviceManager::QueryDeviceForObject( vtkCudaObject* object ){
	this->regularLock->Lock();
	int device = -1;
	if( this->ObjectToDeviceMap.count(object) == 1 )
		device = this->ObjectToDeviceMap.find(object)->second;
	else
		vtkErrorMacro(<<"No unique mapping exists.");
	this->regularLock->Unlock();
	return device;
}

int vtkCudaDeviceManager::QueryDeviceForStream( cudaStream_t* stream ){
	this->regularLock->Lock();
	int device = -1;
	if( this->StreamToDeviceMap.count(stream) == 1 )
		device = this->StreamToDeviceMap.at(stream);
	else
		vtkErrorMacro(<<"No mapping exists.");
	this->regularLock->Unlock();
	return device;
}