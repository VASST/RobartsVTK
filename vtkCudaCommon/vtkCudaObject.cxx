#include "vtkCudaObject.h"
#include "vtkObjectFactory.h"
#include "cuda_runtime_api.h"

vtkStandardNewMacro(vtkCudaObject);

vtkCudaObject::vtkCudaObject(){
	DeviceNumber = -1;
	DeviceManager = vtkCudaDeviceManager::Singleton();
}

vtkCudaObject::~vtkCudaObject(){
	//synchronize remainder of stream
}

void vtkCudaObject::SetDevice( int d ){
	int numberOfDevices = this->DeviceManager->GetNumberOfDevices();

	if( d < 0 || d >= numberOfDevices ){
		vtkErrorMacro(<<"Device selected does not exist.");
		return;
	}

	//set up a purely new device
	if( this->DeviceNumber == -1 ){
		this->DeviceNumber = d;
		bool result = this->DeviceManager->GetDevice(this, this->DeviceNumber);
		if(result){
			vtkErrorMacro(<<"Device selected cannot be retrieved.");
			this->DeviceNumber = -1;
			return;
		}
		result = this->DeviceManager->GetStream(this, &(this->DeviceStream), this->DeviceNumber );
		if(result){
			vtkErrorMacro(<<"Device selected cannot be retrieved.");
			this->DeviceManager->ReturnDevice(this, this->DeviceNumber );
			this->DeviceNumber = -1;
			return;
		}

	//if we are currently using that device, don't change anything
	}else if(this->DeviceNumber == d){
		return;

	//finish all device business and set up a new device
	}else{
		this->DeviceManager->ReturnStream(this, &(this->DeviceStream), this->DeviceNumber );
		this->DeviceManager->ReturnDevice(this, this->DeviceNumber );
		this->DeviceNumber = d;
		bool result = this->DeviceManager->GetDevice(this, this->DeviceNumber);
		if(result){
			vtkErrorMacro(<<"Device selected cannot be retrieved.");
			this->DeviceNumber = -1;
			return;
		}
		result = this->DeviceManager->GetStream(this, &(this->DeviceStream), this->DeviceNumber );
		if(result){
			vtkErrorMacro(<<"Device selected cannot be retrieved.");
			this->DeviceManager->ReturnDevice(this, this->DeviceNumber );
			this->DeviceNumber = -1;
			return;
		}
	}
}

void vtkCudaObject::CallKernel( kernelFunction* k, dim3 grid, dim3 threads, bool synchronized ){

}

void vtkCudaObject::CallSyncThreads( ){
	if( this->DeviceNumber == -1 ){
		vtkErrorMacro(<<"No device set selected does not exist.");
		return;
	}
	this->DeviceManager->SynchronizeStream(&(this->DeviceStream));
}
