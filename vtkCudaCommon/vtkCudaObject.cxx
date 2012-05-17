#include "vtkCudaObject.h"
#include "vtkObjectFactory.h"
#include "cuda_runtime_api.h"

void errorOut(vtkCudaObject* self, const char* message){
	if (vtkObject::GetGlobalWarningDisplay())                    
		 {                                                          
		 vtkOStreamWrapper::EndlType endl;                          
		 vtkOStreamWrapper::UseEndl(endl);                          
		 vtkOStrStreamWrapper vtkmsg;                               
		 vtkmsg << "ERROR: In " __FILE__ ", line " << __LINE__      
				<< "\n" << "vtkCudaObject" << " (" << self     
				<< "): " << message << "\n\n";                                          
		 vtkmsg.rdbuf()->freeze(0); vtkObject::BreakOnError();      
		 }                                                          
}

vtkCudaObject::vtkCudaObject(){
	this->DeviceManager = vtkCudaDeviceManager::Singleton();
	this->DeviceStream = 0;
	this->DeviceNumber = 0;
	bool result = this->DeviceManager->GetDevice(this, this->DeviceNumber);
	if(result){
		errorOut(this,"Device selected cannot be retrieved.");
		this->DeviceNumber = -1;
		return;
	}
	result = this->DeviceManager->GetStream(this, &(this->DeviceStream), this->DeviceNumber );
	if(result){
		errorOut(this,"Device selected cannot be retrieved.");
		this->DeviceManager->ReturnDevice(this, this->DeviceNumber );
		this->DeviceNumber = -1;
		return;
	}
}

vtkCudaObject::~vtkCudaObject(){
	//synchronize remainder of stream and return control of the device
	this->CallSyncThreads();
	this->DeviceManager->ReturnDevice( this, this->DeviceNumber );
}

void vtkCudaObject::SetDevice( int d, int withData ){
	int numberOfDevices = this->DeviceManager->GetNumberOfDevices();

	if( d < 0 || d >= numberOfDevices ){
		errorOut(this,"Device selected does not exist.");
		return;
	}

	//set up a purely new device
	if( this->DeviceNumber == -1 ){
		this->DeviceNumber = d;
		bool result = this->DeviceManager->GetDevice(this, this->DeviceNumber);
		if(result){
			errorOut(this,"Device selected cannot be retrieved.");
			this->DeviceNumber = -1;
			return;
		}
		result = this->DeviceManager->GetStream(this, &(this->DeviceStream), this->DeviceNumber );
		if(result){
			errorOut(this,"Device selected cannot be retrieved.");
			this->DeviceManager->ReturnDevice(this, this->DeviceNumber );
			this->DeviceNumber = -1;
			return;
		}
		this->Reinitialize(withData);

	//if we are currently using that device, don't change anything
	}else if(this->DeviceNumber == d){
		return;

	//finish all device business and set up a new device
	}else{
		this->Deinitialize(withData);
		this->DeviceManager->ReturnStream(this, this->DeviceStream, this->DeviceNumber );
		this->DeviceStream = 0;
		this->DeviceManager->ReturnDevice(this, this->DeviceNumber );
		this->DeviceNumber = d;
		bool result = this->DeviceManager->GetDevice(this, this->DeviceNumber);
		if(result){
			errorOut(this,"Device selected cannot be retrieved.");
			this->DeviceNumber = -1;
			return;
		}
		result = this->DeviceManager->GetStream(this, &(this->DeviceStream), this->DeviceNumber );
		if(result){
			errorOut(this,"Device selected cannot be retrieved.");
			this->DeviceManager->ReturnDevice(this, this->DeviceNumber );
			this->DeviceNumber = -1;
			return;
		}
		this->Reinitialize(withData);
	}
}

void vtkCudaObject::ReserveGPU( ){
	if( this->DeviceNumber == -1 ){
		errorOut(this,"No device set selected does not exist.");
		return;
	}
	if(this->DeviceManager->ReserveGPU(this->DeviceStream)){
		errorOut(this,"Error REserving GPU");
		return;
	}
}

void vtkCudaObject::CallSyncThreads( ){
	if( this->DeviceNumber == -1 ){
		errorOut(this,"No device set selected does not exist.");
		return;
	}
	if(this->DeviceManager->SynchronizeStream(this->DeviceStream)){
		errorOut(this,"Error Synchronizing Streams");
		return;
	}
}

cudaStream_t* vtkCudaObject::GetStream( ){
	return this->DeviceStream;
}

void vtkCudaObject::ReplicateObject( vtkCudaObject* object, int withData ){
	int oldDeviceNumber = this->DeviceNumber;
	this->SetDevice( object->DeviceNumber, withData );
	if(	this->DeviceStream != object->DeviceStream ){
		this->CallSyncThreads();
		this->DeviceManager->ReturnStream(this, this->DeviceStream, oldDeviceNumber);
		this->DeviceStream = 0;
		this->DeviceStream = object->DeviceStream;
		this->DeviceManager->GetStream( this, &(object->DeviceStream), object->DeviceNumber );
	}
}