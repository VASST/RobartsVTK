// Type
#include "vtkCudaDualImageVolumeMapper.h"
#include "vtkObjectFactory.h"

// Volume
#include "vtkVolume.h"
#include "vtkImageData.h"

// Rendering
#include "vtkCamera.h"
#include "vtkRenderer.h"

// VTKCUDA
#include "CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.h"


vtkStandardNewMacro(vtkCudaDualImageVolumeMapper);

vtkMutexLock* vtkCudaDualImageVolumeMapper::tfLock = 0;
vtkCudaDualImageVolumeMapper::vtkCudaDualImageVolumeMapper()
{
	this->transferFunctionInfoHandler = vtkCudaDualImageTransferFunctionInformationHandler::New();
	if( this->tfLock == 0 ) this->tfLock = vtkMutexLock::New();
	else this->tfLock->Register(this);
	this->Reinitialize();
}

vtkCudaDualImageVolumeMapper::~vtkCudaDualImageVolumeMapper(){
	this->Deinitialize();
	this->transferFunctionInfoHandler->Delete();
	this->tfLock->UnRegister(this);
}

void vtkCudaDualImageVolumeMapper::Deinitialize(int withData){
	this->vtkCudaVolumeMapper::Deinitialize(withData);
	this->ReserveGPU();
	CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_clearImageArray(this->GetStream());
}

void vtkCudaDualImageVolumeMapper::Reinitialize(int withData){
	this->vtkCudaVolumeMapper::Reinitialize(withData);
	this->transferFunctionInfoHandler->ReplicateObject(this, withData);
	this->ReserveGPU();
	CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_initImageArray(this->GetStream());
	CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_changeFrame(this->currFrame, this->GetStream());
}

void vtkCudaDualImageVolumeMapper::SetInputInternal(vtkImageData * input, int index){
	
	if( input->GetNumberOfScalarComponents() != 2 ){
		vtkErrorMacro(<<"Input must have 2 components.");
		return;
	}

	//convert data to float
	const cudaVolumeInformation& VolumeInfo = this->VolumeInfoHandler->GetVolumeInfo();
	float* buffer = new float[2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z];
	if(input->GetScalarType() == VTK_CHAR){
		char* tempPtr = (char*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_UNSIGNED_CHAR){
		unsigned char* tempPtr = (unsigned char*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_SIGNED_CHAR){
		signed char* tempPtr = (signed char*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_INT){
		int* tempPtr = (int*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_UNSIGNED_INT){
		unsigned int* tempPtr = (unsigned int*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_SHORT){
		short* tempPtr = (short*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_UNSIGNED_SHORT){
		unsigned short* tempPtr = (unsigned short*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_LONG){
		long* tempPtr = (long*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_UNSIGNED_LONG){
		unsigned long* tempPtr = (unsigned long*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_FLOAT){
		delete buffer;
		buffer = (float*) input->GetScalarPointer();
	}else if(input->GetScalarType() == VTK_DOUBLE){
		double* tempPtr = (double*) input->GetScalarPointer();
		for(int i = 0; i < 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else{
		vtkErrorMacro(<<"Input cannot be of that type.");
		return;
	}

	//load data onto the GPU and clean up the CPU
	if(!this->erroredOut){
		this->ReserveGPU();
		this->erroredOut = !CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_loadImageInfo( buffer, VolumeInfoHandler->GetVolumeInfo(), index, this->GetStream());
	}
	if(input->GetScalarType() != VTK_FLOAT) delete buffer;

	//inform transfer function handler of the data
	this->transferFunctionInfoHandler->SetInputData(input,index);
}

void vtkCudaDualImageVolumeMapper::ChangeFrameInternal(unsigned int frame){
	if(!this->erroredOut){
		this->ReserveGPU();
		this->erroredOut = !CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_changeFrame(frame, this->GetStream());
	}
}

void vtkCudaDualImageVolumeMapper::InternalRender (	vtkRenderer* ren, vtkVolume* vol,
												const cudaRendererInformation& rendererInfo,
												const cudaVolumeInformation& volumeInfo,
												const cudaOutputImageInformation& outputInfo ){
	//handle the transfer function changes
	this->transferFunctionInfoHandler->Update();

	//perform the render
	this->tfLock->Lock();
	this->ReserveGPU();
	this->erroredOut = !CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
		this->transferFunctionInfoHandler->GetTransferFunctionInfo(), this->GetStream() );
	this->tfLock->Unlock();

}

void vtkCudaDualImageVolumeMapper::ClearInputInternal(){
	this->ReserveGPU();
	CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_clearImageArray(this->GetStream());
}

//give the function to the transfer function handler
void vtkCudaDualImageVolumeMapper::SetFunction(vtkCuda2DTransferFunction* funct){
	this->transferFunctionInfoHandler->SetTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCudaDualImageVolumeMapper::GetFunction(){
	return this->transferFunctionInfoHandler->GetTransferFunction();
}