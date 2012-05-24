// Type
#include "vtkCuda2DInExLogicVolumeMapper.h"
#include "vtkCudaRendererInformationHandler.h"
#include "vtkObjectFactory.h"

// Volume
#include "vtkVolume.h"
#include "vtkImageData.h"

// Rendering
#include "vtkRenderer.h"
#include "vtkPlanes.h"

// VTKCUDA
#include "CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.h"


vtkStandardNewMacro(vtkCuda2DInExLogicVolumeMapper);

vtkMutexLock* vtkCuda2DInExLogicVolumeMapper::tfLock = vtkMutexLock::New();
vtkCuda2DInExLogicVolumeMapper::vtkCuda2DInExLogicVolumeMapper(){
	this->transferFunctionInfoHandler = vtkCuda2DInExLogicTransferFunctionInformationHandler::New();
	this->Reinitialize();
}

vtkCuda2DInExLogicVolumeMapper::~vtkCuda2DInExLogicVolumeMapper(){
	this->Deinitialize();
	this->transferFunctionInfoHandler->Delete();
}

void vtkCuda2DInExLogicVolumeMapper::Reinitialize(int withData){
	this->vtkCudaVolumeMapper::Reinitialize(withData);
	this->transferFunctionInfoHandler->ReplicateObject(this, withData);
	this->ReserveGPU();
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_initImageArray(this->GetStream());
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_changeFrame(this->currFrame, this->GetStream());
}

void vtkCuda2DInExLogicVolumeMapper::Deinitialize(int withData){
	this->vtkCudaVolumeMapper::Deinitialize(withData);
	this->ReserveGPU();
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_clearImageArray(this->GetStream());
}

void vtkCuda2DInExLogicVolumeMapper::SetInputInternal(vtkImageData * input, int index){
	
	//convert data to float
	const cudaVolumeInformation& VolumeInfo = this->VolumeInfoHandler->GetVolumeInfo();
	float* buffer = new float[VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z];
	if(input->GetScalarType() == VTK_CHAR){
		char* tempPtr = (char*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_UNSIGNED_CHAR){
		unsigned char* tempPtr = (unsigned char*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_SIGNED_CHAR){
		signed char* tempPtr = (signed char*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_INT){
		int* tempPtr = (int*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_UNSIGNED_INT){
		unsigned int* tempPtr = (unsigned int*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_SHORT){
		short* tempPtr = (short*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_UNSIGNED_SHORT){
		unsigned short* tempPtr = (unsigned short*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_LONG){
		long* tempPtr = (long*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_UNSIGNED_LONG){
		unsigned long* tempPtr = (unsigned long*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else if(input->GetScalarType() == VTK_FLOAT){
		delete buffer;
		buffer = (float*) input->GetScalarPointer();
	}else if(input->GetScalarType() == VTK_DOUBLE){
		double* tempPtr = (double*) input->GetScalarPointer();
		for(int i = 0; i < VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z; i++)
			buffer[i] = (float)(tempPtr[i]);
	}else{
		vtkErrorMacro(<<"Input cannot be of that type.");
		return;
	}

	//load data onto the GPU and clean up the CPU
	this->ReserveGPU();
	this->erroredOut = !CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_loadImageInfo( buffer, VolumeInfoHandler->GetVolumeInfo(), index, this->GetStream());
	if(input->GetScalarType() != VTK_FLOAT) delete buffer;

	//inform transfer function handler of the data
	this->transferFunctionInfoHandler->SetInputData(input,index);
}

void vtkCuda2DInExLogicVolumeMapper::ChangeFrameInternal(unsigned int frame){
	if(!this->erroredOut){
		this->ReserveGPU();
		this->erroredOut = !CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_changeFrame(frame, this->GetStream());
	}
}

void vtkCuda2DInExLogicVolumeMapper::InternalRender (	vtkRenderer* ren, vtkVolume* vol,
												const cudaRendererInformation& rendererInfo,
												const cudaVolumeInformation& volumeInfo,
												const cudaOutputImageInformation& outputInfo ){
	//handle the transfer function changes
	this->transferFunctionInfoHandler->Update();

	//perform the render
	this->tfLock->Lock();
	this->ReserveGPU();
	this->erroredOut = !CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
		this->transferFunctionInfoHandler->GetTransferFunctionInfo(), this->GetStream() );
	this->tfLock->Unlock();

}

void vtkCuda2DInExLogicVolumeMapper::ClearInputInternal(){
	this->ReserveGPU();
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_clearImageArray(this->GetStream());
}

//give the function to the transfer function handler
void vtkCuda2DInExLogicVolumeMapper::SetVisualizationFunction(vtkCuda2DTransferFunction* funct){
	this->transferFunctionInfoHandler->SetVisualizationTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCuda2DInExLogicVolumeMapper::GetVisualizationFunction(){
	return this->transferFunctionInfoHandler->GetVisualizationTransferFunction();
}

//give the function to the transfer function handler
void vtkCuda2DInExLogicVolumeMapper::SetInExLogicFunction(vtkCuda2DTransferFunction* funct){
	this->transferFunctionInfoHandler->SetInExLogicTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCuda2DInExLogicVolumeMapper::GetInExLogicFunction(){
	return this->transferFunctionInfoHandler->GetInExLogicTransferFunction();
}