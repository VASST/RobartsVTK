// Type
#include "vtkCuda2DVolumeMapper.h"
#include "vtkObjectFactory.h"

// Volume
#include "vtkVolume.h"
#include "vtkImageData.h"

// Rendering
#include "vtkCamera.h"
#include "vtkRenderer.h"

// VTKCUDA
#include "CUDA_vtkCuda2DVolumeMapper_renderAlgo.h"


vtkStandardNewMacro(vtkCuda2DVolumeMapper);

vtkMutexLock* vtkCuda2DVolumeMapper::tfLock = 0;
vtkCuda2DVolumeMapper::vtkCuda2DVolumeMapper()
{
	this->transferFunctionInfoHandler = vtkCuda2DTransferFunctionInformationHandler::New();
	if( this->tfLock == 0 ) this->tfLock = vtkMutexLock::New();
	else this->tfLock->Register(this);
	this->Reinitialize();
}

vtkCuda2DVolumeMapper::~vtkCuda2DVolumeMapper(){
	this->Deinitialize();
	this->transferFunctionInfoHandler->Delete();
	this->tfLock->UnRegister(this);
}

void vtkCuda2DVolumeMapper::Deinitialize(int withData){
	this->vtkCudaVolumeMapper::Deinitialize(withData);
	this->ReserveGPU();
	
	for( int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++ ){
		if( this->SourceData[i] )
			CUDA_vtkCuda2DVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]), this->GetStream());
	}
}

void vtkCuda2DVolumeMapper::Reinitialize(int withData){
	this->vtkCudaVolumeMapper::Reinitialize(withData);
	this->transferFunctionInfoHandler->ReplicateObject(this, withData);
	for( int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++ ) this->SourceData[i] = 0;
	this->ReserveGPU();
	CUDA_vtkCuda2DVolumeMapper_renderAlgo_changeFrame(this->SourceData[this->currFrame], this->GetStream());
}

void vtkCuda2DVolumeMapper::SetInputInternal(vtkImageData * input, int index){
	
	if( input->GetNumberOfScalarComponents() != 1 ){
		vtkErrorMacro(<<"Input must have 1 components.");
		return;
	}

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
	if(!this->erroredOut){
		this->ReserveGPU();
		this->erroredOut = !CUDA_vtkCuda2DVolumeMapper_renderAlgo_loadImageInfo( buffer, VolumeInfoHandler->GetVolumeInfo(),
			&(this->SourceData[index]), this->GetStream());
	}
	if(input->GetScalarType() != VTK_FLOAT) delete buffer;

	//inform transfer function handler of the data
	this->transferFunctionInfoHandler->SetInputData(input,index);
}

void vtkCuda2DVolumeMapper::ChangeFrameInternal(int frame){
}

void vtkCuda2DVolumeMapper::InternalRender (	vtkRenderer* ren, vtkVolume* vol,
												const cudaRendererInformation& rendererInfo,
												const cudaVolumeInformation& volumeInfo,
												const cudaOutputImageInformation& outputInfo ){
	//handle the transfer function changes
	this->transferFunctionInfoHandler->Update();

	//perform the render
	this->tfLock->Lock();
	this->ReserveGPU();
	this->erroredOut = !CUDA_vtkCuda2DVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
		this->transferFunctionInfoHandler->GetTransferFunctionInfo(), this->SourceData[this->currFrame], this->GetStream() );
	this->tfLock->Unlock();

}

void vtkCuda2DVolumeMapper::ClearInputInternal(){
	this->ReserveGPU();
	for( int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++ ){
		if( this->SourceData[i] )
			CUDA_vtkCuda2DVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]), this->GetStream());
	}
}

//give the function to the transfer function handler
void vtkCuda2DVolumeMapper::SetFunction(vtkCuda2DTransferFunction* funct){
	this->transferFunctionInfoHandler->SetTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCuda2DVolumeMapper::GetFunction(){
	return this->transferFunctionInfoHandler->GetTransferFunction();
}

//give the function to the transfer function handler
void vtkCuda2DVolumeMapper::SetKeyholeFunction(vtkCuda2DTransferFunction* funct){
	this->transferFunctionInfoHandler->SetKeyholeTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCuda2DVolumeMapper::GetKeyholeFunction(){
	return this->transferFunctionInfoHandler->GetKeyholeTransferFunction();
}