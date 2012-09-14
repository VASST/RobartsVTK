/** @file vtkCudaVolumeMapper.cxx
 *
 *  @brief Implementation of a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on May 12, 2012
 *
 */

// Type
#include "vtkCuda1DVolumeMapper.h"
#include "vtkObjectFactory.h"

// Volume
#include "vtkVolume.h"
#include "vtkImageData.h"

// Rendering
#include "vtkCamera.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkColorTransferFunction.h"
#include "vtkPiecewiseFunction.h"

// VTKCUDA
#include "CUDA_vtkCuda1DVolumeMapper_renderAlgo.h"

vtkStandardNewMacro(vtkCuda1DVolumeMapper);

vtkMutexLock* vtkCuda1DVolumeMapper::tfLock = 0;
vtkCuda1DVolumeMapper::vtkCuda1DVolumeMapper()
{
	this->transferFunctionInfoHandler = vtkCuda1DTransferFunctionInformationHandler::New();
	if( this->tfLock == 0 ) this->tfLock = vtkMutexLock::New();
	else this->tfLock->Register(this);
	this->Reinitialize();
}

void vtkCuda1DVolumeMapper::Deinitialize(int withData){
	this->vtkCudaVolumeMapper::Deinitialize(withData);
	this->ReserveGPU();
	for( int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++ )
		CUDA_vtkCuda1DVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]), this->GetStream());
}

void vtkCuda1DVolumeMapper::Reinitialize(int withData){
	this->vtkCudaVolumeMapper::Reinitialize(withData);
	this->transferFunctionInfoHandler->ReplicateObject(this, withData);
	for( int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++ ) this->SourceData[i] = 0;
}

vtkCuda1DVolumeMapper::~vtkCuda1DVolumeMapper(){
	this->Deinitialize();
	transferFunctionInfoHandler->UnRegister( this );
	this->tfLock->UnRegister(this);
}

void vtkCuda1DVolumeMapper::SetInputInternal(vtkImageData * input, int index){
	
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
		this->erroredOut = !CUDA_vtkCuda1DVolumeMapper_renderAlgo_loadImageInfo( buffer, VolumeInfoHandler->GetVolumeInfo(), &(this->SourceData[index]), this->GetStream());
	}
	if(input->GetScalarType() != VTK_FLOAT) delete buffer;

	//inform transfer function handler of the data
	this->transferFunctionInfoHandler->SetInputData(input,index);
}

void vtkCuda1DVolumeMapper::ChangeFrameInternal(unsigned int frame){
}

void vtkCuda1DVolumeMapper::InternalRender (	vtkRenderer* ren, vtkVolume* vol,
												const cudaRendererInformation& rendererInfo,
												const cudaVolumeInformation& volumeInfo,
												const cudaOutputImageInformation& outputInfo ){
	//handle the transfer function changes
	this->transferFunctionInfoHandler->SetColourTransferFunction( vol->GetProperty()->GetRGBTransferFunction() );
	this->transferFunctionInfoHandler->SetOpacityTransferFunction( vol->GetProperty()->GetScalarOpacity() );
	this->transferFunctionInfoHandler->SetGradientOpacityTransferFunction( vol->GetProperty()->GetGradientOpacity() );
	this->transferFunctionInfoHandler->UseGradientOpacity( !vol->GetProperty()->GetDisableGradientOpacity() );
	this->transferFunctionInfoHandler->Update(vol);

	//perform the render
	this->tfLock->Lock();
	this->ReserveGPU();
	this->erroredOut = !CUDA_vtkCuda1DVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
		this->transferFunctionInfoHandler->GetTransferFunctionInfo(), this->SourceData[this->currFrame], this->GetStream());
	this->tfLock->Unlock();

}

void vtkCuda1DVolumeMapper::ClearInputInternal(){
	this->ReserveGPU();
	for(int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++ )
		CUDA_vtkCuda1DVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]),this->GetStream());
}