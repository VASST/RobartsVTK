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

vtkCuda2DVolumeMapper::vtkCuda2DVolumeMapper()
{
	CUDA_vtkCuda2DVolumeMapper_renderAlgo_initImageArray();
	this->transferFunctionInfoHandler = vtkCuda2DTransferFunctionInformationHandler::New();
}

vtkCuda2DVolumeMapper::~vtkCuda2DVolumeMapper(){
	this->transferFunctionInfoHandler->Delete();
}

void vtkCuda2DVolumeMapper::SetInputInternal(vtkImageData * input, int index){
	
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
	CUDA_vtkCuda2DVolumeMapper_renderAlgo_loadImageInfo( buffer, VolumeInfoHandler->GetVolumeInfo(), index);
	if(input->GetScalarType() != VTK_FLOAT) delete buffer;

	//inform transfer function handler of the data
	this->transferFunctionInfoHandler->SetInputData(input,index);
}

void vtkCuda2DVolumeMapper::ChangeFrameInternal(unsigned int frame){
		CUDA_vtkCuda2DVolumeMapper_renderAlgo_changeFrame(frame);
}

void vtkCuda2DVolumeMapper::InternalRender (	vtkRenderer* ren, vtkVolume* vol,
												const cudaRendererInformation& rendererInfo,
												const cudaVolumeInformation& volumeInfo,
												const cudaOutputImageInformation& outputInfo ){
	//handle the transfer function changes
	this->transferFunctionInfoHandler->Update();

	//perform the render
	CUDA_vtkCuda2DVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
		this->transferFunctionInfoHandler->GetTransferFunctionInfo() );

}

void vtkCuda2DVolumeMapper::ClearInputInternal(){
	CUDA_vtkCuda2DVolumeMapper_renderAlgo_clearImageArray();
}

//give the function to the transfer function handler
void vtkCuda2DVolumeMapper::SetFunction(vtkCuda2DTransferFunction* funct){
	this->transferFunctionInfoHandler->SetTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCuda2DVolumeMapper::GetFunction(){
	return this->transferFunctionInfoHandler->GetTransferFunction();
}