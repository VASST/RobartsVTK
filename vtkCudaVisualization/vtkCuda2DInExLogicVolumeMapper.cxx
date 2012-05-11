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

vtkCuda2DInExLogicVolumeMapper::vtkCuda2DInExLogicVolumeMapper()
{
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_initImageArray();
	this->transferFunctionInfoHandler = vtkCuda2DInExLogicTransferFunctionInformationHandler::New();
	this->sliceInfo.NumberOfSlicingPlanes = 0;
	this->SlicingPlanes = 0;
}

vtkCuda2DInExLogicVolumeMapper::~vtkCuda2DInExLogicVolumeMapper(){
	this->transferFunctionInfoHandler->Delete();
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
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_loadImageInfo( buffer, VolumeInfoHandler->GetVolumeInfo(), index);
	if(input->GetScalarType() != VTK_FLOAT) delete buffer;

	//inform transfer function handler of the data
	this->transferFunctionInfoHandler->SetInputData(input,index);
}

void vtkCuda2DInExLogicVolumeMapper::ChangeFrameInternal(unsigned int frame){
		CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_changeFrame(frame);
}

void vtkCuda2DInExLogicVolumeMapper::InternalRender (	vtkRenderer* ren, vtkVolume* vol,
												const cudaRendererInformation& rendererInfo,
												const cudaVolumeInformation& volumeInfo,
												const cudaOutputImageInformation& outputInfo ){
	//handle the transfer function changes
	this->transferFunctionInfoHandler->Update();

	this->RendererInfoHandler->FigurePlanes(this->SlicingPlanes, this->sliceInfo.SlicingPlanes,
											&(this->sliceInfo.NumberOfSlicingPlanes) );

	//perform the render
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
		this->transferFunctionInfoHandler->GetTransferFunctionInfo(), this->sliceInfo );

}

void vtkCuda2DInExLogicVolumeMapper::ClearInputInternal(){
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_clearImageArray();
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

vtkCxxSetObjectMacro(vtkCuda2DInExLogicVolumeMapper,SlicingPlanes,vtkPlaneCollection);

void vtkCuda2DInExLogicVolumeMapper::AddSlicingPlane(vtkPlane *plane){
  if (this->SlicingPlanes == NULL){
    this->SlicingPlanes = vtkPlaneCollection::New();
    this->SlicingPlanes->Register(this);
    this->SlicingPlanes->Delete();
  }

  this->SlicingPlanes->AddItem(plane);
  this->Modified();
}

void vtkCuda2DInExLogicVolumeMapper::RemoveSlicingPlane(vtkPlane *plane){
  if (this->SlicingPlanes == NULL) vtkErrorMacro(<< "Cannot remove Slicing plane: mapper has none");
  this->SlicingPlanes->RemoveItem(plane);
  this->Modified();
}

void vtkCuda2DInExLogicVolumeMapper::RemoveAllSlicingPlanes(){
  if ( this->SlicingPlanes ) this->SlicingPlanes->RemoveAllItems();
}

void vtkCuda2DInExLogicVolumeMapper::SetSlicingPlanes(vtkPlanes *planes){
  vtkPlane *plane;
  if (!planes) return;

  int numPlanes = planes->GetNumberOfPlanes();

  this->RemoveAllSlicingPlanes();
  for (int i=0; i<numPlanes && i<6; i++){
    plane = vtkPlane::New();
    planes->GetPlane(i, plane);
    this->AddSlicingPlane(plane);
    plane->Delete();
  }
}