/** @file vtkCudaVolumeInformationHandler.cxx
 *
 *  @brief An internal class for vtkCudaVolumeMapper which manages information regarding the volume and transfer function
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 28, 2011
 *
 */

#include "vtkCudaVolumeInformationHandler.h"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"

//Volume and Property
#include "vtkVolume.h"
#include "vtkImageData.h"

vtkStandardNewMacro(vtkCudaVolumeInformationHandler);

vtkCudaVolumeInformationHandler::vtkCudaVolumeInformationHandler(){
	this->lastModifiedTime = 0;
	this->Volume = NULL;
	this->InputData = NULL;
}

vtkCudaVolumeInformationHandler::~vtkCudaVolumeInformationHandler(){
	this->SetVolume(NULL);
	this->SetInputData(NULL, 0);
}

void vtkCudaVolumeInformationHandler::Deinitialize(int withData){
	//TODO
}

void vtkCudaVolumeInformationHandler::Reinitialize(int withData){
	//TODO
}

vtkVolume* vtkCudaVolumeInformationHandler::GetVolume(){
	return this->Volume;
}

void vtkCudaVolumeInformationHandler::SetVolume(vtkVolume* volume){
	this->Volume = volume;
	if (Volume != NULL)
		this->Update();
}

void vtkCudaVolumeInformationHandler::SetInputData(vtkImageData* inputData, int index){
	if (inputData == NULL)
	{
		this->InputData = NULL;
	}
	else if (inputData != this->InputData)
	{
		this->InputData = inputData;
		this->UpdateImageData(index);
		this->Modified();
	}
}

void vtkCudaVolumeInformationHandler::UpdateImageData(int index){
	this->InputData->Update();

	int* dims = this->InputData->GetDimensions();
	double* spacing = this->InputData->GetSpacing();

	this->VolumeInfo.VolumeSize.x = dims[0];
	this->VolumeInfo.VolumeSize.y = dims[1];
	this->VolumeInfo.VolumeSize.z = dims[2];
	
	this->VolumeInfo.SpacingReciprocal.x = 0.5f / spacing[0];
	this->VolumeInfo.SpacingReciprocal.y = 0.5f / spacing[1];
	this->VolumeInfo.SpacingReciprocal.z = 0.5f / spacing[2];
	this->VolumeInfo.Spacing.x = spacing[0];
	this->VolumeInfo.Spacing.y = spacing[1];
	this->VolumeInfo.Spacing.z = spacing[2];
	this->VolumeInfo.MinSpacing = spacing[0];
	this->VolumeInfo.MinSpacing = (this->VolumeInfo.MinSpacing > spacing[1]) ? spacing[1] : this->VolumeInfo.MinSpacing;
	this->VolumeInfo.MinSpacing = (this->VolumeInfo.MinSpacing > spacing[2]) ? spacing[2] : this->VolumeInfo.MinSpacing;

	//calculate the bounds
	this->VolumeInfo.Bounds[0] = 0.0f;
	this->VolumeInfo.Bounds[1] = (float) dims[0] - 1.0f;
	this->VolumeInfo.Bounds[2] = 0.0f;
	this->VolumeInfo.Bounds[3] = (float) dims[1] - 1.0f;
	this->VolumeInfo.Bounds[4] = 0.0f;
	this->VolumeInfo.Bounds[5] = (float) dims[2] - 1.0f;
}

void vtkCudaVolumeInformationHandler::Update(){
	if(this->InputData){
		this->InputData->Update();
		this->Modified();
	}
}

void vtkCudaVolumeInformationHandler::ClearInput(){
	this->Modified();
	this->Volume = NULL;
	this->InputData = NULL;
}
