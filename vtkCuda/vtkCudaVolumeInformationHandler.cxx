#include "vtkCudaVolumeInformationHandler.h"
#include "vtkObjectFactory.h"

//Volume and Property
#include "vtkVolumeProperty.h"
#include "vtkVolume.h"
#include "vtkImageData.h"
#include "vtkPointData.h"
#include "vtkCuda2DTransferClassificationFunction.h"

extern "C" {
#include "CUDA_renderAlgo.h"
}

vtkStandardNewMacro(vtkCudaVolumeInformationHandler);

vtkCudaVolumeInformationHandler::vtkCudaVolumeInformationHandler()
{
	this->function = NULL;

    this->FunctionSize = 128;

    this->Volume = NULL;
    this->InputData = NULL;
}

vtkCudaVolumeInformationHandler::~vtkCudaVolumeInformationHandler()
{
    this->SetVolume(NULL);
    this->SetInputData(NULL, 0);
}

void vtkCudaVolumeInformationHandler::SetVolume(vtkVolume* volume)
{
    this->Volume = volume;
    if (Volume != NULL)
        this->UpdateVolume();
    this->Modified();
}

void vtkCudaVolumeInformationHandler::SetInputData(vtkImageData* inputData, int index)
{
    if (inputData == NULL)
    {
        this->InputData = NULL;
    }
    else if (inputData != this->InputData)
    {
        this->InputData = inputData;

        double range[2];
        inputData->GetPointData()->GetScalars()->GetRange(range);
        this->FunctionRange[0] = range[0];
        this->FunctionRange[1] = range[1];

        this->UpdateImageData(index);
        this->Modified();
    }
}

void vtkCudaVolumeInformationHandler::SetSampleDistance(float sampleDistance)
{
}

void vtkCudaVolumeInformationHandler::SetTransferFunction(vtkCuda2DTransferClassificationFunction* f){
	this->function = f;
}

/**
* @brief Updates the transfer functions on local and global memory.
* @param property: The property that holds the transfer function information.
*/
void vtkCudaVolumeInformationHandler::UpdateTransferFunction()
{
	//if we don't need to update the transfer function, don't
	if(!this->function->NeedsUpdate()){
		return;
	}

	//figure out the multipliers for applying the transfer function in GPU
	this->VolumeInfo.intensityLow = this->FunctionRange[0];
	this->VolumeInfo.intensityMultiplier = (float) this->FunctionSize / ( this->FunctionRange[1] - this->FunctionRange[0] );
	this->VolumeInfo.twiceGradientMultiplier = 0.5f * (float) this->FunctionSize / 50.0f;

	//create local buffers to house the transfer function
	float* LocalColorRedTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalColorGreenTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalColorBlueTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalAlphaTransferFunction = new float[this->FunctionSize * this->FunctionSize];

	memset( (void*) LocalColorRedTransferFunction, 0.0f, sizeof(float) * this->FunctionSize * this->FunctionSize);
	memset( (void*) LocalColorGreenTransferFunction, 0.0f, sizeof(float) * this->FunctionSize * this->FunctionSize);
	memset( (void*) LocalColorBlueTransferFunction, 0.0f, sizeof(float) * this->FunctionSize * this->FunctionSize);
	memset( (void*) LocalAlphaTransferFunction, 0.0f, sizeof(float) * this->FunctionSize * this->FunctionSize);

	//populate the table
	this->function->GetTransferTable(LocalColorRedTransferFunction, LocalColorGreenTransferFunction, LocalColorBlueTransferFunction, LocalAlphaTransferFunction,
		this->FunctionSize, this->FunctionSize, this->FunctionRange[0], this->FunctionRange[1], 0.0f, 50.0f);

	//map the trasfer functions to textures for fast access
	CUDAkernelsetup_loadTextures(this->VolumeInfo, this->FunctionSize,
		LocalColorRedTransferFunction,
		LocalColorGreenTransferFunction,
		LocalColorBlueTransferFunction,
		LocalAlphaTransferFunction);

	//clean up the garbage
	delete LocalColorRedTransferFunction;
	delete LocalColorGreenTransferFunction;
	delete LocalColorBlueTransferFunction;
	delete LocalAlphaTransferFunction;

	//signal to the transfer function that the update has been recorded
	this->function->SatisfyUpdate();

}

#include "vtkMatrix4x4.h"
void vtkCudaVolumeInformationHandler::UpdateVolume()
{
  this->UpdateTransferFunction();
}

void vtkCudaVolumeInformationHandler::UpdateImageData(int index)
{
	this->InputData->Update();

    int* dims = this->InputData->GetDimensions();
    double* spacing = this->InputData->GetSpacing();

    this->VolumeInfo.VolumeSize.x = dims[0];
    this->VolumeInfo.VolumeSize.y = dims[1];
    this->VolumeInfo.VolumeSize.z = dims[2];

	this->VolumeInfo.SpacingReciprocal.x = 0.5f / spacing[0];
	this->VolumeInfo.SpacingReciprocal.y = 0.5f / spacing[1];
	this->VolumeInfo.SpacingReciprocal.z = 0.5f / spacing[2];

	//calculate the bounds
	this->VolumeInfo.Bounds[0] = 1.0f;
	this->VolumeInfo.Bounds[1] = (float) dims[0] - 2.0f;
	this->VolumeInfo.Bounds[2] = 1.0f;
	this->VolumeInfo.Bounds[3] = (float) dims[1] - 2.0f;
	this->VolumeInfo.Bounds[4] = 1.0f;
	this->VolumeInfo.Bounds[5] = (float) dims[2] - 2.0f;

	float* buffer = new float[this->VolumeInfo.VolumeSize.x*this->VolumeInfo.VolumeSize.y*this->VolumeInfo.VolumeSize.z];
	//switch(this->InputData->GetScalarType()){
	if(this->InputData->GetScalarType() == VTK_SHORT){
		short* tempPtr = (short*) this->InputData->GetScalarPointer();
		for(int i = 0; i < this->VolumeInfo.VolumeSize.x*this->VolumeInfo.VolumeSize.y*this->VolumeInfo.VolumeSize.z; i++){
			buffer[i] = (float)(tempPtr[i]);
		}
		this->VolumeInfo.SourceData = buffer;
	}else if(this->InputData->GetScalarType() == VTK_UNSIGNED_SHORT){
		unsigned short* tempPtr = (unsigned short*) this->InputData->GetScalarPointer();
		for(int i = 0; i < this->VolumeInfo.VolumeSize.x*this->VolumeInfo.VolumeSize.y*this->VolumeInfo.VolumeSize.z; i++){
			buffer[i] = (float)(tempPtr[i]);
		}
		this->VolumeInfo.SourceData = buffer;
	}else if(this->InputData->GetScalarType() == VTK_CHAR){
		char* tempPtr = (char*) this->InputData->GetScalarPointer();
		for(int i = 0; i < this->VolumeInfo.VolumeSize.x*this->VolumeInfo.VolumeSize.y*this->VolumeInfo.VolumeSize.z; i++){
			buffer[i] = (float)(tempPtr[i]);
		}
		this->VolumeInfo.SourceData = buffer;
	}else if(this->InputData->GetScalarType() == VTK_UNSIGNED_CHAR){
		unsigned char* tempPtr = (unsigned char*) this->InputData->GetScalarPointer();
		for(int i = 0; i < this->VolumeInfo.VolumeSize.x*this->VolumeInfo.VolumeSize.y*this->VolumeInfo.VolumeSize.z; i++){
			buffer[i] = (float)(tempPtr[i]);
		}
		this->VolumeInfo.SourceData = buffer;
	}else if(this->InputData->GetScalarType() == VTK_INT){
		int* tempPtr = (int*) this->InputData->GetScalarPointer();
		for(int i = 0; i < this->VolumeInfo.VolumeSize.x*this->VolumeInfo.VolumeSize.y*this->VolumeInfo.VolumeSize.z; i++){
			buffer[i] = (float)(tempPtr[i]);
		}
		this->VolumeInfo.SourceData = buffer;
	}else if(this->InputData->GetScalarType() == VTK_UNSIGNED_INT){
		unsigned int* tempPtr = (unsigned int*) this->InputData->GetScalarPointer();
		for(int i = 0; i < this->VolumeInfo.VolumeSize.x*this->VolumeInfo.VolumeSize.y*this->VolumeInfo.VolumeSize.z; i++){
			buffer[i] = (float)(tempPtr[i]);
		}
		this->VolumeInfo.SourceData = buffer;
	}else if(this->InputData->GetScalarType() == VTK_FLOAT){
		this->VolumeInfo.SourceData = (float*) this->InputData->GetScalarPointer();
	}else{
		vtkErrorMacro("Input cannot be of that type.");
	}

	CUDAkernelsetup_loadImageInfo( this->VolumeInfo, index);
	//delete buffer;
}

/**
* @brief Updates the volume information that is being sent to the Cuda Card.
*/
void vtkCudaVolumeInformationHandler::Update()
{
	if(this->InputData){
		this->UpdateVolume();
		this->Modified();
	}
}

void vtkCudaVolumeInformationHandler::PrintSelf(std::ostream& os, vtkIndent indent)
{

}
