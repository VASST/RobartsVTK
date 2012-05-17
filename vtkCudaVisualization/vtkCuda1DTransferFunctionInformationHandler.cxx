#include "vtkCuda1DTransferFunctionInformationHandler.h"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"

//Volume and Property
#include "vtkPiecewiseFunction.h"
#include "vtkColorTransferFunction.h"
#include "vtkImageData.h"

#include "CUDA_vtkCuda1DVolumeMapper_renderAlgo.h"

vtkStandardNewMacro(vtkCuda1DTransferFunctionInformationHandler);

vtkCuda1DTransferFunctionInformationHandler::vtkCuda1DTransferFunctionInformationHandler(){
	this->colourFunction = NULL;
	this->opacityFunction = NULL;

	this->FunctionSize = 512;
	this->lastModifiedTime = 0;

	this->InputData = NULL;
	this->Reinitialize();
}

vtkCuda1DTransferFunctionInformationHandler::~vtkCuda1DTransferFunctionInformationHandler(){
	this->Deinitialize();
	this->SetInputData(NULL, 0);
}

void vtkCuda1DTransferFunctionInformationHandler::Deinitialize(int withData){
	this->ReserveGPU();
	CUDA_vtkCuda1DVolumeMapper_renderAlgo_UnloadTextures( this->GetStream() );
}

void vtkCuda1DTransferFunctionInformationHandler::Reinitialize(int withData){
	lastModifiedTime = 0;
	UpdateTransferFunction();
}

void vtkCuda1DTransferFunctionInformationHandler::SetInputData(vtkImageData* inputData, int index){
	if (inputData == NULL){
		this->InputData = NULL;
	}else if (inputData != this->InputData){
		this->InputData = inputData;
		this->Modified();
	}
}

void vtkCuda1DTransferFunctionInformationHandler::SetColourTransferFunction(vtkColorTransferFunction* f){
	if( f != this->colourFunction ){
		this->colourFunction = f;
		this->lastModifiedTime = 0;
		this->Modified();
	}
}

void vtkCuda1DTransferFunctionInformationHandler::SetOpacityTransferFunction(vtkPiecewiseFunction* f){
	if( f != this->opacityFunction ){
		this->opacityFunction = f;
		this->lastModifiedTime = 0;
		this->Modified();
	}
}

void vtkCuda1DTransferFunctionInformationHandler::UpdateTransferFunction(){
	//if we don't need to update the transfer function, don't
	if(this->colourFunction || this->opacityFunction ||
		(this->colourFunction->GetMTime() <= lastModifiedTime &&
		this->opacityFunction->GetMTime() <= lastModifiedTime) ) return;
	lastModifiedTime = (this->colourFunction->GetMTime() > this->opacityFunction->GetMTime()) ?
		this->colourFunction->GetMTime() : this->opacityFunction->GetMTime();

	//get the ranges from the transfer function
	double minIntensity; 
	double maxIntensity;
	this->opacityFunction->GetRange( minIntensity, maxIntensity );
	minIntensity = (this->InputData->GetScalarRange()[0] > minIntensity ) ? this->InputData->GetScalarRange()[0] : minIntensity;
	maxIntensity = (this->InputData->GetScalarRange()[1] < maxIntensity ) ? this->InputData->GetScalarRange()[1] : maxIntensity;

	//figure out the multipliers for applying the transfer function in GPU
	this->TransInfo.intensityLow = minIntensity;
	this->TransInfo.intensityMultiplier = 1.0 / ( maxIntensity - minIntensity );

	//create local buffers to house the transfer function
	float* LocalColorRedTransferFunction = new float[this->FunctionSize];
	float* LocalColorGreenTransferFunction = new float[this->FunctionSize];
	float* LocalColorBlueTransferFunction = new float[this->FunctionSize];
	float* LocalColorWholeTransferFunction = new float[3*this->FunctionSize];
	float* LocalAlphaTransferFunction = new float[this->FunctionSize];

	memset( (void*) LocalColorRedTransferFunction, 1.0f, sizeof(float) * this->FunctionSize);
	memset( (void*) LocalColorGreenTransferFunction, 1.0f, sizeof(float) * this->FunctionSize);
	memset( (void*) LocalColorBlueTransferFunction, 1.0f, sizeof(float) * this->FunctionSize);
	memset( (void*) LocalColorWholeTransferFunction, 1.0f, sizeof(float) * this->FunctionSize);
	memset( (void*) LocalAlphaTransferFunction, 1.0f, sizeof(float) * this->FunctionSize);

	//populate the table
	this->opacityFunction->GetTable( minIntensity, maxIntensity, this->FunctionSize,
		LocalAlphaTransferFunction );
	this->colourFunction->GetTable( minIntensity, maxIntensity, this->FunctionSize,
		LocalColorWholeTransferFunction );
	for( int i = 0; i < this->FunctionSize; i++ ){
		LocalColorRedTransferFunction[i] = LocalColorWholeTransferFunction[3*i];
		LocalColorGreenTransferFunction[i] = LocalColorWholeTransferFunction[3*i+1];
		LocalColorBlueTransferFunction[i] = LocalColorWholeTransferFunction[3*i+2];
	}

	//map the trasfer functions to textures for fast access
	this->TransInfo.functionSize = this->FunctionSize;

	this->ReserveGPU();
	CUDA_vtkCuda1DVolumeMapper_renderAlgo_loadTextures(this->TransInfo,
		LocalColorRedTransferFunction,
		LocalColorGreenTransferFunction,
		LocalColorBlueTransferFunction,
		LocalAlphaTransferFunction,
		this->GetStream() );

	//clean up the garbage
	delete LocalColorRedTransferFunction;
	delete LocalColorGreenTransferFunction;
	delete LocalColorBlueTransferFunction;
	delete LocalColorWholeTransferFunction;
	delete LocalAlphaTransferFunction;
}

void vtkCuda1DTransferFunctionInformationHandler::Update(){
	if(this->InputData){
		this->InputData->Update();
		this->Modified();
	}
	if(this->colourFunction && this->opacityFunction){
		this->UpdateTransferFunction();
		this->Modified();
	}
}

