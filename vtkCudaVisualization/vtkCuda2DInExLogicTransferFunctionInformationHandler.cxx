#include "vtkCuda2DInExLogicTransferFunctionInformationHandler.h"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"

//Volume and Property
#include "vtkImageData.h"
#include "vtkImageGradientMagnitude.h"
#include "vtkCuda2DTransferFunction.h"

#include "CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.h"

vtkStandardNewMacro(vtkCuda2DInExLogicTransferFunctionInformationHandler);

vtkCuda2DInExLogicTransferFunctionInformationHandler::vtkCuda2DInExLogicTransferFunctionInformationHandler(){
	this->function = NULL;

	this->FunctionSize = 512;
	this->LowGradient = 0;
	this->HighGradient = 10;
	this->lastModifiedTime = 0;

	this->InputData = NULL;

	this->Reinitialize();
}

vtkCuda2DInExLogicTransferFunctionInformationHandler::~vtkCuda2DInExLogicTransferFunctionInformationHandler(){
	this->Deinitialize();
	this->SetInputData(NULL, 0);
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::Deinitialize(){
	this->ReserveGPU();
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_unloadTextures(this->GetStream());
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::Reinitialize(){
	this->lastModifiedTime = 0;
	this->UpdateTransferFunction();
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::SetInputData(vtkImageData* inputData, int index){
	if (inputData == NULL)
	{
		this->InputData = NULL;
	}
	else if (inputData != this->InputData)
	{
		this->InputData = inputData;
		this->Modified();
	}
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::SetVisualizationTransferFunction(vtkCuda2DTransferFunction* f){
	this->function = f;
	this->lastModifiedTime = 0;
	this->Modified();
}

vtkCuda2DTransferFunction* vtkCuda2DInExLogicTransferFunctionInformationHandler::GetVisualizationTransferFunction(){
	return this->function;
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::SetInExLogicTransferFunction(vtkCuda2DTransferFunction* f){
	this->inExFunction = f;
	this->lastModifiedTime = 0;
	this->Modified();
}

vtkCuda2DTransferFunction* vtkCuda2DInExLogicTransferFunctionInformationHandler::GetInExLogicTransferFunction(){
	return this->inExFunction;
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::UpdateTransferFunction(){
	//if we don't need to update the transfer functions, don't
	if(!this->function || this->function->GetMTime() <= lastModifiedTime ||
		!this->inExFunction || this->inExFunction->GetMTime() <= lastModifiedTime) return;
	lastModifiedTime = (this->function->GetMTime() < this->inExFunction->GetMTime()) ?
		this->inExFunction->GetMTime() : this->function->GetMTime();
	
	//calculate the gradient (to get max gradient values)
	double gradRange[2];
	vtkImageGradientMagnitude* gradientCalculator = vtkImageGradientMagnitude::New();
	gradientCalculator->SetInput(this->InputData);
	gradientCalculator->SetDimensionality(3);
	gradientCalculator->Update();
	gradientCalculator->GetOutput()->GetScalarRange(gradRange);
	this->LowGradient = gradRange[0];
	this->HighGradient = gradRange[1];
	gradientCalculator->Delete();

	//get the ranges from the transfer function
	double minIntensity = (this->InputData->GetScalarRange()[0] > this->function->getMinIntensity() ) ? this->InputData->GetScalarRange()[0] : this->function->getMinIntensity();
	double maxIntensity = (this->InputData->GetScalarRange()[1] < this->function->getMaxIntensity() ) ? this->InputData->GetScalarRange()[1] : this->function->getMaxIntensity();
	double minGradient = (this->LowGradient > this->function->getMinGradient() ) ? this->LowGradient : this->function->getMinGradient();
	double maxGradient = (this->HighGradient < this->function->getMaxGradient() ) ? this->HighGradient : this->function->getMaxGradient();
	minGradient = (minGradient > this->inExFunction->getMinGradient() ) ? minGradient : this->inExFunction->getMinGradient();
	maxGradient = (maxGradient < this->inExFunction->getMaxGradient() ) ? maxGradient : this->inExFunction->getMaxGradient();
	double gradientOffset = this->HighGradient * 0.8;
	maxGradient = (log(maxGradient*maxGradient+gradientOffset) - log(gradientOffset) )/ log(2.0) + 1.0;
	minGradient = (log(minGradient*minGradient+gradientOffset) - log(gradientOffset) )/ log(2.0);

	//figure out the multipliers for applying the transfer function in GPU
	this->TransInfo.intensityLow = minIntensity;
	this->TransInfo.intensityMultiplier = 1.0 / ( maxIntensity - minIntensity );
	this->TransInfo.gradientMultiplier = 1.0 / (maxGradient-minGradient);
	this->TransInfo.gradientOffset = gradientOffset;
	this->TransInfo.gradientLow = - minGradient - log(gradientOffset) / log(2.0);

	//create local buffers to house the transfer function
	float* LocalColorRedTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalColorGreenTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalColorBlueTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalAlphaTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalInExTransferFunction = new float[this->FunctionSize * this->FunctionSize];

	//populate the table
	this->function->GetTransferTable(LocalColorRedTransferFunction, LocalColorGreenTransferFunction, LocalColorBlueTransferFunction, LocalAlphaTransferFunction,
		this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, minGradient, maxGradient, gradientOffset);
	this->inExFunction->GetTransferTable(0, 0, 0, LocalInExTransferFunction,
		this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, minGradient, maxGradient, gradientOffset);

	//map the trasfer functions to textures for fast access
	this->TransInfo.functionSize = this->FunctionSize;
	this->ReserveGPU();
	CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_loadTextures(this->TransInfo,
		LocalColorRedTransferFunction,
		LocalColorGreenTransferFunction,
		LocalColorBlueTransferFunction,
		LocalAlphaTransferFunction,
		LocalInExTransferFunction,
		this->GetStream() );

	//clean up the garbage
	delete LocalColorRedTransferFunction;
	delete LocalColorGreenTransferFunction;
	delete LocalColorBlueTransferFunction;
	delete LocalAlphaTransferFunction;
	delete LocalInExTransferFunction;
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::Update(){
	if(this->InputData){
		this->InputData->Update();
		this->Modified();
	}
	if(this->function && this->inExFunction){
		this->UpdateTransferFunction();
		this->Modified();
	}
}

