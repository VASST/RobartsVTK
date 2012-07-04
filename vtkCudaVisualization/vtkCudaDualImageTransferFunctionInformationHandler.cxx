#include "vtkCudaDualImageTransferFunctionInformationHandler.h"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"

//Volume and Property
#include "vtkImageData.h"
#include "vtkPointData.h"
#include "vtkCuda2DTransferFunction.h"

#include "CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.h"

vtkStandardNewMacro(vtkCudaDualImageTransferFunctionInformationHandler);

vtkCudaDualImageTransferFunctionInformationHandler::vtkCudaDualImageTransferFunctionInformationHandler(){
	this->function = NULL;

	this->FunctionSize = 512;
	this->lastModifiedTime = 0;
	
	this->TransInfo.alphaTransferArrayDualImage = 0;
	this->TransInfo.ambientTransferArrayDualImage = 0;
	this->TransInfo.diffuseTransferArrayDualImage = 0;
	this->TransInfo.specularTransferArrayDualImage = 0;
	this->TransInfo.specularPowerTransferArrayDualImage = 0;
	this->TransInfo.colorRTransferArrayDualImage = 0;
	this->TransInfo.colorGTransferArrayDualImage = 0;
	this->TransInfo.colorBTransferArrayDualImage = 0;

	this->InputData = NULL;
}

vtkCudaDualImageTransferFunctionInformationHandler::~vtkCudaDualImageTransferFunctionInformationHandler(){
	this->Deinitialize();
	this->SetInputData(NULL, 0);
	if( this->function ) this->function->UnRegister( this );
}

void vtkCudaDualImageTransferFunctionInformationHandler::Deinitialize(int withData){
	this->ReserveGPU();
	CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_unloadTextures(this->TransInfo, this->GetStream());
}

void vtkCudaDualImageTransferFunctionInformationHandler::Reinitialize(int withData){
	this->lastModifiedTime = 0;
	this->UpdateTransferFunction();
}

void vtkCudaDualImageTransferFunctionInformationHandler::SetInputData(vtkImageData* inputData, int index){
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

void vtkCudaDualImageTransferFunctionInformationHandler::SetTransferFunction(vtkCuda2DTransferFunction* f){
	if( this->function ==  f ) return;
	if( this->function ) this->function->UnRegister( this );
	this->function = f;
	if( this->function ) this->function->Register( this );
	this->lastModifiedTime = 0;
	this->Modified();
}

vtkCuda2DTransferFunction* vtkCudaDualImageTransferFunctionInformationHandler::GetTransferFunction(){
	return this->function;
}

void vtkCudaDualImageTransferFunctionInformationHandler::UpdateTransferFunction(){
	//if we don't need to update the transfer function, don't
	if(!this->function || this->function->GetMTime() <= lastModifiedTime) return;
	lastModifiedTime = this->function->GetMTime();

	//get the ranges from the transfer function
	int num = this->InputData->GetNumberOfScalarComponents();
	double scalarRange[4];
	this->InputData->GetPointData()->GetScalars()->GetRange(scalarRange,0);
	this->InputData->GetPointData()->GetScalars()->GetRange(scalarRange+2,1);
	double functionRange[] = {	this->function->getMinIntensity(), this->function->getMaxIntensity(), 
								this->function->getMinGradient(), this->function->getMaxGradient() };
	double minIntensity1 = (scalarRange[0] > functionRange[0] ) ? scalarRange[0] : functionRange[0];
	double maxIntensity1 = (scalarRange[1] < functionRange[1] ) ? scalarRange[1] : functionRange[1];
	double minIntensity2 = (scalarRange[2] > functionRange[2] ) ? scalarRange[2] : functionRange[2];
	double maxIntensity2 = (scalarRange[3] < functionRange[3] ) ? scalarRange[3] : functionRange[3];

	//figure out the multipliers for applying the transfer function in GPU
	this->TransInfo.intensity1Low = minIntensity1;
	this->TransInfo.intensity1Multiplier = 1.0 / ( maxIntensity1 - minIntensity1 );
	this->TransInfo.intensity2Low = minIntensity2;
	this->TransInfo.intensity2Multiplier = 1.0 / ( maxIntensity2 - minIntensity2 );

	//create local buffers to house the transfer function
	float* LocalColorRedTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalColorGreenTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalColorBlueTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalAlphaTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalAmbientTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalDiffuseTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalSpecularTransferFunction = new float[this->FunctionSize * this->FunctionSize];
	float* LocalSpecularPowerTransferFunction = new float[this->FunctionSize * this->FunctionSize];

	//populate the table
	this->function->GetTransferTable(LocalColorRedTransferFunction, LocalColorGreenTransferFunction, LocalColorBlueTransferFunction, LocalAlphaTransferFunction,
		this->FunctionSize, this->FunctionSize, minIntensity1, maxIntensity1, 0, minIntensity2, maxIntensity2, 0, 0);
	this->function->GetShadingTable(LocalAmbientTransferFunction, LocalDiffuseTransferFunction, LocalSpecularTransferFunction, LocalSpecularPowerTransferFunction,
		this->FunctionSize, this->FunctionSize, minIntensity1, maxIntensity1, 0, minIntensity2, maxIntensity2, 0, 0);

	//map the trasfer functions to textures for fast access
	this->TransInfo.functionSize = this->FunctionSize;
	this->ReserveGPU();
	CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_loadTextures(this->TransInfo,
		LocalColorRedTransferFunction,
		LocalColorGreenTransferFunction,
		LocalColorBlueTransferFunction,
		LocalAlphaTransferFunction,
		LocalAmbientTransferFunction,
		LocalDiffuseTransferFunction,
		LocalSpecularTransferFunction,
		LocalSpecularPowerTransferFunction,
		this->GetStream());

	//clean up the garbage
	delete LocalColorRedTransferFunction;
	delete LocalColorGreenTransferFunction;
	delete LocalColorBlueTransferFunction;
	delete LocalAlphaTransferFunction;
	delete LocalAmbientTransferFunction;
	delete LocalDiffuseTransferFunction;
	delete LocalSpecularTransferFunction;
	delete LocalSpecularPowerTransferFunction;
}

void vtkCudaDualImageTransferFunctionInformationHandler::Update(){
	if(this->InputData){
		this->InputData->Update();
		this->Modified();
	}
	if(this->function){
		this->UpdateTransferFunction();
		this->Modified();
	}
}

