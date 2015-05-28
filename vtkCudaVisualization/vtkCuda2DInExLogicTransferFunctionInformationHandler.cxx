#include "vtkCuda2DInExLogicTransferFunctionInformationHandler.h"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"

//Volume and Property
#include "vtkImageData.h"
#include "vtkCuda2DTransferFunction.h"

#include "CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.h"

vtkStandardNewMacro(vtkCuda2DInExLogicTransferFunctionInformationHandler);

vtkCuda2DInExLogicTransferFunctionInformationHandler::vtkCuda2DInExLogicTransferFunctionInformationHandler(){
  this->function = NULL;
  this->inExFunction = NULL;
  this->UseBlackKeyhole = false;
  this->TransInfo.useBlackKeyhole = false;

  this->FunctionSize = 512;
  this->LowGradient = 0;
  this->HighGradient = 10;
  this->lastModifiedTime = 0;

  this->InputData = NULL;

  this->TransInfo.alphaTransferArray2D = 0;
  this->TransInfo.ambientTransferArray2D = 0;
  this->TransInfo.diffuseTransferArray2D = 0;
  this->TransInfo.specularTransferArray2D = 0;
  this->TransInfo.specularPowerTransferArray2D = 0;
  this->TransInfo.colorRTransferArray2D = 0;
  this->TransInfo.colorGTransferArray2D = 0;
  this->TransInfo.colorBTransferArray2D = 0;
  this->TransInfo.inExLogicTransferArray2D = 0;

  this->Reinitialize();
}

vtkCuda2DInExLogicTransferFunctionInformationHandler::~vtkCuda2DInExLogicTransferFunctionInformationHandler(){
  this->Deinitialize();
  this->SetInputData(NULL, 0);
  if(this->inExFunction) this->inExFunction->UnRegister(this);
  if(this->function) this->function->UnRegister(this);
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::Deinitialize(int withData){
  this->ReserveGPU();
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_unloadTextures(this->TransInfo, this->GetStream());
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::Reinitialize(int withData){
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
  if( this->function ==  f ) return;
  if( this->function ) this->function->UnRegister( this );
  this->function = f;
  if( this->function ) this->function->Register( this );
  this->lastModifiedTime = 0;
  this->Modified();
}

vtkCuda2DTransferFunction* vtkCuda2DInExLogicTransferFunctionInformationHandler::GetVisualizationTransferFunction(){
  return this->function;
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::SetInExLogicTransferFunction(vtkCuda2DTransferFunction* f){
  if( this->inExFunction ==  f ) return;
  if( this->inExFunction ) this->inExFunction->UnRegister( this );
  this->inExFunction = f;
  if( this->inExFunction ) this->inExFunction->Register( this );
  this->lastModifiedTime = 0;
  this->Modified();
}

vtkCuda2DTransferFunction* vtkCuda2DInExLogicTransferFunctionInformationHandler::GetInExLogicTransferFunction(){
  return this->inExFunction;
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::UpdateTransferFunction(){
  //if we don't need to update the transfer functions, don't
  if(!this->InputData ) return;
  if(!this->function || this->function->GetMTime() <= lastModifiedTime ||
    !this->inExFunction || this->inExFunction->GetMTime() <= lastModifiedTime) return;
  lastModifiedTime = (this->function->GetMTime() < this->inExFunction->GetMTime()) ?
    this->inExFunction->GetMTime() : this->function->GetMTime();

  //get the ranges from the transfer function
  double minIntensity = (this->InputData->GetScalarRange()[0] > this->function->getMinIntensity() ) ? this->InputData->GetScalarRange()[0] : this->function->getMinIntensity();
  double maxIntensity = (this->InputData->GetScalarRange()[1] < this->function->getMaxIntensity() ) ? this->InputData->GetScalarRange()[1] : this->function->getMaxIntensity();
  double minGradient = (this->LowGradient > this->function->getMinGradient() ) ? this->LowGradient : this->function->getMinGradient();
  double maxGradient = (this->HighGradient < this->function->getMaxGradient() ) ? this->HighGradient : this->function->getMaxGradient();
  
  //estimate the gradient ranges
  this->LowGradient = 0;
  this->HighGradient = abs(this->InputData->GetScalarRange()[0]-this->InputData->GetScalarRange()[1]) / std::min( this->InputData->GetSpacing()[0], std::min(this->InputData->GetSpacing()[1], this->InputData->GetSpacing()[2] ) );
  minGradient = (minGradient > this->inExFunction->getMinGradient() ) ? minGradient : this->inExFunction->getMinGradient();
  maxGradient = (maxGradient < this->inExFunction->getMaxGradient() ) ? maxGradient : this->inExFunction->getMaxGradient();
  double gradientOffset = maxGradient * 0.8;
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
  float* LocalAmbientTransferFunction = new float[this->FunctionSize * this->FunctionSize];
  float* LocalDiffuseTransferFunction = new float[this->FunctionSize * this->FunctionSize];
  float* LocalSpecularTransferFunction = new float[this->FunctionSize * this->FunctionSize];
  float* LocalSpecularPowerTransferFunction = new float[this->FunctionSize * this->FunctionSize];

  //populate the table
  this->function->GetTransferTable(LocalColorRedTransferFunction, LocalColorGreenTransferFunction, LocalColorBlueTransferFunction, LocalAlphaTransferFunction,
    this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);
  this->function->GetShadingTable(LocalAmbientTransferFunction, LocalDiffuseTransferFunction, LocalSpecularTransferFunction, LocalSpecularPowerTransferFunction,
    this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);
  this->inExFunction->GetTransferTable(0, 0, 0, LocalInExTransferFunction,
    this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);

  //map the trasfer functions to textures for fast access
  this->TransInfo.functionSize = this->FunctionSize;
  this->ReserveGPU();
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_loadTextures(this->TransInfo,
    LocalColorRedTransferFunction,
    LocalColorGreenTransferFunction,
    LocalColorBlueTransferFunction,
    LocalAlphaTransferFunction,
    LocalAmbientTransferFunction,
    LocalDiffuseTransferFunction,
    LocalSpecularTransferFunction,
    LocalSpecularPowerTransferFunction,
    LocalInExTransferFunction,
    this->GetStream() );

  //clean up the garbage
  delete LocalColorRedTransferFunction;
  delete LocalColorGreenTransferFunction;
  delete LocalColorBlueTransferFunction;
  delete LocalAlphaTransferFunction;
  delete LocalAmbientTransferFunction;
  delete LocalDiffuseTransferFunction;
  delete LocalSpecularTransferFunction;
  delete LocalSpecularPowerTransferFunction;
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

void vtkCuda2DInExLogicTransferFunctionInformationHandler::SetUseBlackKeyhole(bool t){
  this->UseBlackKeyhole = t;
  this->TransInfo.useBlackKeyhole = t;
  this->Modified();
}