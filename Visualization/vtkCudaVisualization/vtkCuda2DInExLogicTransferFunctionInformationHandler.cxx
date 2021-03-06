/*=========================================================================

Program:   Robarts Visualization Toolkit

Copyright (c) John Stuart Haberl Baxter, Robarts Research Institute

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.h"
#include "vtkCuda2DInExLogicTransferFunctionInformationHandler.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkImageData.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include <vtkVersion.h>
#include <algorithm>

vtkStandardNewMacro(vtkCuda2DInExLogicTransferFunctionInformationHandler);

vtkCuda2DInExLogicTransferFunctionInformationHandler::vtkCuda2DInExLogicTransferFunctionInformationHandler()
{
  this->Function = NULL;
  this->InExFunction = NULL;
  this->UseBlackKeyhole = false;
  this->TransInfo.useBlackKeyhole = false;

  this->FunctionSize = 512;
  this->LowGradient = 0;
  this->HighGradient = 10;
  this->LastModifiedTime = 0;

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

vtkCuda2DInExLogicTransferFunctionInformationHandler::~vtkCuda2DInExLogicTransferFunctionInformationHandler()
{
  this->Deinitialize();
  this->SetInputData(NULL, 0);
  if (this->InExFunction)
  {
    this->InExFunction->UnRegister(this);
  }
  if (this->Function)
  {
    this->Function->UnRegister(this);
  }
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::Deinitialize(bool withData /*= false*/)
{
  this->ReserveGPU();
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_unloadTextures(this->TransInfo, this->GetStream());
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::Reinitialize(bool withData /*= false*/)
{
  this->LastModifiedTime = 0;
  this->UpdateTransferFunction();
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::SetInputData(vtkImageData* inputData, int index)
{
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

void vtkCuda2DInExLogicTransferFunctionInformationHandler::SetVisualizationTransferFunction(vtkCuda2DTransferFunction* f)
{
  if (this->Function ==  f)
  {
    return;
  }
  if (this->Function)
  {
    this->Function->UnRegister(this);
  }
  this->Function = f;
  if (this->Function)
  {
    this->Function->Register(this);
  }
  this->LastModifiedTime = 0;
  this->Modified();
}

vtkCuda2DTransferFunction* vtkCuda2DInExLogicTransferFunctionInformationHandler::GetVisualizationTransferFunction()
{
  return this->Function;
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::SetInExLogicTransferFunction(vtkCuda2DTransferFunction* f)
{
  if (this->InExFunction ==  f)
  {
    return;
  }
  if (this->InExFunction)
  {
    this->InExFunction->UnRegister(this);
  }
  this->InExFunction = f;
  if (this->InExFunction)
  {
    this->InExFunction->Register(this);
  }
  this->LastModifiedTime = 0;
  this->Modified();
}

vtkCuda2DTransferFunction* vtkCuda2DInExLogicTransferFunctionInformationHandler::GetInExLogicTransferFunction()
{
  return this->InExFunction;
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::UpdateTransferFunction()
{
  //if we don't need to update the transfer functions, don't
  if (!this->InputData)
  {
    return;
  }
  if (!this->Function || this->Function->GetMTime() <= LastModifiedTime ||
      !this->InExFunction || this->InExFunction->GetMTime() <= LastModifiedTime)
  {
    return;
  }
  LastModifiedTime = (this->Function->GetMTime() < this->InExFunction->GetMTime()) ?
                     this->InExFunction->GetMTime() : this->Function->GetMTime();

  //get the ranges from the transfer function
  double minIntensity = (this->InputData->GetScalarRange()[0] > this->Function->getMinIntensity()) ? this->InputData->GetScalarRange()[0] : this->Function->getMinIntensity();
  double maxIntensity = (this->InputData->GetScalarRange()[1] < this->Function->getMaxIntensity()) ? this->InputData->GetScalarRange()[1] : this->Function->getMaxIntensity();
  double minGradient = (this->LowGradient > this->Function->getMinGradient()) ? this->LowGradient : this->Function->getMinGradient();
  double maxGradient = (this->HighGradient < this->Function->getMaxGradient()) ? this->HighGradient : this->Function->getMaxGradient();

  //estimate the gradient ranges
  this->LowGradient = 0;
  this->HighGradient = abs(this->InputData->GetScalarRange()[0] - this->InputData->GetScalarRange()[1]) / std::min(this->InputData->GetSpacing()[0], std::min(this->InputData->GetSpacing()[1], this->InputData->GetSpacing()[2]));
  minGradient = (minGradient > this->InExFunction->getMinGradient()) ? minGradient : this->InExFunction->getMinGradient();
  maxGradient = (maxGradient < this->InExFunction->getMaxGradient()) ? maxGradient : this->InExFunction->getMaxGradient();
  double gradientOffset = maxGradient * 0.8;
  maxGradient = (log(maxGradient * maxGradient + gradientOffset) - log(gradientOffset)) / log(2.0) + 1.0;
  minGradient = (log(minGradient * minGradient + gradientOffset) - log(gradientOffset)) / log(2.0);

  //figure out the multipliers for applying the transfer function in GPU
  this->TransInfo.intensityLow = minIntensity;
  this->TransInfo.intensityMultiplier = 1.0 / (maxIntensity - minIntensity);
  this->TransInfo.gradientMultiplier = 1.0 / (maxGradient - minGradient);
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
  this->Function->GetTransferTable(LocalColorRedTransferFunction, LocalColorGreenTransferFunction, LocalColorBlueTransferFunction, LocalAlphaTransferFunction,
                                   this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);
  this->Function->GetShadingTable(LocalAmbientTransferFunction, LocalDiffuseTransferFunction, LocalSpecularTransferFunction, LocalSpecularPowerTransferFunction,
                                  this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);
  this->InExFunction->GetTransferTable(0, 0, 0, LocalInExTransferFunction,
                                       this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);

  //map the transfer functions to textures for fast access
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
  delete LocalInExTransferFunction;
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::Update()
{
  if (this->InputData)
  {
    this->Modified();
  }
  if (this->Function && this->InExFunction)
  {
    this->UpdateTransferFunction();
    this->Modified();
  }
}

void vtkCuda2DInExLogicTransferFunctionInformationHandler::SetUseBlackKeyhole(bool t)
{
  this->UseBlackKeyhole = t;
  this->TransInfo.useBlackKeyhole = t;
  this->Modified();
}