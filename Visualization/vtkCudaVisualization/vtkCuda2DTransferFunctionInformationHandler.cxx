/*=========================================================================

Program:   Robarts Visualization Toolkit

Copyright (c) John Stuart Haberl Baxter, Robarts Research Institute

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "CUDA_container2DTransferFunctionInformation.h"
#include "CUDA_vtkCuda2DVolumeMapper_renderAlgo.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCuda2DTransferFunctionInformationHandler.h"
#include "vtkImageData.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include <vtkVersion.h>
#include <algorithm>

vtkStandardNewMacro(vtkCuda2DTransferFunctionInformationHandler);

vtkCuda2DTransferFunctionInformationHandler::vtkCuda2DTransferFunctionInformationHandler()
{
  this->Function = NULL;
  this->KeyholeFunction = NULL;

  this->FunctionSize = 512;
  this->LowGradient = 0;
  this->HighGradient = 10;
  this->LastModifiedTime = 0;

  this->TransInfo.alphaTransferArray2D = 0;
  this->TransInfo.ambientTransferArray2D = 0;
  this->TransInfo.diffuseTransferArray2D = 0;
  this->TransInfo.specularTransferArray2D = 0;
  this->TransInfo.specularPowerTransferArray2D = 0;
  this->TransInfo.colorRTransferArray2D = 0;
  this->TransInfo.colorGTransferArray2D = 0;
  this->TransInfo.colorBTransferArray2D = 0;

  this->TransInfo.useSecondTransferFunction = false;
  this->TransInfo.K_alphaTransferArray2D = 0;
  this->TransInfo.K_ambientTransferArray2D = 0;
  this->TransInfo.K_diffuseTransferArray2D = 0;
  this->TransInfo.K_specularTransferArray2D = 0;
  this->TransInfo.K_specularPowerTransferArray2D = 0;
  this->TransInfo.K_colorRTransferArray2D = 0;
  this->TransInfo.K_colorGTransferArray2D = 0;
  this->TransInfo.K_colorBTransferArray2D = 0;

  this->InputData = NULL;
}

vtkCuda2DTransferFunctionInformationHandler::~vtkCuda2DTransferFunctionInformationHandler()
{
  this->Deinitialize();
  this->SetInputData(NULL, 0);
  if (this->Function)
  {
    this->Function->UnRegister(this);
  }
  if (this->KeyholeFunction)
  {
    this->KeyholeFunction->UnRegister(this);
  }
}

void vtkCuda2DTransferFunctionInformationHandler::Deinitialize(bool withData /*= false*/)
{
  this->ReserveGPU();
  CUDA_vtkCuda2DVolumeMapper_renderAlgo_unloadTextures(this->TransInfo, this->GetStream());
}

void vtkCuda2DTransferFunctionInformationHandler::Reinitialize(bool withData /*= false*/)
{
  this->LastModifiedTime = 0;
  this->UpdateTransferFunction();
}

void vtkCuda2DTransferFunctionInformationHandler::SetInputData(vtkImageData* inputData, int index)
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

vtkImageData* vtkCuda2DTransferFunctionInformationHandler::GetInputData() const
{
  return InputData;
}

const cuda2DTransferFunctionInformation& vtkCuda2DTransferFunctionInformationHandler::GetTransferFunctionInfo() const
{
  return (this->TransInfo);
}

void vtkCuda2DTransferFunctionInformationHandler::SetTransferFunction(vtkCuda2DTransferFunction* f)
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

vtkCuda2DTransferFunction* vtkCuda2DTransferFunctionInformationHandler::GetTransferFunction()
{
  return this->Function;
}

void vtkCuda2DTransferFunctionInformationHandler::SetKeyholeTransferFunction(vtkCuda2DTransferFunction* f)
{
  this->TransInfo.useSecondTransferFunction = (f != 0);
  if (this->KeyholeFunction ==  f)
  {
    return;
  }
  if (this->KeyholeFunction)
  {
    this->KeyholeFunction->UnRegister(this);
  }
  this->KeyholeFunction = f;
  if (this->KeyholeFunction)
  {
    this->KeyholeFunction->Register(this);
  }
  this->LastModifiedTime = 0;
  this->Modified();
}

vtkCuda2DTransferFunction* vtkCuda2DTransferFunctionInformationHandler::GetKeyholeTransferFunction()
{
  return this->KeyholeFunction;
}

void vtkCuda2DTransferFunctionInformationHandler::UpdateTransferFunction()
{
  //if we don't need to update the transfer function, don't
  if (!this->Function || !this->InputData)
  {
    return;
  }
  if (this->KeyholeFunction == 0 && this->Function->GetMTime() <= LastModifiedTime)
  {
    return;
  }
  if (this->KeyholeFunction != 0 && this->KeyholeFunction->GetMTime() <= LastModifiedTime && this->Function->GetMTime() <= LastModifiedTime)
  {
    return;
  }
  if (this->KeyholeFunction)
  {
    LastModifiedTime = (this->KeyholeFunction->GetMTime() > this->Function->GetMTime()) ? this->KeyholeFunction->GetMTime() : this->Function->GetMTime();
  }
  else
  {
    LastModifiedTime = this->Function->GetMTime();
  }

  //tell if we can safely ignore the second function
  if (this->KeyholeFunction)
  {
    if (this->KeyholeFunction->GetNumberOfFunctionObjects() == 0)
    {
      this->TransInfo.useSecondTransferFunction = false;
    }
    else
    {
      this->TransInfo.useSecondTransferFunction = true;
    }
  }

  //get the ranges from the transfer function
  double functionRange[] = {  this->Function->getMinIntensity(), this->Function->getMaxIntensity(),
                              this->Function->getMinGradient(), this->Function->getMaxGradient()
                           };
  if (this->TransInfo.useSecondTransferFunction)
  {
    double kfunctionRange[] = {  this->KeyholeFunction->getMinIntensity(), this->KeyholeFunction->getMaxIntensity(),
                                 this->KeyholeFunction->getMinGradient(), this->KeyholeFunction->getMaxGradient()
                              };
    functionRange[0] = (kfunctionRange[0] < functionRange[0]) ? kfunctionRange[0] : functionRange[0];
    functionRange[1] = (kfunctionRange[1] > functionRange[1]) ? kfunctionRange[1] : functionRange[1];
    functionRange[2] = (kfunctionRange[2] < functionRange[2]) ? kfunctionRange[2] : functionRange[2];
    functionRange[3] = (kfunctionRange[3] > functionRange[3]) ? kfunctionRange[3] : functionRange[3];
  }
  double minIntensity = (this->InputData->GetScalarRange()[0] > functionRange[0]) ? this->InputData->GetScalarRange()[0] : functionRange[0];
  double maxIntensity = (this->InputData->GetScalarRange()[1] < functionRange[1]) ? this->InputData->GetScalarRange()[1] : functionRange[1];

  //estimate the gradient (to get max gradient values)
  this->LowGradient = 0;
  this->HighGradient = abs(this->InputData->GetScalarRange()[0] - this->InputData->GetScalarRange()[1]) / std::min(this->InputData->GetSpacing()[0], std::min(this->InputData->GetSpacing()[1], this->InputData->GetSpacing()[2]));
  double minGradient = (this->LowGradient > functionRange[2]) ? this->LowGradient : functionRange[2];
  double maxGradient = (this->HighGradient < functionRange[3]) ? this->HighGradient : functionRange[3];
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
  float* LocalColorRedTransferFunction =      new float[this->FunctionSize * this->FunctionSize];
  float* LocalColorGreenTransferFunction =    new float[this->FunctionSize * this->FunctionSize];
  float* LocalColorBlueTransferFunction =      new float[this->FunctionSize * this->FunctionSize];
  float* LocalAlphaTransferFunction =        new float[this->FunctionSize * this->FunctionSize];
  float* LocalAmbientTransferFunction =      new float[this->FunctionSize * this->FunctionSize];
  float* LocalDiffuseTransferFunction =      new float[this->FunctionSize * this->FunctionSize];
  float* LocalSpecularTransferFunction =      new float[this->FunctionSize * this->FunctionSize];
  float* LocalSpecularPowerTransferFunction =    new float[this->FunctionSize * this->FunctionSize];
  float* KLocalColorRedTransferFunction =      0;
  float* KLocalColorGreenTransferFunction =    0;
  float* KLocalColorBlueTransferFunction =    0;
  float* KLocalAlphaTransferFunction =      0;
  float* KLocalAmbientTransferFunction =      0;
  float* KLocalDiffuseTransferFunction =      0;
  float* KLocalSpecularTransferFunction =      0;
  float* KLocalSpecularPowerTransferFunction =  0;
  if (this->TransInfo.useSecondTransferFunction)
  {
    KLocalColorRedTransferFunction =    new float[this->FunctionSize * this->FunctionSize];
    KLocalColorGreenTransferFunction =    new float[this->FunctionSize * this->FunctionSize];
    KLocalColorBlueTransferFunction =    new float[this->FunctionSize * this->FunctionSize];
    KLocalAlphaTransferFunction =      new float[this->FunctionSize * this->FunctionSize];
    KLocalAmbientTransferFunction =      new float[this->FunctionSize * this->FunctionSize];
    KLocalDiffuseTransferFunction =      new float[this->FunctionSize * this->FunctionSize];
    KLocalSpecularTransferFunction =    new float[this->FunctionSize * this->FunctionSize];
    KLocalSpecularPowerTransferFunction =  new float[this->FunctionSize * this->FunctionSize];
  }

  //populate the table
  this->Function->GetTransferTable(LocalColorRedTransferFunction, LocalColorGreenTransferFunction, LocalColorBlueTransferFunction, LocalAlphaTransferFunction,
                                   this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);
  this->Function->GetShadingTable(LocalAmbientTransferFunction, LocalDiffuseTransferFunction, LocalSpecularTransferFunction, LocalSpecularPowerTransferFunction,
                                  this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);
  if (this->TransInfo.useSecondTransferFunction)
  {
    this->KeyholeFunction->GetTransferTable(KLocalColorRedTransferFunction, KLocalColorGreenTransferFunction, KLocalColorBlueTransferFunction, KLocalAlphaTransferFunction,
                                            this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);
    this->KeyholeFunction->GetShadingTable(KLocalAmbientTransferFunction, KLocalDiffuseTransferFunction, KLocalSpecularTransferFunction, KLocalSpecularPowerTransferFunction,
                                           this->FunctionSize, this->FunctionSize, minIntensity, maxIntensity, 0, minGradient, maxGradient, gradientOffset, 2);
  }

  //map the transfer functions to textures for fast access
  this->TransInfo.functionSize = this->FunctionSize;
  this->ReserveGPU();
  CUDA_vtkCuda2DVolumeMapper_renderAlgo_loadTextures(this->TransInfo,
      LocalColorRedTransferFunction,
      LocalColorGreenTransferFunction,
      LocalColorBlueTransferFunction,
      LocalAlphaTransferFunction,
      LocalAmbientTransferFunction,
      LocalDiffuseTransferFunction,
      LocalSpecularTransferFunction,
      LocalSpecularPowerTransferFunction,
      KLocalColorRedTransferFunction,
      KLocalColorGreenTransferFunction,
      KLocalColorBlueTransferFunction,
      KLocalAlphaTransferFunction,
      KLocalAmbientTransferFunction,
      KLocalDiffuseTransferFunction,
      KLocalSpecularTransferFunction,
      KLocalSpecularPowerTransferFunction,
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
  if (this->TransInfo.useSecondTransferFunction)
  {
    delete KLocalColorRedTransferFunction;
    delete KLocalColorGreenTransferFunction;
    delete KLocalColorBlueTransferFunction;
    delete KLocalAlphaTransferFunction;
    delete KLocalAmbientTransferFunction;
    delete KLocalDiffuseTransferFunction;
    delete KLocalSpecularTransferFunction;
    delete KLocalSpecularPowerTransferFunction;
  }
}

void vtkCuda2DTransferFunctionInformationHandler::Update()
{
  if (this->InputData)
  {
    this->Modified();
  }
  if (this->Function)
  {
    this->UpdateTransferFunction();
    this->Modified();
  }
}