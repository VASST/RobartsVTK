/*=========================================================================

Program:   Robarts Visualization Toolkit

Copyright (c) John Stuart Haberl Baxter, Robarts Research Institute

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaDualImageTransferFunctionInformationHandler.h"
#include "vtkDataArray.h"
#include "vtkImageData.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"

vtkStandardNewMacro(vtkCudaDualImageTransferFunctionInformationHandler);

vtkCudaDualImageTransferFunctionInformationHandler::vtkCudaDualImageTransferFunctionInformationHandler()
{
  this->Function = NULL;
  this->KeyholeFunction = NULL;

  this->FunctionSize = 512;
  this->LastModifiedTime = 0;

  this->TransInfo.alphaTransferArrayDualImage = 0;
  this->TransInfo.ambientTransferArrayDualImage = 0;
  this->TransInfo.diffuseTransferArrayDualImage = 0;
  this->TransInfo.specularTransferArrayDualImage = 0;
  this->TransInfo.specularPowerTransferArrayDualImage = 0;
  this->TransInfo.colorRTransferArrayDualImage = 0;
  this->TransInfo.colorGTransferArrayDualImage = 0;
  this->TransInfo.colorBTransferArrayDualImage = 0;

  this->TransInfo.useSecondTransferFunction = false;
  this->TransInfo.K_alphaTransferArrayDualImage = 0;
  this->TransInfo.K_ambientTransferArrayDualImage = 0;
  this->TransInfo.K_diffuseTransferArrayDualImage = 0;
  this->TransInfo.K_specularTransferArrayDualImage = 0;
  this->TransInfo.K_specularPowerTransferArrayDualImage = 0;
  this->TransInfo.K_colorRTransferArrayDualImage = 0;
  this->TransInfo.K_colorGTransferArrayDualImage = 0;
  this->TransInfo.K_colorBTransferArrayDualImage = 0;

  this->InputData = NULL;
}

vtkCudaDualImageTransferFunctionInformationHandler::~vtkCudaDualImageTransferFunctionInformationHandler()
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

void vtkCudaDualImageTransferFunctionInformationHandler::Deinitialize(bool withData /*= false*/)
{
  this->ReserveGPU();
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_unloadTextures(this->TransInfo, this->GetStream());
}

void vtkCudaDualImageTransferFunctionInformationHandler::Reinitialize(bool withData /*= false*/)
{
  this->LastModifiedTime = 0;
  this->UpdateTransferFunction();
}

void vtkCudaDualImageTransferFunctionInformationHandler::SetInputData(vtkImageData* inputData, int index)
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

void vtkCudaDualImageTransferFunctionInformationHandler::SetTransferFunction(vtkCuda2DTransferFunction* f)
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

vtkCuda2DTransferFunction* vtkCudaDualImageTransferFunctionInformationHandler::GetTransferFunction()
{
  return this->Function;
}

void vtkCudaDualImageTransferFunctionInformationHandler::SetKeyholeTransferFunction(vtkCuda2DTransferFunction* f)
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

vtkCuda2DTransferFunction* vtkCudaDualImageTransferFunctionInformationHandler::GetKeyholeTransferFunction()
{
  return this->KeyholeFunction;
}

void vtkCudaDualImageTransferFunctionInformationHandler::UpdateTransferFunction()
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
  double scalarRange[4];
  this->InputData->GetPointData()->GetScalars()->GetRange(scalarRange, 0);
  this->InputData->GetPointData()->GetScalars()->GetRange(scalarRange + 2, 1);
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

  double minIntensity1 = (scalarRange[0] > functionRange[0]) ? scalarRange[0] : functionRange[0];
  double maxIntensity1 = (scalarRange[1] < functionRange[1]) ? scalarRange[1] : functionRange[1];
  double minIntensity2 = (scalarRange[2] > functionRange[2]) ? scalarRange[2] : functionRange[2];
  double maxIntensity2 = (scalarRange[3] < functionRange[3]) ? scalarRange[3] : functionRange[3];

  //figure out the multipliers for applying the transfer function in GPU
  this->TransInfo.intensity1Low = minIntensity1;
  this->TransInfo.intensity1Multiplier = 1.0 / (maxIntensity1 - minIntensity1);
  this->TransInfo.intensity2Low = minIntensity2;
  this->TransInfo.intensity2Multiplier = 1.0 / (maxIntensity2 - minIntensity2);

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
                                   this->FunctionSize, this->FunctionSize, minIntensity1, maxIntensity1, 0, minIntensity2, maxIntensity2, 0, 0);
  this->Function->GetShadingTable(LocalAmbientTransferFunction, LocalDiffuseTransferFunction, LocalSpecularTransferFunction, LocalSpecularPowerTransferFunction,
                                  this->FunctionSize, this->FunctionSize, minIntensity1, maxIntensity1, 0, minIntensity2, maxIntensity2, 0, 0);
  if (this->TransInfo.useSecondTransferFunction)
  {
    this->KeyholeFunction->GetTransferTable(KLocalColorRedTransferFunction, KLocalColorGreenTransferFunction, KLocalColorBlueTransferFunction, KLocalAlphaTransferFunction,
                                            this->FunctionSize, this->FunctionSize, minIntensity1, maxIntensity1, 0, minIntensity2, maxIntensity2, 0, 0);
    this->KeyholeFunction->GetShadingTable(KLocalAmbientTransferFunction, KLocalDiffuseTransferFunction, KLocalSpecularTransferFunction, KLocalSpecularPowerTransferFunction,
                                           this->FunctionSize, this->FunctionSize, minIntensity1, maxIntensity1, 0, minIntensity2, maxIntensity2, 0, 0);
  }

  //map the transfer functions to textures for fast access
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

void vtkCudaDualImageTransferFunctionInformationHandler::Update()
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