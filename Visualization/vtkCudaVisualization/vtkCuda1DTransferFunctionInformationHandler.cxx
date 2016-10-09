/*=========================================================================

Program:   Robarts Visualization Toolkit

Copyright (c) John Stuart Haberl Baxter, Robarts Research Institute

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/** @file vtkCuda1DTransferFunctionInformationHandler.cxx
 *
 *  @brief Implementation of an internal class for vtkCudaVolumeMapper which manages information regarding the volume and transfer function
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on May 11, 2012
 *
 */

#include "CUDA_vtkCuda1DVolumeMapper_renderAlgo.h"
#include "vtkColorTransferFunction.h"
#include "vtkColorTransferFunction.h"
#include "vtkCuda1DTransferFunctionInformationHandler.h"
#include "vtkImageData.h"
#include "vtkMatrix4x4.h"
#include "vtkObjectFactory.h"
#include "vtkPiecewiseFunction.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include <vtkVersion.h>

vtkStandardNewMacro(vtkCuda1DTransferFunctionInformationHandler);

vtkCuda1DTransferFunctionInformationHandler::vtkCuda1DTransferFunctionInformationHandler()
{
  this->ColourFunction = NULL;
  this->OpacityFunction = NULL;
  this->GradientOpacityFunction = NULL;
  this->UseGradientOpacity = false;

  this->FunctionSize = 512;
  this->LastModifiedTime = 0;

  this->TransInfo.alphaTransferArray1D = 0;
  this->TransInfo.galphaTransferArray1D = 0;
  this->TransInfo.colorRTransferArray1D = 0;
  this->TransInfo.colorGTransferArray1D = 0;
  this->TransInfo.colorBTransferArray1D = 0;

  this->InputData = NULL;
  this->Reinitialize();
}

vtkCuda1DTransferFunctionInformationHandler::~vtkCuda1DTransferFunctionInformationHandler()
{
  this->Deinitialize();
  this->SetInputData(NULL, 0);
  if( this->ColourFunction )
  {
    this->ColourFunction->UnRegister(this);
  }
  if( this->OpacityFunction )
  {
    this->OpacityFunction->UnRegister(this);
  }
  if( this->GradientOpacityFunction )
  {
    this->GradientOpacityFunction->UnRegister(this);
  }
}

void vtkCuda1DTransferFunctionInformationHandler::Deinitialize(int withData)
{
  this->ReserveGPU();
  CUDA_vtkCuda1DVolumeMapper_renderAlgo_UnloadTextures( this->TransInfo, this->GetStream() );
}

void vtkCuda1DTransferFunctionInformationHandler::Reinitialize(int withData)
{
  LastModifiedTime = 0;
  UpdateTransferFunction();
}

void vtkCuda1DTransferFunctionInformationHandler::SetInputData(vtkImageData* inputData, int index)
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

//----------------------------------------------------------------------------
vtkImageData* vtkCuda1DTransferFunctionInformationHandler::GetInputData() const
{
  return InputData;
}

//----------------------------------------------------------------------------
const cuda1DTransferFunctionInformation& vtkCuda1DTransferFunctionInformationHandler::GetTransferFunctionInfo() const
{
  return (this->TransInfo);
}

void vtkCuda1DTransferFunctionInformationHandler::SetColourTransferFunction(vtkColorTransferFunction* f)
{
  if( f != this->ColourFunction )
  {
    if(this->ColourFunction)
    {
      this->ColourFunction->UnRegister(this);
    }
    this->ColourFunction = f;
    if(this->ColourFunction)
    {
      this->ColourFunction->Register(this);
    }
    this->LastModifiedTime = 0;
    this->Modified();
  }
}

void vtkCuda1DTransferFunctionInformationHandler::SetOpacityTransferFunction(vtkPiecewiseFunction* f)
{
  if( f != this->OpacityFunction )
  {
    if(this->OpacityFunction)
    {
      this->OpacityFunction->UnRegister(this);
    }
    this->OpacityFunction = f;
    if(this->OpacityFunction)
    {
      this->OpacityFunction->Register(this);
    }
    this->LastModifiedTime = 0;
    this->Modified();
  }
}

void vtkCuda1DTransferFunctionInformationHandler::SetGradientOpacityTransferFunction(vtkPiecewiseFunction* f)
{
  if( f != this->GradientOpacityFunction )
  {
    if(this->GradientOpacityFunction)
    {
      this->GradientOpacityFunction->UnRegister(this);
    }
    this->GradientOpacityFunction = f;
    if(this->GradientOpacityFunction)
    {
      this->GradientOpacityFunction->Register(this);
    }
    this->LastModifiedTime = 0;
    this->Modified();
  }
}

void vtkCuda1DTransferFunctionInformationHandler::UpdateTransferFunction()
{
  //if we don't need to update the transfer function, don't
  if(!this->ColourFunction || !this->OpacityFunction ||
      (this->ColourFunction->GetMTime() <= LastModifiedTime &&
       this->OpacityFunction->GetMTime() <= LastModifiedTime) )
  {
    return;
  }
  LastModifiedTime = (this->ColourFunction->GetMTime() > this->OpacityFunction->GetMTime()) ?
                     this->ColourFunction->GetMTime() : this->OpacityFunction->GetMTime();

  //get the ranges from the transfer function
  double minIntensity;
  double maxIntensity;
  this->OpacityFunction->GetRange( minIntensity, maxIntensity );
  minIntensity = (this->InputData->GetScalarRange()[0] > minIntensity ) ? this->InputData->GetScalarRange()[0] : minIntensity;
  maxIntensity = (this->InputData->GetScalarRange()[1] < maxIntensity ) ? this->InputData->GetScalarRange()[1] : maxIntensity;

  //get the gradient ranges from the transfer function
  double minGradient;
  double maxGradient;
  this->GradientOpacityFunction->GetRange( minGradient, maxGradient );

  //figure out the multipliers for applying the transfer function in GPU
  this->TransInfo.intensityLow = minIntensity;
  this->TransInfo.intensityMultiplier = 1.0 / ( maxIntensity - minIntensity );
  this->TransInfo.gradientLow = minGradient;
  this->TransInfo.gradientMultiplier = 1.0 / ( maxGradient - minGradient );

  //create local buffers to house the transfer function
  float* LocalColorRedTransferFunction = new float[this->FunctionSize];
  float* LocalColorGreenTransferFunction = new float[this->FunctionSize];
  float* LocalColorBlueTransferFunction = new float[this->FunctionSize];
  float* LocalColorWholeTransferFunction = new float[3*this->FunctionSize];
  float* LocalAlphaTransferFunction = new float[this->FunctionSize];
  float* LocalGAlphaTransferFunction = new float[this->FunctionSize];

  memset( (void*) LocalColorRedTransferFunction, 0.0f, sizeof(float) * this->FunctionSize);
  memset( (void*) LocalColorGreenTransferFunction, 0.0f, sizeof(float) * this->FunctionSize);
  memset( (void*) LocalColorBlueTransferFunction, 0.0f, sizeof(float) * this->FunctionSize);
  memset( (void*) LocalColorWholeTransferFunction, 0.0f, 3*sizeof(float) * this->FunctionSize);
  memset( (void*) LocalAlphaTransferFunction, 1.0f, sizeof(float) * this->FunctionSize);
  memset( (void*) LocalGAlphaTransferFunction, 1.0f, sizeof(float) * this->FunctionSize);

  //populate the table
  this->OpacityFunction->GetTable( minIntensity, maxIntensity, this->FunctionSize,
                                   LocalAlphaTransferFunction );
  this->GradientOpacityFunction->GetTable( minGradient, maxGradient, this->FunctionSize,
      LocalGAlphaTransferFunction );
  this->ColourFunction->GetTable( minIntensity, maxIntensity, this->FunctionSize,
                                  LocalColorWholeTransferFunction );
  for( int i = 0; i < this->FunctionSize; i++ )
  {
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
      LocalGAlphaTransferFunction,
      this->GetStream() );

  //clean up the garbage
  delete LocalColorRedTransferFunction;
  delete LocalColorGreenTransferFunction;
  delete LocalColorBlueTransferFunction;
  delete LocalColorWholeTransferFunction;
  delete LocalAlphaTransferFunction;
  delete LocalGAlphaTransferFunction;
}

void vtkCuda1DTransferFunctionInformationHandler::Update(vtkVolume* vol)
{

  //get shading params
  this->TransInfo.Ambient = 1.0f;
  this->TransInfo.Diffuse = 0.0f;
  this->TransInfo.Specular.x = 0.0f;
  this->TransInfo.Specular.y = 1.0f;
  if( vol && vol->GetProperty() )
  {
    int shadingType = vol->GetProperty()->GetShade();
    if(shadingType)
    {
      this->TransInfo.Ambient = vol->GetProperty()->GetAmbient();
      this->TransInfo.Diffuse = vol->GetProperty()->GetDiffuse();
      this->TransInfo.Specular.x = vol->GetProperty()->GetSpecular();
      this->TransInfo.Specular.y = vol->GetProperty()->GetSpecularPower();
    }
  }

  if(this->InputData)
  {
    this->Modified();
  }
  if(this->ColourFunction && this->OpacityFunction)
  {
    this->UpdateTransferFunction();
    this->Modified();
  }
}

