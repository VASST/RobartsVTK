/*=========================================================================

Program:   Robarts Visualization Toolkit

Copyright (c) John Stuart Haberl Baxter, Robarts Research Institute

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/** @file vtkCudaVolumeMapper.cxx
 *
 *  @brief Implementation of a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on May 12, 2012
 *
 */

#include "CUDA_container2DTransferFunctionInformation.h"
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_vtkCuda2DVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "vtkCamera.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCuda2DTransferFunctionInformationHandler.h"
#include "vtkCuda2DVolumeMapper.h"
#include "vtkCudaVolumeInformationHandler.h"
#include "vtkImageData.h"
#include "vtkMutexLock.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"

vtkStandardNewMacro(vtkCuda2DVolumeMapper);

vtkMutexLock* vtkCuda2DVolumeMapper::tfLock = 0;
vtkCuda2DVolumeMapper::vtkCuda2DVolumeMapper()
{
  this->transferFunctionInfoHandler = vtkCuda2DTransferFunctionInformationHandler::New();
  if (this->tfLock == 0)
  {
    this->tfLock = vtkMutexLock::New();
  }
  else
  {
    this->tfLock->Register(this);
  }
  this->Reinitialize();
}

vtkCuda2DVolumeMapper::~vtkCuda2DVolumeMapper()
{
  this->Deinitialize();
  this->transferFunctionInfoHandler->Delete();
  this->tfLock->UnRegister(this);
}

void vtkCuda2DVolumeMapper::Deinitialize(bool withData /*= false*/)
{
  this->vtkCudaVolumeMapper::Deinitialize(withData);
  this->ReserveGPU();

  for (int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++)
  {
    if (this->SourceData[i])
    {
      CUDA_vtkCuda2DVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]), this->GetStream());
    }
  }
}

void vtkCuda2DVolumeMapper::Reinitialize(bool withData /*= false*/)
{
  this->vtkCudaVolumeMapper::Reinitialize(withData);
  this->transferFunctionInfoHandler->ReplicateObject(this, withData);
  for (int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++)
  {
    this->SourceData[i] = 0;
  }
  this->ReserveGPU();
  CUDA_vtkCuda2DVolumeMapper_renderAlgo_changeFrame(this->SourceData[this->CurrentFrame], this->GetStream());
}

void vtkCuda2DVolumeMapper::SetInputInternal(vtkImageData* input, int index)
{

  if (input->GetNumberOfScalarComponents() != 1)
  {
    vtkErrorMacro("Input must have 1 components.");
    return;
  }

  //convert data to float
  float* buffer = 0;
  const cudaVolumeInformation& VolumeInfo = this->VolumeInfoHandler->GetVolumeInfo();
  if (input->GetScalarType() == VTK_CHAR)
  {
    this->ReserveGPU();
    CUDA_castBuffer<char, float>((char*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else if (input->GetScalarType() == VTK_UNSIGNED_CHAR)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned char, float>((unsigned char*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else if (input->GetScalarType() == VTK_SIGNED_CHAR)
  {
    this->ReserveGPU();
    CUDA_castBuffer<char, float>((char*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else if (input->GetScalarType() == VTK_INT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<int, float>((int*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else if (input->GetScalarType() == VTK_UNSIGNED_INT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned int, float>((unsigned int*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else if (input->GetScalarType() == VTK_SHORT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<short, float>((short*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else if (input->GetScalarType() == VTK_UNSIGNED_SHORT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned short, float>((unsigned short*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else if (input->GetScalarType() == VTK_LONG)
  {
    this->ReserveGPU();
    CUDA_castBuffer<long, float>((long*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else if (input->GetScalarType() == VTK_UNSIGNED_LONG)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned long, float>((unsigned long*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else if (input->GetScalarType() == VTK_FLOAT)
  {
    this->ReserveGPU();
    CUDA_allocBuffer<float>((float*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x * VolumeInfo.VolumeSize.y * VolumeInfo.VolumeSize.z);
  }
  else
  {
    vtkErrorMacro("Input cannot be of that type.");
    return;
  }

  //load data onto the GPU and clean up the CPU
  if (!this->CanRender)
  {
    this->ReserveGPU();
    this->CanRender = !CUDA_vtkCuda2DVolumeMapper_renderAlgo_loadImageInfo(buffer, VolumeInfoHandler->GetVolumeInfo(),
                      &(this->SourceData[index]), this->GetStream());
  }

  //deallocate memory
  this->ReserveGPU();
  CUDA_deallocateMemory((void*) buffer);

  //inform transfer function handler of the data
  this->transferFunctionInfoHandler->SetInputData(input, index);
}

void vtkCuda2DVolumeMapper::ChangeFrameInternal(int frame)
{
}

void vtkCuda2DVolumeMapper::InternalRender(vtkRenderer* ren, vtkVolume* vol,
    const cudaRendererInformation& rendererInfo,
    const cudaVolumeInformation& volumeInfo,
    const cudaOutputImageInformation& outputInfo)
{
  //handle the transfer function changes
  this->transferFunctionInfoHandler->Update();

  //perform the render
  this->tfLock->Lock();
  this->ReserveGPU();
  this->CanRender = !CUDA_vtkCuda2DVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
                    this->transferFunctionInfoHandler->GetTransferFunctionInfo(), this->SourceData[this->CurrentFrame], this->GetStream());
  this->tfLock->Unlock();

}

void vtkCuda2DVolumeMapper::ClearInputInternal()
{
  this->ReserveGPU();
  for (int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++)
  {
    if (this->SourceData[i])
    {
      CUDA_vtkCuda2DVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]), this->GetStream());
    }
  }
}

//give the function to the transfer function handler
void vtkCuda2DVolumeMapper::SetFunction(vtkCuda2DTransferFunction* funct)
{
  this->transferFunctionInfoHandler->SetTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCuda2DVolumeMapper::GetFunction()
{
  return this->transferFunctionInfoHandler->GetTransferFunction();
}

//give the function to the transfer function handler
void vtkCuda2DVolumeMapper::SetKeyholeFunction(vtkCuda2DTransferFunction* funct)
{
  this->transferFunctionInfoHandler->SetKeyholeTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCuda2DVolumeMapper::GetKeyholeFunction()
{
  return this->transferFunctionInfoHandler->GetKeyholeTransferFunction();
}