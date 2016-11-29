/*=========================================================================

Robarts Visualization Toolkit

Copyright (c) 2016 Virtual Augmentation and Simulation for Surgery and Therapy, Robarts Research Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

=========================================================================*/

/** @file vtkCudaVolumeMapper.cxx
 *
 *  @brief Implementation of a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on May 12, 2012
 *
 */

// Local includes
#include "CUDA_container1DTransferFunctionInformation.h"
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_vtkCuda1DVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "vtkCuda1DTransferFunctionInformationHandler.h"
#include "vtkCuda1DVolumeMapper.h"
#include "vtkCudaVolumeInformationHandler.h"

// VTK includes
#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkImageData.h>
#include <vtkMutexLock.h>
#include <vtkObjectFactory.h>
#include <vtkPiecewiseFunction.h>
#include <vtkRenderer.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>

//----------------------------------------------------------------------------

vtkStandardNewMacro(vtkCuda1DVolumeMapper);

//----------------------------------------------------------------------------

vtkMutexLock* vtkCuda1DVolumeMapper::TransferFunctionMutex = 0;

//----------------------------------------------------------------------------
vtkCuda1DVolumeMapper::vtkCuda1DVolumeMapper()
{
  this->TransferFunctionInfoHandler = vtkCuda1DTransferFunctionInformationHandler::New();
  if (this->TransferFunctionMutex == 0)
  {
    this->TransferFunctionMutex = vtkMutexLock::New();
  }
  else
  {
    this->TransferFunctionMutex->Register(this);
  }
  this->Reinitialize();
}

//----------------------------------------------------------------------------
void vtkCuda1DVolumeMapper::Deinitialize(bool withData /*= false*/)
{
  this->vtkCudaVolumeMapper::Deinitialize(withData);
  this->ReserveGPU();
  for (int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++)
  {
    CUDA_vtkCuda1DVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]), this->GetStream());
  }
}

//----------------------------------------------------------------------------
void vtkCuda1DVolumeMapper::Reinitialize(bool withData /*= false*/)
{
  this->vtkCudaVolumeMapper::Reinitialize(withData);
  this->TransferFunctionInfoHandler->ReplicateObject(this, withData);
  for (int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++)
  {
    this->SourceData[i] = 0;
  }
}

//----------------------------------------------------------------------------
vtkCuda1DVolumeMapper::~vtkCuda1DVolumeMapper()
{
  this->Deinitialize();
  TransferFunctionInfoHandler->UnRegister(this);
  this->TransferFunctionMutex->UnRegister(this);
}

//----------------------------------------------------------------------------
void vtkCuda1DVolumeMapper::SetInputInternal(vtkImageData* input, int index)
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
    this->CanRender = !CUDA_vtkCuda1DVolumeMapper_renderAlgo_loadImageInfo(buffer, VolumeInfoHandler->GetVolumeInfo(), &(this->SourceData[index]), this->GetStream());
  }

  //deallocate memory
  this->ReserveGPU();
  CUDA_deallocateMemory((void*) buffer);

  //inform transfer function handler of the data
  this->TransferFunctionInfoHandler->SetInputData(input, index);
}

//----------------------------------------------------------------------------
void vtkCuda1DVolumeMapper::ChangeFrameInternal(int frame)
{
}

//----------------------------------------------------------------------------
void vtkCuda1DVolumeMapper::InternalRender(vtkRenderer* ren, vtkVolume* vol,
    const cudaRendererInformation& rendererInfo,
    const cudaVolumeInformation& volumeInfo,
    const cudaOutputImageInformation& outputInfo)
{
  //handle the transfer function changes
  this->TransferFunctionInfoHandler->SetColourTransferFunction(vol->GetProperty()->GetRGBTransferFunction());
  this->TransferFunctionInfoHandler->SetOpacityTransferFunction(vol->GetProperty()->GetScalarOpacity());
  this->TransferFunctionInfoHandler->SetGradientOpacityTransferFunction(vol->GetProperty()->GetGradientOpacity());
  this->TransferFunctionInfoHandler->SetUseGradientOpacity(vol->GetProperty()->GetDisableGradientOpacity() == 0);
  this->TransferFunctionInfoHandler->Update(vol);

  //perform the render
  this->TransferFunctionMutex->Lock();
  this->ReserveGPU();
  this->CanRender = !CUDA_vtkCuda1DVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
                    this->TransferFunctionInfoHandler->GetTransferFunctionInfo(), this->SourceData[this->CurrentFrame], this->GetStream());
  this->TransferFunctionMutex->Unlock();

}

//----------------------------------------------------------------------------
void vtkCuda1DVolumeMapper::ClearInputInternal()
{
  this->ReserveGPU();
  for (int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++)
  {
    CUDA_vtkCuda1DVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]), this->GetStream());
  }
}