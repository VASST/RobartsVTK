/** @file vtkVolumeMapper.h
 *
 *  @brief Implementation defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 *
 */

#include "CUDA_containerDualImageTransferFunctionInformation.h"
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "vtkCamera.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaDualImageTransferFunctionInformationHandler.h"
#include "vtkCudaDualImageVolumeMapper.h"
#include "vtkCudaVolumeInformationHandler.h"
#include "vtkImageData.h"
#include "vtkMutexLock.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"

vtkStandardNewMacro(vtkCudaDualImageVolumeMapper);

vtkMutexLock* vtkCudaDualImageVolumeMapper::tfLock = 0;
vtkCudaDualImageVolumeMapper::vtkCudaDualImageVolumeMapper()
{
  this->transferFunctionInfoHandler = vtkCudaDualImageTransferFunctionInformationHandler::New();
  if( this->tfLock == 0 )
  {
    this->tfLock = vtkMutexLock::New();
  }
  else
  {
    this->tfLock->Register(this);
  }
  this->Reinitialize();
}

vtkCudaDualImageVolumeMapper::~vtkCudaDualImageVolumeMapper()
{
  this->Deinitialize();
  this->transferFunctionInfoHandler->Delete();
  this->tfLock->UnRegister(this);
}

void vtkCudaDualImageVolumeMapper::Deinitialize(int withData)
{
  this->vtkCudaVolumeMapper::Deinitialize(withData);
  this->ReserveGPU();
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_clearImageArray(this->GetStream());
}

void vtkCudaDualImageVolumeMapper::Reinitialize(int withData)
{
  this->vtkCudaVolumeMapper::Reinitialize(withData);
  this->transferFunctionInfoHandler->ReplicateObject(this, withData);
  this->ReserveGPU();
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_initImageArray(this->GetStream());
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_changeFrame(this->currFrame, this->GetStream());
}

void vtkCudaDualImageVolumeMapper::SetInputInternal(vtkImageData * input, int index)
{

  if( input->GetNumberOfScalarComponents() != 2 )
  {
    vtkErrorMacro(<<"Input must have 2 components.");
    return;
  }

  //convert data to float
  const cudaVolumeInformation& VolumeInfo = this->VolumeInfoHandler->GetVolumeInfo();
  float* buffer = 0;
  if(input->GetScalarType() == VTK_CHAR)
  {
    this->ReserveGPU();
    CUDA_castBuffer<char,float>( (char*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_UNSIGNED_CHAR)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned char,float>( (unsigned char*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_SIGNED_CHAR)
  {
    this->ReserveGPU();
    CUDA_castBuffer<char,float>( (char*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_INT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<int,float>( (int*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_UNSIGNED_INT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned int,float>( (unsigned int*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_SHORT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<short,float>( (short*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_UNSIGNED_SHORT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned short,float>( (unsigned short*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_LONG)
  {
    this->ReserveGPU();
    CUDA_castBuffer<long,float>( (long*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_UNSIGNED_LONG)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned long,float>( (unsigned long*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_FLOAT)
  {
    this->ReserveGPU();
    CUDA_allocBuffer<float>( (float*) input->GetScalarPointer(), &buffer, 2*VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else
  {
    vtkErrorMacro(<<"Input cannot be of that type.");
    return;
  }

  //load data onto the GPU and clean up the CPU
  if(!this->erroredOut)
  {
    this->ReserveGPU();
    this->erroredOut = !CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_loadImageInfo( buffer, VolumeInfoHandler->GetVolumeInfo(),
                       index, this->GetStream());
  }

  //deallocate memory
  this->ReserveGPU();
  CUDA_deallocateMemory( (void*) buffer );

  //inform transfer function handler of the data
  this->transferFunctionInfoHandler->SetInputData(input,index);
}

void vtkCudaDualImageVolumeMapper::ChangeFrameInternal(int frame)
{
  if(!this->erroredOut)
  {
    this->ReserveGPU();
    this->erroredOut = !CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_changeFrame(frame, this->GetStream());
  }
}

void vtkCudaDualImageVolumeMapper::InternalRender (  vtkRenderer* ren, vtkVolume* vol,
    const cudaRendererInformation& rendererInfo,
    const cudaVolumeInformation& volumeInfo,
    const cudaOutputImageInformation& outputInfo )
{
  //handle the transfer function changes
  this->transferFunctionInfoHandler->Update();

  //perform the render
  this->tfLock->Lock();
  this->ReserveGPU();
  this->erroredOut = !CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
                     this->transferFunctionInfoHandler->GetTransferFunctionInfo(), this->GetStream() );
  this->tfLock->Unlock();

}

void vtkCudaDualImageVolumeMapper::ClearInputInternal()
{
  this->ReserveGPU();
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_clearImageArray(this->GetStream());
}

//give the function to the transfer function handler
void vtkCudaDualImageVolumeMapper::SetFunction(vtkCuda2DTransferFunction* funct)
{
  this->transferFunctionInfoHandler->SetTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCudaDualImageVolumeMapper::GetFunction()
{
  return this->transferFunctionInfoHandler->GetTransferFunction();
}

//give the function to the transfer function handler
void vtkCudaDualImageVolumeMapper::SetKeyholeFunction(vtkCuda2DTransferFunction* funct)
{
  this->transferFunctionInfoHandler->SetKeyholeTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCudaDualImageVolumeMapper::GetKeyholeFunction()
{
  return this->transferFunctionInfoHandler->GetKeyholeTransferFunction();
}