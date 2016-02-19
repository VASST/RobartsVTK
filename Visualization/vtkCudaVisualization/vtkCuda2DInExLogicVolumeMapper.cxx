/** @file vtkVolumeMapper.cxx
 *
 *  @brief Header file defining a volume mapper (ray caster) using CUDA kernels for parallel ray calculation
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 29, 2011
 *
 */

#include "CUDA_container2DTransferFunctionInformation.h"
#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "vtkCuda2DInExLogicTransferFunctionInformationHandler.h"
#include "vtkCuda2DInExLogicVolumeMapper.h"
#include "vtkCudaRendererInformationHandler.h"
#include "vtkCudaVolumeInformationHandler.h"
#include "vtkImageData.h"
#include "vtkMutexLock.h"
#include "vtkObjectFactory.h"
#include "vtkPlanes.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"

vtkStandardNewMacro(vtkCuda2DInExLogicVolumeMapper);

vtkMutexLock* vtkCuda2DInExLogicVolumeMapper::tfLock = 0;
vtkCuda2DInExLogicVolumeMapper::vtkCuda2DInExLogicVolumeMapper()
{
  this->UseBlackKeyhole = false;
  this->transferFunctionInfoHandler = vtkCuda2DInExLogicTransferFunctionInformationHandler::New();
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

vtkCuda2DInExLogicVolumeMapper::~vtkCuda2DInExLogicVolumeMapper()
{
  this->Deinitialize();
  this->transferFunctionInfoHandler->Delete();
  this->tfLock->UnRegister(this);
}

void vtkCuda2DInExLogicVolumeMapper::Reinitialize(int withData)
{
  this->vtkCudaVolumeMapper::Reinitialize(withData);
  this->transferFunctionInfoHandler->ReplicateObject(this, withData);
  this->ReserveGPU();
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_initImageArray(this->GetStream());
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_changeFrame(this->currFrame, this->GetStream());
}

void vtkCuda2DInExLogicVolumeMapper::Deinitialize(int withData)
{
  this->vtkCudaVolumeMapper::Deinitialize(withData);
  this->ReserveGPU();
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_clearImageArray(this->GetStream());
}

void vtkCuda2DInExLogicVolumeMapper::SetInputInternal(vtkImageData * input, int index)
{

  if( input->GetNumberOfScalarComponents() != 1 )
  {
    vtkErrorMacro("Input must have 1 components.");
    return;
  }

  //convert data to float
  float* buffer = 0;
  const cudaVolumeInformation& VolumeInfo = this->VolumeInfoHandler->GetVolumeInfo();
  if(input->GetScalarType() == VTK_CHAR)
  {
    this->ReserveGPU();
    CUDA_castBuffer<char,float>( (char*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_UNSIGNED_CHAR)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned char,float>( (unsigned char*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_SIGNED_CHAR)
  {
    this->ReserveGPU();
    CUDA_castBuffer<char,float>( (char*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_INT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<int,float>( (int*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_UNSIGNED_INT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned int,float>( (unsigned int*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_SHORT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<short,float>( (short*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_UNSIGNED_SHORT)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned short,float>( (unsigned short*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_LONG)
  {
    this->ReserveGPU();
    CUDA_castBuffer<long,float>( (long*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_UNSIGNED_LONG)
  {
    this->ReserveGPU();
    CUDA_castBuffer<unsigned long,float>( (unsigned long*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else if(input->GetScalarType() == VTK_FLOAT)
  {
    this->ReserveGPU();
    CUDA_allocBuffer<float>( (float*) input->GetScalarPointer(), &buffer, VolumeInfo.VolumeSize.x*VolumeInfo.VolumeSize.y*VolumeInfo.VolumeSize.z );
  }
  else
  {
    vtkErrorMacro("Input cannot be of that type.");
    return;
  }

  //load data onto the GPU and clean up the CPU
  this->ReserveGPU();
  this->erroredOut = !CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_loadImageInfo( buffer, VolumeInfoHandler->GetVolumeInfo(), index, this->GetStream());

  //deallocate memory
  this->ReserveGPU();
  CUDA_deallocateMemory( (void*) buffer );

  //inform transfer function handler of the data
  this->transferFunctionInfoHandler->SetInputData(input,index);
}

void vtkCuda2DInExLogicVolumeMapper::ChangeFrameInternal(int frame)
{
  if(!this->erroredOut)
  {
    this->ReserveGPU();
    this->erroredOut = !CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_changeFrame(frame, this->GetStream());
  }
}

void vtkCuda2DInExLogicVolumeMapper::InternalRender (  vtkRenderer* ren, vtkVolume* vol,
    const cudaRendererInformation& rendererInfo,
    const cudaVolumeInformation& volumeInfo,
    const cudaOutputImageInformation& outputInfo )
{
  //handle the transfer function changes
  this->transferFunctionInfoHandler->Update();

  //perform the render
  this->tfLock->Lock();
  this->ReserveGPU();
  this->erroredOut = !CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
                     this->transferFunctionInfoHandler->GetTransferFunctionInfo(), this->GetStream() );
  this->tfLock->Unlock();

}

void vtkCuda2DInExLogicVolumeMapper::ClearInputInternal()
{
  this->ReserveGPU();
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_clearImageArray(this->GetStream());
}

//give the function to the transfer function handler
void vtkCuda2DInExLogicVolumeMapper::SetVisualizationFunction(vtkCuda2DTransferFunction* funct)
{
  this->transferFunctionInfoHandler->SetVisualizationTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCuda2DInExLogicVolumeMapper::GetVisualizationFunction()
{
  return this->transferFunctionInfoHandler->GetVisualizationTransferFunction();
}

//give the function to the transfer function handler
void vtkCuda2DInExLogicVolumeMapper::SetInExLogicFunction(vtkCuda2DTransferFunction* funct)
{
  this->transferFunctionInfoHandler->SetInExLogicTransferFunction(funct);
}

//collect the function from the transfer function handler
vtkCuda2DTransferFunction* vtkCuda2DInExLogicVolumeMapper::GetInExLogicFunction()
{
  return this->transferFunctionInfoHandler->GetInExLogicTransferFunction();
}

void vtkCuda2DInExLogicVolumeMapper::SetUseBlackKeyhole(bool t)
{
  this->UseBlackKeyhole = t;
  this->transferFunctionInfoHandler->SetUseBlackKeyhole(t);
  this->Modified();
}