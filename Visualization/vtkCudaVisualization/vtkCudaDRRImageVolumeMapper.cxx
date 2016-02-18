#include "CUDA_containerOutputImageInformation.h"
#include "CUDA_containerRendererInformation.h"
#include "CUDA_containerVolumeInformation.h"
#include "CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "vtkCamera.h"
#include "vtkCudaDRRImageVolumeMapper.h"
#include "vtkCudaVolumeInformationHandler.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"

vtkStandardNewMacro(vtkCudaDRRImageVolumeMapper);

vtkCudaDRRImageVolumeMapper::vtkCudaDRRImageVolumeMapper()
{
  this->Reinitialize();

  //default to Houndsfield units at 50keV
  this->CTIntercept = 0.022;
  this->CTSlope = 0.000022;
  this->CTOffset = 1.0;
}

vtkCudaDRRImageVolumeMapper::~vtkCudaDRRImageVolumeMapper()
{
  this->Deinitialize();
}

void vtkCudaDRRImageVolumeMapper::Deinitialize(int withData)
{
  this->vtkCudaVolumeMapper::Deinitialize(withData);
  this->ReserveGPU();

  for( int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++ )
  {
    if( this->SourceData[i] )
    {
      CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]), this->GetStream());
    }
  }
}

void vtkCudaDRRImageVolumeMapper::Reinitialize(int withData)
{
  this->vtkCudaVolumeMapper::Reinitialize(withData);
  for( int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++ )
  {
    this->SourceData[i] = 0;
  }
  this->ReserveGPU();
  CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_changeFrame(this->SourceData[this->currFrame], this->GetStream());
}

void vtkCudaDRRImageVolumeMapper::SetInputInternal(vtkImageData * input, int index)
{

  if( input->GetNumberOfScalarComponents() != 1 )
  {
    vtkErrorMacro(<<"Input must have 1 components.");
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
    vtkErrorMacro(<<"Input cannot be of that type.");
    return;
  }

  //load data onto the GPU and clean up the CPU
  if(!this->erroredOut)
  {
    this->ReserveGPU();
    this->erroredOut = !CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_loadImageInfo( buffer, VolumeInfoHandler->GetVolumeInfo(),
                       &(this->SourceData[index]), this->GetStream());
  }

  //deallocate memory
  this->ReserveGPU();
  CUDA_deallocateMemory( (void*) buffer );
}

void vtkCudaDRRImageVolumeMapper::ChangeFrameInternal(int frame)
{
}

void vtkCudaDRRImageVolumeMapper::InternalRender (  vtkRenderer* ren, vtkVolume* vol,
    const cudaRendererInformation& rendererInfo,
    const cudaVolumeInformation& volumeInfo,
    const cudaOutputImageInformation& outputInfo )
{

  //perform the render
  this->ReserveGPU();
  this->erroredOut = !CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_doRender(outputInfo, rendererInfo, volumeInfo,
                     this->CTIntercept, this->CTSlope, this->CTOffset, this->SourceData[this->currFrame], this->GetStream() );
}

void vtkCudaDRRImageVolumeMapper::ClearInputInternal()
{
  this->ReserveGPU();
  for( int i = 0; i < VTKCUDAVOLUMEMAPPER_UPPER_BOUND; i++ )
  {
    if( this->SourceData[i] )
    {
      CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_clearImageArray(&(this->SourceData[i]), this->GetStream());
    }
  }
}
