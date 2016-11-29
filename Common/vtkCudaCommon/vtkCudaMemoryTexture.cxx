/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "RobartsVTKConfigure.h"

#include "vtkCudaMemoryTexture.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include <vtkVersion.h>

// Fixed order
#include "vtkOpenGLError.h"
#include "vtkOpenGL.h"
#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"
// End fixed order

//----------------------------------------------------------------------------

vtkStandardNewMacro(vtkCudaMemoryTexture);

//----------------------------------------------------------------------------

bool vtkCudaMemoryTexture::GLBufferObjectsAvailiable = false;

//----------------------------------------------------------------------------
vtkCudaMemoryTexture::vtkCudaMemoryTexture()
{
  this->Initialize();
}

//----------------------------------------------------------------------------
vtkCudaMemoryTexture::~vtkCudaMemoryTexture()
{
  this->Deinitialize();
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::Deinitialize(bool withData /*= false*/)
{
  this->ReserveGPU();
  if (this->CudaOutputData)
  {
    cudaFree((void*) this->CudaOutputData);
  }
  this->CudaOutputData = 0;

  if (this->TextureID == 0 || !glIsTexture(this->TextureID))
  {
    glGenTextures(1, &this->TextureID);
  }

  if (vtkCudaMemoryTexture::GLBufferObjectsAvailiable == true)
  {
    if (this->BufferObjectID != 0 && glIsBufferARB(this->BufferObjectID))
    {
      glDeleteBuffersARB(1, &this->BufferObjectID);
    }
  }
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::Reinitialize(bool withData /*= false*/)
{
  this->Initialize();
  if (!this->CudaOutputData)
  {
    this->ReserveGPU();
    cudaMalloc((void**) &this->CudaOutputData, sizeof(uchar4) * this->Width * this->Height);
    this->RebuildBuffer();
  }
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::Initialize()
{
  this->TextureID = 0;
  this->BufferObjectID = 0;
  this->Height = this->Width = 0;
  this->RenderDestination = NULL;
  this->CurrentRenderMode = RenderToMemory;

  this->CudaOutputData = 0;
  this->LocalOutputData = 0;

  if (vtkCudaMemoryTexture::GLBufferObjectsAvailiable == false)
  {
    // check for the RenderMode
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
      std::cerr << "GLEW Error. " << std::endl;
      return;
    }

    if (GLEW_ARB_vertex_buffer_object)
    {
      vtkCudaMemoryTexture::GLBufferObjectsAvailiable = true;
      this->CurrentRenderMode = RenderToTexture;
    }
  }
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::SetWidth(unsigned int width)
{
  this->SetSize(width, this->GetHeight());
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::SetHeight(unsigned int height)
{
  this->SetSize(this->GetWidth(), height);
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::SetSize(unsigned int width, unsigned int height)
{
  if (width == this->Width && this->Height == height)
  {
    return;
  }
  else
  {
    this->Width = width;
    this->Height = height;

    this->ReserveGPU();
    if (this->CudaOutputData)
    {
      cudaFree((void*) this->CudaOutputData);
    }
    if (this->LocalOutputData)
    {
      delete this->LocalOutputData;
    }

    // Allocate Memory
    cudaMalloc((void**) &this->CudaOutputData, sizeof(uchar4) * this->Width * this->Height);
    this->LocalOutputData = new uchar4[this->Width * this->Height];

    this->RebuildBuffer();
  }
}

//----------------------------------------------------------------------------
unsigned int vtkCudaMemoryTexture::GetWidth() const
{
  return this->Width;
}

//----------------------------------------------------------------------------
unsigned int vtkCudaMemoryTexture::GetHeight() const
{
  return this->Height;
}

//----------------------------------------------------------------------------
unsigned int vtkCudaMemoryTexture::GetTexture() const
{
  return this->TextureID;
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::RebuildBuffer()
{
  // TEXTURE CODE
  this->ReserveGPU();
  glEnable(GL_TEXTURE_2D);
  if (this->TextureID != 0 && glIsTexture(this->TextureID))
  {
    glDeleteTextures(1, &this->TextureID);
  }
  glGenTextures(1, &this->TextureID);
  glBindTexture(GL_TEXTURE_2D, this->TextureID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, this->Width, this->Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, (void*) this->LocalOutputData);
  glBindTexture(GL_TEXTURE_2D, 0);

  if (this->CurrentRenderMode == RenderToTexture)
  {
    // OpenGL Buffer Code
    this->ReserveGPU();

    if (this->BufferObjectID != 0 && glIsBufferARB(this->BufferObjectID))
    {
      glDeleteBuffersARB(1, &this->BufferObjectID);
    }
    glGenBuffersARB(1, &this->BufferObjectID);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, this->BufferObjectID);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, this->Width * Height * sizeof(uchar4), (void*) this->LocalOutputData, GL_STREAM_COPY);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  }
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::SetRenderMode(int mode)
{
  if (mode == RenderToTexture && vtkCudaMemoryTexture::GLBufferObjectsAvailiable)
  {
    this->CurrentRenderMode = mode;
  }
  else
  {
    this->CurrentRenderMode = RenderToMemory;
  }
  this->RebuildBuffer();
}

//----------------------------------------------------------------------------
int vtkCudaMemoryTexture::GetCurrentRenderMode() const
{
  return this->CurrentRenderMode;
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::BindTexture()
{
  this->ReserveGPU();
  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, this->TextureID);
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::BindBuffer()
{
  if (this->CurrentRenderMode == RenderToTexture)
  {
    this->ReserveGPU();
    cudaGLRegisterBufferObject(this->BufferObjectID) ;
    cudaStreamSynchronize(*(this->GetStream()));
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, this->BufferObjectID);
    cudaGLMapBufferObject((void**)&this->RenderDestination, this->BufferObjectID);
    cudaStreamSynchronize(*(this->GetStream()));
  }
  else
  {
    this->RenderDestination = (unsigned char*)((void*) this->CudaOutputData);
  }
}

//----------------------------------------------------------------------------
unsigned char* vtkCudaMemoryTexture::GetRenderDestination() const
{
  return this->RenderDestination;
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::UnbindBuffer()
{
  if (this->CurrentRenderMode == RenderToTexture)
  {
    this->ReserveGPU();
    cudaGLUnmapBufferObject(this->BufferObjectID);
    cudaStreamSynchronize(*(this->GetStream()));
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->Width, this->Height, GL_RGBA, GL_UNSIGNED_BYTE, (0));
    cudaGLUnregisterBufferObject(this->BufferObjectID) ;
    cudaStreamSynchronize(*(this->GetStream()));
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  }
  else // (this->CurrentRenderMode == RenderToMemory)
  {
    this->ReserveGPU();
    cudaMemcpyAsync(this->LocalOutputData, this->CudaOutputData, sizeof(uchar4) * this->Width * this->Height, cudaMemcpyDeviceToHost, *(this->GetStream()));
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->Width, this->Height, GL_RGBA, GL_UNSIGNED_BYTE, (void*) LocalOutputData);
  }
  this->RenderDestination = NULL;
}

//----------------------------------------------------------------------------
void vtkCudaMemoryTexture::UnbindTexture()
{
  this->ReserveGPU();
  glPopAttrib();
}

//----------------------------------------------------------------------------
bool vtkCudaMemoryTexture::CopyToVtkImageData(vtkImageData* data)
{
  // setting up the data type and size.

  data->SetDimensions(this->Width, Height, 1);
  data->SetExtent(0, this->Width - 1,
                  0, this->Height - 1,
                  0, 1 - 1);

  data->AllocateScalars(VTK_UNSIGNED_CHAR, 4);

  this->ReserveGPU();
  cudaMemcpyAsync((void*) data->GetScalarPointer(), (void*) this->CudaOutputData, sizeof(uchar4) * this->Width * this->Height, cudaMemcpyDeviceToHost, *(this->GetStream()));
  cudaStreamSynchronize(*(this->GetStream()));

  return true;
}