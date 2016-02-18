/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaObject.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaObject.h
 *
 *  @brief Header file defining an abstract class which uses CUDA
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on June 12, 2012
 *
 *  @note Interacts primarily with the vtkCudaDeviceManager
 */

#ifndef __VTKCUDAMEMORYTEXTURE_H__
#define __VTKCUDAMEMORYTEXTURE_H__

#include "vtkCudaCommonModule.h"

#include "vtkObject.h"
#include "vtkCudaObject.h"
#include "vtkImageData.h"
#include "vector_types.h"

class VTKCUDACOMMON_EXPORT vtkCudaMemoryTexture : public vtkObject, public vtkCudaObject
{
public:

  vtkTypeMacro( vtkCudaMemoryTexture, vtkObject );

  static vtkCudaMemoryTexture* New();

  void SetWidth(unsigned int width) { this->SetSize(width, this->GetHeight()); }
  void SetHeight(unsigned int height) { this->SetSize(this->GetWidth(), height); }
  void SetSize(unsigned int width, unsigned int height);

  unsigned int GetWidth() const { return this->Width; }
  unsigned int GetHeight() const { return this->Height; }

  unsigned int GetTexture() const { return this->TextureID; }

  void BindTexture();
  void BindBuffer();
  unsigned char* GetRenderDestination() const { return this->RenderDestination; }
  void UnbindBuffer();
  void UnbindTexture();

  bool CopyToVtkImageData(vtkImageData* data);

  typedef enum 
  {
    RenderToTexture,
    RenderToMemory,
  } RenderMode;
  void SetRenderMode(int mode);
  int GetCurrentRenderMode() const { return this->CurrentRenderMode; }


protected:
  vtkCudaMemoryTexture();
  ~vtkCudaMemoryTexture();
  void Reinitialize(int withData = 0);
  void Deinitialize(int withData = 0);

private:
  void Initialize();
  void RebuildBuffer();

private:
  unsigned char*  RenderDestination;

  unsigned int  TextureID;
  unsigned int  BufferObjectID;

  unsigned int  Width;
  unsigned int  Height;

  int        CurrentRenderMode;

  uchar4*      CudaOutputData;
  uchar4*      LocalOutputData;

  static bool  GLBufferObjectsAvailiable;
};
#endif /* __VTKCUDAMEMORYTEXTURE_H__ */
