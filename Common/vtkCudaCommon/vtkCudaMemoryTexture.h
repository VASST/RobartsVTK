/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkCudaMemoryTexture.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file CudaObject.h
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

#include "vtkCudaCommonExport.h"

#include "vtkObject.h"
#include "CudaObject.h"
#include "vtkImageData.h"
#include "vector_types.h"

class vtkCudaCommonExport vtkCudaMemoryTexture : public vtkObject, public CudaObject
{
public:
  typedef enum
  {
    RenderToTexture,
    RenderToMemory,
  } RenderMode;

public:
  static vtkCudaMemoryTexture* New();
  vtkTypeMacro(vtkCudaMemoryTexture, vtkObject);

  void SetWidth(unsigned int width);
  void SetHeight(unsigned int height);
  void SetSize(unsigned int width, unsigned int height);

  unsigned int GetWidth() const;
  unsigned int GetHeight() const;

  unsigned int GetTexture() const;

  void BindTexture();
  void BindBuffer();
  unsigned char* GetRenderDestination() const;
  void UnbindBuffer();
  void UnbindTexture();

  bool CopyToVtkImageData(vtkImageData* data);

  void SetRenderMode(int mode);
  int GetCurrentRenderMode() const;

protected:
  vtkCudaMemoryTexture();
  ~vtkCudaMemoryTexture();
  virtual void Reinitialize(bool withData = false);
  virtual void Deinitialize(bool withData = false);

  unsigned char*  RenderDestination;

  unsigned int  TextureID;
  unsigned int  BufferObjectID;

  unsigned int  Width;
  unsigned int  Height;

  int        CurrentRenderMode;

  uchar4*      CudaOutputData;
  uchar4*      LocalOutputData;

  static bool  GLBufferObjectsAvailiable;

private:
  void Initialize();
  void RebuildBuffer();
};
#endif /* __VTKCUDAMEMORYTEXTURE_H__ */
