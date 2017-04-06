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

/** @file CudaObject.h
 *
 *  @brief Header file defining an abstract class which uses CUDA
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on June 12, 2012
 *
 *  @note Interacts primarily with the vtkCudaDeviceManager
 */

#ifndef __CudaObject_H__
#define __CudaObject_H__

#include "vtkCudaCommonExport.h"

#include "vtkCudaDeviceManager.h"

typedef struct CUstream_st cudaStream;

class vtkCudaCommonExport CudaObject
{
public:
  void SetDevice(int d, bool withData = false);
  int GetDevice();

  void ReserveGPU();
  void CallSyncThreads();
  cudaStream_t* GetStream();

  void ReplicateObject(CudaObject* object, bool withData = false);

protected:
  CudaObject(int d = 0);
  ~CudaObject();

  virtual void Reinitialize(bool withData = false) = 0;
  virtual void Deinitialize(bool withData = false) = 0;

private:
  int DeviceNumber;
  cudaStream_t* DeviceStream;

  vtkCudaDeviceManager* DeviceManager;
};
#endif /* __CudaObject_H__ */
