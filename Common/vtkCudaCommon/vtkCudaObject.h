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

#ifndef __VTKCUDAOBJECT_H__
#define __VTKCUDAOBJECT_H__

#include "vtkCudaCommonModule.h"

#include "vtkCudaDeviceManager.h"
#include "vector_types.h"
#include "cuda.h"

class VTKCUDACOMMON_EXPORT vtkCudaObject
{
public:
  void SetDevice( int d, int withData = 0 );
  int GetDevice();

  void ReserveGPU( );
  void CallSyncThreads( );
  cudaStream_t* GetStream( );

  void ReplicateObject( vtkCudaObject* object, int withData = 0  );

protected:
  vtkCudaObject(int d = 0);
  ~vtkCudaObject();

  virtual void Reinitialize(int withData = 0) = 0;
  virtual void Deinitialize(int withData = 0) = 0;

private:

  int DeviceNumber;
  cudaStream_t* DeviceStream;

  vtkCudaDeviceManager* DeviceManager;

  int withDataStatus;

};
#endif /* __VTKCUDAOBJECT_H__ */
