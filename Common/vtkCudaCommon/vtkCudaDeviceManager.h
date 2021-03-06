/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkCudaDeviceManager.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaDeviceManager.h
 *
 *  @brief Header file defining a singleton class to manage cards and stream interleaving
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on June 12, 2012
 *
 */

#ifndef __VTKCUDADEVICEMANAGER_H__
#define __VTKCUDADEVICEMANAGER_H__

#include "vtkCudaCommonExport.h"

#include "vtkObject.h"
#include "CudaCommon.h"

#include <map>

class vtkMutexLock;
class CudaObject;

class vtkCudaCommonExport vtkCudaDeviceManager : public vtkObject
{
public:
  vtkTypeMacro( vtkCudaDeviceManager, vtkObject );
  static vtkCudaDeviceManager* Singleton();

  int GetNumberOfDevices();
  bool GetDevice(CudaObject* caller, int device);
  bool ReturnDevice(CudaObject* caller, int device);
  bool GetStream(CudaObject* caller, cudaStream_t** stream, int device);
  bool ReturnStream(CudaObject* caller, cudaStream_t* stream, int device);

  bool SynchronizeStream( cudaStream_t* stream );
  bool ReserveGPU( cudaStream_t* stream );

  int QueryDeviceForObject( CudaObject* object );
  int QueryDeviceForStream( cudaStream_t* stream );

protected:
  vtkCudaDeviceManager();
  ~vtkCudaDeviceManager();

  void DestroyEmptyStream( cudaStream_t* stream );
  bool SynchronizeStreamUnlocked( cudaStream_t* stream );

  std::multimap<CudaObject*,int> ObjectToDeviceMap;
  std::multimap<int,CudaObject*> DeviceToObjectMap;

  std::map<cudaStream_t*,int> StreamToDeviceMap;
  std::multimap<cudaStream_t*, CudaObject*> StreamToObjectMap;

  vtkMutexLock* regularLock;

private:
  vtkCudaDeviceManager operator=(const vtkCudaDeviceManager&);
  vtkCudaDeviceManager(const vtkCudaDeviceManager&);

  static vtkCudaDeviceManager* singletonManager;
};

#endif /* __VTKCUDADEVICEMANAGER_H__ */
