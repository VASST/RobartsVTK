/*=========================================================================

  Program:   Visualization Toolkit
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

#include "vtkCudaCommonModule.h"

#include "vtkObject.h"
#include "vector_types.h"

#include <map>

class vtkMutexLock;
class vtkCudaObject;

class VTKCUDACOMMON_EXPORT vtkCudaDeviceManager : public vtkObject
{
public:
  vtkTypeMacro( vtkCudaDeviceManager, vtkObject );
  static vtkCudaDeviceManager* Singleton();

  int GetNumberOfDevices();
  bool GetDevice(vtkCudaObject* caller, int device);
  bool ReturnDevice(vtkCudaObject* caller, int device);
  bool GetStream(vtkCudaObject* caller, cudaStream_t** stream, int device);
  bool ReturnStream(vtkCudaObject* caller, cudaStream_t* stream, int device);

  bool SynchronizeStream( cudaStream_t* stream );
  bool ReserveGPU( cudaStream_t* stream );

  int QueryDeviceForObject( vtkCudaObject* object );
  int QueryDeviceForStream( cudaStream_t* stream );

protected:
  vtkCudaDeviceManager();
  ~vtkCudaDeviceManager();

  void DestroyEmptyStream( cudaStream_t* stream );
  bool SynchronizeStreamUnlocked( cudaStream_t* stream );

  std::multimap<vtkCudaObject*,int> ObjectToDeviceMap;
  std::multimap<int,vtkCudaObject*> DeviceToObjectMap;

  std::map<cudaStream_t*,int> StreamToDeviceMap;
  std::multimap<cudaStream_t*, vtkCudaObject*> StreamToObjectMap;

  vtkMutexLock* regularLock;

private:
  vtkCudaDeviceManager operator=(const vtkCudaDeviceManager&);
  vtkCudaDeviceManager(const vtkCudaDeviceManager&);

  static vtkCudaDeviceManager* singletonManager;
};

#endif /* __VTKCUDADEVICEMANAGER_H__ */
