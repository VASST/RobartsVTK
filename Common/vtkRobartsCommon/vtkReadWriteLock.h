/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkReadWriteLock.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkReadWriteLock - Video-for-Windows video digitizer
// .SECTION Description
// A lock that allows for multiple readers, but only one writer
// a TCP/IP socket, allowing for multiple process VTK pipelines
// .SECTION Caveats
// Will only work as well as the underlying vtk mutices
//

#ifndef __VTKREADWRITELOCK_H
#define __VTKREADWRITELOCK_H

#include "vtkRobartsCommonModule.h"

#include "vtkObject.h"

class vtkMutexLock;

class VTKROBARTSCOMMON_EXPORT vtkReadWriteLock : public vtkObject
{
public:
  static vtkReadWriteLock *New();

  void ReaderLock();
  void ReaderUnlock();
  void WriterLock();
  void WriterUnlock();

protected:
  vtkReadWriteLock();
  ~vtkReadWriteLock();

  vtkMutexLock* noWriters;
  vtkMutexLock* noReaders;
  vtkMutexLock* counter;

  unsigned int readerCount;
};

#endif