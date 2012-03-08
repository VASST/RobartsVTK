/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMILECGVideoSource2.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkMILECGVideoSource2 - Matrox Imaging Library frame grabbers
// .SECTION Description
// vtkMILECGVideoSource2 provides an interface to Matrox Meteor, MeteorII
// and Corona video digitizers through the Matrox Imaging Library 
// interface.  In order to use this class, you must link VTK with mil.lib,
// MIL version 5.0 or higher is required.
// vtkMILECGVideoSource2 is an updated version of vtkMILVideoSource and uses
// vtkVideoSource2 instead of vtkVideoSource
// .SECTION Caveats
// With some capture cards, if this class is leaked and ReleaseSystemResources 
// is not called, you may have to reboot before you can capture again.
// vtkVideoSource used to keep a global list and delete the video sources
// if your program leaked, due to exit crashes that was removed.
// .SECTION See Also
// vtkWin32VideoSource2 vtkVideoSource2 vtkMILVideoSource

#ifndef __vtkMILECGVideoSource2_h
#define __vtkMILECGVideoSource2_h

#include "vtkMILVideoSource2.h"
class VTK_EXPORT vtkMILECGVideoSource2 : public vtkMILVideoSource2
{
public:
  static vtkMILECGVideoSource2 *New();
  vtkTypeRevisionMacro(vtkMILECGVideoSource2,vtkVideoSource2);
  void PrintSelf(ostream& os, vtkIndent indent);   

  void SetECGPhase(int newPhase);
  int GetECGPhase();

protected:
  vtkMILECGVideoSource2();
  ~vtkMILECGVideoSource2();

  int CurrentPhase;

private:
  vtkMILECGVideoSource2(const vtkMILECGVideoSource2&);  // Not implemented.
  void operator=(const vtkMILECGVideoSource2&);  // Not implemented.
};

#endif
