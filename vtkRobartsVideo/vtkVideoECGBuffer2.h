/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkVideoECGBuffer2.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkVideoECGBuffer2 - Store a collection of video frames
// .SECTION Description
// vtkVideoECGBuffer2 is a structure for holding video frames that are
// captured either from a vtkRenderWindow or from some other video
// source.  The buffer can be locked, to allow some video frames to
// be written to while other frames are being read from.  Hopefully
// an additional class will be written that will take a vtkVideoECGBuffer2
// and compress it into a movie file.
// .SECTION See Also
// vtkVideoFrame2 vtkVideoSource2 vtkWin32VideoSource2 vtkMILVideoSource2

#ifndef __vtkVideoECGBuffer2_h
#define __vtkVideoECGBuffer2_h

#include "vtkObject.h"

class vtkCriticalSection;
class vtkVideoFrame2;
class vtkDoubleArray;

class VTK_EXPORT vtkVideoECGBuffer2 : public vtkVideoBuffer2
{
public:
  static vtkVideoECGBuffer2 *New();
  vtkTypeRevisionMacro(vtkVideoECGBuffer2,vtkVideoBuffer2);
  void PrintSelf(ostream& os, vtkIndent indent);

  void SetBufferSize(int n);

  void AddItem(vtkVideoFrame2* frame, double timestamp, int phase);

protected:
  vtkVideoECGBuffer2();
  ~vtkVideoECGBuffer2();

  vtkIntArray *ECGPhaseArray;

private:
  vtkVideoECGBuffer2(const vtkVideoECGBuffer2&);
  void operator=(const vtkVideoECGBuffer2&);
};

#endif
