/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageSquaredDifference.h,v $
  Language:  C++
  Date:      $Date: 2007/05/04 14:34:35 $
  Version:   $Revision: 1.1 $

  Copyright (c) 1993-2002 Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageSquaredDifference - Returns the absolute difference of 2 images
// .SECTION Description
// vtkImageSquaredDifference calculates the absolute difference of 2 images

#ifndef __vtkImageSquaredDifference_h
#define __vtkImageSquaredDifference_h

#include "vtkImageMultipleInputFilter.h"
#include "vtkObjectFactory.h"
#include "vtkImageStencilData.h"
#include "vtkImageData.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

// This number must be greater than the number of CPU's
#define THREAD_NUM 2

class VTK_EXPORT vtkImageSquaredDifference : public vtkImageMultipleInputFilter
{
public:
  static vtkImageSquaredDifference *New();
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkImageSquaredDifference,vtkImageMultipleInputFilter);
#else
  vtkTypeMacro(vtkImageSquaredDifference,vtkImageMultipleInputFilter);
#endif
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/get the 2 input images and stencil to specify which voxels to accumulate.
  virtual void SetInput1(vtkImageData *input);
  virtual void SetInput2(vtkImageData *input);
  void SetStencil(vtkImageStencilData *stencil);
  vtkImageData *GetInput1();
  vtkImageData *GetInput2();
  vtkImageStencilData *GetStencil();

  // Description:
  // Reverse the stencil.
  vtkSetMacro(ReverseStencil, int);
  vtkBooleanMacro(ReverseStencil, int);
  vtkGetMacro(ReverseStencil, int);

  // Description:
  // Get the absolute difference
  double GetResult();

  // Description:
  // These allow averaging after threads are finished.
  // Maximum of 4 processors can be handled.
  double ThreadSquaredDifference[THREAD_NUM];

protected:
  vtkImageSquaredDifference();
  ~vtkImageSquaredDifference() {};

  int ReverseStencil;

  void ThreadedExecute(vtkImageData **inDatas, vtkImageData *outData, int extent[6], int id);

private:
  vtkImageSquaredDifference(const vtkImageSquaredDifference&);  // Not implemented.
  void operator=(const vtkImageSquaredDifference&);  // Not implemented.
};

#endif
