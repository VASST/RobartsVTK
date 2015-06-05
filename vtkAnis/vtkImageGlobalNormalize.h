/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageGlobalNormalize.h,v $
  Language:  C++
  Date:      $Date: 2007/04/26 19:16:45 $
  Version:   $Revision: 1.1 $

  Copyright (c) 1993-2002 Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageGlobalNormalize - Normalizes an image to [0,1].
// .SECTION Description
// vtkImageGlobalNormalize normalizes a single component image to the range of
// [0,1] for all pixel in the iamge


#ifndef __vtkImageGlobalNormalize_h
#define __vtkImageGlobalNormalize_h


#include "vtkImageToImageFilter.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

class VTK_EXPORT vtkImageGlobalNormalize : public vtkImageToImageFilter
{
public:
  static vtkImageGlobalNormalize *New();
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkImageGlobalNormalize,vtkImageToImageFilter);
#else
  vtkTypeMacro(vtkImageGlobalNormalize,vtkImageToImageFilter);
#endif

protected:
  vtkImageGlobalNormalize() {};
  ~vtkImageGlobalNormalize() {};

  void ExecuteInformation(vtkImageData *inData, vtkImageData *outData);
  void ExecuteInformation(){this->vtkImageToImageFilter::ExecuteInformation();};
  void ThreadedExecute(vtkImageData *inData, vtkImageData *outData,
                       int extent[6], int id);
private:
  vtkImageGlobalNormalize(const vtkImageGlobalNormalize&);  // Not implemented.
  void operator=(const vtkImageGlobalNormalize&);  // Not implemented.
};

#endif
