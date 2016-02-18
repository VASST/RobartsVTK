/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImagePatternIntensity.h,v $
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
// .NAME vtkImagePatternIntensity - Returns the absolute difference of 2 images
// .SECTION Description
// vtkImagePatternIntensity calculates the absolute difference of 2 images

#ifndef __vtkImagePatternIntensity_h
#define __vtkImagePatternIntensity_h

#include "vtkRobartsRegistrationModule.h"

#include "vtkThreadedImageAlgorithm.h"
#include "vtkObjectFactory.h"
#include "vtkImageStencilData.h"
#include "vtkImageData.h"
#include "vtkImageMathematics.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

class VTKROBARTSREGISTRATION_EXPORT vtkImagePatternIntensity : public vtkThreadedImageAlgorithm
{
public:
  static vtkImagePatternIntensity *New();

#if (VTK_MAJOR_VERSION < 6)
  vtkTypeRevisionMacro(vtkImagePatternIntensity,vtkImageMultipleInputFilter);
#else
  vtkTypeMacro(vtkImagePatternIntensity,vtkThreadedImageAlgorithm);
#endif
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/get the 2 input images and stencil to specify which voxels to accumulate.
#if (VTK_MAJOR_VERSION < 6)
  virtual void SetInput1(vtkImageData *input);
  virtual void SetInput2(vtkImageData *input);
  void SetStencil(vtkImageStencilData *stencil);
#else
  virtual void SetInput1Data(vtkImageData *input);
  virtual void SetInput2Data(vtkImageData *input);
  void SetStencilData(vtkImageStencilData *stencil);
#endif
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
  double ThreadPatternIntensity[4];

protected:
  vtkImagePatternIntensity();
  ~vtkImagePatternIntensity() {};

  int ReverseStencil;

  void ThreadedExecute(vtkImageData **inDatas, vtkImageData *outData, int extent[6], int id);

private:
  vtkImagePatternIntensity(const vtkImagePatternIntensity&);  // Not implemented.
  void operator=(const vtkImagePatternIntensity&);  // Not implemented.
};

#endif