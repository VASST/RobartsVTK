/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageAbsoluteDifference.h,v $
  Language:  C++
  Date:      $Date: 2007/05/04 14:34:34 $
  Version:   $Revision: 1.1 $

  Copyright (c) 1993-2002 Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageAbsoluteDifference - Returns the absolute difference of 2 images
// .SECTION Description
// vtkImageAbsoluteDifference calculates the absolute difference of 2 images

#ifndef __vtkImageAbsoluteDifference_h
#define __vtkImageAbsoluteDifference_h

#include "vtkRobartsRegistrationModule.h"

#include "vtkThreadedImageAlgorithm.h"
#include "vtkObjectFactory.h"
#include "vtkImageStencilData.h"
#include "vtkImageData.h"

// This number must be greater than the number of CPU's
#define THREAD_NUM 2

class VTKROBARTSREGISTRATION_EXPORT vtkImageAbsoluteDifference : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageAbsoluteDifference *New();
  vtkTypeMacro(vtkImageAbsoluteDifference, vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/get the 2 input images and stencil to specify which voxels to accumulate.
  virtual void SetInput1Data(vtkImageData *input);
  virtual void SetInput2Data(vtkImageData *input);

  void SetStencilData(vtkImageStencilData *stencil);

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
  double ThreadAbsoluteDifference[THREAD_NUM];

protected:
  vtkImageAbsoluteDifference();
  ~vtkImageAbsoluteDifference() {};

  int ReverseStencil;

  void ThreadedExecute(vtkImageData **inDatas, vtkImageData *outData, int extent[6], int id);

private:
  vtkImageAbsoluteDifference(const vtkImageAbsoluteDifference&);  // Not implemented.
  void operator=(const vtkImageAbsoluteDifference&);  // Not implemented.
};

#endif
