/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageNormalizedCrossCorrelation.h,v $
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
// .NAME vtkImageNormalizedCrossCorrelation - Returns the absolute difference of 2 images
// .SECTION Description
// vtkImageNormalizedCrossCorrelation calculates the absolute difference of 2 images

#ifndef __vtkImageNormalizedCrossCorrelation_h
#define __vtkImageNormalizedCrossCorrelation_h

#include "vtkImageMultipleInputFilter.h"
#include "vtkObjectFactory.h"
#include "vtkImageStencilData.h"
#include "vtkImageData.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

// Constants used for array declaration.
#define THREAD_NUM 2

class VTK_EXPORT vtkImageNormalizedCrossCorrelation : public vtkImageMultipleInputFilter
{
public:
  static vtkImageNormalizedCrossCorrelation *New();
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkImageNormalizedCrossCorrelation,vtkImageMultipleInputFilter);
#else
  vtkTypeMacro(vtkImageNormalizedCrossCorrelation,vtkImageMultipleInputFilter);
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
  double ThreadSumST[THREAD_NUM];
  double ThreadSumS[THREAD_NUM];
  double ThreadSumT[THREAD_NUM];

protected:
  vtkImageNormalizedCrossCorrelation();
  ~vtkImageNormalizedCrossCorrelation() {};

  int ReverseStencil;

  void ThreadedExecute(vtkImageData **inDatas, vtkImageData *outData, int extent[6], int id);

private:
  vtkImageNormalizedCrossCorrelation(const vtkImageNormalizedCrossCorrelation&);  // Not implemented.
  void operator=(const vtkImageNormalizedCrossCorrelation&);  // Not implemented.
};

#endif
