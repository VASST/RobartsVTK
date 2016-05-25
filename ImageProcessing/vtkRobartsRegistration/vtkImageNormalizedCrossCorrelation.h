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

#include "vtkRobartsRegistrationModule.h"

#include "vtkThreadedImageAlgorithm.h"
#include "vtkObjectFactory.h"
#include "vtkImageStencilData.h"
#include "vtkImageData.h"

// Constants used for array declaration.
#define THREAD_NUM 2

class VTKROBARTSREGISTRATION_EXPORT vtkImageNormalizedCrossCorrelation : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageNormalizedCrossCorrelation *New();
  vtkTypeMacro(vtkImageNormalizedCrossCorrelation,vtkThreadedImageAlgorithm);
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
