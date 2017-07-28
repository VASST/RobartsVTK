/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageLuminance2.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageLuminance2 - Computes the luminance of the input
// .SECTION Description
// vtkImageLuminance2 calculates luminance from an rgb input.

#ifndef __vtkImageLuminance2_h
#define __vtkImageLuminance2_h


#include "vtkThreadedImageAlgorithm.h"
#include "vtkObjectFactory.h"

class  vtkImageLuminance2 : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageLuminance2 *New();
  vtkTypeMacro(vtkImageLuminance2,vtkThreadedImageAlgorithm);
	vtkSetVector3Macro(Coef, double);
	vtkGetVector3Macro(Coef, double);

protected:
  vtkImageLuminance2();
  ~vtkImageLuminance2() {};
  
  virtual int RequestInformation (vtkInformation *, vtkInformationVector**,
                                  vtkInformationVector *);

  void ThreadedExecute (vtkImageData *inData, vtkImageData *outData,
                        int outExt[6], int id);

  double Coef[3];

private:
  vtkImageLuminance2(const vtkImageLuminance2&);  // Not implemented.
  void operator=(const vtkImageLuminance2&);  // Not implemented.
};

#endif










