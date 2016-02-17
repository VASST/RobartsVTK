/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkGBEFixed.h,v $
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
// .NAME vtkGBEFixed - Returns the absolute difference image of 2 images
// .SECTION Description
// vtkGBEFixed Returns the absolute difference image of 2 images

#ifndef __vtkGBEFixed_h
#define __vtkGBEFixed_h

#include "vtkRobartsRegistrationModule.h"

#include "vtkObject.h"
#include "vtkObjectFactory.h"

class VTKROBARTSREGISTRATION_EXPORT vtkGBEFixed : public vtkObject
{
public:
  static vtkGBEFixed *New();
  vtkTypeMacro(vtkGBEFixed,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set the vector for the center of the calculation
  // (the rest are assumed to be 0,0,0)
  virtual void SetCenterVector(double CenterVector[3]);
  virtual void SetGridSpacing(double GridSpacing[3]);
  double CenterVector[3];
  double CenterVectorSquared[3];
  double f1;
  double f2;
  double f3;
  double f4;
  double f5;
  double f6;

  // Description:
  // Get the bending energy
  double GetBendingEnergy();

protected:
  vtkGBEFixed();
  ~vtkGBEFixed() {};

};

#endif













