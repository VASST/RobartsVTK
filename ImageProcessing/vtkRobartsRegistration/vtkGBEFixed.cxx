/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    $RCSfile: vtkGBEFixed.cxx,v $
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
#include "vtkGBEFixed.h"

//----------------------------------------------------------------------------
vtkGBEFixed* vtkGBEFixed::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkGBEFixed");
  if(ret)
    {
    return (vtkGBEFixed*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkGBEFixed;
}

//----------------------------------------------------------------------------
vtkGBEFixed::vtkGBEFixed()
{
  this->CenterVector[0] = 0.0;
  this->CenterVector[1] = 0.0;
  this->CenterVector[2] = 0.0;
}

//----------------------------------------------------------------------------
void vtkGBEFixed::SetCenterVector(double CenterVector[3])
{
  memcpy(this->CenterVector, CenterVector, sizeof(double)*3);
  this->CenterVectorSquared[0] = CenterVector[0] * CenterVector[0];
  this->CenterVectorSquared[1] = CenterVector[1] * CenterVector[1];
  this->CenterVectorSquared[2] = CenterVector[2] * CenterVector[2];
}

//----------------------------------------------------------------------------
void vtkGBEFixed::SetGridSpacing(double GridSpacing[3])
{
  // Loop below where 6.0 and 4.0 appeared from, absolute value is used
  // so that BE is not < 0.
  this->f1 = fabs( 6.0 * (GridSpacing[1] * GridSpacing[2]) / 
       (GridSpacing[0] * GridSpacing[0] * GridSpacing[0]) );
  this->f2 = fabs( 6.0 * (GridSpacing[0] * GridSpacing[2]) / 
       (GridSpacing[1] * GridSpacing[1] * GridSpacing[1]) );
  this->f3 = fabs( 6.0 * (GridSpacing[0] * GridSpacing[1]) / 
       (GridSpacing[2] * GridSpacing[2] * GridSpacing[2]) );
  this->f4 = fabs( 4.0 * GridSpacing[2] / (GridSpacing[0] * GridSpacing[1] * 4.0) );
  this->f5 = fabs( 4.0 * GridSpacing[1] / (GridSpacing[0] * GridSpacing[2] * 4.0) );
  this->f6 = fabs( 4.0 * GridSpacing[0] / (GridSpacing[1] * GridSpacing[2] * 4.0) );
}

//----------------------------------------------------------------------------
double vtkGBEFixed::GetBendingEnergy()
{
  double sum = 0.0;
  
  for (int i = 0; i <= 2; i++)
    {
      // 2^2 * ( 0, 0, 0) (-1, 0, 0) ( 1, 0, 0)
      sum += this->CenterVectorSquared[i] * f1;

      // 2^2 * ( 0, 0, 0) ( 0,-1, 0) ( 0, 1, 0)
      sum += this->CenterVectorSquared[i] * f2;

      // 2^2 * ( 0, 0, 0) ( 0, 0,-1) ( 0, 0, 1) 
      sum += this->CenterVectorSquared[i] * f3;

      // (-1, 1, 0) ( 1, 1, 0) ( 1,-1, 0) (-1,-1, 0)
      sum += this->CenterVectorSquared[i] * f4;

      // ( 1, 0,-1) (-1, 0,-1) ( 1, 0, 1) (-1, 0, 1)
      sum += this->CenterVectorSquared[i] * f5;

      // ( 0, 1,-1) ( 0,-1,-1) ( 0, 1, 1) ( 0,-1, 1)
      sum += this->CenterVectorSquared[i] * f6;
    }

  return sum / 27.0;
}

//----------------------------------------------------------------------------
void vtkGBEFixed::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

