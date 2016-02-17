/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkGBE.cxx,v $
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
#include "vtkGBE.h"

//----------------------------------------------------------------------------
vtkGBE* vtkGBE::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkGBE");
  if(ret)
    {
    return (vtkGBE*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkGBE;
}

//----------------------------------------------------------------------------
vtkGBE::vtkGBE()
{
  this->CenterVector[0] = 0.0;
  this->CenterVector[1] = 0.0;
  this->CenterVector[2] = 0.0;
}

//----------------------------------------------------------------------------
void vtkGBE::SetCenterVector(double CenterVector[3])
{
  // Pre-calculate the vector maginitude squared
  memcpy(this->CenterVector, CenterVector, sizeof(double)*3);
  this->VectorLengthSquared = ( CenterVector[0] * CenterVector[0] +
        CenterVector[1] * CenterVector[1] +
        CenterVector[2] * CenterVector[2] );
}

//----------------------------------------------------------------------------
void vtkGBE::SetGridSpacing(double GridSpacing[3])
{
  // BE = Factor * vector magnitude squared.  Factor contains 6 terms, each is
  // the effect of taking the second derivative with respect to a combination of
  // x,y,z.  To reproduce these, set up a 3X3X3 voxel grid and go through every voxel.
  // For each location calculate numerical 2nd derivatives. Reminder:
  // d^2(F)/dx^2 = (F(x+dx,y,z) - 2F(x,y,z) + F(x-dx,y,z)) / dx^2
  // d^2(F)/dxdy = (F(x+dx,y+dy,z) - F(x-dx,y+dy,z) - F(x+dx,y-dy,z) + F(x-dx,y-dy,z))/4/dxdy
  this->Factor = ( fabs( 6.0 / (GridSpacing[0] * GridSpacing[0] * GridSpacing[0] * GridSpacing[0]) ) +
       fabs( 6.0 / (GridSpacing[1] * GridSpacing[1] * GridSpacing[1] * GridSpacing[1]) ) +
       fabs( 6.0 / (GridSpacing[2] * GridSpacing[2] * GridSpacing[2] * GridSpacing[2]) ) +
       fabs( 4.0 / 8.0 / (GridSpacing[0] * GridSpacing[1] * GridSpacing[0] * GridSpacing[1]) ) +
       fabs( 4.0 / 8.0 / (GridSpacing[0] * GridSpacing[2] * GridSpacing[0] * GridSpacing[2]) ) +
       fabs( 4.0 / 8.0 / (GridSpacing[1] * GridSpacing[2] * GridSpacing[1] * GridSpacing[2]) ) );
  // Divide Factor by the volume - 3*3*3*dxdydz - don't need dxdydz since I didn't
  // multiply Factor by dxdydz for each voxel considered during summing (integration).
  this->Factor = this->Factor / 27.0;
}

//----------------------------------------------------------------------------
double vtkGBE::GetBendingEnergy()
{ 
  return this->Factor * this->VectorLengthSquared;
}

//----------------------------------------------------------------------------
void vtkGBE::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

