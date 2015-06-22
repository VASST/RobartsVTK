/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkDeformedVolume.cxx,v $
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
#include "vtkDeformedVolume.h"

//----------------------------------------------------------------------------
vtkDeformedVolume* vtkDeformedVolume::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkDeformedVolume");
  if(ret)
    {
    return (vtkDeformedVolume*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkDeformedVolume;
}

//----------------------------------------------------------------------------
vtkDeformedVolume::vtkDeformedVolume()
{
  for (int i = 0; i < 3; i++)
    {
      this->Translation[i] = 0.0;
      this->Location[i] = 0;
    }

  this->GridTransform = vtkGridTransformBSpline::New();
  this->GridTransform->SetInterpolationModeToLinear();

  this->Transform = vtkTransformPolyDataFilter::New();
  this->Transform->SetTransform(this->GridTransform);

  this->Volume = vtkMassProperties::New();
  this->Volume->SetInputConnection(this->Transform->GetOutputPort());
}

//----------------------------------------------------------------------------
vtkDeformedVolume::~vtkDeformedVolume()
{
  this->Transform->Delete();
  this->GridTransform->Delete();
  this->Volume->Delete();
}

//----------------------------------------------------------------------------
void vtkDeformedVolume::SetZeroGrid(vtkImageData *grid)
{
  this->Grid = grid;
  this->GridTransform->SetDisplacementGrid(grid);
  this->GridTransform->Inverse();
}

//----------------------------------------------------------------------------
void vtkDeformedVolume::SetDeformedSurface(vtkPolyData *surface)
{
  this->Surface = surface;
  this->Transform->SetInput(surface);
}

//----------------------------------------------------------------------------
double vtkDeformedVolume::GetDeformedVolume()
{
  double deformedvolume;

  this->Grid->SetScalarComponentFromFloat(this->Location[0],
            this->Location[1],
            this->Location[2],
            0, this->Translation[0]);
  this->Grid->SetScalarComponentFromFloat(this->Location[0],
            this->Location[1],
            this->Location[2],
            1, this->Translation[1]);
  this->Grid->SetScalarComponentFromFloat(this->Location[0],
            this->Location[1],
            this->Location[2],
            2, this->Translation[2]);
  this->Grid->Modified();


  this->GridTransform->Update();
  this->Transform->Update();

  deformedvolume = this->Volume->GetVolume();

  this->Grid->SetScalarComponentFromFloat(this->Location[0],
            this->Location[1],
            this->Location[2],
            0, 0);
  this->Grid->SetScalarComponentFromFloat(this->Location[0],
            this->Location[1],
            this->Location[2],
            1, 0);
  this->Grid->SetScalarComponentFromFloat(this->Location[0],
            this->Location[1],
            this->Location[2],
            2, 0);
  this->Grid->Modified();

  return deformedvolume;

}

//----------------------------------------------------------------------------
void vtkDeformedVolume::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

