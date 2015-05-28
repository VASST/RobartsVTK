/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkPolyDataSurfaceArea.cxx,v $
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
#include "vtkPolyDataSurfaceArea.h"
//----------------------------------------------------------------------------

vtkPolyDataSurfaceArea* vtkPolyDataSurfaceArea::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkPolyDataSurfaceArea");
  if(ret)
    {
    return (vtkPolyDataSurfaceArea*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkPolyDataSurfaceArea;
}

//----------------------------------------------------------------------------
vtkPolyDataSurfaceArea::vtkPolyDataSurfaceArea()
{
}

//----------------------------------------------------------------------------
vtkPolyDataSurfaceArea::~vtkPolyDataSurfaceArea()
{
}

//----------------------------------------------------------------------------
void vtkPolyDataSurfaceArea::SetInput(vtkPolyData *input)
{
  this->inData = input;
  this->inData->Update();
  this->NumCells = this->inData->GetNumberOfCells();

  if (this->NumCells < 1 || this->inData->GetNumberOfPoints() < 1)
    {
      vtkErrorMacro(<<"No data to measure...!");
    }
}

//----------------------------------------------------------------------------
double vtkPolyDataSurfaceArea::GetSurfaceArea()
{
  double area,surfacearea;
  double *p;
  double x[3],y[3],z[3];
  double i[3],j[3],k[3];
  double ii[3],jj[3],kk[3];
  double a,b,c,s;  

  this->inData->Update();

  vtkIdList *pointIds = vtkIdList::New();
  pointIds->Allocate(VTK_CELL_SIZE);  

  // Remember, Polygons must be triangles!!
  surfacearea = 0.0;
  for (int cellId=0; cellId < this->NumCells; cellId++)
    {
      this->inData->GetCellPoints(cellId,pointIds);

      // store current vertix (x,y,z) coordinates ...
      for (int pointId=0; pointId < 3; pointId++)
  {
    p = this->inData->GetPoint(pointIds->GetId(pointId));
    x[pointId] = (double)p[0]; y[pointId] = (double)p[1]; z[pointId] = (double)p[2];
  }
      
      i[0] = ( x[1] - x[0]); j[0] = (y[1] - y[0]); k[0] = (z[1] - z[0]);
      i[1] = ( x[2] - x[0]); j[1] = (y[2] - y[0]); k[1] = (z[2] - z[0]);
      i[2] = ( x[2] - x[1]); j[2] = (y[2] - y[1]); k[2] = (z[2] - z[1]);
      
      ii[0] = i[0] * i[0]; ii[1] = i[1] * i[1]; ii[2] = i[2] * i[2];
      jj[0] = j[0] * j[0]; jj[1] = j[1] * j[1]; jj[2] = j[2] * j[2];
      kk[0] = k[0] * k[0]; kk[1] = k[1] * k[1]; kk[2] = k[2] * k[2];
      
      // area of a triangle...
      a = sqrt(ii[1] + jj[1] + kk[1]);
      b = sqrt(ii[0] + jj[0] + kk[0]);
      c = sqrt(ii[2] + jj[2] + kk[2]);
      s = 0.5 * (a + b + c);
      area = sqrt( fabs(s*(s-a)*(s-b)*(s-c)));
      surfacearea += area;
    }
  
  pointIds->Delete();
  
  return surfacearea;

}

//----------------------------------------------------------------------------
void vtkPolyDataSurfaceArea::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

}
