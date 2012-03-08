/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTriangleMeshQuality.cxx,v $
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
#include "vtkTriangleMeshQuality.h"

#include "vtkCellData.h"
#include "vtkDataSet.h"
#include "vtkFieldData.h"
#include "vtkFloatArray.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkTetra.h"

vtkCxxRevisionMacro(vtkTriangleMeshQuality, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkTriangleMeshQuality);

//----------------------------------------------------------------------------
// Constructor
vtkTriangleMeshQuality::vtkTriangleMeshQuality() 
{
 this->GeometryOff();
 this->TopologyOff();
 this->FieldDataOff();
 this->PointDataOff();
 this->CellDataOn();
 
 this->Quality = 0.0;
}

//----------------------------------------------------------------------------
//destructor
vtkTriangleMeshQuality::~vtkTriangleMeshQuality() 
{ 
}

//----------------------------------------------------------------------------
void vtkTriangleMeshQuality::Execute()
{
  double p1[3],p2[3],p3[3];
  float a,b,c,R,r,ratio;

  vtkDataSet *input = this->GetInput();
  vtkIdType numCells=input->GetNumberOfCells();
  vtkIdList *id = vtkIdList::New();
  vtkCellData *celld = vtkCellData::New();
  vtkFloatArray *scalars = vtkFloatArray::New();
  scalars->SetNumberOfComponents(1);
  scalars->SetNumberOfTuples(numCells);
  
  for (int j = 0; j < numCells; j++)
    {
      input->GetCellPoints(j,id);
      input->GetPoint(id->GetId(0),p1);
      input->GetPoint(id->GetId(1),p2);
      input->GetPoint(id->GetId(2),p3);
 
      a = sqrt( (p2[0] - p1[0]) * (p2[0] - p1[0]) +
		(p2[1] - p1[1]) * (p2[1] - p1[1]) +
		(p2[2] - p1[2]) * (p2[2] - p1[2]) );
      b = sqrt( (p3[0] - p2[0]) * (p3[0] - p2[0]) +
		(p3[1] - p2[1]) * (p3[1] - p2[1]) +
		(p3[2] - p2[2]) * (p3[2] - p2[2]) );
      c = sqrt( (p1[0] - p3[0]) * (p1[0] - p3[0]) +
		(p1[1] - p3[1]) * (p1[1] - p3[1]) +
		(p1[2] - p3[2]) * (p1[2] - p3[2]) );

      R = a * b * c / sqrt ( (a+b+c)*(b+c-a)*(c+a-b)*(a+b-c) );	
      r = 1.0 / 2.0 * sqrt ( (b+c-a)*(c+a-b)*(a+b-c) / (a+b+c) );

      ratio = R/r/2.0;
      this->Quality += ratio;
      scalars->SetTuple1(j,R/r/2.0);
    }
  
  this->Quality = this->Quality/numCells;
   
  celld->SetScalars(scalars);
  this->GetOutput()->SetFieldData(celld);
  celld->Delete();
  id->Delete();
  scalars->Delete();
}

//----------------------------------------------------------------------------
void vtkTriangleMeshQuality::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkDataSetToDataObjectFilter::PrintSelf(os,indent);

  os << indent << "Input: " << this->GetInput() << "\n";
  os << indent << "Quality: " << this->Quality << "\n";
}
