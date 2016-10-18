/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    $RCSfile: vtkPolyDataSurfaceArea.h,v $
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
// .NAME vtkPolyDataSurfaceArea - estimate volume, area, shape index of triangle mesh
// .SECTION Description
// vtkPolyDataSurfaceArea estimates the volume, the surface area, and the
// normalized shape index of a triangle mesh.  The algorithm
// implemented here is based on the discrete form of the divergence
// theorem.  The general assumption here is that the model is of
// closed surface.  For more details see the following reference
// (Alyassin A.M. et al, "Evaluation of new algorithms for the
// interactive measurement of surface area and volume", Med Phys 21(6)
// 1994.).  

// .SECTION Caveats
// Currently only triangles are processed. Use vtkTriangleFilter to
// convert any strips or polygons to triangles.

// .SECTION See Also
// vtkTriangleFilter

#ifndef __vtkPolyDataSurfaceArea_h
#define __vtkPolyDataSurfaceArea_h

#include "vtkRobartsRegistrationExport.h"
#include "vtkVersionMacros.h"

#include "vtkObjectFactory.h"
#include "vtkPolyData.h"
#include "vtkCell.h"
#include "vtkIdList.h"

class vtkRobartsRegistrationExport vtkPolyDataSurfaceArea : public vtkObject
{
public:
  // Description:
  // Constructs with initial values of zero.
  static vtkPolyDataSurfaceArea *New();
  vtkTypeMacro(vtkPolyDataSurfaceArea,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  void SetInput(vtkPolyData *input);

  // Description:
  // Compute and return the area.
  double GetSurfaceArea();

protected:
  vtkPolyDataSurfaceArea();
  ~vtkPolyDataSurfaceArea();

  vtkPolyData *inData;
  int NumCells;

};

#endif


