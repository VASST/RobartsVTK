/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTriangleMeshQuality.h,v $
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
// .NAME vtkTriangleMeshQuality - calculate quality of tetrahedral meshes
// .SECTION Description
// vtkTriangleMeshQuality will calculate the normalized quality ratio of the cells
// in a tetrahedral mesh according to the equation:
// <p> ratio = (radius of circumscribed sphere)/(radius of inscribed sphere)/3.
// <p> The minumum (and ideal) quality ratio is 1.0 for regular tetrahedra,
// i.e. all sides of equal length.  Larger values indicate poorer mesh
// quality.  The resulting quality values (and the tetrahedron volumes)
// are set as the Scalars of the FieldData of the output.

// .SECTION Thanks
// This class was developed by Leila Baghdadi, Hanif Ladak, and
// David Steinman at the Imaging Research Labs, Robarts Research Institute.

#ifndef __vtkTriangleMeshQuality_h
#define __vtkTriangleMeshQuality_h

#include "vtkDataSetToDataObjectFilter.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

class VTK_EXPORT vtkTriangleMeshQuality : public vtkDataSetToDataObjectFilter
{
public:
  static vtkTriangleMeshQuality *New();
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkTriangleMeshQuality,vtkDataSetToDataObjectFilter);
#else
  vtkTypeMacro(vtkTriangleMeshQuality,vtkDataSetToDataObjectFilter);
#endif
  void PrintSelf(ostream& os, vtkIndent indent);

  // Get the quality of the mesh
  vtkGetMacro(Quality, double);

protected:
  vtkTriangleMeshQuality();
  ~vtkTriangleMeshQuality();
  void Execute();

  double Quality;


private:
  vtkTriangleMeshQuality(const vtkTriangleMeshQuality&);  // Not implemented.
  void operator=(const vtkTriangleMeshQuality&);  // Not implemented.

};

#endif
