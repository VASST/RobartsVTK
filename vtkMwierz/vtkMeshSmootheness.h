/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMeshSmootheness.h,v $
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
// .NAME vtkMeshSmootheness - compute curvatures (Gauss and mean) of a Polydata object
// .SECTION Description
// vtkMeshSmootheness takes a polydata input and computes the curvature of the
// mesh at each point. Two possible methods of computation are available :
//
// Gauss Curvature
// discrete Gauss curvature (K) computation,
// K(vertex v) = 2*PI-\sum_{facet neighbs f of v} (angle_f at v)
// The contribution of every facet is for the moment weighted by Area(facet)/3
// The units of Gaussian Curvature are [1/m^2]
//
// Mean Curvature
// H(vertex v) = average over edges neighbs e of H(e)
// H(edge e) = length(e)*dihedral_angle(e)
// NB: dihedral_angle is the ORIENTED angle between -PI and PI,
// this means that the surface is assumed to be orientable
// the computation creates the orientation
// The units of Mean Curvature are [1/m]
//
// NB. The sign of the Gauss curvature is a geometric ivariant, it should be +ve
// when the surface looks like a sphere, -ve when it looks like a saddle,
// however, the sign of the Mean curvature is not, it depends on the
// convention for normals - This code assumes that normals point outwards (ie
// from the surface of a sphere outwards). If a given mesh produces curvatures
// of opposite senses then the flag InvertMeanCurvature can be set and the
// Curvature reported by the Mean calculation will be inverted.
//
// .SECTION Thanks
// Philip Batchelor philipp.batchelor@kcl.ac.uk for creating and contributing
// the class and Andrew Maclean a.maclean@acfr.usyd.edu.au for cleanups and
// fixes
//
// .SECTION See Also
//

#ifndef __vtkMeshSmootheness_h
#define __vtkMeshSmootheness_h

#include "vtkPolyDataAlgorithm.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

#define VTK_CURVATURE_GAUSS 0
#define VTK_CURVATURE_MEAN  1

class VTK_EXPORT vtkMeshSmootheness : public vtkPolyDataAlgorithm
{
public:
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkMeshSmootheness,vtkPolyDataToPolyDataFilter);
#else
  vtkTypeMacro(vtkMeshSmootheness,vtkPolyDataAlgorithm);
#endif
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Construct with curvature type set to Gauss
  static vtkMeshSmootheness *New();

  // Description:
  // Set/Get Curvature type
  // VTK_CURVATURE_GAUSS: Gaussian curvature, stored as
  // DataArray "Gauss_Curvature"
  // VTK_CURVATURE_MEAN : Mean curvature, stored as
  // DataArray "Mean_Curvature"
  vtkSetMacro(CurvatureType,int);
  vtkGetMacro(CurvatureType,int);
  void SetCurvatureTypeToGaussian()
  { this->SetCurvatureType(VTK_CURVATURE_GAUSS); }
  void SetCurvatureTypeToMean()
  { this->SetCurvatureType(VTK_CURVATURE_MEAN); }

#if (VTK_MAJOR_VERSION <= 5)
  virtual void SetInput(vtkPolyData *input);
#else
  virtual void SetInputConnection(vtkAlgorithmOutput *input);
#endif

  vtkGetMacro(Smootheness,double);
  double Smootheness;

protected:
  vtkMeshSmootheness();
  ~vtkMeshSmootheness();

  vtkPolyData *mesh;
  int NumPts;
  int NumCls;

  double *MeanCurvature;
  int *NumNeighb;

  // Usual data generation method
  void Execute();

  // Description:
  // discrete Gauss curvature (K) computation,
  // cf http://www-ipg.umds.ac.uk/p.batchelor/curvatures/curvatures.html
  void GetGaussCurvature();

  // discrete Mean curvature (H) computation,
  // cf http://www-ipg.umds.ac.uk/p.batchelor/curvatures/curvatures.html
  void GetMeanCurvature();

  // Vars
  int CurvatureType;

private:
  vtkMeshSmootheness(const vtkMeshSmootheness&);  // Not implemented.
  void operator=(const vtkMeshSmootheness&);  // Not implemented.

};

#endif
