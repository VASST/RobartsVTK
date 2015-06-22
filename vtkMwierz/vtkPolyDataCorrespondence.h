/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkPolyDataCorrespondence.h,v $
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

#ifndef __vtkPolyDataCorrespondence_h
#define __vtkPolyDataCorrespondence_h

#include "vtkPolyDataAlgorithm.h"
#include "vtkObjectFactory.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"
#include "vtkPoints.h"
#include "vtkDataArray.h"
#include "vtkDoubleArray.h"
#include "vtkLongArray.h"
#include "vtkCurvatures.h"
#include "vtkPolyDataNormals.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

class VTK_EXPORT vtkPolyDataCorrespondence : public vtkPolyDataAlgorithm
{
public:
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkPolyDataCorrespondence,vtkPolyDataToPolyDataFilter);
#else
  vtkTypeMacro(vtkPolyDataCorrespondence,vtkPolyDataAlgorithm);
#endif
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Construct with curvature type set to Gauss
  static vtkPolyDataCorrespondence *New();

  // Description:
  // Set/get the 2 input images.
  virtual void SetInput1(vtkPolyData *input);
  virtual void SetInput2(vtkPolyData *input);

  vtkSetMacro(ConstantN,double);
  vtkSetMacro(ConstantS,double);
  vtkSetMacro(ConstantC,double);
  vtkSetMacro(SliceAxis,int);
  vtkSetMacro(PrintOutput,int);

  void Update2D();
  void Update3D();
  void Update3DFast();
  vtkDoubleArray *GetDistances();
  vtkLongArray *GetPairings();
  void GetAxialCorrespondenceStats();
  double GetMeanDistance();
  double GetRMSDistance();

protected:
  vtkPolyDataCorrespondence();
  ~vtkPolyDataCorrespondence() {};

  vtkPolyData *poly1;
  vtkPolyData *poly2;

  double ConstantN;
  double ConstantS;
  double ConstantC;
  int SliceAxis;
  int PrintOutput;

  vtkDoubleArray *Distances;
  vtkLongArray *Pairings;

  double MeanDistance;
  double RMSDistance;

private:
  vtkPolyDataCorrespondence(const vtkPolyDataCorrespondence&);  // Not implemented.
  void operator=(const vtkPolyDataCorrespondence&);  // Not implemented.

};

#endif
