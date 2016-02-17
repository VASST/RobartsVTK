/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkDeformedVolume.h,v $
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
// .NAME vtkDeformedVolume - Returns the absolute difference image of 2 images
// .SECTION Description
// vtkDeformedVolume Returns the absolute difference image of 2 images

#ifndef __vtkDeformedVolume_h
#define __vtkDeformedVolume_h

#include "vtkRobartsRegistrationModule.h"

#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkMassProperties.h"
#include "vtkTransformPolyDataFilter.h"
#include "vtkGridTransformBSpline.h"

class VTKROBARTSREGISTRATION_EXPORT vtkDeformedVolume : public vtkObject
{
public:
  static vtkDeformedVolume *New();
  vtkTypeMacro(vtkDeformedVolume,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual void SetZeroGrid(vtkImageData *grid);
  virtual void SetDeformedSurface(vtkPolyData *surface);
  vtkSetVector3Macro(Translation, double);
  vtkSetVector3Macro(Location, int);

  double GetDeformedVolume();

protected:
  vtkDeformedVolume();
  ~vtkDeformedVolume();

  vtkImageData *Grid;
  vtkPolyData *Surface;

  double Translation[3];
  int Location[3];

  vtkTransformPolyDataFilter *Transform;
  vtkGridTransformBSpline *GridTransform;
  vtkMassProperties *Volume;

};

#endif













