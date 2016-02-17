/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkGridTransformBSpline.h,v $
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
// .NAME vtkGridTransform - a nonlinear warp transformation
// .SECTION Description
// vtkGridTransform describes a nonlinear warp transformation as a set
// of displacement vectors sampled along a uniform 3D grid.
// .SECTION Caveats
// The inverse grid transform is calculated using an iterative method,
// and is several times more expensive than the forward transform.
// .SECTION see also
// vtkThinPlateSplineTransform vtkGeneralTransform vtkTransformToGrid


#ifndef __vtkGridTransformBSpline_h
#define __vtkGridTransformBSpline_h

#include "vtkRobartsRegistrationModule.h"

#include "vtkWarpTransform.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

class vtkImageData;

#define VTK_GRID_NEAREST 0
#define VTK_GRID_LINEAR 1
#define VTK_GRID_CUBIC 3
#define VTK_GRID_BSPLINE 4

#define LookupTableSize 1000

class VTKROBARTSREGISTRATION_EXPORT vtkGridTransformBSpline : public vtkWarpTransform
{
public:
  static vtkGridTransformBSpline *New();
#if (VTK_MAJOR_VERSION < 6)
  vtkTypeRevisionMacro(vtkGridTransformBSpline,vtkWarpTransform);
#else
  vtkTypeMacro(vtkGridTransformBSpline,vtkWarpTransform);
#endif
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/Get the grid transform (the grid transform must have three
  // components for displacement in x, y, and z respectively).
  // The vtkGridTransform class will never modify the data.
  virtual void SetDisplacementGrid(vtkImageData*);
  vtkGetObjectMacro(DisplacementGrid,vtkImageData);

  // Description:
  // Set scale factor to be applied to the displacements.
  // This is used primarily for grids which contain integer
  // data types.  Default: 1
  vtkSetMacro(DisplacementScale,double);
  vtkGetMacro(DisplacementScale,double);

  // Description:
  // Set a shift to be applied to the displacements.  The shift
  // is applied after the scale, i.e. x = scale*y + shift.
  // Default: 0
  vtkSetMacro(DisplacementShift,double);
  vtkGetMacro(DisplacementShift,double);

  // Description:
  // Set interpolation mode for sampling the grid.  Higher-order
  // interpolation allows you to use a sparser grid.
  // Default: Linear.
  void SetInterpolationMode(int mode);
  vtkGetMacro(InterpolationMode,int);
  void SetInterpolationModeToNearestNeighbor()
  {
    this->SetInterpolationMode(VTK_GRID_NEAREST);
  };
  void SetInterpolationModeToLinear()
  {
    this->SetInterpolationMode(VTK_GRID_LINEAR);
  };
  void SetInterpolationModeToCubic()
  {
    this->SetInterpolationMode(VTK_GRID_CUBIC);
  };
  void SetInterpolationModeToBSpline()
  {
    this->SetInterpolationMode(VTK_GRID_BSPLINE);
  };
  const char *GetInterpolationModeAsString();

  // Description:
  // Make another transform of the same type.
  vtkAbstractTransform *MakeTransform();

  // Description:
  // Get the MTime.
  unsigned long GetMTime();

//BTX
  /// Memory for lookup table for B-spline basis function values
  static    double LookupTable[LookupTableSize][4];
//ETX

protected:
  vtkGridTransformBSpline();
  ~vtkGridTransformBSpline();

  // Description:
  // Update the displacement grid.
  void InternalUpdate();

  // Description:
  // Copy this transform from another of the same type.
  void InternalDeepCopy(vtkAbstractTransform *transform);

  // Description:
  // Internal functions for calculating the transformation.
  void ForwardTransformPoint(const float in[3], float out[3]);
  void ForwardTransformPoint(const double in[3], double out[3]);

  void ForwardTransformDerivative(const float in[3], float out[3],
                                  float derivative[3][3]);
  void ForwardTransformDerivative(const double in[3], double out[3],
                                  double derivative[3][3]);

  void InverseTransformPoint(const float in[3], float out[3]);
  void InverseTransformPoint(const double in[3], double out[3]);

  void InverseTransformDerivative(const float in[3], float out[3],
                                  float derivative[3][3]);
  void InverseTransformDerivative(const double in[3], double out[3],
                                  double derivative[3][3]);

//BTX
  void (*InterpolationFunction)(float point[3], float displacement[3],
                                float derivatives[3][3],
                                void *gridPtr, int grdType,
                                int inExt[6], int inInc[3]);

  /// Returns the value of the i-th B-spline basis function
  static double B (int, double);
  static double B0(double);
  static double B1(double);
  static double B2(double);
  static double B3(double);

  /// Returns the derivation of the i-th B-spline basis function
  static double dB (int, double);
  static double dB0(double);
  static double dB1(double);
  static double dB2(double);
  static double dB3(double);

//ETX
  int InterpolationMode;
  vtkImageData *DisplacementGrid;
  double DisplacementScale;
  double DisplacementShift;
private:
  vtkGridTransformBSpline(const vtkGridTransformBSpline&);  // Not implemented.
  void operator=(const vtkGridTransformBSpline&);  // Not implemented.
};

//BTX

//----------------------------------------------------------------------------
inline const char *vtkGridTransformBSpline::GetInterpolationModeAsString()
{
  switch (this->InterpolationMode)
  {
  case VTK_GRID_NEAREST:
    return "NearestNeighbor";
  case VTK_GRID_LINEAR:
    return "Linear";
  case VTK_GRID_CUBIC:
    return "Cubic";
  case VTK_GRID_BSPLINE:
    return "B-Spline";
  default:
    return "";
  }
}
//ETX

inline double vtkGridTransformBSpline::B(int i, double t)
{
  switch (i)
  {
  case 0:
    return (1-t)*(1-t)*(1-t)/6.0;
  case 1:
    return (3*t*t*t - 6*t*t + 4)/6.0;
  case 2:
    return (-3*t*t*t + 3*t*t + 3*t + 1)/6.0;
  case 3:
    return (t*t*t)/6.0;
  }
  return 0;
}

inline double vtkGridTransformBSpline::B0(double t)
{
  return (1-t)*(1-t)*(1-t)/6.0;
}

inline double vtkGridTransformBSpline::B1(double t)
{
  return (3*t*t*t - 6*t*t + 4)/6.0;
}

inline double vtkGridTransformBSpline::B2(double t)
{
  return (-3*t*t*t + 3*t*t + 3*t + 1)/6.0;
}

inline double vtkGridTransformBSpline::B3(double t)
{
  return (t*t*t)/6.0;
}


inline double vtkGridTransformBSpline::dB(int i, double t)
{
  switch (i)
  {
  case 0:
    return dB0(t);
  case 1:
    return dB1(t);
  case 2:
    return dB3(t);
  case 3:
    return dB3(t);
  }
  return 0;
}

inline double vtkGridTransformBSpline::dB0(double t)
{
  return -(1-t)*(1-t)/2.0;
}

inline double vtkGridTransformBSpline::dB1(double t)
{
  return (9*t*t - 12*t)/6.0;
}

inline double vtkGridTransformBSpline::dB2(double t)
{
  return (-9*t*t + 6*t + 3)/6.0;
}

inline double vtkGridTransformBSpline::dB3(double t)
{
  return (t*t)/2.0;
}

#endif
