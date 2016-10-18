/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    $RCSfile: vtkCompactSupportRBFTransform.h,v $
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
// .NAME vtkCompactSupportRBFTransform - a nonlinear warp transformation
// .SECTION Description
// vtkCompactSupportRBFTransform describes a nonlinear warp transform defined
// by a set of source and target landmarks. Any point on the mesh close to a
// source landmark will be moved to a place close to the corresponding target
// landmark. The points in between are interpolated smoothly using
// Bookstein's Thin Plate Spline algorithm.
// .SECTION Caveats
// 1) The inverse transform is calculated using an iterative method,
// and is several times more expensive than the forward transform.
// 2) Whenever you add, subtract, or set points you must call Modified()
// on the vtkPoints object, or the transformation might not update.
// 3) Collinear point configurations (except those that lie in the XY plane)
// result in an unstable transformation.
// .SECTION see also
// vtkGridTransform vtkGeneralTransform


#ifndef __vtkCompactSupportRBFTransform_h
#define __vtkCompactSupportRBFTransform_h

#include "vtkRobartsRegistrationExport.h"

#include "vtkWarpTransform.h"

#define VTK_RBF_CUSTOM 0
#define VTK_RBF_CS3D0C 1
#define VTK_RBF_CS3D2C 2
#define VTK_RBF_CS3D4C 3

class vtkRobartsRegistrationExport vtkCompactSupportRBFTransform : public vtkWarpTransform
{
public:
  vtkTypeMacro(vtkCompactSupportRBFTransform,vtkWarpTransform);
  void PrintSelf(ostream& os, vtkIndent indent);
  static vtkCompactSupportRBFTransform *New();

  // Description:
  // Specify the 'stiffness' of the spline. The default is 1.0.
  vtkGetMacro(Sigma,double);
  vtkSetMacro(Sigma,double);

  // Description:
  // Specify the radial basis function to use.  The default is
  // R2LogR which is what most people use as the thin plate spline.
  void SetBasis(int basis);
  vtkGetMacro(Basis,int);
  void SetBasisToCS3D0C() { this->SetBasis(VTK_RBF_CS3D0C); };
  void SetBasisToCS3D2C() { this->SetBasis(VTK_RBF_CS3D2C); };
  void SetBasisToCS3D4C() { this->SetBasis(VTK_RBF_CS3D4C); };
  const char *GetBasisAsString();

//BTX
  // Description:
  // Set the radial basis function to a custom function.  You must
  // supply both the function and its derivative with respect to r.
  void SetBasisFunction(double (*U)(double r)) {
    if (this->BasisFunction == U) { return; }
    this->SetBasis(VTK_RBF_CUSTOM);
    this->BasisFunction = U;
    this->Modified(); };
  void SetBasisDerivative(double (*dUdr)(double r, double &dU)) {
    this->BasisDerivative = dUdr;
    this->Modified(); };
//ETX

  // Description:
  // Set the source landmarks for the warp.  If you add or change the
  // vtkPoints object, you must call Modified() on it or the transformation
  // might not update.
  void SetSourceLandmarks(vtkPoints *source);
  vtkGetObjectMacro(SourceLandmarks,vtkPoints);

  // Description:
  // Set the target landmarks for the warp.  If you add or change the
  // vtkPoints object, you must call Modified() on it or the transformation
  // might not update.
  void SetTargetLandmarks(vtkPoints *target);
  vtkGetObjectMacro(TargetLandmarks,vtkPoints);

  // Description:
  // Get the MTime.
  unsigned long GetMTime();

  // Description:
  // Make another transform of the same type.
  vtkAbstractTransform *MakeTransform();

protected:
  vtkCompactSupportRBFTransform();
  ~vtkCompactSupportRBFTransform();

  // Description:
  // Prepare the transformation for application.
  void InternalUpdate();

  // Description:
  // This method does no type checking, use DeepCopy instead.
  void InternalDeepCopy(vtkAbstractTransform *transform);

  void ForwardTransformPoint(const float in[3], float out[3]);
  void ForwardTransformPoint(const double in[3], double out[3]);

  void ForwardTransformDerivative(const float in[3], float out[3],
                                  float derivative[3][3]);
  void ForwardTransformDerivative(const double in[3], double out[3],
                                  double derivative[3][3]);

  double Sigma;
  vtkPoints *SourceLandmarks;
  vtkPoints *TargetLandmarks;

//BTX
  // the radial basis function to use
  double (*BasisFunction)(double r);
  double (*BasisDerivative)(double r, double& dUdr);
//ETX
  int Basis;

  int NumberOfPoints;
  double **MatrixW;
private:
  vtkCompactSupportRBFTransform(const vtkCompactSupportRBFTransform&);  // Not implemented.
  void operator=(const vtkCompactSupportRBFTransform&);  // Not implemented.
};

#endif
