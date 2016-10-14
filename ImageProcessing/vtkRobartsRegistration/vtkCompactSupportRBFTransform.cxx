/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    $RCSfile: vtkCompactSupportRBFTransform.cxx,v $
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
#include "vtkCompactSupportRBFTransform.h"

#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPoints.h"

vtkStandardNewMacro(vtkCompactSupportRBFTransform);

//------------------------------------------------------------------------
// some dull matrix things

inline double** vtkNewMatrix(int rows, int cols)
{
  double *matrix = new double[rows*cols];
  double **m = new double *[rows];
  for(int i = 0; i < rows; i++)
    {
    m[i] = &matrix[i*cols];
    }
  return m;
}

//------------------------------------------------------------------------
inline void vtkDeleteMatrix(double **m)
{
  delete [] *m;
  delete [] m;
}

//------------------------------------------------------------------------
inline void vtkZeroMatrix(double **m, int rows, int cols)
{
  for(int i = 0; i < rows; i++)
    {
    for(int j = 0; j < cols; j++)
      {
      m[i][j] = 0.0;
      }
    }
}

//------------------------------------------------------------------------
inline void vtkMatrixMultiply(double **a, double **b, double **c,
                              int arows, int acols, int brows, int bcols)
{
  if(acols != brows)
    {
    return;     // acols must equal br otherwise we can't proceed
    }

  // c must have size arows*bcols (we assume this)

  for(int i = 0; i < arows; i++)
    {
    for(int j = 0; j < bcols; j++)
      {
      c[i][j] = 0.0;
      for(int k = 0; k < acols; k++)
        {
        c[i][j] += a[i][k]*b[k][j];
        }
      }
    }
}

//------------------------------------------------------------------------
inline void vtkMatrixTranspose(double **a, double **b, int rows, int cols)
{
  for(int i = 0; i < rows; i++)
    {
    for(int j = 0; j < cols; j++)
      {
      double tmp = a[i][j];
      b[i][j] = a[j][i];
      b[j][i] = tmp;
      }
    }
}

//------------------------------------------------------------------------
vtkCompactSupportRBFTransform::vtkCompactSupportRBFTransform()
{
  this->SourceLandmarks=NULL;
  this->TargetLandmarks=NULL;
  this->Sigma = 1.0;

  // If the InverseFlag is set, then we use an iterative
  // method to invert the transformation.
  // The InverseTolerance sets the precision to which we want to
  // calculate the inverse.
  this->InverseTolerance = 0.001;
  this->InverseIterations = 500;

  this->Basis = -1;
  this->SetBasisToCS3D0C();

  this->NumberOfPoints = 0;
  this->MatrixW = NULL;
}

//------------------------------------------------------------------------
vtkCompactSupportRBFTransform::~vtkCompactSupportRBFTransform()
{
  if (this->SourceLandmarks)
    {
    this->SourceLandmarks->Delete();
    }
  if (this->TargetLandmarks)
    {
    this->TargetLandmarks->Delete();
    }
  if (this->MatrixW)
    {
    vtkDeleteMatrix(this->MatrixW);
    this->MatrixW = NULL;
    }
}

//------------------------------------------------------------------------
void vtkCompactSupportRBFTransform::SetSourceLandmarks(vtkPoints *source)
{
  if (this->SourceLandmarks == source)
    {
    return;
    }

  if (this->SourceLandmarks)
    {
    this->SourceLandmarks->Delete();
    }

  source->Register(this);
  this->SourceLandmarks = source;

  this->Modified();
}

//------------------------------------------------------------------------
void vtkCompactSupportRBFTransform::SetTargetLandmarks(vtkPoints *target)
{
  if (this->TargetLandmarks == target)
    {
    return;
    }

  if (this->TargetLandmarks)
    {
    this->TargetLandmarks->Delete();
    }

  target->Register(this);
  this->TargetLandmarks = target;
  this->Modified();
}

//------------------------------------------------------------------------
unsigned long vtkCompactSupportRBFTransform::GetMTime()
{
  unsigned long result = this->vtkWarpTransform::GetMTime();
  unsigned long mtime;

  if (this->SourceLandmarks)
    {
    mtime = this->SourceLandmarks->GetMTime();
    if (mtime > result)
      {
      result = mtime;
      }
    }
  if (this->TargetLandmarks)
    {
    mtime = this->TargetLandmarks->GetMTime();
    if (mtime > result)
      {
      result = mtime;
      }
    }
  return result;
}

//------------------------------------------------------------------------
void vtkCompactSupportRBFTransform::InternalUpdate()
{
  if (this->SourceLandmarks == NULL || this->TargetLandmarks == NULL)
    {
    if (this->MatrixW)
      {
      vtkDeleteMatrix(this->MatrixW);
      }
    this->MatrixW = NULL;
    this->NumberOfPoints = 0;
    return;
    }

  if (this->SourceLandmarks->GetNumberOfPoints() !=
      this->TargetLandmarks->GetNumberOfPoints())
    {
    vtkErrorMacro("Update: Source and Target Landmarks contain a different number of points");
    return;
    }

  const vtkIdType N = this->SourceLandmarks->GetNumberOfPoints();
  const int D = 3; // dimensions

  // Notation and inspiration from:
  // Fred L. Bookstein (1997) "Shape and the Information in Medical Images:
  // A Decade of the Morphometric Synthesis" Computer Vision and Image
  // Understanding 66(2):97-118
  // and online work published by Tim Cootes (http://www.wiau.man.ac.uk/~bim)

  // the output weights and input matrices
  double **W = vtkNewMatrix(N,D);
  double **L = vtkNewMatrix(N,N);
  double **X = vtkNewMatrix(N,D);

  int q,c;
  double p[3],p2[3];
  double dx,dy,dz;
  double r;
  double (*phi)(double) = this->BasisFunction;

  for(q = 0; q < N; q++)
    {
      this->SourceLandmarks->GetPoint(q,p);
      // fill in the diagonal of L
      L[q][q] = phi(0.0);
      // fill the rest of L using symmetry
      for(c = 0; c < q; c++)
        {
        this->SourceLandmarks->GetPoint(c,p2);
        dx = p[0]-p2[0]; dy = p[1]-p2[1]; dz = p[2]-p2[2];
        r = sqrt(dx*dx + dy*dy + dz*dz);
        L[q][c] = L[c][q] = phi(r/this->Sigma);
        }
      }

  // build X - matrix of DISPLACEMENTS
  for (q = 0; q < N; q++)
    {
      this->SourceLandmarks->GetPoint(q,p);
      this->TargetLandmarks->GetPoint(q,p2);
      X[q][0] = p2[0] - p[0];
      X[q][1] = p2[1] - p[1];
      X[q][2] = p2[2] - p[2];
    }

  // solve for W, where W = Inverse(L)*X;
  // this is done via eigenvector decomposition so
  // that we can avoid singular values
  // W = V*Inverse(w)*U*X
  double **w = vtkNewMatrix(N,N);
  double **V = vtkNewMatrix(N,N);
  double **U = L;  // reuse the space
  double *values = new double[N];
  vtkMath::JacobiN(L,N,values,V);
  vtkMatrixTranspose(V,U,N,N);

  vtkIdType i, j;
  double maxValue = 0.0; // maximum eigenvalue
  for (i = 0; i < N; i++)
    {
      double tmp = fabs(values[i]);
      if (tmp > maxValue)
        {
    maxValue = tmp;
        }
    }

  for (i = 0; i < N; i++)
    {
      for (j = 0; j < N; j++)
        {
    w[i][j] = 0.0;
        }
      // here's the trick: don't invert the singular values
      if (fabs(values[i]/maxValue) > 1e-16)
  {
    w[i][i] = 1.0/values[i];
  }
    }
  delete [] values;


  vtkMatrixMultiply(U,X,W,N,N,N,D);
  vtkMatrixMultiply(w,W,X,N,N,N,D);
  vtkMatrixMultiply(V,X,W,N,N,N,D);

  vtkDeleteMatrix(V);
  vtkDeleteMatrix(w);
  vtkDeleteMatrix(U);
  vtkDeleteMatrix(X);

  if (this->MatrixW)
    {
      vtkDeleteMatrix(this->MatrixW);
    }
  this->MatrixW = W;
  this->NumberOfPoints = N;
}

//------------------------------------------------------------------------
// The matrix W was created by Update.  Not much has to be done to
// apply the transform:  do an affine transformation, then do
// perturbations based on the landmarks.
template<class T>
inline void vtkCompactSupportRBFForwardTransformPoint(vtkCompactSupportRBFTransform *self,
                                                    double **W, int N,
                                                    double (*phi)(double),
                                                    const T point[3], T output[3])
{
  if (N == 0)
    {
      output[0] = point[0];
      output[1] = point[1];
      output[2] = point[2];
      return;
    }

  double dx,dy,dz;
  double p[3];
  double U,r;
  double invSigma = 1.0/self->GetSigma();
  double sigma = self->GetSigma();
  double x = 0, y = 0, z = 0;

  vtkPoints *sourceLandmarks = self->GetSourceLandmarks();

  // Do the nonlinear stuff
  for(vtkIdType i = 0; i < N; i++)
    {
      sourceLandmarks->GetPoint(i,p);
      dx = point[0]-p[0]; dy = point[1]-p[1]; dz = point[2]-p[2];
      r = sqrt(dx*dx + dy*dy + dz*dz);
      U = phi(r*invSigma);
      x += U*W[i][0];
      y += U*W[i][1];
      z += U*W[i][2];
    }

  // finish off with adding the displacement to the starting location
  x += point[0];
  y += point[1];
  z += point[2];

  output[0] = x;
  output[1] = y;
  output[2] = z;

}

void vtkCompactSupportRBFTransform::ForwardTransformPoint(const double point[3],
                double output[3])
{
  vtkCompactSupportRBFForwardTransformPoint(this, this->MatrixW,
              this->NumberOfPoints,
              this->BasisFunction,
              point, output);
}

void vtkCompactSupportRBFTransform::ForwardTransformPoint(const float point[3],
                float output[3])
{
  vtkCompactSupportRBFForwardTransformPoint(this, this->MatrixW,
              this->NumberOfPoints,
              this->BasisFunction,
              point, output);
}

//----------------------------------------------------------------------------
// calculate the thin plate spline as well as the jacobian
template<class T>
inline void vtkCompactSupportRBFForwardTransformDerivative(
  vtkCompactSupportRBFTransform *self,
  double **W, int N,
  double (*phi)(double, double&),
  const T point[3], T output[3],
  T derivative[3][3])
{
  if (N == 0)
    {
      for (int i = 0; i < 3; i++)
  {
    output[i] = point[i];
    derivative[i][0] = derivative[i][1] = derivative[i][2] = 0.0;
    derivative[i][i] = 1.0;
  }
      return;
    }

  double dx,dy,dz;
  double p[3];
  double r, U, f, Ux, Uy, Uz;
  double x = 0, y = 0, z = 0;
  double invSigma = 1.0/self->GetSigma();
  double sigma = self->GetSigma();

  derivative[0][0] = derivative[0][1] = derivative[0][2] = 0;
  derivative[1][0] = derivative[1][1] = derivative[1][2] = 0;
  derivative[2][0] = derivative[2][1] = derivative[2][2] = 0;

  vtkPoints *sourceLandmarks = self->GetSourceLandmarks();

  // do the nonlinear stuff
  for(vtkIdType i = 0; i < N; i++)
    {
      sourceLandmarks->GetPoint(i,p);
      dx = point[0]-p[0]; dy = point[1]-p[1]; dz = point[2]-p[2];
      r = sqrt(dx*dx + dy*dy + dz*dz);

      // get both U and its derivative and do the sigma-mangling
      U = phi(r*invSigma, f);
      if (r != 0) f *= invSigma/r;

      Ux = f*dx;
      Uy = f*dy;
      Uz = f*dz;

      x += U*W[i][0];
      y += U*W[i][1];
      z += U*W[i][2];

      derivative[0][0] += Ux*W[i][0];
      derivative[0][1] += Uy*W[i][0];
      derivative[0][2] += Uz*W[i][0];
      derivative[1][0] += Ux*W[i][1];
      derivative[1][1] += Uy*W[i][1];
      derivative[1][2] += Uz*W[i][1];
      derivative[2][0] += Ux*W[i][2];
      derivative[2][1] += Uy*W[i][2];
      derivative[2][2] += Uz*W[i][2];
    }

  // finish off with adding the displacement to the starting location
  x += point[0];
  y += point[1];
  z += point[2];

  output[0] = x;
  output[1] = y;
  output[2] = z;

  derivative[0][0] += 1;
  derivative[1][1] += 1;
  derivative[2][2] += 1;

}

void vtkCompactSupportRBFTransform::ForwardTransformDerivative(
                                                  const double point[3],
                                                  double output[3],
                                                  double derivative[3][3])
{
  vtkCompactSupportRBFForwardTransformDerivative(this, this->MatrixW,
             this->NumberOfPoints,
             this->BasisDerivative,
             point, output, derivative);
}

void vtkCompactSupportRBFTransform::ForwardTransformDerivative(
                     const float point[3],
                     float output[3],
                     float derivative[3][3])
{
  vtkCompactSupportRBFForwardTransformDerivative(this, this->MatrixW,
             this->NumberOfPoints,
             this->BasisDerivative,
             point, output, derivative);
}

//------------------------------------------------------------------------
void vtkCompactSupportRBFTransform::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Sigma: " << this->Sigma << "\n";
  os << indent << "Basis: " << this->GetBasisAsString() << "\n";
  os << indent << "Source Landmarks: " << this->SourceLandmarks << "\n";
  if (this->SourceLandmarks)
    {
    this->SourceLandmarks->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "Target Landmarks: " << this->TargetLandmarks << "\n";
  if (this->TargetLandmarks)
    {
    this->TargetLandmarks->PrintSelf(os,indent.GetNextIndent());
    }
}

//----------------------------------------------------------------------------
vtkAbstractTransform *vtkCompactSupportRBFTransform::MakeTransform()
{
  return vtkCompactSupportRBFTransform::New();
}

//----------------------------------------------------------------------------
void vtkCompactSupportRBFTransform::InternalDeepCopy(
                                      vtkAbstractTransform *transform)
{
  vtkCompactSupportRBFTransform *t = (vtkCompactSupportRBFTransform *)transform;

  this->SetInverseTolerance(t->InverseTolerance);
  this->SetInverseIterations(t->InverseIterations);
  this->SetSigma(t->Sigma);
  this->SetBasis(t->GetBasis());
  this->SetSourceLandmarks(t->SourceLandmarks);
  this->SetTargetLandmarks(t->TargetLandmarks);

  if (this->InverseFlag != t->InverseFlag)
    {
    this->InverseFlag = t->InverseFlag;
    this->Modified();
    }
}

//------------------------------------------------------------------------
// Wendland's compact support RBF for 3D and C0 continuity
double vtkRBFCS3D0C(double r)
{
  if (r > 1.0)
    {
      return 0.0;
    }
  else
    {
      return pow((1.0-r),2.0);
    }
}

// calculate both phi(r) its derivative wrt r
double vtkRBFDRCS3D0C(double r, double &dUdr)
{
  if (r == 0)
    {
      dUdr = 0;
      return 1.0;
    }
  else if (r > 1.0)
    {
      dUdr = 0.0;
      return 0.0;
    }
  else
    {
      dUdr = -2.0 * (1.0 - r);
      return pow((1.0-r),2.0);
    }
}



//------------------------------------------------------------------------
// Wendland's compact support RBF for 3D and C2 continuity
double vtkRBFCS3D2C(double r)
{
  if (r > 1.0)
    {
      return 0.0;
    }
  else
    {
      return pow((1.0-r),4.0) * (4.0*r + 1.0);
    }
}

// calculate both phi(r) its derivative wrt r
double vtkRBFDRCS3D2C(double r, double &dUdr)
{
  if (r == 0)
    {
      dUdr = 0;
      return 1.0;
    }
  else if (r > 1.0)
    {
      dUdr = 0.0;
      return 0.0;
    }
  else
    {
      dUdr = -20.0 * pow((1.0-r),3.0) * r;
      return pow((1.0-r),4.0) * (4.0*r + 1.0);
    }
}

//------------------------------------------------------------------------
// Wendland's compact support RBF for 3D and C4 continuity
double vtkRBFCS3D4C(double r)
{
  if (r > 1.0)
    {
      return 0.0;
    }
  else
    {
      return pow((1.0-r),6.0) * (35.0*r*r + 18.0*r + 3.0);
    }
}

// calculate both phi(r) its derivative wrt r
double vtkRBFDRCS3D4C(double r, double &dUdr)
{
  if (r == 0)
    {
      dUdr = 0;
      return 3.0;
    }
  else if ( r > 1.0)
    {
      dUdr = 0.0;
      return 0.0;
    }
  else
    {
      dUdr = pow((1.0-r),5.0) * (-56.0*r) * (5.0*r+1.0);
      return pow((1.0-r),6.0) * (35.0*r*r + 18.0*r + 3.0);
    }
}


//------------------------------------------------------------------------
void vtkCompactSupportRBFTransform::SetBasis(int basis)
{
  if (basis == this->Basis)
    {
    return;
    }

  switch (basis)
    {
    case VTK_RBF_CUSTOM:
      break;
    case VTK_RBF_CS3D0C:
      this->BasisFunction = &vtkRBFCS3D0C;
      this->BasisDerivative = &vtkRBFDRCS3D0C;
      break;
    case VTK_RBF_CS3D2C:
      this->BasisFunction = &vtkRBFCS3D2C;
      this->BasisDerivative = &vtkRBFDRCS3D2C;
      break;
    case VTK_RBF_CS3D4C:
      this->BasisFunction = &vtkRBFCS3D4C;
      this->BasisDerivative = &vtkRBFDRCS3D4C;
      break;
    default:
      vtkErrorMacro( "SetBasisFunction: Unrecognized basis function");
      break;
    }

  this->Basis = basis;
  this->Modified();
}

//------------------------------------------------------------------------
const char *vtkCompactSupportRBFTransform::GetBasisAsString()
{
  switch (this->Basis)
    {
    case VTK_RBF_CUSTOM:
      return "Custom";
    case VTK_RBF_CS3D0C:
      return "CS3D0C";
    case VTK_RBF_CS3D2C:
      return "CS3D2C";
    case VTK_RBF_CS3D4C:
      return "CS3D4C";
     }
  return "Unknown";
}
