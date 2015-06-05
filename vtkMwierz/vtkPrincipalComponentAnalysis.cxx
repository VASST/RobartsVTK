/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkPrincipalComponentAnalysis.cxx,v $
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
#include "vtkPrincipalComponentAnalysis.h"
#include "vtkObjectFactory.h"
#include "vtkMath.h"
#include "vtkImageAccumulate.h"

#include <vtkVersion.h> //for VTK_MAJOR_VERSION

#if (VTK_MAJOR_VERSION <= 5)
vtkCxxRevisionMacro(vtkPrincipalComponentAnalysis, "$Revision: 1.1 $");
#endif
vtkStandardNewMacro(vtkPrincipalComponentAnalysis);

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
    b[j][i] = a[i][j];
  }
    }
}

//------------------------------------------------------------------------
inline void vtkReduceMatrix(double **a, double **b, int arows, int acols,
          int brows, int bcols)
{
  if ((arows < brows) | (acols < bcols))
  {
      cout << "\n Error in vtkReduceMatrix: a is smaller than b.\n";
      return;      // matrix b must be smaller or the same size as a
  }

  for(int i = 0; i < brows; i++)
  {
    for(int j = 0; j < bcols; j++)
  {
    b[i][j] = a[i][j];
  }
  }
}

//----------------------------------------------------------------------------
vtkPrincipalComponentAnalysis::vtkPrincipalComponentAnalysis()
{
  this->M = 0;
  this->N = 0;
  this->NumberOfModes = 1;
  this->MaskPoints = vtkPoints::New();
  this->MaskPoints->SetDataTypeToInt();
}

//----------------------------------------------------------------------------
void vtkPrincipalComponentAnalysis::AddImage(vtkImageData *input)
{
  if (this->M > (MAX_M - 1))
    {
      cout << "\n Warning: Max # of images fitable is " << MAX_M << ", ignoring this image\n";
    }
  else
    {
      this->Images[this->M] = input;
      this->M++;

      if (this->M == 1)
  {

    input->GetExtent(this->ext);
    input->GetSpacing(this->spa);
    input->GetOrigin(this->ori);

    this->OutputImage = vtkImageData::New();
    this->OutputImage->SetWholeExtent(this->ext);
    this->OutputImage->SetExtent(this->ext);
    this->OutputImage->SetSpacing(this->spa);
    this->OutputImage->SetOrigin(this->ori);
    this->OutputImage->SetNumberOfScalarComponents(1);
    this->OutputImage->SetScalarType(this->Images[0]->GetScalarType());
    this->OutputImage->AllocateScalars();

  }
    }
}

//----------------------------------------------------------------------------
void vtkPrincipalComponentAnalysis::SetMask(vtkImageData *mask)
{
  this->MaskImage = mask;
}

//----------------------------------------------------------------------------
void vtkPrincipalComponentAnalysis::SetMaskPoints(vtkPoints *maskPoints)
{
  this->MaskPoints = maskPoints;
}

//----------------------------------------------------------------------------
vtkPoints *vtkPrincipalComponentAnalysis::GetMaskPoints()
{
  return this->MaskPoints;
}

//----------------------------------------------------------------------------
void vtkPrincipalComponentAnalysis::Fit()
{
  int ext[6];
  int N,M = this->M,n = 0;

  // Determine N from the mask image.  Rounding at the end instead of flooring,
  // to avoid round off errors with this hokey method.
  vtkImageAccumulate *accumulate;
  accumulate = vtkImageAccumulate::New();
  accumulate->SetInput(this->MaskImage);
  accumulate->Update();
  N = int(0.5 + accumulate->GetVoxelCount() * accumulate->GetMean()[0]);
  this->N = N;

  // N: the number of voxels.
  // n: current voxel.
  // M: the number of images.
  // A: a N by M matrix with intensity values in from a particular
  //    voxel and image in the corresponding row and column.
  // AT: transpose of A (M by N).
  // L: AT * A (M by M).
  // psi: mean values for each voxel over all images (N by 1).
  // mu: eigenvalues of L.
  // v: eigenvectors of L (M by M).
  // u: eigenvectors of A * AT - the covarience matrix (N by M).
  double **A   = vtkNewMatrix(N,M);
  double **AT  = vtkNewMatrix(M,N);
  double **L   = vtkNewMatrix(M,M);
  double **psi = vtkNewMatrix(N,1);
  double *mu = new double[M];
  double **v = vtkNewMatrix(M,M);
  double **u = vtkNewMatrix(N,M);

  // Fill in A and psi.
  vtkZeroMatrix(psi,N,1);
  this->Images[0]->GetExtent(ext);
  for (int i = ext[0]; i <= ext[1]; i++)
  {
    for (int j = ext[2]; j <= ext[3]; j++)
   {
     for (int k = ext[4]; k <= ext[5]; k++)
     {
      if (this->MaskImage->GetScalarComponentAsFloat(i,j,k,0))
    {
       for (int l = 0; l < M; l++)
      {
          A[n][l] = double(this->Images[l]->GetScalarComponentAsFloat(i,j,k,0));
          psi[n][0] = psi[n][0] + A[n][l];
      }
       psi[n][0] = psi[n][0] / M;

       for (int lGN = 0; lGN < M; lGN++)
       {
          A[n][lGN] = A[n][lGN] - psi[n][0];
       }
      n++;
      this->MaskPoints->InsertNextPoint(i,j,k);
    }
    }
  }
  }

  // Calculate AT and the matrix L.
  vtkMatrixTranspose(A,AT,N,M);
  vtkMatrixMultiply(AT,A,L,M,N,N,M);

  // Find the eigenvalues and vectors of L (mu and v).
  vtkMath::JacobiN(L,M,mu,v);

  // Eigenvectors of A * AT (the PCA modes) are calculated using
  // single value decomposition: u = A * v * mu^(-1/2)
  double **muinvsqrt = vtkNewMatrix(M,M);
  double **vtimesmu = vtkNewMatrix(M,M);
  vtkZeroMatrix(muinvsqrt,M,M);
  cout << "\n Eigenvalues: ";
  for (int iGN = 0; iGN < M; iGN++)
  {
      cout << mu[iGN] << " ";
      if (mu[iGN] > 0)
    {
       muinvsqrt[iGN][iGN] = 1.0 / sqrt(mu[iGN]);
    }
      else
    {
        muinvsqrt[iGN][iGN] = 0.0;
    }
  }
  cout << "\n\n";
  vtkMatrixMultiply(v,muinvsqrt,vtimesmu,M,M,M,M);
  vtkMatrixMultiply(A,vtimesmu,u,N,M,M,M);

  // Keep only NumberOfModes eigenvectors
  double **uprime = vtkNewMatrix(N,this->NumberOfModes);
  vtkReduceMatrix(u,uprime,N,M,N,this->NumberOfModes);

  this->EigenVectors = uprime;
  this->MeanIntensities = psi;

  vtkDeleteMatrix(A);
  vtkDeleteMatrix(AT);
  vtkDeleteMatrix(L);
  vtkDeleteMatrix(v);
  vtkDeleteMatrix(u);
  vtkDeleteMatrix(muinvsqrt);
  vtkDeleteMatrix(vtimesmu);
  delete [] mu;
}

//----------------------------------------------------------------------------
vtkImageData *vtkPrincipalComponentAnalysis::GetEigenVectorsImage()
{
  int M = this->NumberOfModes;
  int N = this->N;

  vtkImageData *EVI;
  EVI = vtkImageData::New();
  EVI->SetWholeExtent(0,N-1,0,M-1,0,0);
  EVI->SetExtent(0,N-1,0,M-1,0,0);
  EVI->SetNumberOfScalarComponents(1);
  EVI->SetScalarTypeToFloat();
  EVI->AllocateScalars();

  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < M; j++)
  {
    EVI->SetScalarComponentFromFloat(i,j,0,0,this->EigenVectors[i][j]);
  }
    }

  return EVI;

}

//----------------------------------------------------------------------------
vtkImageData *vtkPrincipalComponentAnalysis::GetMeanIntensitiesImage()
{
  int N = this->N;

  vtkImageData *MII;
  MII = vtkImageData::New();
  MII->SetWholeExtent(0,N-1,0,0,0,0);
  MII->SetExtent(0,N-1,0,0,0,0);
  MII->SetNumberOfScalarComponents(1);
  MII->SetScalarTypeToFloat();
  MII->AllocateScalars();

  for (int i = 0; i < N; i++)
    {
      MII->SetScalarComponentFromFloat(i,0,0,0,this->MeanIntensities[i][0]);
    }

  return MII;

}

//----------------------------------------------------------------------------
void vtkPrincipalComponentAnalysis::SetEigenVectorsImage(vtkImageData *EVI)
{
  int ext[6];

  EVI->GetExtent(ext);

  int N = ext[1] + 1;
  int M = ext[3] + 1;

  this->N = N;
  this->M = M;
  this->NumberOfModes = M;

  this->EigenVectors = vtkNewMatrix(N,M);

  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < M; j++)
   {
     this->EigenVectors[i][j] = EVI->GetScalarComponentAsFloat(i,j,0,0);
   }
    }
}

//----------------------------------------------------------------------------
void vtkPrincipalComponentAnalysis::SetMeanIntensitiesImage(vtkImageData *MII)
{
  int ext[6];

  MII->GetExtent(ext);

  int N = ext[1] + 1;
  this->N = N;

  this->MeanIntensities = vtkNewMatrix(N,1);

  for (int i = 0; i < N; i++)
    {
     this->MeanIntensities[i][0] = MII->GetScalarComponentAsFloat(i,0,0,0);
    }
}

//----------------------------------------------------------------------------
vtkDoubleArray *vtkPrincipalComponentAnalysis::GetWeightsForImage(vtkImageData *image)
{
  int M = this->NumberOfModes;
  int N = this->N;
  double locs[3];

  double **ut = vtkNewMatrix(M,N);
  double **gammaMinusPsi = vtkNewMatrix(N,1);
  double **w = vtkNewMatrix(M,1);

  for (int i = 0; i < this->MaskPoints->GetNumberOfPoints(); i++)
    {
      this->MaskPoints->GetPoint(i,locs);
      gammaMinusPsi[i][0] = image->GetScalarComponentAsFloat(int(locs[0]),int(locs[1]),int(locs[2]),0);
      gammaMinusPsi[i][0] = gammaMinusPsi[i][0] - this->MeanIntensities[i][0];
    }

  vtkMatrixTranspose(this->EigenVectors,ut,N,M);
  vtkMatrixMultiply(ut,gammaMinusPsi,w,M,N,N,1);

  vtkDoubleArray *wRet;
  wRet = vtkDoubleArray::New();

  for (int iGN = 0; iGN < M; iGN++)
    {
      wRet->InsertNextValue(w[iGN][0]);
    }

  vtkDeleteMatrix(ut);
  vtkDeleteMatrix(gammaMinusPsi);
  vtkDeleteMatrix(w);

  return wRet;
}


//----------------------------------------------------------------------------
vtkImageData *vtkPrincipalComponentAnalysis::GetOutput(vtkDoubleArray *weights)
{
  int M = this->NumberOfModes;
  int N = this->N;
  int n = 0;
  double **w = vtkNewMatrix(M,1);
  double **gamma = vtkNewMatrix(N,1);

  double locs[3];

  for (int i = 0; i < M; i++) { w[i][0] = weights->GetValue(i); }
  vtkMatrixMultiply(this->EigenVectors,w,gamma,N,M,M,1);

  for (int iGN = 0; iGN < this->MaskPoints->GetNumberOfPoints(); iGN++)
    {
      this->MaskPoints->GetPoint(iGN,locs);
      this->OutputImage->SetScalarComponentFromFloat(int(locs[0]),int(locs[1]),int(locs[2]),0,
                 int(0.5 + gamma[iGN][0] + this->MeanIntensities[iGN][0]));
    }

  vtkDeleteMatrix(w);
  vtkDeleteMatrix(gamma);

  return this->OutputImage;
}
