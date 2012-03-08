/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkGridTransform.cxx,v $
  Language:  C++
  Date:      $Date: 2002/06/17 14:10:53 $
  Version:   $Revision: 1.17 $

  Copyright (c) 1993-2002 Ken Martin, Will Schroeder, Bill Lorensen 
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGridTransformBSpline.h"

#include "vtkImageData.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"

#include "math.h"

vtkCxxRevisionMacro(vtkGridTransformBSpline, "$Revision: 1.17 $");
//vtkStandardNewMacro(vtkGridTransformBSpline);

vtkCxxSetObjectMacro(vtkGridTransformBSpline,DisplacementGrid,vtkImageData);

/// Size of lookup table for B-spline basis function values
#define LUTSIZE (double)(LookupTableSize-1)

/// Memory for lookup table for B-spline basis function values
double vtkGridTransformBSpline::LookupTable[LookupTableSize][4];
static  double dLookupTable[LookupTableSize][4];

inline int roundit(double x)
{
  return x > 0 ? int(x + 0.5) : int(x - 0.5);
}

//----------------------------------------------------------------------------
// fast floor() function for converting a double to an int
// (the floor() implementation on some computers is much slower than this,
// because they require some 'exact' behaviour that we don't).

inline int vtkGridFloor(double x, double &f)
{
  int ix = int(x);
  f = x-ix;
  if (f < 0) { f = x - (--ix); }

  return ix;
}

inline int vtkGridFloor(double x)
{
  int ix = int(x);
  if (x-ix < 0) { ix--; }

  return ix;
}

//----------------------------------------------------------------------------
// Nearest-neighbor interpolation of a displacement grid.
// The displacement as well as the derivatives are returned.
// There are two versions: one which computes the derivatives,
// and one which doesn't.

template <class T>
inline void vtkNearestHelper(double displacement[3], T *gridPtr, int increment)
{
  gridPtr += increment;
  displacement[0] = gridPtr[0];
  displacement[1] = gridPtr[1];
  displacement[2] = gridPtr[2];
}

inline void vtkNearestNeighborInterpolation(double point[3], 
                                            double displacement[3],
                                            void *gridPtr, int gridType,
                                            int gridExt[6], int gridInc[3])
{
  int gridId[3];
  gridId[0] = vtkGridFloor(point[0]+0.5f)-gridExt[0];
  gridId[1] = vtkGridFloor(point[1]+0.5f)-gridExt[2];
  gridId[2] = vtkGridFloor(point[2]+0.5f)-gridExt[4];
  
  int ext[3];
  ext[0] = gridExt[1]-gridExt[0];
  ext[1] = gridExt[3]-gridExt[2];
  ext[2] = gridExt[5]-gridExt[4];

  // do bounds check, most points will be inside so optimize for that
  if ((gridId[0] | (ext[0] - gridId[0]) |
       gridId[1] | (ext[1] - gridId[1]) |
       gridId[2] | (ext[2] - gridId[2])) < 0)
    {
    for (int i = 0; i < 3; i++)
      {
      if (gridId[i] < 0)
        {
        gridId[i] = 0; 
        }
      else if (gridId[i] > ext[i])
        {
        gridId[i] = ext[i];
        }
      }
    }

  // do nearest-neighbor interpolation
  int increment = gridId[0]*gridInc[0] + 
                  gridId[1]*gridInc[1] + 
                  gridId[2]*gridInc[2];

  switch (gridType)
    {
    case VTK_CHAR:
      vtkNearestHelper(displacement, (char *)gridPtr, increment);
      break;
    case VTK_UNSIGNED_CHAR:
      vtkNearestHelper(displacement, (unsigned char *)gridPtr, increment); 
      break;
    case VTK_SHORT:
      vtkNearestHelper(displacement, (short *)gridPtr, increment);
      break;
    case VTK_UNSIGNED_SHORT:
      vtkNearestHelper(displacement, (unsigned short *)gridPtr, increment);
      break;
    case VTK_FLOAT:
      vtkNearestHelper(displacement, (double *)gridPtr, increment);
      break;
    }
}

template <class T>
inline void vtkNearestHelper(double displacement[3], double derivatives[3][3],
                             T *gridPtr, int gridId[3], int gridId0[3], 
                             int gridId1[3], int gridInc[3])
{
  int incX = gridId[0]*gridInc[0];
  int incY = gridId[1]*gridInc[1];
  int incZ = gridId[2]*gridInc[2];

  T *gridPtr0;
  T *gridPtr1 = gridPtr + incX + incY + incZ;

  displacement[0] = gridPtr1[0];
  displacement[1] = gridPtr1[1];
  displacement[2] = gridPtr1[2];

  int incX0 = gridId0[0]*gridInc[0];
  int incX1 = gridId1[0]*gridInc[0];
  int incY0 = gridId0[1]*gridInc[1];

  int incY1 = gridId1[1]*gridInc[1];
  int incZ0 = gridId0[2]*gridInc[2];
  int incZ1 = gridId1[2]*gridInc[2];

  gridPtr0 = gridPtr + incX0 + incY + incZ;
  gridPtr1 = gridPtr + incX1 + incY + incZ;

  derivatives[0][0] = gridPtr1[0] - gridPtr0[0];
  derivatives[1][0] = gridPtr1[1] - gridPtr0[1];
  derivatives[2][0] = gridPtr1[2] - gridPtr0[2];

  gridPtr0 = gridPtr + incX + incY0 + incZ;
  gridPtr1 = gridPtr + incX + incY1 + incZ;

  derivatives[0][1] = gridPtr1[0] - gridPtr0[0];
  derivatives[1][1] = gridPtr1[1] - gridPtr0[1];
  derivatives[2][1] = gridPtr1[2] - gridPtr0[2];

  gridPtr0 = gridPtr + incX + incY + incZ0;
  gridPtr1 = gridPtr + incX + incY + incZ1;

  derivatives[0][2] = gridPtr1[0] - gridPtr0[0];
  derivatives[1][2] = gridPtr1[1] - gridPtr0[1];
  derivatives[2][2] = gridPtr1[2] - gridPtr0[2];
}

void vtkNearestNeighborInterpolation(double point[3], double displacement[3],
                                     double derivatives[3][3], void *gridPtr, 
                                     int gridType, int gridExt[6], int gridInc[3])
{
  if (derivatives == NULL)
    {
    vtkNearestNeighborInterpolation(point,displacement,gridPtr,gridType,
                                    gridExt,gridInc);
    return;
    }

  double f[3];
  int gridId0[3];
  gridId0[0] = vtkGridFloor(point[0],f[0])-gridExt[0];
  gridId0[1] = vtkGridFloor(point[1],f[1])-gridExt[2];
  gridId0[2] = vtkGridFloor(point[2],f[2])-gridExt[4];

  int gridId[3], gridId1[3];
  gridId[0] = gridId1[0] = gridId0[0] + 1;
  gridId[1] = gridId1[1] = gridId0[1] + 1;
  gridId[2] = gridId1[2] = gridId0[2] + 1;

  if (f[0] < 0.5) 
    {
    gridId[0] = gridId0[0];
    }
  if (f[1] < 0.5) 
    {
    gridId[1] = gridId0[1];
    }
  if (f[2] < 0.5) 
    {
    gridId[2] = gridId0[2];
    }
  
  int ext[3];
  ext[0] = gridExt[1] - gridExt[0];
  ext[1] = gridExt[3] - gridExt[2];
  ext[2] = gridExt[5] - gridExt[4];

  // do bounds check, most points will be inside so optimize for that
  if ((gridId0[0] | (ext[0] - gridId1[0]) |
       gridId0[1] | (ext[1] - gridId1[1]) |
       gridId0[2] | (ext[2] - gridId1[2])) < 0)
    {
    for (int i = 0; i < 3; i++) 
      {
      if (gridId0[i] < 0)
        {
        gridId[i] = 0;
        gridId0[i] = 0;
        gridId1[i] = 0;
        }
      else if (gridId1[i] > ext[i])
        {
        gridId[i] = ext[i];
        gridId0[i] = ext[i];
        gridId1[i] = ext[i];
        }
      }
    }

  // do nearest-neighbor interpolation
  switch (gridType)
    {
    case VTK_CHAR:
      vtkNearestHelper(displacement, derivatives, (char *)gridPtr, 
                       gridId, gridId0, gridId1, gridInc);
      break;
    case VTK_UNSIGNED_CHAR:
      vtkNearestHelper(displacement, derivatives, (unsigned char *)gridPtr, 
                       gridId, gridId0, gridId1, gridInc);
      break;
    case VTK_SHORT:
      vtkNearestHelper(displacement, derivatives, (short *)gridPtr, 
                       gridId, gridId0, gridId1, gridInc);
      break;
    case VTK_UNSIGNED_SHORT:
      vtkNearestHelper(displacement, derivatives, (unsigned short *)gridPtr, 
                       gridId, gridId0, gridId1, gridInc);
      break;
    case VTK_FLOAT:
      vtkNearestHelper(displacement, derivatives, (double *)gridPtr, 
                       gridId, gridId0, gridId1, gridInc);
      break;
    }
}

//----------------------------------------------------------------------------
// Trilinear interpolation of a displacement grid.
// The displacement as well as the derivatives are returned.

template <class T>
inline void vtkLinearHelper(double displacement[3], double derivatives[3][3],
                            double fx, double fy, double fz, T *gridPtr, 
                            int i000, int i001, int i010, int i011,
                            int i100, int i101, int i110, int i111)
{
  double rx = 1 - fx;
  double ry = 1 - fy;
  double rz = 1 - fz;
  
  double ryrz = ry*rz;
  double ryfz = ry*fz;
  double fyrz = fy*rz;
  double fyfz = fy*fz;

  double rxryrz = rx*ryrz;
  double rxryfz = rx*ryfz;
  double rxfyrz = rx*fyrz;
  double rxfyfz = rx*fyfz;
  double fxryrz = fx*ryrz;
  double fxryfz = fx*ryfz;
  double fxfyrz = fx*fyrz;
  double fxfyfz = fx*fyfz;

  if (!derivatives)
    {
    int i = 3;
    do
      {
      *displacement++ = (rxryrz*gridPtr[i000] + rxryfz*gridPtr[i001] +
                         rxfyrz*gridPtr[i010] + rxfyfz*gridPtr[i011] +
                         fxryrz*gridPtr[i100] + fxryfz*gridPtr[i101] +
                         fxfyrz*gridPtr[i110] + fxfyfz*gridPtr[i111]);
      gridPtr++;
      }
    while (--i);
    }
  else
    {
    double rxrz = rx*rz;
    double rxfz = rx*fz;
    double fxrz = fx*rz;
    double fxfz = fx*fz;
    
    double rxry = rx*ry;
    double rxfy = rx*fy;
    double fxry = fx*ry;
    double fxfy = fx*fy;
    
    double *derivative = *derivatives;

    int i = 3;
    do
      {
      *displacement++ = (rxryrz*gridPtr[i000] + rxryfz*gridPtr[i001] +
                         rxfyrz*gridPtr[i010] + rxfyfz*gridPtr[i011] +
                         fxryrz*gridPtr[i100] + fxryfz*gridPtr[i101] +
                         fxfyrz*gridPtr[i110] + fxfyfz*gridPtr[i111]);

      *derivative++ = (ryrz*(gridPtr[i100] - gridPtr[i000]) +
                       ryfz*(gridPtr[i101] - gridPtr[i001]) +
                       fyrz*(gridPtr[i110] - gridPtr[i010]) +
                       fyfz*(gridPtr[i111] - gridPtr[i011]));
      
      *derivative++ = (rxrz*(gridPtr[i010] - gridPtr[i000]) +
                       rxfz*(gridPtr[i011] - gridPtr[i001]) +
                       fxrz*(gridPtr[i110] - gridPtr[i100]) +
                       fxfz*(gridPtr[i111] - gridPtr[i101]));
      
      *derivative++ = (rxry*(gridPtr[i001] - gridPtr[i000]) +
                       rxfy*(gridPtr[i011] - gridPtr[i010]) +
                       fxry*(gridPtr[i101] - gridPtr[i100]) +
                       fxfy*(gridPtr[i111] - gridPtr[i110]));

      gridPtr++;
      }
    while (--i);
    }
}

void vtkTrilinearInterpolation(double point[3], double displacement[3],
                               double derivatives[3][3], void *gridPtr, int gridType, 
                               int gridExt[6], int gridInc[3])
{
  // change point into integer plus fraction
  double f[3];
  int floorX = vtkGridFloor(point[0],f[0]);
  int floorY = vtkGridFloor(point[1],f[1]);
  int floorZ = vtkGridFloor(point[2],f[2]);

  int gridId0[3];
  gridId0[0] = floorX - gridExt[0];
  gridId0[1] = floorY - gridExt[2];
  gridId0[2] = floorZ - gridExt[4];

  int gridId1[3];
  gridId1[0] = gridId0[0] + 1;
  gridId1[1] = gridId0[1] + 1;
  gridId1[2] = gridId0[2] + 1;

  int ext[3];
  ext[0] = gridExt[1] - gridExt[0];
  ext[1] = gridExt[3] - gridExt[2];
  ext[2] = gridExt[5] - gridExt[4];

  // do bounds check, most points will be inside so optimize for that
  if ((gridId0[0] | (ext[0] - gridId1[0]) |
       gridId0[1] | (ext[1] - gridId1[1]) |
       gridId0[2] | (ext[2] - gridId1[2])) < 0)
    {
    for (int i = 0; i < 3; i++)
      {
      if (gridId0[i] < 0)
        {
        gridId0[i] = 0;
        gridId1[i] = 0;
        f[i] = 0;
        }
      else if (gridId1[i] > ext[i])
        {
        gridId0[i] = ext[i];
        gridId1[i] = ext[i];
        f[i] = 0;
        }
      }
    }

  // do trilinear interpolation
  int factX0 = gridId0[0]*gridInc[0];
  int factY0 = gridId0[1]*gridInc[1];
  int factZ0 = gridId0[2]*gridInc[2];

  int factX1 = gridId1[0]*gridInc[0];
  int factY1 = gridId1[1]*gridInc[1];
  int factZ1 = gridId1[2]*gridInc[2];
    
  int i000 = factX0+factY0+factZ0;
  int i001 = factX0+factY0+factZ1;
  int i010 = factX0+factY1+factZ0;
  int i011 = factX0+factY1+factZ1;
  int i100 = factX1+factY0+factZ0;
  int i101 = factX1+factY0+factZ1;
  int i110 = factX1+factY1+factZ0;
  int i111 = factX1+factY1+factZ1;
  
  switch (gridType)
    {
    case VTK_CHAR:
      vtkLinearHelper(displacement, derivatives, f[0], f[1], f[2], 
                      (char *)gridPtr,
                      i000, i001, i010, i011, i100, i101, i110, i111);
      break;
    case VTK_UNSIGNED_CHAR:
      vtkLinearHelper(displacement, derivatives, f[0], f[1], f[2], 
                      (unsigned char *)gridPtr,
                      i000, i001, i010, i011, i100, i101, i110, i111);
      break;
    case VTK_SHORT:
      vtkLinearHelper(displacement, derivatives, f[0], f[1], f[2], 
                      (short *)gridPtr, 
                      i000, i001, i010, i011, i100, i101, i110, i111);
      break;
    case VTK_UNSIGNED_SHORT:
      vtkLinearHelper(displacement, derivatives, f[0], f[1], f[2], 
                      (unsigned short *)gridPtr,
                      i000, i001, i010, i011, i100, i101, i110, i111);
      break;
    case VTK_FLOAT:
      vtkLinearHelper(displacement, derivatives, f[0], f[1], f[2], 
                      (double *)gridPtr,
                      i000, i001, i010, i011, i100, i101, i110, i111);
      break;
    }
}

//----------------------------------------------------------------------------
// Do tricubic interpolation of the input data 'gridPtr' of extent 'gridExt' 
// at the 'point'.  The result is placed at 'outPtr'.  
// The number of scalar components in the data is 'numscalars'

// The tricubic interpolation ensures that both the intensity and
// the first derivative of the intensity are smooth across the
// image.  The first derivative is estimated using a 
// centered-difference calculation.


// helper function: set up the lookup indices and the interpolation 
// coefficients

void vtkSetTricubicInterpCoeffs(double F[4], int *l, int *m, double f, 
                                int interpMode)
{   
  double fp1,fm1,fm2;

  switch (interpMode)
    {
    case 7:     // cubic interpolation
      *l = 0; *m = 4; 
      fm1 = f-1;
      F[0] = -f*fm1*fm1/2;
      F[1] = ((3*f-2)*f-2)*fm1/2;
      F[2] = -((3*f-4)*f-1)*f/2;
      F[3] = f*f*fm1/2;
      break;
    case 0:     // no interpolation
    case 2:
    case 4:
    case 6:
      *l = 1; *m = 2;
      F[0] = 0;
      F[1] = 1;
      F[2] = 0;
      F[3] = 0;
      break;
    case 1:     // linear interpolation
      *l = 1; *m = 3;
      F[0] = 0;
      F[1] = 1-f;
      F[2] = f;
      F[3] = 0;
      break;
    case 3:     // quadratic interpolation
      *l = 1; *m = 4; 
      fm1 = f-1; fm2 = fm1-1;
      F[0] = 0;
      F[1] = fm1*fm2/2;
      F[2] = -f*fm2;
      F[3] = f*fm1/2;
      break;
    case 5:     // quadratic interpolation
      *l = 0; *m = 3; 
      fp1 = f+1; fm1 = f-1; 
      F[0] = f*fm1/2;
      F[1] = -fp1*fm1;
      F[2] = fp1*f/2;
      F[3] = 0;
      break;
    }
}

// set coefficients to be used to find the derivative of the cubic
void vtkSetTricubicDerivCoeffs(double F[4], double G[4], int *l, int *m, 
                               double f, int interpMode)
{   
  double fp1,fm1,fm2;

  switch (interpMode)
    {
    case 7:     // cubic interpolation
      *l = 0; *m = 4; 
      fm1 = f-1;
      F[0] = -f*fm1*fm1/2;
      F[1] = ((3*f-2)*f-2)*fm1/2;
      F[2] = -((3*f-4)*f-1)*f/2;
      F[3] = f*f*fm1/2;
      G[0] = -((3*f-4)*f+1)/2;
      G[1] =  (9*f-10)*f/2;
      G[2] = -((9*f-8)*f-1)/2;
      G[3] =  (3*f-2)*f/2;
      break;
    case 0:     // no interpolation
    case 2:
    case 4:
    case 6:
      *l = 1; *m = 2;
      F[0] = 0;
      F[1] = 1;
      F[2] = 0;
      F[3] = 0;
      G[0] = 0;
      G[1] = 0;
      G[2] = 0;
      G[3] = 0;
      break;
    case 1:     // linear interpolation
      *l = 1; *m = 3;
      F[0] = 0;
      F[1] = 1-f;
      F[2] = f;
      F[3] = 0;
      G[0] =  0;
      G[1] = -1;
      G[2] =  1;
      G[3] =  0;
      break;
    case 3:     // quadratic interpolation
      *l = 1; *m = 4; 
      fm1 = f-1; fm2 = fm1-1;
      F[0] = 0;
      F[1] = fm1*fm2/2;
      F[2] = -f*fm2;
      F[3] = f*fm1/2;
      G[0] = 0;
      G[1] = f-1.5;
      G[2] = 2-2*f;
      G[3] = f-0.5;
      break;
    case 5:     // quadratic interpolation
      *l = 0; *m = 3; 
      fp1 = f+1; fm1 = f-1; 
      F[0] = f*fm1/2;
      F[1] = -fp1*fm1;
      F[2] = fp1*f/2;
      F[3] = 0;
      G[0] = f-0.5;
      G[1] = -2*f;
      G[2] = f+0.5;
      G[3] = 0;
      break;
    }
}

// tricubic interpolation of a warp grid with derivatives
// (set derivatives to NULL to avoid computing them).

template <class T>
inline void vtkCubicHelper(double displacement[3], double derivatives[3][3],
                           double fx, double fy, double fz, T *gridPtr,
                           int interpModeX, int interpModeY, int interpModeZ,
                           int factX[4], int factY[4], int factZ[4])
{
  double fX[4],fY[4],fZ[4];
  double gX[4],gY[4],gZ[4];
  int jl,jm,kl,km,ll,lm;

  if (derivatives)
    {
    for (int i = 0; i < 3; i++)
      {
      derivatives[i][0] = 0.0f; 
      derivatives[i][1] = 0.0f; 
      derivatives[i][2] = 0.0f;
      }
    vtkSetTricubicDerivCoeffs(fX,gX,&ll,&lm,fx,interpModeX);
    vtkSetTricubicDerivCoeffs(fY,gY,&kl,&km,fy,interpModeY);
    vtkSetTricubicDerivCoeffs(fZ,gZ,&jl,&jm,fz,interpModeZ);
    }
  else
    {
    vtkSetTricubicInterpCoeffs(fX,&ll,&lm,fx,interpModeX);
    vtkSetTricubicInterpCoeffs(fY,&kl,&km,fy,interpModeY);
    vtkSetTricubicInterpCoeffs(fZ,&jl,&jm,fz,interpModeZ);
    }

  // Here is the tricubic interpolation
  // (or cubic-cubic-linear, or cubic-nearest-cubic, etc)
  double vY[3],vZ[3];
  displacement[0] = 0;
  displacement[1] = 0;
  displacement[2] = 0;
  for (int j = jl; j < jm; j++)
    {
    T *gridPtr1 = gridPtr + factZ[j];
    vZ[0] = 0;
    vZ[1] = 0;
    vZ[2] = 0;
    for (int k = kl; k < km; k++)
      {
      T *gridPtr2 = gridPtr1 + factY[k];
      vY[0] = 0;
      vY[1] = 0;
      vY[2] = 0;
      if (!derivatives)
        {
        for (int l = ll; l < lm; l++)
          {
          T *gridPtr3 = gridPtr2 + factX[l];
          double f = fX[l];
          vY[0] += gridPtr3[0] * f;
          vY[1] += gridPtr3[1] * f;
          vY[2] += gridPtr3[2] * f;
          }
        }
      else
        {
        for (int l = ll; l < lm; l++)
          {
          T *gridPtr3 = gridPtr2 + factX[l];
          double f = fX[l];
          double gff = gX[l]*fY[k]*fZ[j];
          double fgf = fX[l]*gY[k]*fZ[j];
          double ffg = fX[l]*fY[k]*gZ[j];
          double inVal = gridPtr3[0];
          vY[0] += inVal * f;
          derivatives[0][0] += inVal * gff;
          derivatives[0][1] += inVal * fgf;
          derivatives[0][2] += inVal * ffg;
          inVal = gridPtr3[1];
          vY[1] += inVal * f;
          derivatives[1][0] += inVal * gff;
          derivatives[1][1] += inVal * fgf;
          derivatives[1][2] += inVal * ffg;
          inVal = gridPtr3[2];
          vY[2] += inVal * f;
          derivatives[2][0] += inVal * gff;
          derivatives[2][1] += inVal * fgf;
          derivatives[2][2] += inVal * ffg;
          }
        }
        vZ[0] += vY[0]*fY[k];
        vZ[1] += vY[1]*fY[k];
        vZ[2] += vY[2]*fY[k];
      }
    displacement[0] += vZ[0]*fZ[j];
    displacement[1] += vZ[1]*fZ[j];
    displacement[2] += vZ[2]*fZ[j];
    }
}

void vtkTricubicInterpolation(double point[3], double displacement[3], 
                              double derivatives[3][3], void *gridPtr, 
                              int gridType, int gridExt[6], int gridInc[3])
{
  int factX[4],factY[4],factZ[4];

  // change point into integer plus fraction
  double f[3];
  int floorX = vtkGridFloor(point[0],f[0]);
  int floorY = vtkGridFloor(point[1],f[1]);
  int floorZ = vtkGridFloor(point[2],f[2]);

  int gridId0[3];
  gridId0[0] = floorX - gridExt[0];
  gridId0[1] = floorY - gridExt[2];
  gridId0[2] = floorZ - gridExt[4];

  int gridId1[3];
  gridId1[0] = gridId0[0] + 1;
  gridId1[1] = gridId0[1] + 1;
  gridId1[2] = gridId0[2] + 1;

  int ext[3];
  ext[0] = gridExt[1] - gridExt[0];
  ext[1] = gridExt[3] - gridExt[2];
  ext[2] = gridExt[5] - gridExt[4];

  // the doInterpX,Y,Z variables are 0 if interpolation
  // does not have to be done in the specified direction.
  int doInterp[3];
  doInterp[0] = 1;
  doInterp[1] = 1;
  doInterp[2] = 1;

  // do bounds check, most points will be inside so optimize for that
  if ((gridId0[0] | (ext[0] - gridId1[0]) |
       gridId0[1] | (ext[1] - gridId1[1]) |
       gridId0[2] | (ext[2] - gridId1[2])) < 0)
    {
    for (int i = 0; i < 3; i++)
      {
      if (gridId0[i] < 0)
        {
        gridId0[i] = 0;
        gridId1[i] = 0;
        doInterp[i] = 0;
        f[i] = 0;
        }
      else if (gridId1[i] > ext[i])
        {
        gridId0[i] = ext[i];
        gridId1[i] = ext[i];
        doInterp[i] = 0;
        f[i] = 0;
        }
      }
    }

  // do tricubic interpolation
  
  for (int i = 0; i < 4; i++)
    {
    factX[i] = (gridId0[0]-1+i)*gridInc[0];
    factY[i] = (gridId0[1]-1+i)*gridInc[1];
    factZ[i] = (gridId0[2]-1+i)*gridInc[2];
    }

  // depending on whether we are at the edge of the 
  // input extent, choose the appropriate interpolation
  // method to use

  int interpModeX = ((gridId0[0] > 0) << 2) + 
                    ((gridId1[0] < ext[0]) << 1) +
                    doInterp[0];
  int interpModeY = ((gridId0[1] > 0) << 2) + 
                    ((gridId1[1] < ext[1]) << 1) +
                    doInterp[1];
  int interpModeZ = ((gridId0[2] > 0) << 2) + 
                    ((gridId1[2] < ext[2]) << 1) +
                    doInterp[2];

  switch (gridType)
    {
    case VTK_CHAR:
      vtkCubicHelper(displacement, derivatives, f[0], f[1], f[2],
                     (char *)gridPtr,
                     interpModeX, interpModeY, interpModeZ,
                     factX, factY, factZ);
      break;
    case VTK_UNSIGNED_CHAR:
      vtkCubicHelper(displacement, derivatives, f[0], f[1], f[2],
                     (unsigned char *)gridPtr,
                     interpModeX, interpModeY, interpModeZ,
                     factX, factY, factZ);
      break;
    case VTK_SHORT:
      vtkCubicHelper(displacement, derivatives, f[0], f[1], f[2],
                     (short *)gridPtr,
                     interpModeX, interpModeY, interpModeZ,
                     factX, factY, factZ);
      break;
    case VTK_UNSIGNED_SHORT:
      vtkCubicHelper(displacement, derivatives, f[0], f[1], f[2],
                     (unsigned short *)gridPtr,
                     interpModeX, interpModeY, interpModeZ,
                     factX, factY, factZ);
      break;
    case VTK_FLOAT:
      vtkCubicHelper(displacement, derivatives, f[0], f[1], f[2],
                     (double *)gridPtr,
                     interpModeX, interpModeY, interpModeZ,
                     factX, factY, factZ);
      break;
    }
}

//----------------------------------------------------------------------------
// B-Spline grid interpolation

template <class Type>
static inline void vtkBSplineHelper(double displacement[3], 
				    double derivatives[3][3],
				    int l,int m,int n,
				    int S,int T,int U,
				    Type *gridPtr, int gridInc[3],int ext[3]){
  int i, j, k,K,J,I,offset;
  double B_K, B_J, B_I;
  double v, dx,dy,dz;

  displacement[0]=0;
  displacement[1]=0;
  displacement[2]=0;

  if (!derivatives) {
    for (k = 0; k < 4; k++){
      K = k + n - 1;
      if ((K >= 0) && (K <= ext[2])){
	B_K   = vtkGridTransformBSpline::LookupTable[U][k];
	for (j = 0; j < 4; j++){
	  J = j + m - 1;
	  if ((J >= 0) && (J <= ext[1])){
	    B_J   = vtkGridTransformBSpline::LookupTable[T][j];
	    for (i = 0; i < 4; i++){
	      I = i + l - 1;
	      if ((I >= 0) && (I <= ext[0])){
		B_I   = vtkGridTransformBSpline::LookupTable[S][i];

		offset=I*gridInc[0]+J*gridInc[1]+K*gridInc[2];
		dx=((double) gridPtr[offset  ]);
		dy=((double) gridPtr[offset+1]);
		dz=((double) gridPtr[offset+2]);

		//		if (dx>2.0) cout << " " << dx << " ";
		
		v = B_I * B_J * B_K;
		displacement[0] += dx * v;
		displacement[1] += dy * v;
		displacement[2] += dz * v;

	      } 
	    }
	  } 
	}
      }
    }

  } else {
    double dB_K, dB_J, dB_I;
    for (i=0; i<3; i++) {
      for (j=0; j<3; j++) {
	derivatives[i][j]=0.0;
      }
    }
    
    for (k = 0; k < 4; k++){
      K = k + n - 1;
      if ((K >= 0) && (K <= ext[2])){
	B_K   = vtkGridTransformBSpline::LookupTable[U][k];
	dB_K = dLookupTable[U][k];
	for (j = 0; j < 4; j++){
	  J = j + m - 1;
	  if ((J >= 0) && (J <= ext[1])){
	    B_J   = vtkGridTransformBSpline::LookupTable[T][j];
	    dB_J = dLookupTable[T][j];
	    for (i = 0; i < 4; i++){
	      I = i + l - 1;
	      if ((I >= 0) && (I <= ext[0])){
		B_I   = vtkGridTransformBSpline::LookupTable[S][i];
		dB_I = dLookupTable[S][i];

		offset=I*gridInc[0]+J*gridInc[1]+K*gridInc[2];
		dx=((double) gridPtr[offset  ]);
		dy=((double) gridPtr[offset+1]);
		dz=((double) gridPtr[offset+2]);

		v = B_I * B_J * B_K;
		displacement[0] += dx * v;
		displacement[1] += dy * v;
		displacement[2] += dz * v;

		v = B_I * B_J * dB_K;
		derivatives[0][2] += dx * v;
		derivatives[1][2] += dy * v;
		derivatives[2][2] += dz * v;

		v = B_I * dB_J * B_K;
		derivatives[0][1] += dx * v;
		derivatives[1][1] += dy * v;
		derivatives[2][1] += dz * v;

		v = dB_I * B_J * B_K;
		derivatives[0][0] += dx * v;
		derivatives[1][0] += dy * v;
		derivatives[2][0] += dz * v;
	      } 
	    }
	  } 
	}
      }
    }


  }

}


// point: point position in the coordinate system of the grid
// displacement: result of this function
// derivatives: in the first step NULL
// gridPtr: pointer to the displacement vectors of the grid
// gridType: Type of the grid, e.g. VTK_CHAR, VTK_FLOAT
// gridExt: min and max index of the control points (can be negative)
// gridInc: grid increments

static void vtkBSplineInterpolation(double point[3], double displacement[3], 
				    double derivatives[3][3], void *gridPtr, 
				    int gridType, int gridExt[6], int gridInc[3])
{
  double s, t, u;
  // double v, B_K, B_J, B_I;
  int l, m, n, S, T, U;

  l = (int)floor(point[0]);
  m = (int)floor(point[1]);
  n = (int)floor(point[2]);
  s = point[0] - l;
  t = point[1] - m;
  u = point[2] - n;
  l -= gridExt[0];
  m -= gridExt[2];
  n -= gridExt[4];
  S = roundit(LUTSIZE*s);
  T = roundit(LUTSIZE*t);
  U = roundit(LUTSIZE*u);

  int ext[3];
  ext[0] = gridExt[1]-gridExt[0];
  ext[1] = gridExt[3]-gridExt[2];
  ext[2] = gridExt[5]-gridExt[4];


  switch (gridType) {
  case VTK_CHAR:
    vtkBSplineHelper(displacement, derivatives, 
		     l,m,n,S,T,U,
		     (char *)gridPtr,gridInc,ext);
    break;
  case VTK_UNSIGNED_CHAR:
    vtkBSplineHelper(displacement, derivatives, 
		     l,m,n,S,T,U,
		     (unsigned char *)gridPtr,gridInc,ext);
    break;
  case VTK_SHORT:
    vtkBSplineHelper(displacement, derivatives, 
		     l,m,n,S,T,U,
		     (short *)gridPtr,gridInc,ext);
    break;
  case VTK_UNSIGNED_SHORT:
    vtkBSplineHelper(displacement, derivatives, 
		     l,m,n,S,T,U, 
		     (unsigned short *)gridPtr,gridInc,ext);
    break;
  case VTK_FLOAT:
    vtkBSplineHelper(displacement, derivatives, 
		     l,m,n,S,T,U,
		     (double *)gridPtr,gridInc,ext);
    break;
  }

}		  
                
//----------------------------------------------------------------------------
vtkGridTransformBSpline::vtkGridTransformBSpline()
{
  this->InterpolationMode = VTK_GRID_LINEAR;
  this->InterpolationFunction = &vtkTrilinearInterpolation;
  this->DisplacementGrid = NULL;
  this->DisplacementScale = 1.0;
  this->DisplacementShift = 0.0;
  // the grid warp has a fairly large tolerance
  this->InverseTolerance = 0.01;

  int i;
  // Initialize lookup table
  for (i = 0; i < LookupTableSize; i++){
    this->LookupTable[i][0] = this->B0(i/LUTSIZE);
    this->LookupTable[i][1] = this->B1(i/LUTSIZE);
    this->LookupTable[i][2] = this->B2(i/LUTSIZE);
    this->LookupTable[i][3] = this->B3(i/LUTSIZE);
    dLookupTable[i][0] = this->dB0(i/LUTSIZE);
    dLookupTable[i][1] = this->dB1(i/LUTSIZE);
    dLookupTable[i][2] = this->dB2(i/LUTSIZE);
    dLookupTable[i][3] = this->dB3(i/LUTSIZE);
  }

}

//----------------------------------------------------------------------------
vtkGridTransformBSpline::~vtkGridTransformBSpline()
{
  this->SetDisplacementGrid(NULL);
}

//----------------------------------------------------------------------------
void vtkGridTransformBSpline::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "InterpolationMode: " 
     << this->GetInterpolationModeAsString() << "\n";
  os << indent << "DisplacementScale: " << this->DisplacementScale << "\n";
  os << indent << "DisplacementShift: " << this->DisplacementShift << "\n";
  os << indent << "DisplacementGrid: " << this->DisplacementGrid << "\n";
  if(this->DisplacementGrid)
    {
    this->DisplacementGrid->PrintSelf(os,indent.GetNextIndent());
    }
}

//----------------------------------------------------------------------------
// need to check the input image data to determine MTime
unsigned long vtkGridTransformBSpline::GetMTime()
{
  unsigned long mtime,result;
  result = vtkWarpTransform::GetMTime();
  if (this->DisplacementGrid)
    {
    this->DisplacementGrid->UpdateInformation();

    mtime = this->DisplacementGrid->GetPipelineMTime();
    result = ( mtime > result ? mtime : result );    

    mtime = this->DisplacementGrid->GetMTime();
    result = ( mtime > result ? mtime : result );
    }

  return result;
}

//----------------------------------------------------------------------------
void vtkGridTransformBSpline::SetInterpolationMode(int mode)
{
  if (mode == this->InterpolationMode)
    {
    return;
    }
  this->InterpolationMode = mode;

  switch(mode)
    {
    case VTK_GRID_NEAREST:
      this->InterpolationFunction = &vtkNearestNeighborInterpolation;
      break;
    case VTK_GRID_LINEAR:
      this->InterpolationFunction = &vtkTrilinearInterpolation;
      break;
    case VTK_GRID_CUBIC:
      this->InterpolationFunction = &vtkTricubicInterpolation;
      break;
    case VTK_GRID_BSPLINE:
      this->InterpolationFunction = &vtkBSplineInterpolation;
      break;
    default:
      vtkErrorMacro( << "SetInterpolationMode: Illegal interpolation mode");
    }
  this->Modified();
}

//----------------------------------------------------------------------------
void vtkGridTransformBSpline::ForwardTransformPoint(const double inPoint[3], 
						    double outPoint[3])
{
  if (this->DisplacementGrid == NULL)
    {
    outPoint[0] = inPoint[0]; 
    outPoint[1] = inPoint[1]; 
    outPoint[2] = inPoint[2]; 
    return;
    }

  vtkImageData *grid = this->DisplacementGrid;
  void *gridPtr = grid->GetScalarPointer();
  int gridType = grid->GetScalarType();

  double *spacing = grid->GetSpacing();
  double *origin = grid->GetOrigin();
  int *extent = grid->GetExtent();
  int *increments = grid->GetIncrements();

  double scale = this->DisplacementScale;
  double shift = this->DisplacementShift;

  double point[3];
  double displacement[3];

  // Convert the inPoint to i,j,k indices into the deformation grid 
  // plus fractions
  point[0] = (inPoint[0] - origin[0])/spacing[0];
  point[1] = (inPoint[1] - origin[1])/spacing[1];
  point[2] = (inPoint[2] - origin[2])/spacing[2];

  this->InterpolationFunction(point,displacement,NULL,
                              gridPtr,gridType,extent,increments);

  outPoint[0] = inPoint[0] + (displacement[0]*scale + shift);
  outPoint[1] = inPoint[1] + (displacement[1]*scale + shift);
  outPoint[2] = inPoint[2] + (displacement[2]*scale + shift);
}


//----------------------------------------------------------------------------
// calculate the derivative of the grid transform: only cubic interpolation
// provides well-behaved derivative so we always use that.
void vtkGridTransformBSpline::ForwardTransformDerivative(const double inPoint[3],
							 double outPoint[3],
							 double derivative[3][3])
{
  if (this->DisplacementGrid == NULL)
    {
    outPoint[0] = inPoint[0]; 
    outPoint[1] = inPoint[1]; 
    outPoint[2] = inPoint[2]; 
    vtkMath::Identity3x3(derivative);
    return;
    }

  vtkImageData *grid = this->DisplacementGrid;
  void *gridPtr = grid->GetScalarPointer();
  int gridType = grid->GetScalarType();

  double *spacing = grid->GetSpacing();
  double *origin = grid->GetOrigin();
  int *extent = grid->GetExtent();
  int *increments = grid->GetIncrements();

  double scale = this->DisplacementScale;
  double shift = this->DisplacementShift;

  double point[3];
  double displacement[3];

  // convert the inPoint to i,j,k indices plus fractions
  point[0] = (inPoint[0] - origin[0])/spacing[0];
  point[1] = (inPoint[1] - origin[1])/spacing[1];
  point[2] = (inPoint[2] - origin[2])/spacing[2];

  this->InterpolationFunction(point,displacement,derivative,
                              gridPtr,gridType,extent,increments);

  for (int i = 0; i < 3; i++)
    {
    derivative[i][0] = derivative[i][0]*scale/spacing[0];
    derivative[i][1] = derivative[i][1]*scale/spacing[1];
    derivative[i][2] = derivative[i][2]*scale/spacing[2];
    derivative[i][i] += 1.0f;
    }

  outPoint[0] = inPoint[0] + (displacement[0]*scale + shift);
  outPoint[1] = inPoint[1] + (displacement[1]*scale + shift);
  outPoint[2] = inPoint[2] + (displacement[2]*scale + shift);
}  

//----------------------------------------------------------------------------
// We use Newton's method to iteratively invert the transformation.  
// This is actally quite robust as long as the Jacobian matrix is never
// singular.
// Note that this is similar to vtkWarpTransform::InverseTransformPoint()
// but has been optimized specifically for grid transforms.
void vtkGridTransformBSpline::InverseTransformDerivative(const double inPoint[3], 
							 double outPoint[3],
							 double derivative[3][3])
{
  if (this->DisplacementGrid == NULL)
    {
    outPoint[0] = inPoint[0]; 
    outPoint[1] = inPoint[1]; 
    outPoint[2] = inPoint[2]; 
    return;
    }

  vtkImageData *grid = this->DisplacementGrid;
  void *gridPtr = grid->GetScalarPointer();
  int gridType = grid->GetScalarType();

  double *spacing = grid->GetSpacing();
  double *origin = grid->GetOrigin();
  int *extent = grid->GetExtent();
  int *increments = grid->GetIncrements();

  double invSpacing[3];
  invSpacing[0] = 1.0f/spacing[0];
  invSpacing[1] = 1.0f/spacing[1];
  invSpacing[2] = 1.0f/spacing[2];

  double shift = this->DisplacementShift;
  double scale = this->DisplacementScale;

  double point[3], inverse[3], lastInverse[3];
  double deltaP[3], deltaI[3];

  double functionValue = 0;
  double functionDerivative = 0;
  double lastFunctionValue = VTK_FLOAT_MAX;

  double errorSquared = 0.0;
  double toleranceSquared = this->InverseTolerance;
  toleranceSquared *= toleranceSquared;

  double f = 1.0f;
  double a;

  // convert the inPoint to i,j,k indices plus fractions
  point[0] = (inPoint[0] - origin[0])*invSpacing[0];
  point[1] = (inPoint[1] - origin[1])*invSpacing[1];
  point[2] = (inPoint[2] - origin[2])*invSpacing[2];

  // first guess at inverse point, just subtract displacement
  // (the inverse point is given in i,j,k indices plus fractions)
  this->InterpolationFunction(point, deltaP, NULL,
                              gridPtr, gridType, extent, increments);

  inverse[0] = point[0] - (deltaP[0]*scale + shift)*invSpacing[0];
  inverse[1] = point[1] - (deltaP[1]*scale + shift)*invSpacing[1];
  inverse[2] = point[2] - (deltaP[2]*scale + shift)*invSpacing[2];
  lastInverse[0] = inverse[0];
  lastInverse[1] = inverse[1];
  lastInverse[2] = inverse[2];

  // do a maximum 500 iterations, usually less than 10 are required
  int n = this->InverseIterations;
  int i, j;

  for (i = 0; i < n; i++)
    {
    this->InterpolationFunction(inverse, deltaP, derivative,
                                gridPtr, gridType, extent, increments);

    // convert displacement 
    deltaP[0] = (inverse[0] - point[0])*spacing[0] + deltaP[0]*scale + shift;
    deltaP[1] = (inverse[1] - point[1])*spacing[1] + deltaP[1]*scale + shift;
    deltaP[2] = (inverse[2] - point[2])*spacing[2] + deltaP[2]*scale + shift;

    // convert derivative
    for (j = 0; j < 3; j++)
      {
      derivative[j][0] = derivative[j][0]*scale*invSpacing[0];
      derivative[j][1] = derivative[j][1]*scale*invSpacing[1];
      derivative[j][2] = derivative[j][2]*scale*invSpacing[2];
      derivative[j][j] += 1.0f;
      }

    // get the current function value
    functionValue = (deltaP[0]*deltaP[0] +
                     deltaP[1]*deltaP[1] +
                     deltaP[2]*deltaP[2]);

    // if the function value is decreasing, do next Newton step
    // (the f < 1.0 is there because I found that convergence
    // is more stable if only a single reduction step is done)
    if (functionValue < lastFunctionValue || f < 1.0)
      {
      // here is the critical step in Newton's method
      vtkMath::LinearSolve3x3(derivative,deltaP,deltaI);

      // get the error value in the output coord space
      errorSquared = (deltaI[0]*deltaI[0] +
                      deltaI[1]*deltaI[1] +
                      deltaI[2]*deltaI[2]);

      // break if less than tolerance in both coordinate systems
      if (errorSquared < toleranceSquared && 
          functionValue < toleranceSquared)
        {
        break;
        }

      // save the last inverse point
      lastInverse[0] = inverse[0];
      lastInverse[1] = inverse[1];
      lastInverse[2] = inverse[2];

      // save error at last inverse point
      lastFunctionValue = functionValue;

      // derivative of functionValue at last inverse point
      functionDerivative = (deltaP[0]*derivative[0][0]*deltaI[0] +
                            deltaP[1]*derivative[1][1]*deltaI[1] +
                            deltaP[2]*derivative[2][2]*deltaI[2])*2;

      // calculate new inverse point
      inverse[0] -= deltaI[0]*invSpacing[0];
      inverse[1] -= deltaI[1]*invSpacing[1];
      inverse[2] -= deltaI[2]*invSpacing[2];

      // reset f to 1.0 
      f = 1.0;

      continue;
      }      

    // the error is increasing, so take a partial step 
    // (see Numerical Recipes 9.7 for rationale, this code
    //  is a simplification of the algorithm provided there)

    // quadratic approximation to find best fractional distance
    a = -functionDerivative/(2*(functionValue - 
                                lastFunctionValue -
                                functionDerivative));

    // clamp to range [0.1,0.5]
    f *= (a < 0.1 ? 0.1 : (a > 0.5 ? 0.5 : a));

    // re-calculate inverse using fractional distance
    inverse[0] = lastInverse[0] - f*deltaI[0]*invSpacing[0];
    inverse[1] = lastInverse[1] - f*deltaI[1]*invSpacing[1];
    inverse[2] = lastInverse[2] - f*deltaI[2]*invSpacing[2];
    }

  vtkDebugMacro("Inverse Iterations: " << (i+1));

  if (i >= n)
    {
    // didn't converge: back up to last good result
    inverse[0] = lastInverse[0];
    inverse[1] = lastInverse[1];
    inverse[2] = lastInverse[2];    

//     vtkWarningMacro("InverseTransformPoint: no convergence (" <<
//                     inPoint[0] << ", " << inPoint[1] << ", " << inPoint[2] << 
//                     ") error = " << sqrt(errorSquared) << " after " <<
//                     i << " iterations.");
    }

  // convert point
  outPoint[0] = inverse[0]*spacing[0] + origin[0];
  outPoint[1] = inverse[1]*spacing[1] + origin[1];
  outPoint[2] = inverse[2]*spacing[2] + origin[2];
}

//----------------------------------------------------------------------------
void vtkGridTransformBSpline::InverseTransformPoint(const double point[3], 
						    double output[3])
{
  // the derivative won't be used, but it is required for Newton's method
  double derivative[3][3];
  this->InverseTransformDerivative(point,output,derivative);
}

//----------------------------------------------------------------------------
void vtkGridTransformBSpline::InternalDeepCopy(vtkAbstractTransform *transform)
{
  vtkGridTransformBSpline *gridTransform = (vtkGridTransformBSpline *)transform;

  this->SetInverseTolerance(gridTransform->InverseTolerance);
  this->SetInverseIterations(gridTransform->InverseIterations);
  this->SetInterpolationMode(gridTransform->InterpolationMode);
  this->InterpolationFunction = gridTransform->InterpolationFunction;
  this->SetDisplacementScale(gridTransform->DisplacementScale);
  this->SetDisplacementGrid(gridTransform->DisplacementGrid);
  this->SetDisplacementShift(gridTransform->DisplacementShift);
  this->SetDisplacementScale(gridTransform->DisplacementScale);

  if (this->InverseFlag != gridTransform->InverseFlag)
    {
    this->InverseFlag = gridTransform->InverseFlag;
    this->Modified();
    }
}

//----------------------------------------------------------------------------
void vtkGridTransformBSpline::InternalUpdate()
{
  vtkImageData *grid = this->DisplacementGrid;

  if (grid == 0)
    {
    return;
    }

  grid->UpdateInformation();

  if (grid->GetNumberOfScalarComponents() != 3)
    {
    vtkErrorMacro(<< "TransformPoint: displacement grid must have 3 components");
    return;
    }
  if (grid->GetScalarType() != VTK_CHAR &&
      grid->GetScalarType() != VTK_UNSIGNED_CHAR &&
      grid->GetScalarType() != VTK_SHORT &&
      grid->GetScalarType() != VTK_UNSIGNED_SHORT &&
      grid->GetScalarType() != VTK_FLOAT)
    {
    vtkErrorMacro(<< "TransformPoint: displacement grid is of unsupported numerical type");
    return;
    }
 
  grid->SetUpdateExtent(grid->GetWholeExtent());
  grid->Update();
}

//----------------------------------------------------------------------------
vtkAbstractTransform *vtkGridTransformBSpline::MakeTransform()
{
  return vtkGridTransformBSpline::New();
}
