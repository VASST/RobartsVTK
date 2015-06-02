/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkUltrasoundSphereDetectionRayCaster.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkUltrasoundSphereDetectionRayCaster.h"

#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkMath.h"
#include "vtkExtractVOI.h"
#include "vtkImageSeedConnectivity.h"
//#include "vtkImageThreshold.h"
#include "vtkImageMask.h"
#include <math.h>
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

vtkCxxRevisionMacro(vtkUltrasoundSphereDetectionRayCaster, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkUltrasoundSphereDetectionRayCaster);

//----------------------------------------------------------------------------
// Construct an instance of vtkUltrasoundSphereDetectionRayCaster filter.
vtkUltrasoundSphereDetectionRayCaster::vtkUltrasoundSphereDetectionRayCaster()
{
  this->Seed[0] = 0.0;
  this->Seed[1] = 0.0;
  this->Seed[1] = 0.0;
  this->IntensityDifferenceThreshold = 0.0;
  this->AngleIncrement = 10;
  this->MinimumDistanceFromSeed = 0;
  this->MaximumDistanceFromSeed = 0;
  this->EdgeValue = 255;
  this->Dimensionality = 3; // deal with 3D case only for now
}

//----------------------------------------------------------------------------
void vtkUltrasoundSphereDetectionRayCaster::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Dimensionality: " << this->Dimensionality << "\n";

  os << indent << "Seed: ( "
     << this->Seed[0] << ", "
     << this->Seed[1] << ", "
     << this->Seed[2] << " )\n";

  os << indent << "IntensityDifferenceThreshold: " << this->IntensityDifferenceThreshold << "\n";
}

//----------------------------------------------------------------------------
// rounding functions, split and optimized for each type
// (because we don't want to round if the result is a float!)

static inline int vtkUltraFloor(double x)
{
#if defined mips || defined sparc || defined __ppc__
  x += 2147483648.0;
  unsigned int i = (unsigned int)(x);
  return (int)(i - 2147483648U);
#elif defined i386 || defined _M_IX86
  union { double d; unsigned short s[4]; unsigned int i[2]; } dual;
  dual.d = x + 103079215104.0;  // (2**(52-16))*1.5
  return (int)((dual.i[1]<<16)|((dual.i[0])>>16));
#elif defined ia64 || defined __ia64__ || defined IA64
  x += 103079215104.0;
  long long i = (long long)(x);
  return (int)(i - 103079215104LL);
#else
  double y = floor(x);
  return (int)(y);
#endif
}

// convert a float into an integer plus a fraction
static inline int vtkUltraFloor(double x, double &f)
{
  int ix = vtkUltraFloor(x);
  f = x - ix;

  return ix;
}

//----------------------------------------------------------------------------
// Performs trilinear interpolation-  use the pixel values to interpolate something
// in the middle
// Returns the intensity at the interpolated value
// point is in indices, relative to inPtr, inPtr should be the pointer to point
// if edgeValue != 0, then thresholds every pixel above zero to edgeValue
template <class T>
static T vtkTrilinearInterpolation(vtkUltrasoundSphereDetectionRayCaster *self, double *point, T *inPtr, int inExt[6], vtkIdType inInc[3], T edgeValue)
  {
  int idX0, idY0, idZ0; // floors
  double fx, fy, fz; // fractions
  int idX1, idY1, idZ1; // ceilings
  double rx, ry, rz; // 1-fractions
  double floorX, floorY, floorZ;
  double ceilX, ceilY, ceilZ;
  int factX0, factY0, factZ0, factX1, factY1, factZ1, factY0Z0, factY0Z1, factY1Z0, factY1Z1; // temps
  int factX0Y0Z0, factX0Y0Z1, factX0Y1Z0, factX0Y1Z1, factX1Y0Z0, factX1Y0Z1, factX1Y1Z0, factX1Y1Z1; // increments to the pixel that we are working on
  T intX0Y0Z0, intX0Y0Z1, intX0Y1Z0, intX0Y1Z1, intX1Y0Z0, intX1Y0Z1, intX1Y1Z0, intX1Y1Z1; // intensities for 8 points on cube
  T i1, i2, j1, j2, w1, w2; // temps for trilinear interp
  T interpIntensity = -1;

  // convert point into integer component and a fraction
  // point is unchanged, idX0 is the integer (floor), fx is the float
  floorX = vtkUltraFloor(point[0], fx);
  floorY = vtkUltraFloor(point[1], fy);
  floorZ = vtkUltraFloor(point[2], fz);
  ceilX = floorX + (fx != 0);
  ceilY = floorY + (fy != 0);
  ceilZ = floorZ + (fz != 0);
  idX0 = 0;
  idY0 = 0;
  idZ0 = 0;
  // ceiling
  idX1 = idX0 + (fx != 0);
  idY1 = idY0 + (fy != 0);
  idZ1 = idZ0 + (fz != 0);
  // 1-fractions
  rx = 1-fx;
  ry = 1-fy;
  rz = 1-fz;

  if (floorX >= 0 && floorY >= 0 && floorZ >= 0 &&
    ceilX <= (inExt[1]-inExt[0]) && ceilY <= (inExt[3]-inExt[2]) && (inExt[5]-inExt[4]))
    {

    // increments to corners
    factX0 = idX0 * inInc[0]; // = 0
    factY0 = idY0 * inInc[1]; // = 0
    factZ0 = idZ0 * inInc[2]; // = 0
    factX1 = idX1 * inInc[0]; // = inInc[0]
    factY1 = idY1 * inInc[1]; // = inInc[1]
    factZ1 = idZ1 * inInc[2]; // = inInc[2]

    factY0Z0 = factY0 + factZ0;
    factY0Z1 = factY0 + factZ1;
    factY1Z0 = factY1 + factZ0;
    factY1Z1 = factY1 + factZ1;

    factX0Y0Z0 = factX0 + factY0Z0;
    factX0Y0Z1 = factX0 + factY0Z1;
    factX0Y1Z0 = factX0 + factY1Z0;
    factX0Y1Z1 = factX0 + factY1Z1;
    factX1Y0Z0 = factX1 + factY0Z0;
    factX1Y0Z1 = factX1 + factY0Z1;
    factX1Y1Z0 = factX1 + factY1Z0;
    factX1Y1Z1 = factX1 + factY1Z1;

    // calculate intensities
    if (edgeValue == 0)
      {
      intX0Y0Z0 = *(inPtr + factX0Y0Z0);
      intX0Y0Z1 = *(inPtr + factX0Y0Z1);
      intX0Y1Z0 = *(inPtr + factX0Y1Z0);
      intX0Y1Z1 = *(inPtr + factX0Y1Z1);
      intX1Y0Z0 = *(inPtr + factX1Y0Z0);
      intX1Y0Z1 = *(inPtr + factX1Y0Z1);
      intX1Y1Z0 = *(inPtr + factX1Y1Z0);
      intX1Y1Z1 = *(inPtr + factX1Y1Z1);
      }
    else
      {
      intX0Y0Z0 = (*(inPtr + factX0Y0Z0) > 0) * edgeValue;
      intX0Y0Z1 = (*(inPtr + factX0Y0Z1) > 0) * edgeValue;
      intX0Y1Z0 = (*(inPtr + factX0Y1Z0) > 0) * edgeValue;
      intX0Y1Z1 = (*(inPtr + factX0Y1Z1) > 0) * edgeValue;
      intX1Y0Z0 = (*(inPtr + factX1Y0Z0) > 0) * edgeValue;
      intX1Y0Z1 = (*(inPtr + factX1Y0Z1) > 0) * edgeValue;
      intX1Y1Z0 = (*(inPtr + factX1Y1Z0) > 0) * edgeValue;
      intX1Y1Z1 = (*(inPtr + factX1Y1Z1) > 0) * edgeValue;

      //std::cout << (int)intX0Y0Z0 << " " << (int)intX0Y0Z1 << " " << (int)intX0Y1Z0 << " " << (int)intX0Y1Z1 << " " << (int)intX1Y0Z0 << " " << (int)intX1Y0Z1 << " " << (int)intX1Y1Z0 << " " << (int)intX1Y1Z1 << std::endl;
      }

    // do trilinear interpolation
    i1 = intX0Y0Z0*rz + intX0Y0Z1*fz;
    i2 = intX0Y1Z0*rz + intX0Y1Z1*fz;
    j1 = intX1Y0Z0*rz + intX1Y0Z1*fz;
    j2 = intX1Y1Z0*rz + intX1Y1Z1*fz;
    w1 = i1*ry + i2*fy;
    w2 = j1*ry + j2*fy;
    interpIntensity = w1*rx + w2*fx;
    }

  return interpIntensity;
  }


//----------------------------------------------------------------------------
// Goes along each ray from the seed point out, and keeps the voxels that have the
// first or second highest canny intensity
void vtkUltrasoundSphereDetectionRayCasterExecute(vtkUltrasoundSphereDetectionRayCaster *self,
                                    vtkImageData *inData, unsigned char *inPtr,
                                    vtkImageData *outData, unsigned char *outPtr,
                                    int outExt[6])
{
  int maxX, maxY, maxZ;
  vtkIdType *inIncs;
  vtkIdType *outIncs;
  int *wholeExtent;
  double *spacing;
  double spacingMagnitude;
  double *origin;
  double pointWorld[3];
  int pointInc[3];
  double gradX, gradY, gradZ;
  double awayWorld[3];
  double awayInc[3];
  double towardsWorld[3];
  double towardsInc[3];
  double seed[3];
  unsigned char awayIntensity, towardsIntensity;
  unsigned char awayIntensityDifference, towardsIntensityDifference;
  unsigned char *pointPtr;
  unsigned char *outPointPtr;
  unsigned char *optAwayPtrFirst = 0;
  unsigned char *optTowardsPtrFirst = 0;
  unsigned char maxAwayIntensityFirst;
  unsigned char maxTowardsIntensityFirst;
  unsigned char *optAwayPtrSecond = 0;
  unsigned char *optTowardsPtrSecond = 0;
  unsigned char maxAwayIntensitySecond;
  unsigned char maxTowardsIntensitySecond;
  unsigned char maxAwayDifferenceFirst;
  unsigned char maxTowardsDifferenceFirst;
  unsigned char maxAwayDifferenceSecond;
  unsigned char maxTowardsDifferenceSecond;
  double distAwayFirst, distTowardsFirst, distAwaySecond, distTowardsSecond, currDist;
  double threshold;
  double angleInc, angleAround, angleUpDown, angleAroundRad, angleUpDownRad, tempLength;
  int haveNotReachedTheEdge;
  double maxDistance, minDistance;
  double edgeValue;

  // find the region to loop over
  maxX = outExt[1] - outExt[0];
  maxY = outExt[3] - outExt[2];
  maxZ = outExt[5] - outExt[4];

  // Get some more information we will need
  inIncs = inData->GetIncrements();
  outIncs = outData->GetIncrements();
  wholeExtent = inData->GetExtent();
  spacing = inData->GetSpacing();
  spacingMagnitude = spacing[0]; // assuming isotropic spacing
  origin = inData->GetOrigin();
  self->GetSeed(seed);
  threshold = self->GetIntensityDifferenceThreshold();
  maxDistance = self->GetMaximumDistanceFromSeed();
  minDistance = self->GetMinimumDistanceFromSeed();
  edgeValue = self->GetEdgeValue();

  // Convert angles to radians
  angleInc = self->GetAngleIncrement();

  //for (angleAround = 0; angleAround <= 360.0; angleAround += angleInc) //before
    //{
    //for (angleUpDown = -90; angleUpDown <= 90; angleUpDown += angleInc) //before
      //{
  for (towardsInc[2] = 0; towardsInc[2] <= maxZ; towardsInc[2]++)
    {
    for (angleAround = 0; angleAround <= 360.0; angleAround += angleInc)
      {
      // convert degrees to radians
      angleAroundRad = angleAround * vtkMath::Pi() / 180.0;
      //angleUpDownRad = angleUpDown * vtkMath::Pi() / 180.0; //before

      // calculate vector (magnitude = spacingMagnitude)
      //gradY = sin(angleUpDownRad); //before
      //tempLength = sqrt(1.0 - gradY*gradY); //before
      //gradX = tempLength * cos(angleAroundRad); //before
      //gradZ = tempLength * sin(angleAroundRad); //before
      //gradX = gradX * spacingMagnitude; //before
      //gradY = gradY * spacingMagnitude; //before
      //gradZ = gradZ * spacingMagnitude; //before

      gradX = cos(angleAround) * spacingMagnitude;
      gradY = sin(angleAround) * spacingMagnitude;
      gradZ = 0;

      // initial "towards" point = seed (in world coordinates)
      towardsWorld[0] = seed[0];
      towardsWorld[1] = seed[1];
      //towardsWorld[2] = seed[2];  //before

      // calculate "towards" neighbor point (in "increment" coordinates)
      towardsInc[0] = (towardsWorld[0] - origin[0] - (0.5*spacing[0])) / spacing[0] - wholeExtent[0];
      towardsInc[1] = (towardsWorld[1] - origin[1] - (0.5*spacing[1])) / spacing[1] - wholeExtent[2];
      //towardsInc[2] = (towardsWorld[2] - origin[2] - (0.5*spacing[2])) / spacing[2] - wholeExtent[4];  //before

      towardsWorld[2] = ((towardsInc[2] + wholeExtent[4]) * spacing[2]) + origin[2] + (0.5*spacing[2]);

      // get the current point (in world coordinates)
      pointWorld[0] = towardsWorld[0] + gradX;
      pointWorld[1] = towardsWorld[1] + gradY;
      pointWorld[2] = towardsWorld[2] + gradZ;

      // get this point (in "increment" coordinates)
      pointInc[0] = (pointWorld[0] - origin[0]) / spacing[0] - wholeExtent[0];
      pointInc[1] = (pointWorld[1] - origin[1]) / spacing[1] - wholeExtent[2];
      pointInc[2] = (pointWorld[2] - origin[2]) / spacing[2] - wholeExtent[4];

      // iterate along the vector away from the seed, until we hit the edge
      haveNotReachedTheEdge = 1;
      maxAwayIntensityFirst = 0;
      maxTowardsIntensityFirst = 0;
      maxAwayIntensitySecond = 0;
      maxTowardsIntensitySecond = 0;
      distAwayFirst = 0;
      distTowardsFirst = 0;
      distAwaySecond = 0;
      distTowardsSecond = 0;
      currDist = spacingMagnitude;
      maxAwayDifferenceFirst = 0;
      maxTowardsDifferenceFirst = 0;
      maxAwayDifferenceSecond = 0;
      maxTowardsDifferenceSecond = 0;
      while (haveNotReachedTheEdge)
        {
        // pointer to this point
        pointPtr = inPtr + (pointInc[0]*inIncs[0]) + (pointInc[1]*inIncs[1]) + (pointInc[2]*inIncs[2]);

        // calculate "away" neighbor point (in world coordinates)
        awayWorld[0] = pointWorld[0] + gradX;
        awayWorld[1] = pointWorld[1] + gradY;
        awayWorld[2] = pointWorld[2] + gradZ;

        // calculate "away" neighbor point (in "increment" coordinates)
        awayInc[0] = (awayWorld[0] - origin[0] - (0.5*spacing[0])) / spacing[0] - wholeExtent[0];
        awayInc[1] = (awayWorld[1] - origin[1] - (0.5*spacing[1])) / spacing[1] - wholeExtent[2];
        awayInc[2] = (awayWorld[2] - origin[2] - (0.5*spacing[2])) / spacing[2] - wholeExtent[4];

        // break out of loop if we've hit the edge
        if (awayInc[0] > maxX || awayInc[1] > maxY || awayInc[2] > maxZ || awayInc[0] < 0 || awayInc[1] < 0 || awayInc[2] < 0)
          {
          haveNotReachedTheEdge = 0;
          }
        else
          {
          // skip points with intensity 0 - they cannot be optimal
          if (*pointPtr != 0)
            {
            // calculate the pixel intensity for the neighbors
            awayIntensity = vtkTrilinearInterpolation(self, awayInc, pointPtr, wholeExtent, inIncs, unsigned char(0));
            awayIntensityDifference = *pointPtr - awayIntensity;
            towardsIntensity = vtkTrilinearInterpolation(self, towardsInc, pointPtr, wholeExtent, inIncs, unsigned char(0));
            towardsIntensityDifference = *pointPtr - towardsIntensity;

            // find the first and second intensity differences
            if (awayIntensityDifference > maxAwayDifferenceFirst && awayIntensityDifference > threshold)
              {
              outPointPtr = outPtr + (pointInc[0]*outIncs[0]) + (pointInc[1]*outIncs[1]) + (pointInc[2]*outIncs[2]);
              // swap second with first
              maxAwayIntensitySecond = maxAwayIntensityFirst;
              optAwayPtrSecond = optAwayPtrFirst;
              distAwaySecond = distAwayFirst;
              maxAwayDifferenceSecond = maxAwayDifferenceFirst;
              // update first
              maxAwayIntensityFirst = *pointPtr;
              optAwayPtrFirst = outPointPtr;
              distAwayFirst = currDist;
              maxAwayDifferenceFirst = awayIntensityDifference;
              }
            else if (awayIntensityDifference > maxAwayDifferenceSecond && awayIntensityDifference > threshold)
              {
              outPointPtr = outPtr + (pointInc[0]*outIncs[0]) + (pointInc[1]*outIncs[1]) + (pointInc[2]*outIncs[2]);
              // update second
              maxAwayIntensitySecond = *pointPtr;
              optAwayPtrSecond = outPointPtr;
              distAwaySecond = currDist;
              maxAwayDifferenceSecond = awayIntensityDifference;
              }

            if (towardsIntensityDifference > maxTowardsDifferenceFirst && towardsIntensityDifference > threshold)
              {
              outPointPtr = outPtr + (pointInc[0]*outIncs[0]) + (pointInc[1]*outIncs[1]) + (pointInc[2]*outIncs[2]);
              // swap second with first
              maxTowardsIntensitySecond = maxTowardsIntensityFirst;
              optTowardsPtrSecond = optTowardsPtrFirst;
              distTowardsSecond = distTowardsFirst;
              maxTowardsDifferenceSecond = maxTowardsDifferenceFirst;
              // update first
              maxTowardsIntensityFirst = *pointPtr;
              optTowardsPtrFirst = outPointPtr;
              distTowardsFirst = currDist;
              maxTowardsDifferenceFirst = towardsIntensityDifference;
              }
            else if (towardsIntensityDifference > maxTowardsDifferenceSecond && towardsIntensityDifference > threshold)
              {
              outPointPtr = outPtr + (pointInc[0]*outIncs[0]) + (pointInc[1]*outIncs[1]) + (pointInc[2]*outIncs[2]);
              // update second
              maxTowardsIntensitySecond = *pointPtr;
              optTowardsPtrSecond = outPointPtr;
              distTowardsSecond = currDist;
              maxTowardsDifferenceSecond = towardsIntensityDifference;
              }

            }

          // go to the next point
          towardsWorld[0] = pointWorld[0];
          towardsWorld[1] = pointWorld[1];
          towardsWorld[2] = pointWorld[2];
          towardsInc[0] = towardsInc[0];
          towardsInc[1] = towardsInc[1];
          towardsInc[2] = towardsInc[2];
          pointWorld[0] = awayWorld[0];
          pointWorld[1] = awayWorld[1];
          pointWorld[2] = awayWorld[2];
          pointInc[0] = awayInc[0];
          pointInc[1] = awayInc[1];
          pointInc[2] = awayInc[2];
          }

        currDist += spacingMagnitude;
        }

      if (maxAwayIntensityFirst != 0 && distAwayFirst >= minDistance && distAwayFirst <= maxDistance)
        {
        *optAwayPtrFirst = edgeValue;
        }
      if (maxTowardsIntensityFirst != 0 && distTowardsFirst >= minDistance && distTowardsFirst <= maxDistance)
        {
        *optTowardsPtrFirst = edgeValue;
        }
      if (maxAwayIntensitySecond != 0 && distAwaySecond >= minDistance && distAwaySecond <= maxDistance)
        {
        *optAwayPtrSecond = edgeValue;
        }
      if (maxTowardsIntensitySecond != 0 && distTowardsSecond >= minDistance && distTowardsSecond <= maxDistance)
        {
        *optTowardsPtrSecond = edgeValue;
        }

      }
    }
}

//----------------------------------------------------------------------------
// Goes through each Z slice and picks the points that have the
// highest number of neighbors in the plane
void vtkUltrasoundSphereDetectionPickEdgePointsAccordingToSeedConnectivity(vtkUltrasoundSphereDetectionRayCaster *self,
                                    vtkImageData *inData, unsigned char *inPtr,
                                    vtkImageData *outData, unsigned char *outPtr,
                                    int outExt[6])
{
  int maxX, maxY, maxZ;
  double seed[3];
  double seedX, seedY, seedZ;
  vtkIdType *outIncs;
  double origin[3];
  double spacing[3];
  int wholeExtent[6];
  int pointX, pointY, pointZ;
  unsigned char *currPtrIn;
  unsigned char *currPtrOut;
  double currX, currY, currZ;
  int currIndX, currIndY, currIndZ;
  double vectorX, vectorY, vectorZ; // from current point to seed, in "indices" coordinates
  double vectorMagnitude;
  unsigned char *firstPtr = 0;
  unsigned char firstValue = 0;
  unsigned char *secondPtr = 0;
  unsigned char secondValue = 0;
  unsigned char edgeValue;
  vtkExtractVOI *extractVOI;
  vtkImageSeedConnectivity *seedConnectivity;
  vtkImageData *seedData = vtkImageData::New();
  vtkImageData *tempData = vtkImageData::New();
  unsigned char *seedPtr;
  unsigned char *currSeedPtr;
  int haveNotReachedTheEdge;
  unsigned char firstNumConnections, secondNumConnections;

  // get some stuff we'll need
  self->GetSeed(seed);
  seedX = seed[0];
  seedY = seed[1];
  seedZ = seed[2];
  outData->GetOrigin(origin);
  outData->GetSpacing(spacing);
  outData->GetExtent(wholeExtent);
  outIncs = outData->GetIncrements();
  maxX = outExt[1] - outExt[0];
  maxY = outExt[3] - outExt[2];
  maxZ = outExt[5] - outExt[4];
  edgeValue = (unsigned char)self->GetEdgeValue();

  // convert the seed point into indices (in "increment" coordinates)
  seedX = (seedX - origin[0] - (0.5*spacing[0])) / spacing[0] - wholeExtent[0];
  seedY = (seedY - origin[1] - (0.5*spacing[1])) / spacing[1] - wholeExtent[2];
  seedZ = (seedZ - origin[2] - (0.5*spacing[2])) / spacing[2] - wholeExtent[4];

  // setup the extract voi filter
  extractVOI = vtkExtractVOI::New();

#if (VTK_MAJOR_VERSION <= 5)
  extractVOI->SetInput(inData);
#else
  extractVOI->SetInputData(inData);
#endif

  extractVOI->SetSampleRate(1,1,1);

  // setup the seed connectivity filter
  seedConnectivity = vtkImageSeedConnectivity::New();
  seedConnectivity->SetInputConnection(extractVOI->GetOutputPort());
  seedConnectivity->SetInputConnectValue(edgeValue);
  seedConnectivity->SetOutputConnectedValue(edgeValue);
  seedConnectivity->SetOutputUnconnectedValue(0);
  seedConnectivity->SetDimensionality(3);

  // holds the results of the seed connectivity filter, because it produces an output
  tempData = vtkImageData::New();
  tempData->SetExtent(outExt[0],outExt[1],outExt[2],outExt[3],0,0);
  tempData->SetWholeExtent(tempData->GetExtent());
  tempData->SetUpdateExtent(tempData->GetExtent());
  tempData->SetOrigin(outData->GetOrigin());
  tempData->SetSpacing(outData->GetSpacing());
  tempData->SetScalarTypeToUnsignedChar();
  tempData->AllocateScalars();
  tempData->DeepCopy(outData);
  tempData = seedConnectivity->GetOutput();

  // the seed connectivity image stores how many seeds each pixel is connected to
  seedData = vtkImageData::New();
  seedData->SetExtent(outData->GetExtent());
  seedData->SetWholeExtent(outData->GetWholeExtent());
  seedData->SetUpdateExtent(outData->GetUpdateExtent());
  seedData->SetOrigin(outData->GetOrigin());
  seedData->SetSpacing(outData->GetSpacing());
  seedData->SetScalarTypeToUnsignedChar();
  seedData->AllocateScalars();
  seedData->DeepCopy(outData);
  seedPtr = (unsigned char *)seedData->GetScalarPointerForExtent(outExt);

  //-----------
  // create the seed connectivity image
  for (pointX = 0; pointX <= maxX; pointX++)
    {
    for (pointY = 0; pointY <= maxY; pointY++)
      {
      for (pointZ = 0; pointZ <= maxZ; pointZ++)
        {
        currPtrIn = inPtr + (pointX*outIncs[0]) + (pointY*outIncs[1]) + (pointZ*outIncs[2]);
        currSeedPtr = seedPtr + (pointX*outIncs[0]) + (pointY*outIncs[1]) + (pointZ*outIncs[2]);
        if (*currPtrIn != 0)
          {
          tempData->SetExtent(outExt[0],outExt[1],outExt[2],outExt[3],pointZ+outExt[4],pointZ+outExt[4]);
          tempData->SetWholeExtent(tempData->GetExtent());
          tempData->SetUpdateExtent(tempData->GetUpdateExtent());

          extractVOI->SetVOI(outExt[0],outExt[1],outExt[2],outExt[3],pointZ+outExt[4],pointZ+outExt[4]);
          seedConnectivity->RemoveAllSeeds();
          seedConnectivity->AddSeed(pointX+outExt[0], pointY+outExt[2], pointZ+outExt[4]);
          tempData->Update();
          *currSeedPtr = (unsigned char) seedConnectivity->GetNumberOfConnectedPixels();
          }
        }
      }
    std::cout << (((double)pointX) / ((double)(maxX+1)) * 100.0) << "% ";
    }
  std::cout << "100.000%" << std::endl;

  //-----------
  // for each pixel in the image, find the pixels between it and the seed (in the 2D Z plane)
  // and pick the pixels with the two highest number of neighbors
  for (pointX = 0; pointX <= maxX; pointX++)
    {
    for (pointY = 0; pointY <= maxY; pointY++)
      {
      for (pointZ = 0; pointZ <= maxZ; pointZ++)
        {
        // update the seed for this slice
        seedZ = pointZ;

        // calculate the vector (in "increment" coordinates) - magnitude = 1
        vectorX = seedX - pointX;
        vectorY = seedY - pointY;
        vectorZ = seedZ - pointZ;
        vectorMagnitude = sqrt(vectorX*vectorX + vectorY*vectorY + vectorZ*vectorZ);
        vectorX = vectorX/vectorMagnitude;
        vectorY = vectorY/vectorMagnitude;
        vectorZ = vectorZ/vectorMagnitude;

        //----------
        // find the top two number of connections along the line

        // double values for the points along the line
        currX = (double)pointX;
        currY = (double)pointY;
        currZ = (double)pointZ;
        // indices for the points along the line - rounding makes no difference, b/c
        // currX, currY, currZ already integers
        currIndX = vtkUltraFloor(currX+0.5); // ok because currX already in "increment" coords
        currIndY = vtkUltraFloor(currY+0.5);
        currIndZ = vtkUltraFloor(currZ+0.5);

        // go along the line from this point to the seed, and keep the two pixels that have the most
        // non-zero pixels connected to it
        firstNumConnections = 0;
        secondNumConnections = 0;
        if (seedX == pointX && seedY == pointY && seedZ == pointZ)
          {
          haveNotReachedTheEdge = 0;
          }
        else
          {
          haveNotReachedTheEdge = 1;
          }
        while (haveNotReachedTheEdge)
          {
          currSeedPtr = seedPtr + (pointX*outIncs[0]) + (pointY*outIncs[1]) + (pointZ*outIncs[2]);

          if (*currSeedPtr > firstNumConnections)
            {
            secondNumConnections = firstNumConnections;
            firstNumConnections = *currSeedPtr;
            }
          else if (*currSeedPtr > secondNumConnections)
            {
            secondNumConnections = *currSeedPtr;
            }

          // calculate the next point
          currX += vectorX;
          currY += vectorY;
          currZ += vectorZ;
          currIndX = vtkUltraFloor(currX+0.5); // ok because currX already in "increment" coords
          currIndY = vtkUltraFloor(currY+0.5);
          currIndZ = vtkUltraFloor(currZ+0.5);

          // check to see if we've reached the end
          // know that vector[0] = 0 - not moving along the z direction
          if ( (vectorX < 0 && currIndX < seedX) || (vectorX > 0 && currIndX > seedX)
            || (vectorY < 0 && currIndY < seedY) || (vectorY > 0 && currIndY > seedY)
            || (vectorZ < 0 && currIndZ < seedZ) || (vectorZ > 0 && currIndZ > seedZ))
            {
            haveNotReachedTheEdge = 0;
            }
          }

        //----------
        // copy pixels with the top two values along the line

        // double values for the points along the line
        currX = (double)pointX;
        currY = (double)pointY;
        currZ = (double)pointZ;
        // indices for the points along the line - rounding makes no difference, b/c
        // currX, currY, currZ already integers
        currIndX = vtkUltraFloor(currX+0.5); // ok because currX already in "increment" coords
        currIndY = vtkUltraFloor(currY+0.5);
        currIndZ = vtkUltraFloor(currZ+0.5);

        // go along the line from this point to the seed, and keep the two pixels that have the most
        // non-zero pixels connected to it
        if (seedX == pointX && seedY == pointY && seedZ == pointZ)
          {
          haveNotReachedTheEdge = 0;
          }
        else
          {
          haveNotReachedTheEdge = 1;
          }
        while (haveNotReachedTheEdge)
          {
          currSeedPtr = seedPtr + (pointX*outIncs[0]) + (pointY*outIncs[1]) + (pointZ*outIncs[2]);
          currPtrOut = outPtr + (pointX*outIncs[0]) + (pointY*outIncs[1]) + (pointZ*outIncs[2]);

          if (*currSeedPtr == firstNumConnections)
            {
            *currPtrOut = *currSeedPtr;
            //*currPtrOut = edgeValue;
            }

          // calculate the next point
          currX += vectorX;
          currY += vectorY;
          currZ += vectorZ;
          currIndX = vtkUltraFloor(currX+0.5); // ok because currX already in "increment" coords
          currIndY = vtkUltraFloor(currY+0.5);
          currIndZ = vtkUltraFloor(currZ+0.5);

          // check to see if we've reached the end
          // know that vector[0] = 0 - not moving along the x direction
          if ( (vectorX < 0 && currIndX < seedX) || (vectorX > 0 && currIndX > seedX)
            || (vectorY < 0 && currIndY < seedY) || (vectorY > 0 && currIndY > seedY)
            || (vectorZ < 0 && currIndZ < seedZ) || (vectorZ > 0 && currIndZ > seedZ))
            {
            haveNotReachedTheEdge = 0;
            }
          }

        }
      }
    }
}

//----------------------------------------------------------------------------
// Picks the closest and farthest points
void vtkUltrasoundSphereDetectionFinalEdgeSelection(vtkUltrasoundSphereDetectionRayCaster *self,
                                    vtkImageData *inData, unsigned char *inPtr,
                                    vtkImageData *outData, unsigned char *outPtr,
                                    int outExt[6])
{
  int pointIndX, pointIndY, pointIndZ;
  double pointX, pointY, pointZ;
  int maxX, maxY, maxZ;
  vtkIdType *outIncs;
  double seed[3];
  double seedX, seedY, seedZ;
  int seedIndX, seedIndY, seedIndZ;
  double origin[3];
  double spacing[3];
  int wholeExtent[6];
  unsigned char *currPtrIn;
  unsigned char *currPtrOut;
  unsigned char *currDistPtr;
  unsigned char *ptPtrIn;
  unsigned char *ptPtrOut;
  vtkImageData *distData;
  vtkImageData *innerData;
  vtkImageData *outerData;
  unsigned char *distPtr;
  unsigned char *innerPtr;
  //double currX, currY, currZ;
  double curr[3];
  //int currIndX, currIndY, currIndZ;
  int currInd[3];
  double distance;
  double vectorX, vectorY, vectorZ, vectorMagnitude;
  double gradX, gradY, gradZ;
  double spacingMagnitude;
  double angleInc, angleAround, angleUpDown, angleAroundRad, angleUpDownRad, tempLength;
  int foundPtAlongVector, haveNotReachedTheEdge;
  unsigned char *lastPtr;
  unsigned char lastVal;
  unsigned char edgeValue;
  unsigned char currValue;
  vtkImageMask *innerMask;
  double innerDistance;
  double thisInnerDistance;
  double numInner;
  double innerStdDev;

  // get some things we'll need
  maxX = outExt[1] - outExt[0];
  maxY = outExt[3] - outExt[2];
  maxZ = outExt[5] - outExt[4];
  outIncs = outData->GetIncrements();
  self->GetSeed(seed);
  seedX = seed[0];
  seedY = seed[1];
  seedZ = seed[2];
  outData->GetOrigin(origin);
  outData->GetSpacing(spacing);
  spacingMagnitude = spacing[0]; // assuming isotropic spacing
  outData->GetExtent(wholeExtent);
  edgeValue = (unsigned char)self->GetEdgeValue();

  // the distance image stores the distance from the seed to the voxel
  distData = vtkImageData::New();
  distData->SetExtent(outData->GetExtent());
  distData->SetWholeExtent(outData->GetWholeExtent());
  distData->SetUpdateExtent(outData->GetUpdateExtent());
  distData->SetOrigin(outData->GetOrigin());
  distData->SetSpacing(outData->GetSpacing());
  distData->SetScalarTypeToUnsignedChar();
  distData->AllocateScalars();
  distData->DeepCopy(outData);
  distPtr = (unsigned char *)distData->GetScalarPointerForExtent(outExt);

  // the inner data image stores the pixels that are identified as inside pixels
  innerData = vtkImageData::New();
  innerData->SetExtent(outData->GetExtent());
  innerData->SetWholeExtent(outData->GetWholeExtent());
  innerData->SetUpdateExtent(outData->GetUpdateExtent());
  innerData->SetOrigin(outData->GetOrigin());
  innerData->SetSpacing(outData->GetSpacing());
  innerData->SetScalarTypeToUnsignedChar();
  innerData->AllocateScalars();
  innerData->DeepCopy(outData);
  innerPtr = (unsigned char *)innerData->GetScalarPointerForExtent(outExt);


  //--------
  // create distance map
  for (pointIndX = 0; pointIndX <= maxX; pointIndX++)
    {
    for (pointIndY = 0; pointIndY <= maxY; pointIndY++)
      {
      for (pointIndZ = 0; pointIndZ <= maxZ; pointIndZ++)
        {
        currPtrIn = inPtr + (pointIndX*outIncs[0]) + (pointIndY*outIncs[1]) + (pointIndZ*outIncs[2]);
        currDistPtr = distPtr + (pointIndX*outIncs[0]) + (pointIndY*outIncs[1]) + (pointIndZ*outIncs[2]);
        curr[0] = ((pointIndX + wholeExtent[0]) * spacing[0]) + origin[0] + (0.5*spacing[0]);
        curr[1] = ((pointIndY + wholeExtent[2]) * spacing[1]) + origin[1] + (0.5*spacing[1]);
        curr[2] = ((pointIndZ + wholeExtent[4]) * spacing[2]) + origin[2] + (0.5*spacing[2]);
        distance = sqrt((curr[0]-seedX)*(curr[0]-seedX) + (curr[1]-seedY)*(curr[1]-seedY) + (curr[2]-seedZ)*(curr[2]-seedZ));
        *currDistPtr = (unsigned char) vtkUltraFloor(distance+0.5);
        }
      }
    }

  //-------
  // for each pixel, see if it's the closest or farthest from the seed (in Z plane)

  // convert the seed point into indices (in "increment" coordinates)
  seedIndX = (seed[0] - origin[0] - (0.5*spacing[0])) / spacing[0] - wholeExtent[0];
  seedIndY = (seed[1] - origin[1] - (0.5*spacing[1])) / spacing[1] - wholeExtent[2];
  seedIndZ = (seed[2] - origin[2] - (0.5*spacing[2])) / spacing[2] - wholeExtent[4];

  innerDistance = 0;
  for (pointIndX = 0; pointIndX <= maxX; pointIndX++)
    {
    for (pointIndY = 0; pointIndY <= maxY; pointIndY++)
      {
      for (pointIndZ = 0; pointIndZ <= maxZ; pointIndZ++)
        {

        // find point in world coordinates
        pointX = ((pointIndX + wholeExtent[0]) * spacing[0]) + origin[0] + (0.5*spacing[0]);
        pointY = ((pointIndY + wholeExtent[2]) * spacing[1]) + origin[1] + (0.5*spacing[1]);
        pointZ = ((pointIndZ + wholeExtent[4]) * spacing[2]) + origin[2] + (0.5*spacing[2]);

        // update seed
        seedIndZ = pointIndZ;

        // convert current point to double
        curr[0] = double(pointIndX);
        curr[1] = double(pointIndY);
        curr[2] = double(pointIndZ);

        // calculate vector (towards)
        vectorX = seedIndX - curr[0];
        vectorY = seedIndY - curr[1];
        vectorZ = seedIndZ - curr[2];
        vectorMagnitude = sqrt(vectorX*vectorX + vectorY*vectorY + vectorZ*vectorZ);
        vectorX = vectorX/vectorMagnitude;
        vectorY = vectorY/vectorMagnitude;
        vectorZ = vectorZ/vectorMagnitude;
        // with rounding
        vectorX = vtkUltraFloor(vectorX+0.5);
        vectorY = vtkUltraFloor(vectorY+0.5);
        vectorZ = vtkUltraFloor(vectorZ+0.5);

        // see if this pixel is valid
        ptPtrIn = inPtr + (pointIndX*outIncs[0]) + (pointIndY*outIncs[1]) + (pointIndZ*outIncs[2]);
        ptPtrOut = outPtr + (pointIndX*outIncs[0]) + (pointIndY*outIncs[1]) + (pointIndZ*outIncs[2]);

        if (*ptPtrIn > 0)
          {

          *ptPtrOut = edgeValue / 2.0;

          // calculate the next point
          curr[0] += vectorX;
          curr[1] += vectorY;
          curr[2] += vectorZ;
          currInd[0] = vtkUltraFloor(curr[0]+0.5); // ok because currX already in "increment" coords
          currInd[1] = vtkUltraFloor(curr[1]+0.5);
          currInd[2] = vtkUltraFloor(curr[2]+0.5);

          // search towards
          foundPtAlongVector = 0;
          if (seedIndX == currInd[0] && seedIndY == currInd[1] &&seedIndZ == currInd[2])
            {
            haveNotReachedTheEdge = 0;
            }
          else
            {
            haveNotReachedTheEdge = 1;
            }
          while (haveNotReachedTheEdge && !foundPtAlongVector)
            {
            currPtrIn = inPtr + (currInd[0]*outIncs[0]) + (currInd[1]*outIncs[1]) + (currInd[2]*outIncs[2]);
            //currPtrOut = outPtr + (currInd[0]*outIncs[0]) + (currInd[1]*outIncs[1]) + (currInd[2]*outIncs[2]);
            //currValue = vtkTrilinearInterpolation(self, curr, ptPtrIn, outExt, outIncs, edgeValue);

            //std::cout << (double)currValue << " ";

            //if (currValue > 150)
            //  {
            //  foundPtAlongVector = 1;
            //  }
            if (*currPtrIn != 0)
              {
              foundPtAlongVector = 1;
              }

            // calculate the next point
            curr[0] += vectorX;
            curr[1] += vectorY;
            curr[2] += vectorZ;
            currInd[0] = vtkUltraFloor(curr[0]+0.5); // ok because currX already in "increment" coords
            currInd[1] = vtkUltraFloor(curr[1]+0.5);
            currInd[2] = vtkUltraFloor(curr[2]+0.5);

            // check to see if we've reached the end
            // know that vector[0] = 0 - not moving along the z direction
            if ( (vectorX < 0 && curr[0] < seedIndX) || (vectorX > 0 && curr[0] > seedIndX)
              || (vectorY < 0 && curr[1] < seedIndY) || (vectorY > 0 && curr[1] > seedIndY)
              || (vectorZ < 0 && curr[2] < seedIndZ) || (vectorZ > 0 && curr[2] > seedIndZ))
              {
              haveNotReachedTheEdge = 0;
              }
            }

          if (!foundPtAlongVector)
            {
            //*ptPtrOut = *ptPtrIn;
            *ptPtrOut = edgeValue;
            thisInnerDistance = sqrt((pointX-seedX)*(pointX-seedX) + (pointY-seedY)*(pointY-seedY) + (pointZ-seedZ)*(pointZ-seedZ));
            innerDistance += thisInnerDistance;
            numInner++;
            }
          }
        }
      }
    }

  innerDistance = innerDistance / numInner;
  std::cout << "inner distance = " << innerDistance << std::endl;

  // find the standard deviation
  innerStdDev = 0;
  for (pointIndX = 0; pointIndX <= maxX; pointIndX++)
    {
    for (pointIndY = 0; pointIndY <= maxY; pointIndY++)
      {
      for (pointIndZ = 0; pointIndZ <= maxZ; pointIndZ++)
        {
        ptPtrOut = outPtr + (pointIndX*outIncs[0]) + (pointIndY*outIncs[1]) + (pointIndZ*outIncs[2]);

        // find point in world coordinates
        pointX = ((pointIndX + wholeExtent[0]) * spacing[0]) + origin[0] + (0.5*spacing[0]);
        pointY = ((pointIndY + wholeExtent[2]) * spacing[1]) + origin[1] + (0.5*spacing[1]);
        pointZ = ((pointIndZ + wholeExtent[4]) * spacing[2]) + origin[2] + (0.5*spacing[2]);

        if (*ptPtrOut != 0)
          {
          thisInnerDistance = sqrt((pointX-seedX)*(pointX-seedX) + (pointY-seedY)*(pointY-seedY) + (pointZ-seedZ)*(pointZ-seedZ));
          innerStdDev += (thisInnerDistance-innerDistance)*(thisInnerDistance-innerDistance);
          }
        }
      }
    }
  innerStdDev = sqrt(innerStdDev / numInner);
  std::cout << "inner std dev = " << innerStdDev << std::endl;

/*  //--------
  // for each Z slice,
  // go through each ray, and keep the points that have only two along the vector

  // get angle increment
  angleInc = self->GetAngleIncrement();

  for (pointZ = 0; pointZ <= maxZ; pointZ++)
  //for (angleAround = 0; angleAround <= 360; angleAround += angleInc)
    {

    for (angleAround = 0; angleAround <= 360; angleAround += angleInc)
    //for (angleUpDown = -90; angleUpDown <= 90; angleUpDown += angleInc)
      {

      // convert degrees to radians
      angleAroundRad = angleAround * vtkMath::Pi() / 180.0;
      angleUpDownRad = angleUpDown * vtkMath::Pi() / 180.0;

      // calculate vector (magnitude = spacingMagnitude)
      gradX = cos(angleAround) * spacingMagnitude;
      gradY = sin(angleAround) * spacingMagnitude;
      //gradY = sin(angleUpDownRad);
      //tempLength = sqrt(1.0 - gradY*gradY);
      //gradX = tempLength * cos(angleAroundRad);
      //gradZ = tempLength * sin(angleAroundRad);
      //gradX = gradX * spacingMagnitude;
      //gradY = gradY * spacingMagnitude;
      //gradZ = gradZ * spacingMagnitude;

      // initial point = seed (in world coords)
      currX = seed[0];
      currY = seed[1];
      // no z here because we have the z slice in "increment" coords already
      //currZ = seed[2];

      // initial point (in "increment" coordinates)
      currIncX = (currX - origin[0] - (0.5*spacing[0])) / spacing[0] - wholeExtent[0];
      currIncY = (currY - origin[1] - (0.5*spacing[1])) / spacing[1] - wholeExtent[2];
      currIncZ = pointZ;
      //currIncZ = (currZ - origin[2] - (0.5*spacing[2])) / spacing[2] - wholeExtent[4];

      // iterate along the vector away from the seed, until we hit the edge
      haveNotReachedTheEdge = 1;
      haveFoundFirst = 0;
      lastVal = 0; // initialize in case there are no points along the vector

      while (haveNotReachedTheEdge)
        {
        // pointer to this point
        currPtrIn = inPtr + (currIncX*outIncs[0]) + (currIncY*outIncs[1]) + (currIncZ*outIncs[2]);
        currPtrOut = outPtr + (currIncX*outIncs[0]) + (currIncY*outIncs[1]) + (currIncZ*outIncs[2]);
        currDistPtr = distPtr + (currIncX*outIncs[0]) + (currIncY*outIncs[1]) + (currIncZ*outIncs[2]);

        // break out of loop if we've hit the edge
        if (currIncX > maxX || currIncY > maxY || currIncZ > maxZ || currIncX < 0 || currIncY < 0 || currIncZ < 0)
          {
          haveNotReachedTheEdge = 0;
          }

        else
          {
          // skip points with intensity 0
          if (*currPtrIn != 0)
            {
            // leave the first one alone - copy to output
            if (!haveFoundFirst)
              {
              haveFoundFirst = 1;
              //*currPtrOut = *currPtrIn;
              *currPtrOut = edgeValue;
              }
            // if we've already found the first, then delete this pixel in the output
            else
              {
              lastPtr = currPtrOut;
              lastVal = *currPtrIn;

              *currPtrOut = edgeValue / 2.0;
              }
            }

          // go to next point
          currX += gradX;
          currY += gradY;
          // no z because we already have it
          //currZ += gradZ;
          currIncX = (currX - origin[0] - (0.5*spacing[0])) / spacing[0] - wholeExtent[0];
          currIncY = (currY - origin[1] - (0.5*spacing[1])) / spacing[1] - wholeExtent[2];
          // no z because we already have it
          //currIncZ = (currZ - origin[2] - (0.5*spacing[2])) / spacing[2] - wholeExtent[4];
          }
        }

        // copy the last point to the output
        if (lastVal != 0)
          {
          //*lastPtr = lastVal;
          *lastPtr = edgeValue;
          }
      }
    }
    */
}

//----------------------------------------------------------------------------
void vtkCalculateIdealGradient(vtkUltrasoundSphereDetectionRayCaster *self,
                               vtkImageData *inData, unsigned char *inPtr,
                               vtkImageData *outData, unsigned char *outPtr,
                               int outExt[6])
{

int maxX, maxY, maxZ;
int pointIndX, pointIndY, pointIndZ;
double seed[3];
int seedIndX, seedIndY;
double distIndX, distIndY, distIndZ;
double normDistIndX, normDistIndY, normDistIndZ;
int towardsXIndX, towardsXIndY, towardsXIndZ; // away and towards notation assumes pt on right, but
int awayXIndX, awayXIndY, awayXIndZ;           // works regardless
double distMagnitude;
double origin[3];
double spacing[3];
int wholeExtent[6];
vtkIdType *outIncs;
unsigned char *currInPtr;
unsigned char *currOutPtr;
unsigned char *currInPtrTowards;
unsigned char *currInPtrAway;
unsigned char *currOutPtrTowards;
unsigned char *currOutPtrAway;

// some things we'll need
maxX = outExt[1] - outExt[0];
maxY = outExt[3] - outExt[2];
maxZ = outExt[5] - outExt[4];
self->GetSeed(seed);
outData->GetOrigin(origin);
outData->GetSpacing(spacing);
outData->GetExtent(wholeExtent);
outIncs = outData->GetIncrements();

// seed point in increment coordinates
seedIndX = (seed[0] - origin[0] - (0.5*spacing[0])) / spacing[0] - wholeExtent[0];
seedIndY = (seed[1] - origin[1] - (0.5*spacing[1])) / spacing[1] - wholeExtent[2];

// go through all pixels
for (pointIndX = 0; pointIndX <= maxX; pointIndX++)
  {
  for (pointIndY = 0; pointIndY <= maxY; pointIndY++)
    {
    for (pointIndZ = 0; pointIndZ <= maxZ; pointIndZ++)
      {
      currInPtr = inPtr + (pointIndX*outIncs[0]) + (pointIndY*outIncs[1]) + (pointIndZ*outIncs[2]);
      currOutPtr = outPtr + (pointIndX*outIncs[0]) + (pointIndY*outIncs[1]) + (pointIndZ*outIncs[2]);

      // for visualization
      if (*currOutPtr == 0)
        {
        *currOutPtr = unsigned char(128);
        }


      // look at in points that are non zero
      //if (*currInPtr != 0)
      //  {
        // calculate distance from seed to point in x and y directions only (point - seed)
        // (in increment coords, magnitude 1)
        distIndX = abs(pointIndX - seedIndX);
        distIndY = 0;
        distIndZ = 0;
        distMagnitude = sqrt((distIndX*distIndX) + (distIndY*distIndY) + (distIndZ*distIndZ));
        normDistIndX = distIndX / distMagnitude;
        normDistIndY = distIndY / distMagnitude;
        normDistIndZ = distIndZ / distMagnitude;
        distIndX = vtkUltraFloor(normDistIndX + 0.5);
        distIndY = vtkUltraFloor(normDistIndY + 0.5);
        distIndZ = vtkUltraFloor(normDistIndZ + 0.5);

        // calculate towards point and away point (in increment coords)
        towardsXIndX = pointIndX - distIndX;
        towardsXIndY = pointIndY - distIndY;
        towardsXIndZ = pointIndZ - distIndZ;
        awayXIndX = pointIndX + distIndX;
        awayXIndY = pointIndY + distIndY;
        awayXIndZ = pointIndZ + distIndZ;

        // update expected gradient for towards point
        if (towardsXIndX <= maxX && towardsXIndY <= maxY && towardsXIndZ <= maxZ )
          {
          currOutPtrTowards = outPtr + (towardsXIndX*outIncs[0]) + (towardsXIndY*outIncs[1]) + (towardsXIndZ*outIncs[2]);
          currInPtrTowards = inPtr + (towardsXIndX*outIncs[0]) + (towardsXIndY*outIncs[1]) + (towardsXIndZ*outIncs[2]);
          if ((*currInPtr != 0 && *currInPtrTowards == 0) || (*currInPtr == 0 && *currInPtrTowards != 0))
            {
            *currOutPtrTowards = (unsigned char)200;
            }
          }

        // update expected gradient for away point
        if (awayXIndX <= maxX && awayXIndY <= maxY && awayXIndZ <= maxZ)
          {
          currOutPtrAway = outPtr + (awayXIndX*outIncs[0]) + (awayXIndY*outIncs[1]) + (awayXIndZ*outIncs[2]);
          currInPtrAway = inPtr + (awayXIndX*outIncs[0]) + (awayXIndY*outIncs[1]) + (awayXIndZ*outIncs[2]);
          if ((*currInPtr != 0 && *currInPtrAway == 0) || (*currInPtr == 0 && *currInPtrAway != 0))
            {
            *currOutPtrAway = (unsigned char)1;
            }
          }
       //}


      }
    }
  }

}







//----------------------------------------------------------------------------
// This method uses the input data to fill the output data.
// It can handle any type data, but the two datas must have the same
// data type.  Assumes that in and out have the same lower extent.
// It just executes a switch statement to call the correct function for
// the regions data types.
int vtkUltrasoundSphereDetectionRayCaster::RequestData(
    vtkInformation *vtkNotUsed(request),
    vtkInformationVector **inputVector,
    vtkInformationVector *outputVector)
{
  // get the data object
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkImageData *inData = vtkImageData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkInformation *outInfo = outputVector->GetInformationObject(0);
  vtkImageData *outData = vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  // we need to allocate our own scalars
  int wholeExtent[6];
  int extent[6];
  outInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), wholeExtent);
  outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), extent);
  extent[0] = wholeExtent[0];
  extent[1] = wholeExtent[1];
  extent[2] = wholeExtent[2];
  extent[3] = wholeExtent[3];
  extent[4] = wholeExtent[4];
  extent[5] = wholeExtent[5];
  outData->SetExtent(extent);
  outData->AllocateScalars();

  // this filter expects that input is the same type as output.
  if (inData->GetScalarType() != outData->GetScalarType())
    {
    vtkErrorMacro(<< "Execute: input ScalarType, "
                  << vtkImageScalarTypeNameMacro(inData->GetScalarType())
                  << ", must match out ScalarType "
                  << vtkImageScalarTypeNameMacro(outData->GetScalarType()));
    return 1;
    }

  // this filter expects only one scalar component
  if (inData->GetNumberOfScalarComponents() != 1 || outData->GetNumberOfScalarComponents() != 1)
    {
    vtkErrorMacro(<< "Execute: number of scalar components, "
                  << inData->GetNumberOfScalarComponents()
                  << ", "
                  << outData->GetNumberOfScalarComponents()
                  << ", must be 1");
    return 1;
    }

  // this filter expects isotropic spacing
  double inSpacing[3];
  inData->GetSpacing(inSpacing);
  double outSpacing[3];
  outData->GetSpacing(outSpacing);
  if (inSpacing[0] != inSpacing[1] || inSpacing[1] != inSpacing[2] || inSpacing[2] != outSpacing[0] ||
    outSpacing[0] != outSpacing[1] || outSpacing[1] != outSpacing[2])
    {
    vtkErrorMacro(<< "Execute: spacing must be isotropic for both input, "
                  << inSpacing[0] << " " << inSpacing[1] << " " << inSpacing[2]
                  << ", and output, "
                  << outSpacing[0] << " " << outSpacing[1] << " " << outSpacing[2]);
    return 1;
    }

  // this filter expects unsigned char (because uses seed connectivity)
  if (inData->GetScalarType() != VTK_UNSIGNED_CHAR ||
      outData->GetScalarType() != VTK_UNSIGNED_CHAR)
    {
    vtkErrorMacro("Execute: Both input and output must have scalar type UnsignedChar");
    return 1;
    }

  // intermediate object - store result from edge detection with ray casting as input to
  // refinement algorithm
  vtkImageData *intermediateData = vtkImageData::New();
  intermediateData->SetExtent(outData->GetExtent());
  intermediateData->SetWholeExtent(outData->GetWholeExtent());
  intermediateData->SetUpdateExtent(outData->GetUpdateExtent());
  intermediateData->SetOrigin(outData->GetOrigin());
  intermediateData->SetSpacing(outData->GetSpacing());
  intermediateData->SetScalarTypeToUnsignedChar();
  intermediateData->AllocateScalars();
  intermediateData->DeepCopy(outData);

  // another intermediate image - for picking out the final edge points
  vtkImageData *intermediateData2 = vtkImageData::New();
  intermediateData2->SetExtent(outData->GetExtent());
  intermediateData2->SetWholeExtent(outData->GetWholeExtent());
  intermediateData2->SetUpdateExtent(outData->GetUpdateExtent());
  intermediateData2->SetOrigin(outData->GetOrigin());
  intermediateData2->SetSpacing(outData->GetSpacing());
  intermediateData2->SetScalarTypeToUnsignedChar();
  intermediateData2->AllocateScalars();
  intermediateData2->DeepCopy(outData);

  // another intermediate image - for thresholding
  //vtkImageData *intermediateData3 = vtkImageData::New();
  //intermediateData3->SetExtent(outData->GetExtent());
  //intermediateData3->SetWholeExtent(outData->GetWholeExtent());
  //intermediateData3->SetUpdateExtent(outData->GetUpdateExtent());
  //intermediateData3->SetOrigin(outData->GetOrigin());
  //intermediateData3->SetSpacing(outData->GetSpacing());
  //intermediateData3->SetScalarTypeToUnsignedChar();
  //intermediateData3->AllocateScalars();
  //intermediateData3->DeepCopy(outData);


  //vtkImageThreshold *thresholder = vtkImageThreshold::New();
  //thresholder->SetInput(intermediateData2);
  //thresholder->ThresholdByUpper(1);
  //thresholder->SetInValue(this->EdgeValue);
  //thresholder->SetOutValue(0);
  //thresholder->ReplaceInOn();
  //thresholder->ReplaceOutOn();
  //intermediateData3 = thresholder->GetOutput();
  //outData = thresholder->GetOutput();

  int outExt[6];
  outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), outExt);

  unsigned char *inPtr;
  unsigned char *outPtr;
  unsigned char *intermediatePtr;
  unsigned char *intermediatePtr2;
  //unsigned char *intermediatePtr3;
  inPtr = (unsigned char *)inData->GetScalarPointerForExtent(outExt);
  outPtr = (unsigned char *)outData->GetScalarPointerForExtent(outExt);
  intermediatePtr = (unsigned char *)intermediateData->GetScalarPointerForExtent(outExt);
  intermediatePtr2 = (unsigned char *)intermediateData2->GetScalarPointerForExtent(outExt);
  //intermediatePtr3 = (unsigned char *)intermediateData3->GetScalarPointerForExtent(outExt);

  /*vtkUltrasoundSphereDetectionRayCasterExecute(this, inData,
                                                    inPtr, intermediateData,
                                                    intermediatePtr, outExt);

  vtkUltrasoundSphereDetectionPickEdgePointsAccordingToSeedConnectivity(this, intermediateData,
                                                    intermediatePtr, intermediateData2,
                                                    intermediatePtr2, outExt);

  //intermediateData3 = thresholder->GetOutput();
  //intermediateData3->Update();

  vtkUltrasoundSphereDetectionFinalEdgeSelection(this, intermediateData2,
                                                    intermediatePtr2, outData,
                                                    outPtr, outExt);*/

  //vtkUltrasoundSphereDetectionRayCasterExecute(this, inData,
   //                                                 inPtr, outData,
    //                                                outPtr, outExt);

  vtkCalculateIdealGradient(this, inData, inPtr, outData, outPtr, outExt);


  // garbage collection
  intermediateData->Delete();
  intermediateData2->Delete();
  //intermediateData3->Delete();

  return 1;
}
