/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkShapeBasedInterpolation.cxx,v $
  Language:  C++
  Date:      $Date: 2007/05/04 14:34:35 $
  Version:   $Revision: 1.1 $
  Thanks:    Thanks to C. Charles Law who developed this class.

Copyright (c) 1993-2001 Ken Martin, Will Schroeder, Bill Lorensen 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Ken Martin, Will Schroeder, or Bill Lorensen nor the names
   of any contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

 * Modified source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/
#include <stdlib.h>
#include "vtkMath.h"

#include "vtkShapeBasedInterpolation.h"
#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkImageAccumulate.h"
#include "vtkImageReslice.h"
#include "vtkImageAppend.h"
#include "vtkImageThreshold.h"
#include "vtkImageMathematics.h"

#include <vtkVersion.h> // for VTK_MAJOR_VERSION

//----------------------------------------------------------------------------
inline double vtkMinimum(double vals[14])
{
  double min = vals[0];
  for (int i = 0; i <= 13; i ++) {if (vals[i] < min) min = vals[i];}
  return min;
}

//----------------------------------------------------------------------------
inline double vtkMaximum(double vals[14])
{
  double max = vals[0];
  for (int i = 0; i <= 13; i ++) {if (vals[i] > max) max = vals[i];}
  return max;
}

//------------------------------------------------------------------------------
vtkShapeBasedInterpolation* vtkShapeBasedInterpolation::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkShapeBasedInterpolation");
  if(ret)
    {
    return (vtkShapeBasedInterpolation*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkShapeBasedInterpolation;
}

//----------------------------------------------------------------------------
vtkShapeBasedInterpolation::vtkShapeBasedInterpolation()
{
  this->SliceAxis = 2;
  this->SliceAxis = 256;
  this->OutputSpacing[0] = 1.0;
  this->OutputSpacing[1] = 1.0;
  this->OutputSpacing[2] = 1.0;
}

//----------------------------------------------------------------------------
vtkShapeBasedInterpolation::~vtkShapeBasedInterpolation()
{
}

//----------------------------------------------------------------------------
void vtkShapeBasedInterpolation::SetInput(vtkImageData *input)
{
  vtkImageCast *cast;
  cast = vtkImageCast::New();
  cast->SetInput(input);
  cast->SetOutputScalarTypeToFloat();
  cast->Update();

  vtkImageAccumulate *accumulate;
  accumulate = vtkImageAccumulate::New();
#if (VTK_MAJOR_VERSION <= 5)
  accumulate->SetInput(cast->GetOutput());
#else
  accumulate->SetInputConnection(cast->GetOutputPort());
#endif
  accumulate->Update();

  cast->GetOutput()->GetExtent(this->inExt);
  cast->GetOutput()->GetSpacing(this->inSpa);
  cast->GetOutput()->GetOrigin(this->inOri);
  this->inMinVal = accumulate->GetMin()[0];
  this->inMaxVal = accumulate->GetMax()[0];
  this->inData = cast->GetOutput();
}

//----------------------------------------------------------------------------
vtkImageData *vtkShapeBasedInterpolation::GetOutput()
{
  return this->outData;
}

//----------------------------------------------------------------------------
void CalculateSpacingsExtents(double inSpa[3], int inExt[6], int SliceAxis, double OutputSpacing[3], int nBins, 
            double liftSpa[3], double distSpa[3], double inteSpa[3], double slicSpa[3], 
            int liftExt[6], int distExt[6], int inteExt[6], int slicExt[6])
{
  memcpy(slicSpa, OutputSpacing, sizeof(double)*3);
  if (SliceAxis == 0)
    {
      liftSpa[0] = (inSpa[1] + inSpa[2]) / 2.0; liftSpa[1] = inSpa[1]; liftSpa[2] = inSpa[2];
      memcpy(distSpa, liftSpa, sizeof(double)*3);
      inteSpa[0] = (inSpa[1] + inSpa[2]) / 2.0; inteSpa[1] = OutputSpacing[1]; inteSpa[2] = OutputSpacing[2];

      liftExt[0] = -1; liftExt[1] = nBins;
      liftExt[2] = inExt[2] - 1; liftExt[3] = inExt[3] + 1;
      liftExt[4] = inExt[4] - 1; liftExt[5] = inExt[5] + 1;

      distExt[0] = 0; distExt[1] = nBins - 1;
      distExt[2] = inExt[2]; distExt[3] = inExt[3];
      distExt[4] = inExt[4]; distExt[5] = inExt[5];

      inteExt[0] = 0; inteExt[1] = nBins - 1;
      inteExt[2] = inExt[2]; inteExt[3] = int (inExt[3] * fabs(distSpa[1] / inteSpa[1]));
      inteExt[4] = inExt[4]; inteExt[5] = int (inExt[5] * fabs(distSpa[2] / inteSpa[2]));

      slicExt[0] = 0; slicExt[1] = 0;
      slicExt[2] = inteExt[2]; slicExt[3] = inteExt[3];
      slicExt[4] = inteExt[4]; slicExt[5] = inteExt[5];
    }
  else if (SliceAxis == 1)
    {
      liftSpa[0] = inSpa[0]; liftSpa[1] = (inSpa[0] + inSpa[2]) / 2.0; liftSpa[2] = inSpa[2];
      memcpy(distSpa, liftSpa, sizeof(double)*3);
      inteSpa[0] = OutputSpacing[0]; inteSpa[1] = (inSpa[0] + inSpa[2]) / 2.0; inteSpa[2] = OutputSpacing[2];

      liftExt[0] = inExt[0] - 1; liftExt[1] = inExt[1] + 1;
      liftExt[2] = -1; liftExt[3] = nBins;
      liftExt[4] = inExt[4] - 1; liftExt[5] = inExt[5] + 1;

      distExt[0] = inExt[0]; distExt[1] = inExt[1];
      distExt[2] = 0; distExt[3] = nBins - 1;
      distExt[4] = inExt[4]; distExt[5] = inExt[5];

      inteExt[0] = inExt[0]; inteExt[1] = int (inExt[1] * fabs(distSpa[0] / inteSpa[0]));
      inteExt[2] = 0; inteExt[3] = nBins - 1;
      inteExt[4] = inExt[4]; inteExt[5] = int (inExt[5] * fabs(distSpa[2] / inteSpa[2]));

      slicExt[0] = inteExt[0]; slicExt[1] = inteExt[1];
      slicExt[2] = 0; slicExt[3] = 0;
      slicExt[4] = inteExt[4]; slicExt[5] = inteExt[5];
    }
  else
    {
      liftSpa[0] = inSpa[0]; liftSpa[1] = inSpa[1]; liftSpa[2] = (inSpa[0] + inSpa[1]) / 2.0; 
      memcpy(distSpa, liftSpa, sizeof(double)*3);
      inteSpa[0] = OutputSpacing[0]; inteSpa[1] = OutputSpacing[1]; inteSpa[2] = (inSpa[0] + inSpa[1]) / 2.0; 

      liftExt[0] = inExt[0] - 1; liftExt[1] = inExt[1] + 1;
      liftExt[2] = inExt[2] - 1; liftExt[3] = inExt[3] + 1;
      liftExt[4] = -1; liftExt[5] = nBins;

      distExt[0] = inExt[0]; distExt[1] = inExt[1];
      distExt[2] = inExt[2]; distExt[3] = inExt[3];
      distExt[4] = 0; distExt[5] = nBins - 1;

      inteExt[0] = inExt[0]; inteExt[1] = int (inExt[1] * fabs(distSpa[0] / inteSpa[0]));
      inteExt[2] = inExt[2]; inteExt[3] = int (inExt[3] * fabs(distSpa[1] / inteSpa[1]));
      inteExt[4] = 0; inteExt[5] = nBins - 1;

      slicExt[0] = inteExt[0]; slicExt[1] = inteExt[1];
      slicExt[2] = inteExt[2]; slicExt[3] = inteExt[3];
      slicExt[4] = 0; slicExt[5] = 0;
    }
}

//----------------------------------------------------------------------------
void LiftSlice(vtkImageData *input, int inExt[6], double inSpa[3], double inMinVal, 
         double inMaxVal, int SliceAxis, int nBins, int slice, vtkImageData *&lift)
{
  int i,j,k,v;
  int ext[6];
  double value;

  memcpy(ext, inExt, sizeof(int)*6);
  ext[0] = ext[0] - 1; ext[1] = ext[1] + 1;
  ext[2] = ext[2] - 1; ext[3] = ext[3] + 1;
  ext[4] = ext[4] - 1; ext[5] = ext[5] + 1;
  ext[2 * SliceAxis] = slice;
  ext[2 * SliceAxis + 1] = slice;

  for (k = ext[4]; k <= ext[5]; k++)
    {
      for (j = ext[2]; j <= ext[3]; j++)
         {
     for (i = ext[0]; i <= ext[1]; i++)
       {
        if (SliceAxis == 0)
    {
      if ((j != ext[2]) & (j != ext[3] )& (k != ext[4]) & (k != ext[5]))
        {
          value = double(nBins - 1) / (inMaxVal - inMinVal) * (input->GetScalarComponentAsFloat(i,j,k,0) - inMinVal);
          for (v = -1; v <= nBins; v++)
      {
        if (v <= int(value)) lift->SetScalarComponentFromFloat(v,j,k,0,1.0);
        else lift->SetScalarComponentFromFloat(v,j,k,0,0.0);
      }
        }
      else
        {
          for (v = -1; v <= nBins; v++)
      {
        lift->SetScalarComponentFromFloat(v,j,k,0,1.0);
      }
        }
    }
        else if (SliceAxis == 1)
    {
      if ((i != ext[0]) & (i != ext[1]) & (k != ext[4]) & (k != ext[5]))
        {
          value = double(nBins - 1) / (inMaxVal - inMinVal) * (input->GetScalarComponentAsFloat(i,j,k,0) - inMinVal);
          for (v = -1; v <= nBins; v++)
      {
        if (v <= int(value)) lift->SetScalarComponentFromFloat(i,v,k,0,1.0);
        else lift->SetScalarComponentFromFloat(i,v,k,0,0.0);
      }
        }
      else
        {
          for (v = -1; v <= nBins; v++)
      {
        lift->SetScalarComponentFromFloat(i,v,k,0,1.0);
      }
        }
    }
        else if (SliceAxis == 2)
    {
      if ((i != ext[0]) & (i != ext[1]) & (j != ext[2]) & (j != ext[3]))
        {
          value = double(nBins - 1) / (inMaxVal - inMinVal) * (input->GetScalarComponentAsFloat(i,j,k,0) - inMinVal);
          for (v = -1; v <= nBins; v++)
      {
        if (v <= int(value)) lift->SetScalarComponentFromFloat(i,j,v,0,1.0);
        else lift->SetScalarComponentFromFloat(i,j,v,0,0.0);
      }
        }
      else
        {
          for (v = -1; v <= nBins; v++)
      {
        lift->SetScalarComponentFromFloat(i,j,v,0,1.0);
      }
        }
    }
      }
  }
    }
}

//----------------------------------------------------------------------------
void CalculateDistanceMap(vtkImageData *lift, int SliceAxis, vtkImageData *&distanceMap)
{
  int i,j,k,l;
  int ext[6];
  double spa[3];
  int loc011,loc211,loc101,loc121,loc110,loc112;
  double iniVals[7], ini;
  double iniD1 = 1.0, iniD2 = 1000.0;
  double finVals[14];
  double finD1 = 1.0,finD2 = 1.314,finD3 = 1.628;

  lift->GetExtent(ext);
  lift->GetSpacing(spa);

  vtkImageData *temp = vtkImageData::New();
  temp->SetExtent(ext);
#if (VTK_MAJOR_VERSION <= 5)
  temp->SetScalarTypeToFloat();
  temp->SetNumberOfScalarComponents(1);
  temp->SetSpacing(spa);
  temp->AllocateScalars();
#else
  temp->SetSpacing(spa);
  temp->AllocateScalars(VTK_FLOAT, 1);
#endif

  // Initialize distance map
  for (k = ext[4]; k <= ext[5]; k++)
    {
      for (j = ext[2]; j <= ext[3]; j++)
   {
       for (i = ext[0]; i <= ext[1]; i++)
         {
        loc011 = i - 1; loc211 = i + 1;
        loc101 = j - 1; loc121 = j + 1;
        loc110 = k - 1; loc112 = k + 1;

        // Axis 0
        iniVals[0] = 1.0;
        if (loc011 >= ext[0]) iniVals[0] = lift->GetScalarComponentAsFloat(loc011,j,k,0);
        if (SliceAxis == 0) iniVals[1] = 0.0; if (SliceAxis == 1) iniVals[1] = 1.0; if (SliceAxis == 2) iniVals[1] = 1.0;
        if (loc211 <= ext[1]) iniVals[1] = lift->GetScalarComponentAsFloat(loc211,j,k,0);

        // Axis 1
        iniVals[2] = 1.0;
        if (loc101 >= ext[2]) iniVals[2] = lift->GetScalarComponentAsFloat(i,loc101,k,0);
        if (SliceAxis == 0) iniVals[3] = 1.0; if (SliceAxis == 1) iniVals[3] = 0.0; if (SliceAxis == 2) iniVals[3] = 1.0;
        if (loc121 <= ext[3]) iniVals[3] = lift->GetScalarComponentAsFloat(i,loc121,k,0);

        // Axis 2
        iniVals[4] = 1.0;
        if (loc110 >= ext[4]) iniVals[4] = lift->GetScalarComponentAsFloat(i,j,loc110,0);
        if (SliceAxis == 0) iniVals[5] = 1.0; if (SliceAxis == 1) iniVals[5] = 1.0; if (SliceAxis == 2) iniVals[5] = 0.0;
        if (loc112 <= ext[5]) iniVals[5] = lift->GetScalarComponentAsFloat(i,j,loc112,0);

        iniVals[6] = lift->GetScalarComponentAsFloat(i,j,k,0);
        
        if (iniVals[6] == 1.0)
    {
      ini = iniD2;
      for (l = 0; l <= 5; l++)
        {
          if (iniVals[l] == 0.0) ini = iniD1;
        }
    }
        else
    {
      ini = -iniD2;
      for (l = 0; l <= 5; l++)
        {
          if (iniVals[l] == 1.0) ini = -iniD1;
        }
    }

        temp->SetScalarComponentFromFloat(i,j,k,0,ini);
      }
  }
    }

  // First chamfer pass
  for (k = ext[4] + 1; k <= ext[5] - 1; k++)
    {
      for (j = ext[2] + 1; j <= ext[3] - 1; j++)
   {
       for (i = ext[0] + 1; i <= ext[1] - 1; i++)
         {
        finVals[13] = temp->GetScalarComponentAsFloat(i,j,k,0);
        if ((finVals[13] != 1.0) & (finVals[13] != -1.0))
    {
      if (finVals[13] > 0)
        {
          finVals[0]  = temp->GetScalarComponentAsFloat(i-1,j-1,k-1,0) + finD3;
          finVals[1]  = temp->GetScalarComponentAsFloat(i  ,j-1,k-1,0) + finD2;
          finVals[2]  = temp->GetScalarComponentAsFloat(i+1,j-1,k-1,0) + finD3;
          finVals[3]  = temp->GetScalarComponentAsFloat(i-1,j  ,k-1,0) + finD2;
          finVals[4]  = temp->GetScalarComponentAsFloat(i  ,j  ,k-1,0) + finD1;
          finVals[5]  = temp->GetScalarComponentAsFloat(i+1,j  ,k-1,0) + finD2;
          finVals[6]  = temp->GetScalarComponentAsFloat(i-1,j+1,k-1,0) + finD3;
          finVals[7]  = temp->GetScalarComponentAsFloat(i  ,j+1,k-1,0) + finD2;
          finVals[8]  = temp->GetScalarComponentAsFloat(i+1,j+1,k-1,0) + finD3;
          finVals[9]  = temp->GetScalarComponentAsFloat(i-1,j-1,k  ,0) + finD2;
          finVals[10] = temp->GetScalarComponentAsFloat(i  ,j-1,k  ,0) + finD1;
          finVals[11] = temp->GetScalarComponentAsFloat(i+1,j-1,k  ,0) + finD2;
          finVals[12] = temp->GetScalarComponentAsFloat(i-1,j  ,k  ,0) + finD1;
          temp->SetScalarComponentFromFloat(i,j,k,0,vtkMinimum(finVals));
        }
      else
        {        
          finVals[0]  = temp->GetScalarComponentAsFloat(i-1,j-1,k-1,0) - finD3;
          finVals[1]  = temp->GetScalarComponentAsFloat(i  ,j-1,k-1,0) - finD2;
          finVals[2]  = temp->GetScalarComponentAsFloat(i+1,j-1,k-1,0) - finD3;
          finVals[3]  = temp->GetScalarComponentAsFloat(i-1,j  ,k-1,0) - finD2;
          finVals[4]  = temp->GetScalarComponentAsFloat(i  ,j  ,k-1,0) - finD1;
          finVals[5]  = temp->GetScalarComponentAsFloat(i+1,j  ,k-1,0) - finD2;
          finVals[6]  = temp->GetScalarComponentAsFloat(i-1,j+1,k-1,0) - finD3;
          finVals[7]  = temp->GetScalarComponentAsFloat(i  ,j+1,k-1,0) - finD2;
          finVals[8]  = temp->GetScalarComponentAsFloat(i+1,j+1,k-1,0) - finD3;
          finVals[9]  = temp->GetScalarComponentAsFloat(i-1,j-1,k  ,0) - finD2;
          finVals[10] = temp->GetScalarComponentAsFloat(i  ,j-1,k  ,0) - finD1;
          finVals[11] = temp->GetScalarComponentAsFloat(i+1,j-1,k  ,0) - finD2;
          finVals[12] = temp->GetScalarComponentAsFloat(i-1,j  ,k  ,0) - finD1;
          temp->SetScalarComponentFromFloat(i,j,k,0,vtkMaximum(finVals));
        }
    }
      }
  }
    }

  // Second chamfer pass
  for (k = ext[5] - 1; k >= ext[4] + 1; k--)
    {
      for (j = ext[3] - 1; j >= ext[2] + 1; j--)
   {
       for (i = ext[1] - 1; i >= ext[0] + 1; i--)
         {
        finVals[13] = temp->GetScalarComponentAsFloat(i,j,k,0);
        if ((finVals[13] != 1.0 )& (finVals[13] != -1.0))
    {
      if (finVals[13] > 0)
        {
          finVals[0]  = temp->GetScalarComponentAsFloat(i-1,j-1,k+1,0) + finD3;
          finVals[1]  = temp->GetScalarComponentAsFloat(i  ,j-1,k+1,0) + finD2;
          finVals[2]  = temp->GetScalarComponentAsFloat(i+1,j-1,k+1,0) + finD3;
          finVals[3]  = temp->GetScalarComponentAsFloat(i-1,j  ,k+1,0) + finD2;
          finVals[4]  = temp->GetScalarComponentAsFloat(i  ,j  ,k+1,0) + finD1;
          finVals[5]  = temp->GetScalarComponentAsFloat(i+1,j  ,k+1,0) + finD2;
          finVals[6]  = temp->GetScalarComponentAsFloat(i-1,j+1,k+1,0) + finD3;
          finVals[7]  = temp->GetScalarComponentAsFloat(i  ,j+1,k+1,0) + finD2;
          finVals[8]  = temp->GetScalarComponentAsFloat(i+1,j+1,k+1,0) + finD3;
          finVals[9]  = temp->GetScalarComponentAsFloat(i-1,j+1,k  ,0) + finD2;
          finVals[10] = temp->GetScalarComponentAsFloat(i  ,j+1,k  ,0) + finD1;
          finVals[11] = temp->GetScalarComponentAsFloat(i+1,j+1,k  ,0) + finD2;
          finVals[12] = temp->GetScalarComponentAsFloat(i+1,j  ,k  ,0) + finD1;
          temp->SetScalarComponentFromFloat(i,j,k,0,vtkMinimum(finVals));
        }
      else
        {        
          finVals[0]  = temp->GetScalarComponentAsFloat(i-1,j-1,k+1,0) - finD3;
          finVals[1]  = temp->GetScalarComponentAsFloat(i  ,j-1,k+1,0) - finD2;
          finVals[2]  = temp->GetScalarComponentAsFloat(i+1,j-1,k+1,0) - finD3;
          finVals[3]  = temp->GetScalarComponentAsFloat(i-1,j  ,k+1,0) - finD2;
          finVals[4]  = temp->GetScalarComponentAsFloat(i  ,j  ,k+1,0) - finD1;
          finVals[5]  = temp->GetScalarComponentAsFloat(i+1,j  ,k+1,0) - finD2;
          finVals[6]  = temp->GetScalarComponentAsFloat(i-1,j+1,k+1,0) - finD3;
          finVals[7]  = temp->GetScalarComponentAsFloat(i  ,j+1,k+1,0) - finD2;
          finVals[8]  = temp->GetScalarComponentAsFloat(i+1,j+1,k+1,0) - finD3;
          finVals[9]  = temp->GetScalarComponentAsFloat(i-1,j+1,k  ,0) - finD2;
          finVals[10] = temp->GetScalarComponentAsFloat(i  ,j+1,k  ,0) - finD1;
          finVals[11] = temp->GetScalarComponentAsFloat(i+1,j+1,k  ,0) - finD2;
          finVals[12] = temp->GetScalarComponentAsFloat(i+1,j  ,k  ,0) - finD1;
          temp->SetScalarComponentFromFloat(i,j,k,0,vtkMaximum(finVals));
        }
    }
      }
  }
    }

  ext[0] = ext[0] + 1; ext[1] = ext[1] - 1;
  ext[2] = ext[2] + 1; ext[3] = ext[3] - 1;
  ext[4] = ext[4] + 1; ext[5] = ext[5] - 1;

  vtkImageReslice *reslice = vtkImageReslice::New();
  reslice->SetInput(temp);
  reslice->SetOutputExtent(ext);
  reslice->SetOutputSpacing(spa);
  reslice->SetInterpolationModeToLinear();
  reslice->UpdateWholeExtent();

  // Clean up
  distanceMap->DeepCopy(reslice->GetOutput());
  temp->Delete();
  reslice->Delete();
}

//----------------------------------------------------------------------------
void InterpolateDistanceMap(vtkImageData *distanceMaps[2], int SliceAxis, double pos, 
          double OutputSpacing[3], vtkImageData *&interpolatedDistanceMap)
{
  int i,j,k;
  int ext[6];
  double spa[3];
  int outExt[6];
  double outSpa[3];
  double newValue;

  distanceMaps[0]->GetExtent(ext);
  distanceMaps[0]->GetSpacing(spa);

  vtkImageData *temp = vtkImageData::New();
  temp->SetExtent(ext);
#if (VTK_MAJOR_VERSION <= 5)
  temp->SetScalarTypeToFloat();
  temp->SetNumberOfScalarComponents(1);
  temp->SetSpacing(spa);
  temp->AllocateScalars();
#else
  temp->SetSpacing(spa);
  temp->AllocateScalars(VTK_FLOAT, 1);
#endif

  if (pos == 0.0)
    {
      temp->DeepCopy(distanceMaps[0]);
    }
  else if (pos == 1.0)
    {
      temp->DeepCopy(distanceMaps[1]);
    }
  // Interpolate a new slice
  else
    {
      for (k = ext[4]; k <= ext[5]; k++)
  {
    for (j = ext[2]; j <= ext[3]; j++)
      {
        for (i = ext[0]; i <= ext[1]; i++)
    {
      newValue = ( (1.0 - pos) * distanceMaps[0]->GetScalarComponentAsFloat(i,j,k,0) + 
             pos * distanceMaps[1]->GetScalarComponentAsFloat(i,j,k,0) );
      temp->SetScalarComponentFromFloat(i,j,k,0,newValue);
    }
      }
  }
    }      

  // Interpolate pixels
  interpolatedDistanceMap->GetExtent(outExt);
  interpolatedDistanceMap->GetSpacing(outSpa);

  vtkImageReslice *reslice = vtkImageReslice::New();
  reslice->SetInput(temp);
  reslice->SetOutputExtent(outExt);
  reslice->SetOutputSpacing(outSpa);
  reslice->SetInterpolationModeToLinear();
  reslice->UpdateWholeExtent();

  // Clean up
  interpolatedDistanceMap->DeepCopy(reslice->GetOutput());
  temp->Delete();
  reslice->Delete();
}

//----------------------------------------------------------------------------
vtkImageData *ExtractSlice(vtkImageData *distanceMap, int SliceAxis, double OutputSpacing[3], 
         int slicExt[6], double inMinVal, double inMaxVal, int nBins)
{
  int i,j,k,l;
  int ext1[6], ext2[6];
  double value;
  
  vtkImageData *slice1 = vtkImageData::New();
  slice1->SetExtent(slicExt);
  slice1->SetSpacing(OutputSpacing);
#if (VTK_MAJOR_VERSION <= 5)
  slice1->SetScalarTypeToShort();
  slice1->SetNumberOfScalarComponents(1);
  slice1->AllocateScalars();
#else
  slice1->AllocateScalars(VTK_SHORT, 1);
#endif

  vtkImageThreshold *threshold1 = vtkImageThreshold::New();
  threshold1->SetInput(distanceMap);
  threshold1->ThresholdByUpper(0.0);
  threshold1->SetOutValue(0);
  threshold1->SetInValue(1);
  threshold1->ReplaceOutOn();
  threshold1->ReplaceInOn();
  threshold1->SetOutputScalarTypeToShort();
  threshold1->Update();

  threshold1->GetOutput()->GetExtent(ext1);
  threshold1->GetOutput()->GetExtent(ext2);
  ext2[2 * SliceAxis] = 0; ext2[2 * SliceAxis + 1] = 0;

  for (k = ext2[4]; k <= ext2[5]; k++)
    {
      for (j = ext2[2]; j <= ext2[3]; j++)
         {
     for (i = ext2[0]; i <= ext2[1]; i++)
      {
        if (SliceAxis == 0)
    {
      for (l = ext1[0]; l <= ext1[1]; l++)
        {
          if (threshold1->GetOutput()->GetScalarComponentAsFloat(l,j,k,0) == 0)
      {
        value = double(l) * (inMaxVal - inMinVal) / double(nBins - 1) + inMinVal;
        slice1->SetScalarComponentFromFloat(i,j,k,0,value);
        break;
      }
          slice1->SetScalarComponentFromFloat(i,j,k,0,inMaxVal);
        }
    }
        if (SliceAxis == 1)
    {
      for (l = ext1[2]; l <= ext1[3]; l++)
        {
          if (threshold1->GetOutput()->GetScalarComponentAsFloat(i,l,k,0) == 0)
      {
        value = double(l) * (inMaxVal - inMinVal) / double(nBins - 1) + inMinVal;
        slice1->SetScalarComponentFromFloat(i,j,k,0,value);
        break;
      }
          slice1->SetScalarComponentFromFloat(i,j,k,0,inMaxVal);
        }
    }
        if (SliceAxis == 2)
    {
      for (l = ext1[4]; l <= ext1[5]; l++)
        {
          if (threshold1->GetOutput()->GetScalarComponentAsFloat(i,j,l,0) == 0)
      {
        value = double(l) * (inMaxVal - inMinVal) / double(nBins - 1) + inMinVal;
        slice1->SetScalarComponentFromFloat(i,j,k,0,value);
        break;
      }
          slice1->SetScalarComponentFromFloat(i,j,k,0,inMaxVal);
        }
    }
      }
  }
    }

  vtkImageData *slice2 = vtkImageData::New();
  slice2->SetExtent(slicExt);
  slice2->SetSpacing(OutputSpacing);
#if (VTK_MAJOR_VERSION <= 5)
  slice2->SetScalarTypeToShort();
  slice2->SetNumberOfScalarComponents(1);
  slice2->AllocateScalars();
#else
  slice2->AllocateScalars(VTK_SHORT, 1);
#endif

  vtkImageThreshold *threshold2 = vtkImageThreshold::New();
  threshold2->SetInput(distanceMap);
  threshold2->ThresholdByUpper(0.0);
  threshold2->SetOutValue(0);
  threshold2->SetInValue(1);
  threshold2->ReplaceOutOn();
  threshold2->ReplaceInOn();
  threshold2->SetOutputScalarTypeToShort();
  threshold2->Update();

  threshold2->GetOutput()->GetExtent(ext1);
  threshold2->GetOutput()->GetExtent(ext2);
  ext2[2 * SliceAxis] = 0; ext2[2 * SliceAxis + 1] = 0;

  for (k = ext2[4]; k <= ext2[5]; k++)
    {
      for (j = ext2[2]; j <= ext2[3]; j++)
         {
     for (i = ext2[0]; i <= ext2[1]; i++)
      {
        if (SliceAxis == 0)
    {
      for (l = ext1[0]; l <= ext1[1]; l++)
        {
          if (threshold2->GetOutput()->GetScalarComponentAsFloat(l,j,k,0) == 0)
      {
        value = double(l) * (inMaxVal - inMinVal) / double(nBins - 1) + inMinVal;
        slice2->SetScalarComponentFromFloat(i,j,k,0,value);
        break;
      }
          slice2->SetScalarComponentFromFloat(i,j,k,0,inMaxVal);
        }
    }
        if (SliceAxis == 1)
    {
      for (l = ext1[2]; l <= ext1[3]; l++)
        {
          if (threshold2->GetOutput()->GetScalarComponentAsFloat(i,l,k,0) == 0)
      {
        value = double(l) * (inMaxVal - inMinVal) / double(nBins - 1) + inMinVal;
        slice2->SetScalarComponentFromFloat(i,j,k,0,value);
        break;
      }
          slice2->SetScalarComponentFromFloat(i,j,k,0,inMaxVal);
        }
    }
        if (SliceAxis == 2)
    {
      for (l = ext1[4]; l <= ext1[5]; l++)
        {
          if (threshold2->GetOutput()->GetScalarComponentAsFloat(i,j,l,0) == 0)
      {
        value = double(l) * (inMaxVal - inMinVal) / double(nBins - 1) + inMinVal;
        slice2->SetScalarComponentFromFloat(i,j,k,0,value);
        break;
      }
          slice2->SetScalarComponentFromFloat(i,j,k,0,inMaxVal);
        }
    }
      }
  }
    }

  vtkImageMathematics *math1 = vtkImageMathematics::New();
  math1->SetInput1(slice1);
  math1->SetInput2(slice2);
  math1->SetOperationToAdd();
  math1->Update();

  vtkImageMathematics *math2 = vtkImageMathematics::New();
  math2->SetInput1(math1->GetOutput());
  math2->SetConstantK(0.5);
  math2->SetOperationToMultiplyByK();
  math2->Update();

  threshold1->Delete();
  threshold2->Delete();
  return math2->GetOutput();
}

//----------------------------------------------------------------------------
vtkImageData *GetOriginalSlice(vtkImageData *input, int SliceAxis, int iSlice, double OutputSpacing[3])
{
  int i,j,k;
  int ext[6];

  input->GetExtent(ext);
  ext[2 * SliceAxis] = iSlice;
  ext[2 * SliceAxis + 1] = iSlice;

  vtkImageData *slice = vtkImageData::New();
  slice->SetExtent(ext);
#if (VTK_MAJOR_VERSION <= 5)
  slice->SetScalarTypeToShort();
  slice->SetNumberOfScalarComponents(1);
  slice->SetSpacing(OutputSpacing);
  slice->AllocateScalars();
#else
  slice->SetSpacing(OutputSpacing);
  slice->AllocateScalars(VTK_SHORT, 1);
#endif

  for (k = ext[4]; k <= ext[5]; k++)
    {
      for (j = ext[2]; j <= ext[3]; j++)
         {
     for (i = ext[0]; i <= ext[1]; i++)
      {
        slice->SetScalarComponentFromFloat(i,j,k,0,input->GetScalarComponentAsFloat(i,j,k,0));
      }
  }
    }

  ext[2 * SliceAxis] = 0;
  ext[2 * SliceAxis + 1] = 0;
  slice->SetExtent(ext);

  return slice;
}

//----------------------------------------------------------------------------
void vtkShapeBasedInterpolation::Update()
{
  int i,j,iSta,iEnd;
  double pos, nPos, staPos = 0.0;
  int liftExt[6],distExt[6],inteExt[6],slicExt[6];//,ext[6];
  double liftSpa[3],distSpa[3],inteSpa[6],slicSpa[3];
  int interPix = 0;
 
  CalculateSpacingsExtents(this->inSpa, this->inExt, this->SliceAxis, this->OutputSpacing, this->NumberOfBins,
         liftSpa, distSpa, inteSpa, slicSpa, liftExt, distExt, inteExt, slicExt);

  vtkImageData *lift = vtkImageData::New();
  lift->SetExtent(liftExt);
#if (VTK_MAJOR_VERSION <= 5)
  lift->SetScalarTypeToFloat();
  lift->SetNumberOfScalarComponents(1);
  lift->SetSpacing(liftSpa);
  lift->AllocateScalars();
#else
  lift->SetSpacing(liftSpa);
  lift->AllocateScalars(VTK_FLOAT, 1);
#endif

  vtkImageData *distanceMaps[2];
  distanceMaps[0] = vtkImageData::New();
  distanceMaps[0]->SetExtent(distExt);
#if (VTK_MAJOR_VERSION <= 5)
  distanceMaps[0]->SetScalarTypeToFloat();
  distanceMaps[0]->SetNumberOfScalarComponents(1);
  distanceMaps[0]->SetSpacing(distSpa);
  distanceMaps[0]->AllocateScalars();
  distanceMaps[1] = vtkImageData::New();
  distanceMaps[1]->SetExtent(distExt);
  distanceMaps[1]->SetScalarTypeToFloat();
  distanceMaps[1]->SetNumberOfScalarComponents(1);
  distanceMaps[1]->SetSpacing(distSpa);
  distanceMaps[1]->AllocateScalars();
#else
  distanceMaps[0]->SetSpacing(distSpa);
  distanceMaps[0]->AllocateScalars(VTK_FLOAT, 1);
  distanceMaps[1] = vtkImageData::New();
  distanceMaps[1]->SetExtent(distExt);
  distanceMaps[1]->SetSpacing(distSpa);
  distanceMaps[1]->AllocateScalars(VTK_FLOAT, 1);
#endif

  vtkImageData *interpolatedDistanceMap = vtkImageData::New();
  interpolatedDistanceMap->SetExtent(inteExt);
#if (VTK_MAJOR_VERSION <= 5)
  interpolatedDistanceMap->SetScalarTypeToFloat();
  interpolatedDistanceMap->SetNumberOfScalarComponents(1);
  interpolatedDistanceMap->SetSpacing(inteSpa);
  interpolatedDistanceMap->AllocateScalars();
#else
  interpolatedDistanceMap->SetSpacing(inteSpa);
  interpolatedDistanceMap->AllocateScalars(VTK_FLOAT, 1);
#endif

  vtkImageData *slice;

  vtkImageAppend *append = vtkImageAppend::New();
  append->SetAppendAxis(SliceAxis);

  iSta = this->inExt[2 * this->SliceAxis];
  iEnd = this->inExt[2 * this->SliceAxis + 1];
    
  for (i = 0; i <= 2; i++)
    {
      if ((i != this->SliceAxis) & (this->inSpa[i] != this->OutputSpacing[i])) interPix = 1;
    }

  LiftSlice(this->inData, this->inExt, this->inSpa, this->inMinVal, this->inMaxVal, this->SliceAxis, this->NumberOfBins, 0, lift);
  CalculateDistanceMap(lift, this->SliceAxis, distanceMaps[0]);
  for (i = iSta; i <= iEnd - 1; i++)
    {
      cout << "\n Working on original slice " << i+1 << " of " << iEnd + 1;

      LiftSlice(this->inData, this->inExt, this->inSpa, this->inMinVal, this->inMaxVal, this->SliceAxis, this->NumberOfBins, i+1, lift);
      CalculateDistanceMap(lift, this->SliceAxis, distanceMaps[1]);

      pos = staPos; nPos = (pos - i * fabs(this->inSpa[SliceAxis])) / fabs(this->inSpa[SliceAxis]); j = 0;
      while (pos < (i + 1) * fabs(this->inSpa[SliceAxis]))
  {
     if ((nPos <= 0.01) & (interPix == 0))
       {
         slice = GetOriginalSlice(this->inData, this->SliceAxis, i, this->OutputSpacing);
       }
     else
       {
        InterpolateDistanceMap(distanceMaps, this->SliceAxis, nPos, this->OutputSpacing, interpolatedDistanceMap);
        slice = ExtractSlice(interpolatedDistanceMap, this->SliceAxis, this->OutputSpacing, slicExt,
           this->inMinVal, this->inMaxVal, this->NumberOfBins);
       }
    append->AddInput(slice);
    append->Update();
    j++;
    pos = staPos + j * fabs(this->OutputSpacing[SliceAxis]);
    nPos = (pos - i * fabs(this->inSpa[SliceAxis])) / fabs(this->inSpa[SliceAxis]);
  }

      staPos = pos;
      distanceMaps[0]->DeepCopy(distanceMaps[1]);
    }

  cout << "\n Working on original slice " << iEnd + 1 << " of " << iEnd + 1 << "\n\n";
  if (interPix == 0)
    {
      slice = GetOriginalSlice(this->inData, this->SliceAxis, iEnd, this->OutputSpacing);
    }
  else
    {
      InterpolateDistanceMap(distanceMaps, this->SliceAxis, 1, this->OutputSpacing, interpolatedDistanceMap);
      slice = ExtractSlice(interpolatedDistanceMap, this->SliceAxis, this->OutputSpacing, slicExt,
         this->inMinVal, this->inMaxVal, this->NumberOfBins);
    }

  append->AddInput(slice);
  append->Update();
  append->GetOutput()->SetOrigin(this->inOri);
  this->outData = append->GetOutput();

}

