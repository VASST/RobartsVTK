/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageProbeSum.cxx,v $
  Language:  C++
  Date:      $Date: 2007/05/04 14:34:35 $
  Version:   $Revision: 1.1 $


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
#include "vtkImageProbeSum.h"
#include "vtkObjectFactory.h"
#include "vtkImageData.h"

//----------------------------------------------------------------------------
vtkImageProbeSum* vtkImageProbeSum::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkImageProbeSum");
  if(ret)
    {
    return (vtkImageProbeSum*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkImageProbeSum;
}

//----------------------------------------------------------------------------
vtkImageProbeSum::vtkImageProbeSum()
{
  this->Sum = 0.0;
}

//----------------------------------------------------------------------------
vtkImageProbeSum::~vtkImageProbeSum()
{
}

//----------------------------------------------------------------------------
void vtkImageProbeSum::SetInput(vtkDataSet *input)
{
  this->vtkAlgorithm::SetNthInput(0, input);
}

//----------------------------------------------------------------------------
vtkDataSet *vtkImageProbeSum::GetInput()
{
  if (this->NumberOfInputs < 1)
    {
    return NULL;
    }
  
  return (vtkDataSet *)(this->Inputs[0]);
}

//----------------------------------------------------------------------------
void vtkImageProbeSum::SetSource(vtkImageData *input)
{
  if (!input->IsA("vtkImageData"))
    {
    vtkErrorMacro("SetSource: source must be a vtkImageData");
    }

  this->vtkAlgorithm::SetNthInput(1, input);
}

//----------------------------------------------------------------------------
vtkImageData *vtkImageProbeSum::GetSource()
{
  if (this->NumberOfInputs < 2)
    {
    return NULL;
    }
  
  return (vtkImageData *)(this->Inputs[1]);
}

//--------------------------------------------------------------------------
// The 'floor' function on x86 and mips is many times slower than these
// and is used a lot in this code, optimize for different CPU architectures
static inline int vtkResliceFloor(double x)
{
#if defined mips || defined sparc
  return (int)((unsigned int)(x + 2147483648.0) - 2147483648U);
#elif defined i386 || defined _M_IX86
  unsigned int hilo[2];
  *((double *)hilo) = x + 103079215104.0;  // (2**(52-16))*1.5
  return (int)((hilo[1]<<16)|(hilo[0]>>16));
#else
  return int(floor(x));
#endif
}

static inline int vtkResliceCeil(double x)
{
  return -vtkResliceFloor(-x - 1.0) - 1;
}

static inline int vtkResliceRound(double x)
{
  return vtkResliceFloor(x + 0.5);
}

// convert a double into an integer plus a fraction  
static inline int vtkResliceFloor(double x, double &f)
{
  int ix = vtkResliceFloor(x);
  f = x - ix;
  return ix;
}

//----------------------------------------------------------------------------
// Do trilinear interpolation of the input data 'inPtr' of extent 'inExt'
// at the 'point'.  The result is placed at 'outPtr'.  
// If the lookup data is beyond the extent 'inExt', set 'outPtr' to
// the background color 'background'.  
// The number of scalar components in the data is 'numscalars'
template <class T>
static
double vtkTrilinearInterpolation(const T *inPtr,
         const int inExt[6], const int inInc[3],
         const double point[3])
{
  double fx, fy, fz;
  int floorX = vtkResliceFloor(point[0], fx);
  int floorY = vtkResliceFloor(point[1], fy);
  int floorZ = vtkResliceFloor(point[2], fz);

  int inIdX0 = floorX - inExt[0];
  int inIdY0 = floorY - inExt[2];
  int inIdZ0 = floorZ - inExt[4];

  int inIdX1 = inIdX0 + (fx != 0);
  int inIdY1 = inIdY0 + (fy != 0);
  int inIdZ1 = inIdZ0 + (fz != 0);

  int inExtX = inExt[1] - inExt[0] + 1;
  int inExtY = inExt[3] - inExt[2] + 1;
  int inExtZ = inExt[5] - inExt[4] + 1;

  if (inIdX0 < 0 || inIdX1 >= inExtX ||
      inIdY0 < 0 || inIdY1 >= inExtY ||
      inIdZ0 < 0 || inIdZ1 >= inExtZ)
    {
    return 0.0;
    }

  int factX0 = inIdX0*inInc[0];
  int factX1 = inIdX1*inInc[0];
  int factY0 = inIdY0*inInc[1];
  int factY1 = inIdY1*inInc[1];
  int factZ0 = inIdZ0*inInc[2];
  int factZ1 = inIdZ1*inInc[2];

  int i00 = factY0 + factZ0;
  int i01 = factY0 + factZ1;
  int i10 = factY1 + factZ0;
  int i11 = factY1 + factZ1;

  double rx = 1 - fx;
  double ry = 1 - fy;
  double rz = 1 - fz;

  double ryrz = ry*rz;
  double fyrz = fy*rz;
  double ryfz = ry*fz;
  double fyfz = fy*fz;

  const T *inPtr0 = inPtr + factX0;
  const T *inPtr1 = inPtr + factX1;

  return (rx*(ryrz*inPtr0[i00] + ryfz*inPtr0[i01] +
        fyrz*inPtr0[i10] + fyfz*inPtr0[i11]) +
    fx*(ryrz*inPtr1[i00] + ryfz*inPtr1[i01] +
        fyrz*inPtr1[i10] + fyfz*inPtr1[i11]));
}

//----------------------------------------------------------------------------
double vtkImageProbeSum::GetSum()
{
  vtkIdType ptId, numPts;
  double *point;
  double newpoint[3];
  vtkPointData *pd;
  vtkImageData *source = this->GetSource();
  vtkDataSet *input = this->GetInput();
  double *inSpacing, *inOrigin;
  int inExt[6], inInc[3];
  double *inPtr;

  // Check the inputs
  if (source == NULL)
    {
    vtkErrorMacro("GetSum:  Source is NULL");
    return 0.0;
    }
  if (input == NULL)
    {
    vtkErrorMacro("GetSum:  Input is NULL");
    return 0.0;
    }
  if (source->GetScalarType() != VTK_FLOAT)
    {
    vtkErrorMacro("GetSum: Source must be a double image");
    return 0.0;
    }

  // update the image
  source->UpdateInformation();
  source->SetUpdateExtent(source->GetWholeExtent());
  source->Update();

  // update the input
  input->Update();

  // Update information about the image
  inPtr = (double *)source->GetScalarPointerForExtent(source->GetExtent());
  source->GetExtent(inExt);
  source->GetIncrements(inInc);
  inOrigin = source->GetOrigin();
  inSpacing = source->GetSpacing();

  vtkDebugMacro(<<"Probing data");

  pd = source->GetPointData();
  if (pd == NULL)
    {
    vtkErrorMacro(<< "PointData is NULL.");
    return 0.0;
    }

  numPts = input->GetNumberOfPoints();

  // Loop over all input points, interpolating source data
  this->Sum = 0.0;
  int abort=0;
  vtkIdType progressInterval=numPts/20 + 1;
  for (ptId=0; ptId < numPts && !abort; ptId++)
    {
    if ( !(ptId % progressInterval) )
      {
      this->UpdateProgress((double)ptId/numPts);
      abort = GetAbortExecute();
      }

    // Get the xyz coordinate of the point in the input dataset
    point = input->GetPoint(ptId);
    newpoint[0] = (point[0] - inOrigin[0])/inSpacing[0];
    newpoint[1] = (point[1] - inOrigin[1])/inSpacing[1];
    newpoint[2] = (point[2] - inOrigin[2])/inSpacing[2];

    // add image interpolation code here
    this->Sum += vtkTrilinearInterpolation(inPtr, inExt, inInc, newpoint);
    }

  return this->Sum;
}

//----------------------------------------------------------------------------
void vtkImageProbeSum::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkAlgorithm::PrintSelf(os,indent);
  os << indent << "Input: " << this->GetInput() << "\n";
  os << indent << "Source: " << this->GetSource() << "\n";
}
