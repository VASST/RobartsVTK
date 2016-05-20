/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageSMIPVIManipulator.cxx,v $
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
#include "vtkImageSMIPVIManipulator.h"

//--------------------------------------------------------------------------
// The 'floor' function on x86 and mips is many times slower than these
// and is used a lot in this code, optimize for different CPU architectures
inline int vtkResliceFloor(double x)
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

inline int vtkResliceRound(double x)
{
  return vtkResliceFloor(x + 0.5);
}

inline int vtkResliceFloor(float x)
{
  return vtkResliceFloor((double)x);
}

inline int vtkResliceRound(float x)
{
  return vtkResliceRound((double)x);
}

// convert a float into an integer plus a fraction
inline int vtkResliceFloor(double x, double &f)
{
  int ix = vtkResliceFloor(x);
  f = x - ix;
  if (f < 0.0)
  {
    f = 0.0;
  }
  if (f > 1.0)
  {
    f = 1.0;
  }
  return ix;
}

//----------------------------------------------------------------------------
vtkImageSMIPVIManipulator* vtkImageSMIPVIManipulator::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkImageSMIPVIManipulator");
  if(ret)
  {
    return (vtkImageSMIPVIManipulator*)ret;
  }
  // If the factory was unable to create the object, then create it here.
  return new vtkImageSMIPVIManipulator;
}

//----------------------------------------------------------------------------
vtkImageSMIPVIManipulator::vtkImageSMIPVIManipulator()
{
  for (int i = 0; i <= 5; i++)
  {
    this->Extent[i] = 0;
  }
  this->BinNumber[0] = 4096;
  this->BinNumber[1] = 4096;
  this->BinWidth[0] = 1.0;
  this->BinWidth[1] = 1.0;
  this->MaxIntensities[0] = 4095;
  this->MaxIntensities[1] = 4095;
  this->Metric = 0;
}

//----------------------------------------------------------------------------
vtkImageSMIPVIManipulator::~vtkImageSMIPVIManipulator()
{
  delete this->HistS;
  delete this->HistT;
  delete this->HistST;
}

//----------------------------------------------------------------------------
void vtkImageSMIPVIManipulator::SetInput1(vtkImageData *input)
{
  input->GetSpacing(this->inSpa);
  input->GetExtent(this->inExt);
  input->GetIncrements(this->inc[0], this->inc[1], this->inc[2]);
  this->inData[0] = input;
}

//----------------------------------------------------------------------------
void vtkImageSMIPVIManipulator::SetInput2(vtkImageData *input)
{
  this->inData[1] = input;
}

//----------------------------------------------------------------------------
vtkImageData *vtkImageSMIPVIManipulator::GetInput1()
{
  return this->inData[0];
}

//----------------------------------------------------------------------------
vtkImageData *vtkImageSMIPVIManipulator::GetInput2()
{
  return this->inData[1];
}

//----------------------------------------------------------------------------
void vtkImageSMIPVIManipulator::SetBinNumber(int numS, int numT)
{
  this->BinNumber[0] = numS;
  this->BinNumber[1] = numT;
  this->BinWidth[0] = (double)this->MaxIntensities[0] / ((double)this->BinNumber[0] - 1.0);
  this->BinWidth[1] = (double)this->MaxIntensities[1] / ((double)this->BinNumber[1] - 1.0);
  this->HistS = new double[numS];
  this->HistT = new double[numT];
  this->HistST = new double[numS*numT];

  vtkImageShiftScale* scale0 = vtkImageShiftScale::New();
  scale0->SetInputData(this->inData[0]);
  scale0->SetShift(0.5*this->BinWidth[0]); // Shift to round not floor
  scale0->SetScale(1.0/this->BinWidth[0]);
  scale0->Update();
  this->inDataScl[0] = scale0->GetOutput();
  inDataScl[0]->GetIncrements(this->inc[0], this->inc[1], this->inc[2]);

  vtkImageShiftScale* scale1 = vtkImageShiftScale::New();
  scale1->SetInputData(this->inData[1]);
  scale1->SetShift(0.5*this->BinWidth[1]); // Shift to round not floor
  scale1->SetScale(1.0/this->BinWidth[1]);
  scale1->Update();
  this->inDataScl[1] = scale1->GetOutput();
}

//----------------------------------------------------------------------------
void vtkImageSMIPVIManipulator::SetMaxIntensities(int maxS, int maxT)
{
  this->MaxIntensities[0] = maxS;
  this->MaxIntensities[1] = maxT;
  this->BinWidth[0] = (double)this->MaxIntensities[0] / ((double)this->BinNumber[0] - 1.0);
  this->BinWidth[1] = (double)this->MaxIntensities[1] / ((double)this->BinNumber[1] - 1.0);

  vtkImageShiftScale* scale0 = vtkImageShiftScale::New();
  scale0->SetInputData(this->inData[0]);
  scale0->SetShift(0.5*this->BinWidth[0]); // Shift to round not floor
  scale0->SetScale(1.0/this->BinWidth[0]);
  scale0->Update();
  this->inDataScl[0] = scale0->GetOutput();
  inDataScl[0]->GetIncrements(this->inc[0], this->inc[1], this->inc[2]);

  vtkImageShiftScale* scale1 = vtkImageShiftScale::New();
  scale1->SetInputData(this->inData[1]);
  scale1->SetShift(0.5*this->BinWidth[1]); // Shift to round not floor
  scale1->SetScale(1.0/this->BinWidth[1]);
  scale1->Update();
  this->inDataScl[1] = scale1->GetOutput();
}

//----------------------------------------------------------------------------
template <class T>
void vtkImageSMIPVIManipulatorEntropyT(vtkImageSMIPVIManipulator *self,
                                       T  *inPtr, int inc2[2], double count)
{
  short b;
  double temp, entropyT = 0;

  // Loop over data
  for (int idZ = self->Extent[4]; idZ <= self->Extent[5]; idZ++)
  {
    for (int idY = self->Extent[2]; idY <= self->Extent[3]; idY++)
    {
      for (int idX = self->Extent[0]; idX <= self->Extent[1]; idX++)
      {
        b = (short)*inPtr;
        self->HistT[b]++;
        inPtr++;
      }
      inPtr += inc2[0];
    }
    inPtr += inc2[1];
  }

  for (int i = 0; i < self->BinNumber[1]; i++)
  {
    temp = self->HistT[i];
    if (temp > 0.0)
    {
      entropyT += temp * log(temp);
    }
  }

  self->entropyT = -entropyT / count + log(count);
}

//----------------------------------------------------------------------------
void vtkImageSMIPVIManipulator::SetExtent(int ext[6])
{
  this->inc2[0] = this->inc[1] - this->inc[0] * (ext[1] - ext[0] + 1);
  this->inc2[1] = this->inc[2] - this->inc[1] * (ext[3] - ext[2] + 1);

  this->inPtr[0] = this->inDataScl[0]->GetScalarPointerForExtent(ext);
  this->inPtr[1] = this->inDataScl[1]->GetScalarPointerForExtent(ext);

  this->count = (ext[1]-ext[0]+1) * (ext[3]-ext[2]+1) * (ext[5]-ext[4]+1);
  if (this->count == 0)
  {
    vtkErrorMacro( "GetResult: No data to work with.");
  }

  memcpy(this->Extent, ext, sizeof(int)*6);

  memset((void *)this->HistT, 0, this->BinNumber[1]*sizeof(double));
  // Calculate the entropy of image 2
  switch (this->inDataScl[1]->GetScalarType())
  {
    vtkTemplateMacro(vtkImageSMIPVIManipulatorEntropyT(this,
                     (VTK_TT *)(this->inPtr[1]), this->inc2, this->count));
  default:
    vtkErrorMacro( "Execute: Unknown ScalarType");
  }
}

//----------------------------------------------------------------------------
void vtkImageSMIPVIManipulator::SetTranslation(double tran[3])
{
  double f[3];

  // Interpolation is not required.
  if ( (tran[0] == 0.0) && (tran[1] == 0.0) && (tran[2] == 0.0) )
  {
    for (int i = 0; i <= 2; i++)
    {
      this->loc000[i] = 0;
      this->loc111[i] = 0;
      f[i] = 0.0;
    }
  }
  // Interpolation is required.
  else
  {
    for (int i = 0; i <= 2; i++)
    {
      this->loc000[i] = vtkResliceFloor(tran[i]/this->inSpa[i], f[i]);
      this->loc111[i] = this->loc000[i] + 1;
      // No interpolation for this axis
      if (tran[i] == 0.0)
      {
        this->loc111[i] = this->loc000[i];
      }
    }
  }

  this->F000 = (1.0 - f[0]) * (1.0 - f[1]) * (1.0 - f[2]);
  this->F100 =        f[0]  * (1.0 - f[1]) * (1.0 - f[2]);
  this->F010 = (1.0 - f[0]) *        f[1]  * (1.0 - f[2]);
  this->F110 =        f[0]  *        f[1]  * (1.0 - f[2]);
  this->F001 = (1.0 - f[0]) * (1.0 - f[1]) *        f[2];
  this->F101 =        f[0]  * (1.0 - f[1]) *        f[2];
  this->F011 = (1.0 - f[0]) *        f[1]  *        f[2];
  this->F111 =        f[0]  *        f[1]  *        f[2];
}

//----------------------------------------------------------------------------
template <class T>
void vtkImageSMIPVIManipulatorExecute(vtkImageSMIPVIManipulator *self,
                                      T  *in1Ptr, T *in2Ptr,
                                      vtkIdType inc[3], int inc2[2], int inExt[6],
                                      int loc000[3], int loc111[3], double count)
{
  short V000, V100, V010, V110, V001, V101, V011, V111;
  double temp1, temp2, entropyS = 0, entropyST = 0;
  short a, b;
  int i, j;

  // CASE 1: Check if translation takes us out of the input image, in
  // which case set the result to indicate complete dissimilarity and stop.
  if ( (self->Extent[0] + loc000[0] < inExt[0]) ||
       (self->Extent[2] + loc000[1] < inExt[2]) ||
       (self->Extent[4] + loc000[2] < inExt[4]) ||
       (self->Extent[1] + loc111[0] > inExt[1]) ||
       (self->Extent[3] + loc111[1] > inExt[3]) ||
       (self->Extent[5] + loc111[2] > inExt[5]) )
  {
    self->Result = 0.5;
    return;
  }

  // CASE 2: Check if no interpolation is necessary.
  if ( (loc000[0] == 0) && (loc111[0] == 0) &&
       (loc000[1] == 0) && (loc111[1] == 0) &&
       (loc000[2] == 0) && (loc111[2] == 0) )
  {
    for (int idZ = self->Extent[4]; idZ <= self->Extent[5]; idZ++)
    {
      for (int idY = self->Extent[2]; idY <= self->Extent[3]; idY++)
      {
        for (int idX = self->Extent[0]; idX <= self->Extent[1]; idX++)
        {
          a = (short)*in1Ptr;
          b = (short)*in2Ptr;

          self->HistS[a]++;
          self->HistST[b * self->BinNumber[0] + a]++;

          in1Ptr++;
          in2Ptr++;
        }
        in1Ptr += inc2[0];
        in2Ptr += inc2[0];
      }
      in1Ptr += inc2[1];
      in2Ptr += inc2[1];
    }
  }

  // CASE 3: If the above two cases don't apply, do the full calculation.
  else
  {
    in1Ptr += inc[2] * loc000[2] + inc[1] * loc000[1] + inc[0] * loc000[0];

    for (int idZ = self->Extent[4]; idZ <= self->Extent[5]; idZ++)
    {
      for (int idY = self->Extent[2]; idY <= self->Extent[3]; idY++)
      {
        // Initiate previous data
        V000 = (short)*in1Ptr;
        in1Ptr += inc[1];
        V010 = (short)*in1Ptr;
        in1Ptr += inc[2];
        V011 = (short)*in1Ptr;
        in1Ptr -= inc[1];
        V001 = (short)*in1Ptr;
        in1Ptr -= inc[2];

        for (int idX = self->Extent[0]; idX <= self->Extent[1]; idX++)
        {
          in1Ptr++;
          V100 = (short)*in1Ptr;
          in1Ptr += inc[1];
          V110 = (short)*in1Ptr;
          in1Ptr += inc[2];
          V111 = (short)*in1Ptr;
          in1Ptr -= inc[1];
          V101 = (short)*in1Ptr;
          in1Ptr -= inc[2];

          b = (short)*in2Ptr * self->BinNumber[0];

          self->HistST[b + V000] +=  self->F000;
          self->HistST[b + V100] +=  self->F100;
          self->HistST[b + V010] +=  self->F010;
          self->HistST[b + V110] +=  self->F110;
          self->HistST[b + V001] +=  self->F001;
          self->HistST[b + V101] +=  self->F101;
          self->HistST[b + V011] +=  self->F011;
          self->HistST[b + V111] +=  self->F111;

          V000 = V100;
          V010 = V110;
          V001 = V101;
          V011 = V111;

          in2Ptr++;
        }
        in1Ptr += inc2[0];
        in2Ptr += inc2[0];
      }
      in1Ptr += inc2[1];
      in2Ptr += inc2[1];
    }
  }

  // Loop over S and ST histograms.
  for (i = 0; i < self->BinNumber[0]; i++)
  {
    temp2 = 0.0;
    for (j = 0; j < self->BinNumber[1]; j++)
    {
      temp1 = self->HistST[j * self->BinNumber[0] + i];
      temp2 += temp1;
      if (temp1 > 0.0)
      {
        entropyST += temp1 * log(temp1);
      }
    }
    if (temp2 > 0.0)
    {
      entropyS += temp2 * log(temp2);
    }
  }

  // Calculate entropies and the SMIPVI result based on the Metric flag.
  entropyS  = -entropyS /count + log(count);
  entropyST = -entropyST/count + log(count);

  // Normalized Mutual Information
  if (self->Metric == 0)
  {
    if (entropyST == 0)
    {
      self->Result = 1.0;
    }
    else
    {
      self->Result = (entropyS + self->entropyT)/entropyST/2.0;
    }
  }

  // Mutual Information
  else if (self->Metric == 1)
  {
    self->Result = entropyS + self->entropyT - entropyST;
  }

  // Entropy Correlation Coefficient
  else if (self->Metric == 2)
  {
    if (entropyS + self->entropyT == 0)
    {
      self->Result = 0.0;
    }
    else
    {
      self->Result = sqrt ( 2.0 * (1.0 - entropyST / ( self->entropyT + entropyS ) ) );
    }
  }

  // Error
  else
  {
    cout << "ERROR: Wrong Metric chosen\n";
    exit(0);
  }
}

//----------------------------------------------------------------------------
double vtkImageSMIPVIManipulator::GetResult()
{
  // Check inputs.
  if (this->inData[0] == NULL)
  {
    vtkErrorMacro( "Input " << 0 << " must be specified.");
    return 0;
  }
  if (this->inData[1] == NULL)
  {
    vtkErrorMacro( "Input " << 1 << " must be specified.");
    return 0;
  }
  if ((this->inData[0]->GetScalarType() != this->inData[1]->GetScalarType()))
  {
    vtkErrorMacro( "Execute: Inputs must be of the same ScalarType");
    return 0;
  }

  // Zero the histograms and result.
  memset((void *)this->HistS, 0, this->BinNumber[0]*sizeof(double));
  memset((void *)this->HistST, 0, this->BinNumber[0]*this->BinNumber[1]*sizeof(double));
  this->Result = 0;

  // Calculate and return SMIPVI result.
  switch (this->inDataScl[0]->GetScalarType())
  {
    vtkTemplateMacro(vtkImageSMIPVIManipulatorExecute(this,
                     (VTK_TT *)(this->inPtr[0]), (VTK_TT *)(this->inPtr[1]),
                     this->inc, this->inc2, this->inExt,
                     this->loc000, this->loc111, this->count));
  default:
    vtkErrorMacro( "Execute: Unknown ScalarType");
  }
  return this->Result;
}

//----------------------------------------------------------------------------
void vtkImageSMIPVIManipulator::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Input 1: "    << this->inData[0]   << "\n";
  os << indent << "Input 2: "    << this->inData[1]   << "\n";
  os << indent << "BinWidth: ( " << this->BinWidth[0] << ", " << this->BinWidth[1]  << " )\n";
  os << indent << "BinNumber: ( "<< this->BinNumber[0]<< ", " << this->BinNumber[1] << " )\n";
  os << indent << "Extent: "     << this->Extent      << "\n";
  os << indent << "Result: "     << this->Result      << "\n";
}
