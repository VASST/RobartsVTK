/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageSMIManipulator.cxx,v $
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
#include "vtkImageSMIManipulator.h"

//-----------------------`---------------------------------------------------
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
  if (f < 0.0) f = 0.0;
  if (f > 1.0) f = 1.0;
  return ix;
}

//----------------------------------------------------------------------------
vtkImageSMIManipulator* vtkImageSMIManipulator::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkImageSMIManipulator");
  if(ret)
    {
    return (vtkImageSMIManipulator*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkImageSMIManipulator;
}

//----------------------------------------------------------------------------
vtkImageSMIManipulator::vtkImageSMIManipulator()
{
  for (int i = 0; i <= 5; i++) { this->Extent[i] = 0; }
  this->BinNumber[0] = 4096;
  this->BinNumber[1] = 4096;
  this->BinWidth[0] = 1.0;
  this->BinWidth[1] = 1.0;
  this->MaxIntensities[0] = 4095;
  this->MaxIntensities[1] = 4095;
  this->Metric = 0;
}

//----------------------------------------------------------------------------
vtkImageSMIManipulator::~vtkImageSMIManipulator()
{
  delete this->HistS;
  delete this->HistT;
  delete this->HistST;
}

//----------------------------------------------------------------------------
void vtkImageSMIManipulator::SetInput1(vtkImageData *input)
{
  input->GetSpacing(this->inSpa);
  input->GetExtent(this->inExt);
  input->GetIncrements(this->inc[0], this->inc[1], this->inc[2]);
  this->inData[0] = input;
}

//----------------------------------------------------------------------------
void vtkImageSMIManipulator::SetInput2(vtkImageData *input)
{
  this->inData[1] = input;
}

//----------------------------------------------------------------------------
vtkImageData *vtkImageSMIManipulator::GetInput1()
{
  return this->inData[0];
}

//----------------------------------------------------------------------------
vtkImageData *vtkImageSMIManipulator::GetInput2()
{
  return this->inData[1];
}

//----------------------------------------------------------------------------
void vtkImageSMIManipulator::SetBinNumber(int numS, int numT)
{
  this->BinNumber[0] = numS;
  this->BinNumber[1] = numT;
  this->BinWidth[0] = (double)this->MaxIntensities[0] / ((double)this->BinNumber[0] - 1.0);
  this->BinWidth[1] = (double)this->MaxIntensities[1] / ((double)this->BinNumber[1] - 1.0);
  this->HistS = new long[numS];
  this->HistT = new long[numT];
  this->HistST = new long[numS*numT];
}

//----------------------------------------------------------------------------
void vtkImageSMIManipulator::SetMaxIntensities(int maxS, int maxT)
{
  this->MaxIntensities[0] = maxS;
  this->MaxIntensities[1] = maxT;
  this->BinWidth[0] = (double)this->MaxIntensities[0] / ((double)this->BinNumber[0] - 1.0);
  this->BinWidth[1] = (double)this->MaxIntensities[1] / ((double)this->BinNumber[1] - 1.0);
}

//----------------------------------------------------------------------------
template <class T>
void vtkImageSMIManipulatorEntropyT(vtkImageSMIManipulator *self,
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
           b = vtkResliceRound(*inPtr/double(self->BinWidth[1]));

        if (b < 0) b = 0;
        if (b >= self->BinNumber[1]) b = self->BinNumber[1] - 1;

        self->HistT[b]++;
        inPtr++;
      }
    inPtr += inc2[0];
  }
      inPtr += inc2[1];
    }

  for (int i = 0; i < self->BinNumber[1]; i++)
    {  
      temp = (double)self->HistT[i];
      if (temp > 0.0) entropyT += temp * log(temp);
    }

  self->entropyT = -entropyT / count + log(count);

}

//----------------------------------------------------------------------------
void vtkImageSMIManipulator::SetExtent(int ext[6])
{
  void *inPtr;

  this->inc2[0] = this->inc[1] - this->inc[0] * (ext[1] - ext[0] + 1);
  this->inc2[1] = this->inc[2] - this->inc[1] * (ext[3] - ext[2] + 1);

  this->inPtr[0] = this->inData[0]->GetScalarPointerForExtent(ext);
  this->inPtr[1] = this->inData[1]->GetScalarPointerForExtent(ext);
  inPtr = this->inData[1]->GetScalarPointerForExtent(ext);

  this->count = (ext[1]-ext[0]+1) * (ext[3]-ext[2]+1) * (ext[5]-ext[4]+1);
  if (this->count == 0) vtkErrorMacro(<< "GetResult: No data to work with.");

  memcpy(this->Extent, ext, sizeof(int)*6);

  memset((void *)this->HistT, 0, this->BinNumber[1]*sizeof(long));
  // Calculate the entropy of image 2
  switch (this->inData[1]->GetScalarType())
    {
      vtkTemplateMacro4(vtkImageSMIManipulatorEntropyT,this, 
      (VTK_TT *)(inPtr), this->inc2, this->count);
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
    }

}

//----------------------------------------------------------------------------
void vtkImageSMIManipulator::SetTranslation(double tran[3])
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
    if (tran[i] == 0.0) this->loc111[i] = this->loc000[i];
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
void vtkImageSMIManipulatorExecute(vtkImageSMIManipulator *self,
           T  *in1Ptr, T *in2Ptr,
           int inc[3], int inc2[2], int inExt[6],
           int loc000[3], int loc111[3], double count)
{
  double V000, V100, V010, V110, V001, V101, V011, V111, Vxyz;
  double temp, entropyS = 0, entropyST = 0;
  short a, b;
  int i, j;
 
  // CASE 1: Check if translation takes us out of the input image, in 
  // which case set the result to indicate complete disimilarity and stop.
  if ( (self->Extent[0] + loc000[0] < inExt[0]) ||
       (self->Extent[2] + loc000[1] < inExt[2]) ||
       (self->Extent[4] + loc000[2] < inExt[4]) ||
       (self->Extent[1] + loc111[0] > inExt[1]) ||
       (self->Extent[3] + loc111[1] > inExt[3]) ||
       (self->Extent[5] + loc111[2] > inExt[5]) )
    {
      if (self->Metric == 0) self->Result = 0.5; // Lowest NMI
      if (self->Metric == 1) self->Result = 0.0; // Lowest MI
      if (self->Metric == 2) self->Result = 1.0; // Highest ECR
      return;
    }

  // CASE 2: Check if no interpolation is neccessary.
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
      a = vtkResliceRound(*in1Ptr/double(self->BinWidth[0]));
      b = vtkResliceRound(*in2Ptr/double(self->BinWidth[1]));

      if (a < 0) a = 0;
      if (b < 0) b = 0;
      if (a >= self->BinNumber[0]) a = self->BinNumber[0] - 1;
      if (b >= self->BinNumber[1]) b = self->BinNumber[1] - 1;
      
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
        V000 = *in1Ptr;
        in1Ptr += inc[1];
        V010 = *in1Ptr;
        in1Ptr += inc[2];
        V011 = *in1Ptr;
        in1Ptr -= inc[1];
        V001 = *in1Ptr;
        in1Ptr -= inc[2];
        
        for (int idX = self->Extent[0]; idX <= self->Extent[1]; idX++)
    {
      in1Ptr++;
      V100 = *in1Ptr;
      in1Ptr += inc[1];
      V110 = *in1Ptr;
      in1Ptr += inc[2];
      V111 = *in1Ptr;
      in1Ptr -= inc[1];
      V101 = *in1Ptr;
      in1Ptr -= inc[2];
      
      Vxyz = (V000 * self->F000 + V100 * self->F100 + 
        V010 * self->F010 + V110 * self->F110 + 
        V001 * self->F001 + V101 * self->F101 + 
        V011 * self->F011 + V111 * self->F111);
        
      V000 = V100;
      V010 = V110;
      V001 = V101;
      V011 = V111;  

      a = vtkResliceRound(Vxyz/double(self->BinWidth[0]));
      b = vtkResliceRound(*in2Ptr/double(self->BinWidth[1]));

      if (a < 0) a = 0;
      if (b < 0) b = 0;
      if (a >= self->BinNumber[0]) a = self->BinNumber[0] - 1;
      if (b >= self->BinNumber[1]) b = self->BinNumber[1] - 1;

      self->HistS[a]++;
      self->HistST[b * self->BinNumber[0] + a]++;

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
      temp = (double)self->HistS[i];
      if (temp > 0.0) entropyS += temp * log(temp);
      for (j = 0; j < self->BinNumber[1]; j++) 
   {
     temp = (double)self->HistST[j * self->BinNumber[0] + i];
     if (temp > 0.0) entropyST += temp * log(temp);
   }
    }

  // Calculate entropies and the SMI result based on the Metric flag.
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

  // Mutual Informaiton
  else if (self->Metric == 1)
    {
      self->Result = entropyS + self->entropyT - entropyST;
    }

  // Entropy Correlaiton Coefficient
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
double vtkImageSMIManipulator::GetResult()
{
  // Check inputs.
  if (this->inData[0] == NULL)
    {
      vtkErrorMacro(<< "Input " << 0 << " must be specified.");
      return 0;
    }
  if (this->inData[1] == NULL)
    {
      vtkErrorMacro(<< "Input " << 1 << " must be specified.");
      return 0;
    }
  if ((this->inData[0]->GetScalarType() != this->inData[1]->GetScalarType()))
    {
      vtkErrorMacro(<< "Execute: Inputs must be of the same ScalarType");
      return 0;
    }

  // Zero the histograms and result.
  memset((void *)this->HistS, 0, this->BinNumber[0]*sizeof(long));
  memset((void *)this->HistST, 0, this->BinNumber[0]*this->BinNumber[1]*sizeof(long));
  this->Result = 0;

  // Calculate and return SMI result.
  switch (this->inData[0]->GetScalarType())
    {
      vtkTemplateMacro9(vtkImageSMIManipulatorExecute,this, 
             (VTK_TT *)(this->inPtr[0]), (VTK_TT *)(this->inPtr[1]),
      this->inc, this->inc2, this->inExt,
      this->loc000, this->loc111, this->count);
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
    }
  return this->Result;

}

//----------------------------------------------------------------------------
void vtkImageSMIManipulator::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Input 1: "    << this->inData[0]   << "\n";
  os << indent << "Input 2: "    << this->inData[1]   << "\n";
  os << indent << "BinWidth: ( " << this->BinWidth[0] << ", " << this->BinWidth[1]  << " )\n";
  os << indent << "BinNumber: ( "<< this->BinNumber[0]<< ", " << this->BinNumber[1] << " )\n";
  os << indent << "Extent: "     << this->Extent      << "\n";
  os << indent << "Result: "     << this->Result      << "\n";

}
