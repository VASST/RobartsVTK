/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageShannonMutualInformation.cxx,v $
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
#include "vtkImageShannonMutualInformation.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

#if (VTK_MAJOR_VERSION <= 5)
vtkCxxRevisionMacro(vtkImageShannonMutualInformation, "$Revision: 1.1 $");
#endif
vtkStandardNewMacro(vtkImageShannonMutualInformation);

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
  if (f < 0.0) f = 0.0;
  if (f > 1.0) f = 1.0;
  return ix;
}

//----------------------------------------------------------------------------
vtkImageShannonMutualInformation::vtkImageShannonMutualInformation()
{
  this->BinNumber[0] = 256;
  this->BinNumber[1] = 256;
  this->BinWidth[0] = 1.0;
  this->BinWidth[1] = 1.0;
  this->MaxIntensities[0] = 4095;
  this->MaxIntensities[1] = 4095;
  this->Metric = 0;
  this->ReverseStencil = 0;

  this->Threader = vtkMultiThreader::New();
  this->NumberOfThreads = THREAD_NUM;
  this->Threader->SetNumberOfThreads(THREAD_NUM);
  SetNumberOfThreads(THREAD_NUM);
}

//----------------------------------------------------------------------------
vtkImageShannonMutualInformation::~vtkImageShannonMutualInformation()
{
  this->Threader->Delete();
}

//----------------------------------------------------------------------------
void vtkImageShannonMutualInformation::SetInput1(vtkImageData *input)
{
  this->vtkImageMultipleInputFilter::SetNthInput(0,input);
}

//----------------------------------------------------------------------------
void vtkImageShannonMutualInformation::SetInput2(vtkImageData *input)
{
  this->vtkImageMultipleInputFilter::SetNthInput(1,input);
}

//----------------------------------------------------------------------------
void vtkImageShannonMutualInformation::SetStencil(vtkImageStencilData *stencil)
{
  this->vtkProcessObject::SetNthInput(2, stencil);
}

//----------------------------------------------------------------------------
vtkImageData *vtkImageShannonMutualInformation::GetInput1()
{
  if (this->NumberOfInputs < 1)
    {
    return NULL;
    }

  return (vtkImageData *)(this->Inputs[0]);
}

//----------------------------------------------------------------------------
vtkImageData *vtkImageShannonMutualInformation::GetInput2()
{
  if (this->NumberOfInputs < 2)
    {
    return NULL;
    }

  return (vtkImageData *)(this->Inputs[1]);
}

//----------------------------------------------------------------------------
vtkImageStencilData *vtkImageShannonMutualInformation::GetStencil()
{
  if (this->NumberOfInputs < 3)
    {
    return NULL;
    }
  else
    {
    return (vtkImageStencilData *)(this->Inputs[2]);
    }
}

//----------------------------------------------------------------------------
void vtkImageShannonMutualInformation::SetBinNumber(int numS, int numT)
{
  if ((numS > 256) || (numT > 256)) vtkErrorMacro(<< "Number of bins must be <= 256.");
  this->BinNumber[0] = numS;
  this->BinNumber[1] = numT;
  this->BinWidth[0] = (double)this->MaxIntensities[0] / ((double)this->BinNumber[0] - 1.0);
  this->BinWidth[1] = (double)this->MaxIntensities[1] / ((double)this->BinNumber[1] - 1.0);
}

//----------------------------------------------------------------------------
void vtkImageShannonMutualInformation::SetMaxIntensities(int maxS, int maxT)
{
  this->MaxIntensities[0] = maxS;
  this->MaxIntensities[1] = maxT;
  this->BinWidth[0] = (double)this->MaxIntensities[0] / ((double)this->BinNumber[0] - 1.0);
  this->BinWidth[1] = (double)this->MaxIntensities[1] / ((double)this->BinNumber[1] - 1.0);
}


//----------------------------------------------------------------------------
// Need to add histograms from different threads to create joint
// histogram image.
vtkImageData *vtkImageShannonMutualInformation::GetOutput()
{
  if (this->NumberOfOutputs < 1)
    {
    return NULL;
    }

  int i, j, n = GetNumberOfThreads();
  double temp = 0.0;
  vtkImageData *HistST = (vtkImageData *)(this->Outputs[0]);

  for (i = 0; i < this->BinNumber[0]; i++)
    {
      for (j = 0; j < this->BinNumber[1]; j++)
  {
    temp = 0;
    for (int id = 0; id < n; id++)
      {
        temp += (double)this->ThreadHistST[id][i][j];
      }
    HistST->SetScalarComponentFromFloat(i,j,0,0,temp);
  }
    }

  return (vtkImageData *)(this->Outputs[0]);
}

//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
// Handles the two input operations
template <class T>
void vtkImageShannonMutualInformationExecute(vtkImageShannonMutualInformation *self,
            vtkImageData *in1Data, T *in1Ptr,
            vtkImageData *in2Data, T *in2Ptr,
            int inExt[6], int id)
{
  int i,j,a,b;
  int idX, idY, idZ;
  int incX, incY, incZ;
  int maxX, maxY, maxZ;
  int pminX, pmaxX, iter;
  T *temp1Ptr, *temp2Ptr;
  vtkImageStencilData *stencil = self->GetStencil();

  for (i = 0; i < self->BinNumber[0]; i++)
    {
      self->ThreadHistS[id][i] = 0;
      self->ThreadHistT[id][i] = 0;
      for (j = 0; j < self->BinNumber[1]; j++)
   {
     self->ThreadHistST[id][i][j] = 0;
   }
    }

  // Find the region to loop over
  maxX = inExt[1] - inExt[0];
  maxY = inExt[3] - inExt[2];
  maxZ = inExt[5] - inExt[4];

  // Get increments to march through data
  in1Data->GetIncrements(incX, incY, incZ);

  // Loop over data within stencil sub-extents
  for (idZ = 0; idZ <= maxZ; idZ++)
    {
     for (idY = 0; idY <= maxY; idY++)
       {
   // Flag that we want the complementary extents
   iter = 0; if (self->GetReverseStencil()) iter = -1;

   pminX = 0; pmaxX = maxX;
   while ((stencil !=0 &&
     stencil->GetNextExtent(pminX, pmaxX, 0, maxX, idY, idZ+inExt[4], iter)) ||
    (stencil == 0 && iter++ == 0))
     {
       // Set up pointers to the sub-extents
       temp1Ptr = in1Ptr + (incZ * idZ + incY * idY + pminX);
       temp2Ptr = in2Ptr + (incZ * idZ + incY * idY + pminX);
       // Compute over the sub-extent
       for (idX = pminX; idX <= pmaxX; idX++)
         {
     a = vtkResliceRound(*temp1Ptr/double(self->BinWidth[0]));
     b = vtkResliceRound(*temp2Ptr/double(self->BinWidth[1]));

     if (a < 0)
       {
         a = 0;
       }
     if (a >= self->BinNumber[0])
       {
         a = self->BinNumber[0] - 1;
       }
     if (b < 0)
       {
         b = 0;
       }
     if (b >= self->BinNumber[1])
       {
         b = self->BinNumber[1] - 1;
       }

     self->ThreadHistS[id][a]++;
     self->ThreadHistT[id][b]++;
     self->ThreadHistST[id][a][b]++;
     temp1Ptr++;
     temp2Ptr++;
         }
     }
       }
    }
}

//----------------------------------------------------------------------------
// This method is passed a input and output datas, and executes the filter
// algorithm to fill the output from the inputs.
// It just executes a switch statement to call the correct function for
// the datas data types.
void vtkImageShannonMutualInformation::ThreadedExecute1(vtkImageData **inData,
                 vtkImageData *vtkNotUsed(outData),
                 int inExt[6], int id)
{
  void *inPtr1;
  void *inPtr2;
  int ext0[6], ext1[6];

  if (inData[0] == NULL)
    {
    vtkErrorMacro(<< "Input 0 must be specified.");
    return;
    }

  if (inData[1] == NULL)
    {
      vtkErrorMacro(<< "Input 1 must be specified.");
      return;
    }

  inData[0]->GetWholeExtent(ext0);
  inData[1]->GetWholeExtent(ext1);

  if ((ext0[0] - ext1[0]) | (ext0[1] - ext1[1]) | (ext0[2] - ext1[2]) |
      (ext0[3] - ext1[3]) | (ext0[4] - ext1[4]) | (ext0[5] - ext1[5]))
    {
    vtkErrorMacro(<<"Inputs 0 and 1 must have the same extents.");
    return;
    }

  if ((inData[0]->GetNumberOfScalarComponents() > 1) |
      (inData[1]->GetNumberOfScalarComponents() > 1))
    {
      vtkErrorMacro("Inputs 0 and 1 must have 1 component each.");
      return;
    }

  if (inData[0]->GetScalarType() != inData[1]->GetScalarType())
    {
      vtkErrorMacro(<< "Inputs 0 and 1 must be of the same type");
      return;
    }

  // GetScalarPointer() for inData doesn't give the right results.
  inPtr1 = inData[0]->GetScalarPointerForExtent(inExt);
  inPtr2 = inData[1]->GetScalarPointerForExtent(inExt);

  switch (inData[0]->GetScalarType())
    {
      vtkTemplateMacro7(vtkImageShannonMutualInformationExecute,this,
      inData[0], (VTK_TT *)(inPtr1),
      inData[1], (VTK_TT *)(inPtr2),
      inExt, id);
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return;
    }
}

//----------------------------------------------------------------------------
void vtkImageShannonMutualInformation::ThreadedExecute2(int extS[6], int extST[6], int id)
{
  int i, j, n, N = GetNumberOfThreads();
  double temp1 = 0.0, temp2 = 0.0;

  this->ThreadEntropyS[id] = 0.0;
  this->ThreadEntropyT[id] = 0.0;
  this->ThreadEntropyST[id] = 0.0;
  this->ThreadCount[id] = 0.0;

  // Loop over S image histogram.
  for (i = extS[0]; i <= extS[1]; i++)
    {
      temp1 = 0.0;
      for (n = 0; n < N; n++)
  {
    temp1 += (double)this->ThreadHistS[n][i];
  }
      if (temp1 > 0.0) this->ThreadEntropyS[id] += temp1 * log(temp1);
    }

  // Loop over T and ST histograms.
  for (j = extST[2]; j <= extST[3]; j++)
    {
      temp2 = 0.0;
      for (n = 0; n < N; n++)
  {
    temp2 += (double)this->ThreadHistT[n][j];
  }
      if (temp2 > 0.0) this->ThreadEntropyT[id] += temp2 * log(temp2);

      for (i = extST[0]; i <= extST[1]; i++)
  {
     temp1 = 0.0;
     for (n = 0; n < N; n++)
       {
         temp1 += (double)this->ThreadHistST[n][i][j];
         this->ThreadCount[id] += (double)this->ThreadHistST[n][i][j];
       }
    if (temp1 > 0.0) this->ThreadEntropyST[id] += temp1 * log(temp1);
  }
    }
}

//----------------------------------------------------------------------------
double vtkImageShannonMutualInformation::GetResult()
{
  int id, n = GetNumberOfThreads();
  double entropyS = 0, entropyT = 0, entropyST = 0, result = 0;
  double count = 0.0;

  for (id = 0; id < n; id++)
    {
      entropyS += this->ThreadEntropyS[id];
      entropyT += this->ThreadEntropyT[id];
      entropyST += this->ThreadEntropyST[id];
      count += this->ThreadCount[id];
    }

  if (count == 0) vtkErrorMacro( << "GetResult: No data to work with.")

  entropyS  = -entropyS /count + log(count);
  entropyT  = -entropyT /count + log(count);
  entropyST = -entropyST/count + log(count);

  if ((entropyS  < 1E-10) && (entropyS  > -1E-10)) entropyS  = 0.0;
  if ((entropyT  < 1E-10) && (entropyT  > -1E-10)) entropyT  = 0.0;
  if ((entropyST < 1E-10) && (entropyST > -1E-10)) entropyST = 0.0;

  if ((entropyS < 0) || (entropyT < 0) || (entropyST < 0))
    {
      vtkErrorMacro(<< "GetResult: Entropy < 0");
    }

  // Normalized Mutual Information
  if (this->Metric == 0)
    {
      if (entropyST == 0)
  {
    result = 1.0;
  }
      else
  {
    result = (entropyS + entropyT)/entropyST/2.0;
  }
    }

  // Mutual Informaiton
  else if (this->Metric == 1)
    {
      result = entropyS + entropyT - entropyST;
    }

  // Entropy Correlaiton Ratio
  else if (this->Metric == 2)
    {
      if (entropyS + entropyT == 0)
  {
    result = 1.0;
  }
      else
  {
    result = sqrt ( 2.0 * (1.0 - entropyST / ( entropyT + entropyS ) ) );
  }
    }

  // Error
  else
    {
      cout << "ERROR: Wrong Metric chosen\n";
      exit(0);
    }

  return result;

}

//----------------------------------------------------------------------------
// Get ALL of the input.
void vtkImageShannonMutualInformation::ComputeInputUpdateExtent(int inExt[6],
                   int outExt[6],
                   int vtkNotUsed(whichInput))
{
  int *wholeExtent = this->GetInput()->GetWholeExtent();
  memcpy(inExt, wholeExtent, 6*sizeof(int));
}

//----------------------------------------------------------------------------
struct vtkImageMultiThreadStruct
{
  vtkImageShannonMutualInformation *Filter;
  vtkImageData   **Inputs;
  vtkImageData   *Output;
};

//----------------------------------------------------------------------------
// this mess is really a simple function. All it does is call
// the ThreadedExecute method after setting the correct
// extent for this thread. Its just a pain to calculate
// the correct extent.
VTK_THREAD_RETURN_TYPE vtkImageShannonMutualInformationMultiThreadedExecute1( void *arg )
{
  vtkImageMultiThreadStruct *str;
  int ext[6], splitExt[6], total;
  int threadId, threadCount;

  threadId = ((ThreadInfoStruct *)(arg))->ThreadID;
  threadCount = ((ThreadInfoStruct *)(arg))->NumberOfThreads;

  str = (vtkImageMultiThreadStruct *)(((ThreadInfoStruct *)(arg))->UserData);

  // Thread over input images
  memcpy(ext,str->Filter->GetInput()->GetUpdateExtent(),
         sizeof(int)*6);

  // execute the actual method with appropriate extent
  // first find out how many pieces extent can be split into.
  total = str->Filter->SplitExtent(splitExt, ext, threadId, threadCount);

  if (threadId < total)
    {
    str->Filter->ThreadedExecute1(str->Inputs, str->Output, splitExt, threadId);
    }
  // else
  //   {
  //   otherwise don't use this thread. Sometimes the threads dont
  //   break up very well and it is just as efficient to leave a
  //   few threads idle.
  //   }

  return VTK_THREAD_RETURN_VALUE;
}

//----------------------------------------------------------------------------
// this mess is really a simple function. All it does is call
// the ThreadedExecute method after setting the correct
// extent for this thread. Its just a pain to calculate
// the correct extent.
VTK_THREAD_RETURN_TYPE vtkImageShannonMutualInformationMultiThreadedExecute2( void *arg )
{
  vtkImageMultiThreadStruct *str;
  int ext[6], splitExtS[6], splitExtST[6], total;
  int threadId, threadCount;

  threadId = ((ThreadInfoStruct *)(arg))->ThreadID;
  threadCount = ((ThreadInfoStruct *)(arg))->NumberOfThreads;

  str = (vtkImageMultiThreadStruct *)(((ThreadInfoStruct *)(arg))->UserData);

  // Thread over S image histogram.
  ext[0] = 0; ext[1] = str->Filter->BinNumber[0] - 1;
  ext[2] = 0; ext[3] = 0;
  ext[4] = 0; ext[5] = 0;

  total = str->Filter->SplitExtent(splitExtS, ext, threadId, threadCount);

  // Thread over joint and T image histograms.
  ext[0] = 0; ext[1] = str->Filter->BinNumber[0] - 1;
  ext[2] = 0; ext[3] = str->Filter->BinNumber[1] - 1;
  ext[4] = 0; ext[5] = 0;

  // execute the actual method with appropriate extent
  // first find out how many pieces extent can be split into.
  total = str->Filter->SplitExtent(splitExtST, ext, threadId, threadCount);

  if (threadId < total)
    {
    str->Filter->ThreadedExecute2(splitExtS, splitExtST, threadId);
    }
  // else
  //   {
  //   otherwise don't use this thread. Sometimes the threads dont
  //   break up very well and it is just as efficient to leave a
  //   few threads idle.
  //   }

  return VTK_THREAD_RETURN_VALUE;
}


//----------------------------------------------------------------------------
void vtkImageShannonMutualInformation::ExecuteData(vtkDataObject *out)
{
  vtkImageData *output = vtkImageData::SafeDownCast(out);
  if (!output)
    {
    vtkWarningMacro("ExecuteData called without ImageData output");
    return;
    }
  output->SetExtent(output->GetUpdateExtent());
  output->AllocateScalars();

  vtkImageMultiThreadStruct str;

  str.Filter = this;
  str.Inputs = (vtkImageData **)this->GetInputs();
  str.Output = output;

  this->Threader->SetNumberOfThreads(this->NumberOfThreads);

  // setup threading and the invoke threadedExecute
  this->Threader->SetSingleMethod(vtkImageShannonMutualInformationMultiThreadedExecute1, &str);
  this->Threader->SingleMethodExecute();

  this->Threader->SetSingleMethod(vtkImageShannonMutualInformationMultiThreadedExecute2, &str);
  this->Threader->SingleMethodExecute();
}

//----------------------------------------------------------------------------
void vtkImageShannonMutualInformation::ExecuteInformation(vtkImageData **inData,
                   vtkImageData *outData)
{
  // the two inputs are required to be of the same data type and extents.
  inData[0]->Update();
  inData[1]->Update();

  outData->SetWholeExtent(0, this->BinNumber[0]-1, 0, this->BinNumber[1]-1, 0, 0);
  outData->SetOrigin(0.0,0.0,0.0);
  outData->SetSpacing(this->BinWidth[0],this->BinWidth[1],1.0);
  outData->SetNumberOfScalarComponents(1);
  outData->SetScalarType(VTK_INT);

  // need to set the spacing and origin of the stencil to match the output
  vtkImageStencilData *stencil = this->GetStencil();
  if (stencil)
    {
    stencil->SetSpacing(inData[0]->GetSpacing());
    stencil->SetOrigin(inData[0]->GetOrigin());
    }
}

//----------------------------------------------------------------------------
void vtkImageShannonMutualInformation::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Stencil: " << this->GetStencil() << "\n";
  os << indent << "ReverseStencil: " << (this->ReverseStencil ? "On\n" : "Off\n");
  os << indent << "BinWidth: ( " << this->BinWidth[0] << ", " << this->BinWidth[1] << " )\n";
  os << indent << "BinNumber: ( "<<this->BinNumber[0] << ", " << this->BinNumber[1] << " )\n";
}
