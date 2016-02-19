/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageNormalizedCrossCorrelation.cxx,v $
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
#include "vtkImageNormalizedCrossCorrelation.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

#if (VTK_MAJOR_VERSION >= 6)
#include <vtkExecutive.h>
#endif

#if (VTK_MAJOR_VERSION < 6)
vtkCxxRevisionMacro(vtkImageNormalizedCrossCorrelation, "$Revision: 1.1 $");
#endif
vtkStandardNewMacro(vtkImageNormalizedCrossCorrelation);

//----------------------------------------------------------------------------
vtkImageNormalizedCrossCorrelation::vtkImageNormalizedCrossCorrelation()
{
  this->ReverseStencil = 0;
  SetNumberOfThreads(THREAD_NUM);
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION < 6)
void vtkImageNormalizedCrossCorrelation::SetInput1(vtkImageData *input)
{
  this->vtkImageMultipleInputFilter::SetNthInput(0,input);
}
#else
void vtkImageNormalizedCrossCorrelation::SetInput1Data(vtkImageData *input)
{
  this->vtkImageAlgorithm::SetInputData(0,input);
}
#endif

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION < 6)
void vtkImageNormalizedCrossCorrelation::SetInput2(vtkImageData *input)
{
  this->vtkImageMultipleInputFilter::SetNthInput(1,input);
}
#else
void vtkImageNormalizedCrossCorrelation::SetInput2Data(vtkImageData *input)
{
  this->vtkImageAlgorithm::SetInputData(1,input);
}
#endif

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION < 6)
void vtkImageNormalizedCrossCorrelation::SetStencil(vtkImageStencilData *stencil)
{
  this->vtkProcessObject::SetNthInput(2, stencil);
}
#else
void vtkImageNormalizedCrossCorrelation::SetStencilData(vtkImageStencilData *stencil)
{
  this->vtkImageAlgorithm::SetInputData(2, stencil);
}
#endif

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION < 6)
vtkImageData *vtkImageNormalizedCrossCorrelation::GetInput1()
{
  if (this->NumberOfInputs < 1)
  {
    return NULL;
  }
  else
  {
    return (vtkImageData *)(this->Inputs[0]);
  }
}
#else
vtkImageData *vtkImageNormalizedCrossCorrelation::GetInput1()
{
  if (this->GetNumberOfInputConnections(0) < 1)
  {
    return NULL;
  }

  return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, 0) );
}
#endif

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION < 6)
vtkImageData *vtkImageNormalizedCrossCorrelation::GetInput2()
{
  if (this->NumberOfInputs < 2)
  {
    return NULL;
  }
  else
  {
    return (vtkImageData *)(this->Inputs[1]);
  }
}
#else
vtkImageData *vtkImageNormalizedCrossCorrelation::GetInput2()
{
  if (this->GetNumberOfInputConnections(1) < 1)
  {
    return NULL;
  }

  return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(1, 0) );
}
#endif

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION < 6)
vtkImageStencilData *vtkImageNormalizedCrossCorrelation::GetStencil()
{
  if (this->NumberOfInputs < 3)
  {
    return NULL;
  }

  return (vtkImageStencilData *)(this->Inputs[2]);
}
#else
vtkImageStencilData *vtkImageNormalizedCrossCorrelation::GetStencil()
{
  if (this->GetNumberOfInputConnections(2) < 1)
  {
    return NULL;
  }

  return vtkImageStencilData::SafeDownCast( this->GetExecutive()->GetInputData(2, 0) );
}
#endif

//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
// Handles the two input operations
template <class T>
void vtkImageNormalizedCrossCorrelationExecute(vtkImageNormalizedCrossCorrelation *self,
    vtkImageData *in1Data, T *in1Ptr,
    vtkImageData *in2Data, T *in2Ptr,
    int outExt[6], int id)
{
  int idX, idY, idZ;
  vtkIdType incX, incY, incZ;
  int maxX, maxY, maxZ;
  int pminX, pmaxX, iter;
  T *temp1Ptr, *temp2Ptr;
  vtkImageStencilData *stencil = self->GetStencil();

  self->ThreadSumST[id] = 0.0;
  self->ThreadSumS[id] = 0.0;
  self->ThreadSumT[id] = 0.0;

  // Find the region to loop over
  maxX = (outExt[1] - outExt[0])*in1Data->GetNumberOfScalarComponents();
  maxY = outExt[3] - outExt[2];
  maxZ = outExt[5] - outExt[4];

  // Get increments to march through data
  in1Data->GetIncrements(incX, incY, incZ);

  // Loop over data within stencil sub-extents
  for (idZ = 0; idZ <= maxZ; idZ++)
  {
    for (idY = 0; idY <= maxY; idY++)
    {
      // Flag that we want the complementary extents
      iter = 0;
      if (self->GetReverseStencil())
      {
        iter = -1;
      }

      pminX = 0;
      pmaxX = maxX;
      while ((stencil !=0 &&
              stencil->GetNextExtent(pminX, pmaxX, 0, maxX, idY, idZ+outExt[4], iter)) ||
             (stencil == 0 && iter++ == 0))
      {
        // Set up pointers to the sub-extents
        temp1Ptr = in1Ptr + (incZ * idZ + incY * idY + pminX);
        temp2Ptr = in2Ptr + (incZ * idZ + incY * idY + pminX);
        // Compute over the sub-extent
        for (idX = pminX; idX <= pmaxX; idX++)
        {
          self->ThreadSumST[id] += (double)*temp1Ptr * (double)*temp2Ptr;
          self->ThreadSumS[id] += (double)*temp1Ptr * (double)*temp1Ptr;
          self->ThreadSumT[id] += (double)*temp2Ptr * (double)*temp2Ptr;
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
void vtkImageNormalizedCrossCorrelation::ThreadedExecute(vtkImageData **inData,
    vtkImageData *vtkNotUsed(outData),
    int outExt[6], int id)
{
  void *inPtr1;
  void *inPtr2;

  vtkDebugMacro( "Execute: inData = " << inData);

  if (inData[0] == NULL)
  {
    vtkErrorMacro( "Input " << 0 << " must be specified.");
    return;
  }

  if (inData[1] == NULL)
  {
    vtkErrorMacro( "Input " << 1 << " must be specified.");
    return;
  }

  if ((inData[0]->GetScalarType() != inData[1]->GetScalarType()))
  {
    vtkErrorMacro( "Execute: Inputs must be of the same ScalarType");
    return;
  }

  inPtr1 = inData[0]->GetScalarPointerForExtent(outExt);
  inPtr2 = inData[1]->GetScalarPointerForExtent(outExt);

  // this filter expects that inputs that have the same number of components
  if ((inData[0]->GetNumberOfScalarComponents() != inData[1]->GetNumberOfScalarComponents()))
  {
    vtkErrorMacro( "Execute: input1 NumberOfScalarComponents, "
                  << inData[0]->GetNumberOfScalarComponents()
                  << ", must match input2 NumberOfScalarComponents "
                  << inData[1]->GetNumberOfScalarComponents());
    return;
  }

  switch (inData[0]->GetScalarType())
  {
#if (VTK_MAJOR_VERSION < 5)
    vtkTemplateMacro7(vtkImageNormalizedCrossCorrelationExecute,this,
                      inData[0], (VTK_TT *)(inPtr1),
                      inData[1], (VTK_TT *)(inPtr2),
                      outExt, id);
#else
    vtkTemplateMacro(vtkImageNormalizedCrossCorrelationExecute(this,
                     inData[0], (VTK_TT *)(inPtr1),
                     inData[1], (VTK_TT *)(inPtr2),
                     outExt, id));
#endif
  default:
    vtkErrorMacro( "Execute: Unknown ScalarType");
    return;
  }
}

//----------------------------------------------------------------------------
double vtkImageNormalizedCrossCorrelation::GetResult()
{
  int n = GetNumberOfThreads();
  double sumST = 0.0;
  double sumS = 0.0;
  double sumT = 0.0;
  double result;

  for (int id = 0; id < n; id++)
  {
    sumST += this->ThreadSumST[id];
    sumS += this->ThreadSumS[id];
    sumT += this->ThreadSumT[id];
  }

  result = sumST / (sqrt(sumS) * sqrt(sumT));

  if (result < 0)
  {
    vtkErrorMacro( "GetResult: result < 0");
  }

  return result;
}

//----------------------------------------------------------------------------
void vtkImageNormalizedCrossCorrelation::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Input 1: "<< this->GetInput1() << "\n";
  os << indent << "Input 2: "<< this->GetInput2() << "\n";
  os << indent << "Stencil: " << this->GetStencil() << "\n";
  os << indent << "ReverseStencil: " << (this->ReverseStencil ? "On\n" : "Off\n");
}
