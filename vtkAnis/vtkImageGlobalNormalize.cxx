/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageGlobalNormalize.cxx,v $
  Language:  C++
  Date:      $Date: 2007/04/26 19:16:45 $
  Version:   $Revision: 1.1 $

  Copyright (c) 1993-2002 Ken Martin, Will Schroeder, Bill Lorensen 
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageGlobalNormalize.h"

#include "vtkImageData.h"
#include "vtkImageProgressIterator.h"
#include "vtkObjectFactory.h"

#include <math.h>

vtkCxxRevisionMacro(vtkImageGlobalNormalize, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkImageGlobalNormalize);

//----------------------------------------------------------------------------
// This method tells the superclass that the first axis will collapse.
void vtkImageGlobalNormalize::ExecuteInformation(vtkImageData *vtkNotUsed(inData), 
                                           vtkImageData *outData)
{
  outData->SetScalarType(VTK_FLOAT);
}

//----------------------------------------------------------------------------
// This execute method handles boundaries.
// it handles boundaries. Pixels are just replicated to get values 
// out of extent.
template <class T>
void vtkImageGlobalNormalizeExecute(vtkImageGlobalNormalize *self,
                              vtkImageData *inData,
                              vtkImageData *outData,
                              int outExt[6], int id, T *)
{
  float min=1e30, max=-1e30;
  vtkImageIterator<T> inIt1(inData, outExt);
  vtkImageIterator<T> inIt2(inData, outExt);
  vtkImageProgressIterator<float> outIt(outData, outExt, self, id);
  
  // Find the maximum intensity
  while (!inIt1.IsAtEnd())
    {
    T* inSI = inIt1.BeginSpan();
    T* inSIEnd = inIt1.EndSpan();
    while (inSI != inSIEnd)
      {
        if(float(*inSI) > max) max = float(*inSI);
        if(float(*inSI) < min) min = float(*inSI);
        inSI++;
      }
    inIt1.NextSpan();
    }

  // Normalize pixels
  while (!outIt.IsAtEnd())
    {
    T* inSI = inIt2.BeginSpan();
    float *outSI = outIt.BeginSpan();
    float *outSIEnd = outIt.EndSpan();
    while (outSI != outSIEnd)
      {
        *outSI++ = (*inSI++ - min) / (max - min);
      }
    inIt2.NextSpan();
    outIt.NextSpan();
    }
}


//----------------------------------------------------------------------------
// This method contains a switch statement that calls the correct
// templated function for the input data type.  The output data
// must match input type.  This method does handle boundary conditions.
void vtkImageGlobalNormalize::ThreadedExecute(vtkImageData *inData, 
                                        vtkImageData *outData,
                                        int outExt[6], int id)
{
  vtkDebugMacro(<< "Execute: inData = " << inData 
  << ", outData = " << outData);
  
  // this filter expects that input is the same type as output.
  if (outData->GetScalarType() != VTK_FLOAT)
    {
    vtkErrorMacro(<< "Execute: output ScalarType, " << outData->GetScalarType()
    << ", must be float");
    return;
    }
  
  switch (inData->GetScalarType())
    {
    vtkTemplateMacro6(vtkImageGlobalNormalizeExecute, this, inData,
                     outData, outExt, id, static_cast<VTK_TT *>(0));
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return;
    }
}












