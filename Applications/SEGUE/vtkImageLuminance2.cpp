/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageLuminance2.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageLuminance2.h"

#include "vtkImageData.h"
#include "vtkImageProgressIterator.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <math.h>

vtkStandardNewMacro(vtkImageLuminance2);

//----------------------------------------------------------------------------
vtkImageLuminance2::vtkImageLuminance2()
{
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);
Coef[0]=0.30 ;
Coef[1]=0.59 ;
Coef[2]=0.11 ;

}

//----------------------------------------------------------------------------
// This method overrides information set by parent's ExecuteInformation.
int vtkImageLuminance2::RequestInformation (
  vtkInformation       * vtkNotUsed( request ),
  vtkInformationVector** vtkNotUsed( inputVector ),
  vtkInformationVector * outputVector)
{
  vtkDataObject::SetPointDataActiveScalarInfo(
    outputVector->GetInformationObject(0), -1, 1);
  return 1;
}

//----------------------------------------------------------------------------
// This execute method handles boundaries.
// it handles boundaries. Pixels are just replicated to get values 
// out of extent.
template <class T>
void vtkImageLuminanceExecute(vtkImageLuminance2 *self, vtkImageData *inData,
                              vtkImageData *outData,
                              int outExt[6], int id, T *)
{

	double c[3];
	self->GetCoef(c);
  vtkImageIterator<T> inIt(inData, outExt);
  vtkImageProgressIterator<T> outIt(outData, outExt, self, id);
  float luminance;

  // Loop through ouput pixels
  while (!outIt.IsAtEnd())
    {
    T* inSI = inIt.BeginSpan();
    T* outSI = outIt.BeginSpan();
    T* outSIEnd = outIt.EndSpan();
    while (outSI != outSIEnd)
      {
      // now process the components
      luminance =  c[0] * *inSI++;
      luminance += c[1] * *inSI++;
      luminance += c[2]* *inSI++;
      *outSI = static_cast<T>(luminance);
      ++outSI;
      }
    inIt.NextSpan();
    outIt.NextSpan();
    }
}


//----------------------------------------------------------------------------
// This method contains a switch statement that calls the correct
// templated function for the input data type.  The output data
// must match input type.  This method does handle boundary conditions.
void vtkImageLuminance2::ThreadedExecute (vtkImageData *inData, 
                                        vtkImageData *outData,
                                        int outExt[6], int id)
{
  vtkDebugMacro(<< "Execute: inData = " << inData 
  << ", outData = " << outData);
  
  // this filter expects that input is the same type as output.
  if (inData->GetNumberOfScalarComponents() != 3)
    {
    vtkErrorMacro(<< "Execute: input must have 3 components, but has " 
                  << inData->GetNumberOfScalarComponents());
    return;
    }
  
  // this filter expects that input is the same type as output.
  if (inData->GetScalarType() != outData->GetScalarType())
    {
    vtkErrorMacro(<< "Execute: input ScalarType, " << inData->GetScalarType()
    << ", must match out ScalarType " << outData->GetScalarType());
    return;
    }
  
  switch (inData->GetScalarType())
    {
    vtkTemplateMacro(
      vtkImageLuminanceExecute( this, inData, outData, 
                                outExt, id, static_cast<VTK_TT *>(0)));
    default:
      vtkErrorMacro(<< "Execute: Unknown ScalarType");
      return;
    }
}










