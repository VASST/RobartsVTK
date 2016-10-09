/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    $RCSfile: vtkShapeBasedInterpolation.h,v $
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
// .NAME vtkShapeBasedInterpolation - Create an image filled with noise.
// .SECTION Description
// vtkShapeBasedInterpolation just produces images filled with noise.  The only
// option now is uniform noise specified by a min and a max.  There is one
// major problem with this source. Every time it executes, it will output
// different pixel values.  This has important implications when a stream
// requests overlapping regions.  The same pixels will have different values
// on different updates.


#ifndef __vtkShapeBasedInterpolation_h
#define __vtkShapeBasedInterpolation_h

#include "vtkRobartsRegistrationModule.h"

#include "vtkImageAlgorithm.h"

class VTKROBARTSREGISTRATION_EXPORT vtkShapeBasedInterpolation : public vtkImageAlgorithm 
{
public:
  static vtkShapeBasedInterpolation *New();
  vtkTypeMacro(vtkShapeBasedInterpolation,vtkImageAlgorithm);

  vtkSetMacro(SliceAxis, int);
  vtkGetMacro(SliceAxis, int);

  vtkSetMacro(NumberOfBins, int);
  vtkGetMacro(NumberOfBins, int);

  vtkSetVector3Macro(OutputSpacing, double);
  vtkGetVector3Macro(OutputSpacing, double);

  void SetInputConnection(vtkAlgorithmOutput *input);
  vtkImageData *GetOutput();
  void Update();

protected:
  vtkShapeBasedInterpolation();
  ~vtkShapeBasedInterpolation();

  int SliceAxis;
  int NumberOfBins;

  vtkImageData *inData;
  int inExt[6];
  double inSpa[3];
  double inOri[3];
  double inMinVal;
  double inMaxVal;

  vtkImageData *outData;
  double OutputSpacing[3];

};


#endif

  
