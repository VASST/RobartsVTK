/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageProbeSum.h,v $
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
// .NAME vtkImageProbeSum - sample data values at specified point locations
// .SECTION Description
// vtkImageProbeSum is a filter that computes point attributes (e.g., scalars,
// vectors, etc.) at specified point positions. The filter has two inputs:
// the Input and Source. The Input geometric structure is passed through the
// filter. The point attributes are computed at the Input point positions
// by interpolating into the source data. For example, we can compute data
// values on a plane (plane specified as Input) from a volume (Source).

#ifndef __vtkImageProbeSum_h
#define __vtkImageProbeSum_h

#include "vtkProcessObject.h"
#include "vtkImageData.h"
#include "vtkIdTypeArray.h"

class VTK_EXPORT vtkImageProbeSum : public vtkProcessObject
{
public:
  static vtkImageProbeSum *New();
  vtkTypeMacro(vtkImageProbeSum,vtkProcessObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set the image data set.
  void SetSource(vtkImageData *Source);
  vtkImageData *GetSource();

  // Description:
  // Specify the point locations used to probe input. Any geometry
  // can be used.
  void SetInput(vtkDataSet *source);
  vtkDataSet *GetInput();

  // Description:
  // Get the sum of all the values interpolated from the image
  double GetSum();
  
protected:
  vtkImageProbeSum();
  ~vtkImageProbeSum();

  double Sum;

private:
  vtkImageProbeSum(const vtkImageProbeSum&);
  void operator=(const vtkImageProbeSum&);
};

#endif
