/*=========================================================================

  Program:   Shapetape for VTK
  Module:    $RCSfile: vtkTrackerToolSensor.h,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2007/04/19 12:48:53 $
  Version:   $Revision: 1.1 $

==========================================================================

Copyright (c)

Use, modification and redistribution of the software, in source or
binary forms, are permitted provided that the following terms and
conditions are met:

1) Redistribution of the source code, in verbatim or modified
   form, must retain the above copyright notice, this license,
   the following disclaimer, and any notices that refer to this
   license and/or the following disclaimer.  

2) Redistribution in binary form must include the above copyright
   notice, a copy of this license and the following disclaimer
   in the documentation or with other materials provided with the
   distribution.

3) Modified copies of the source code must be clearly marked as such,
   and must not be misrepresented as verbatim copies of the source code.

THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

=========================================================================*/
// .NAME vtkTrackerToolSensor - interfaces VTK to a handheld 3D positioning tool
// .SECTION Description
// The vtkTrackerToolSensor provides an interface between a tracked object in
// the real world and a virtual object.
// .SECTION see also
// vtkTracker vtkPOLARISTracker vtkFlockTracker

#ifndef __vtkTrackerToolSensor_h
#define __vtkTrackerToolSensor_h

#define VTK_MAX_SENSORS 150

#include "vtkObject.h"
#include "vtkTrackerSensor.h"
#include "vtkTrackerTool.h"

class vtkMatrix4x4;
class vtkTransform;
class vtkDoubleArray;
class vtkAmoebaMinimizer;
class vtkTrackerBuffer;
class vtkTrackerSensor;

class VTK_EXPORT vtkTrackerToolSensor : public vtkTrackerTool
{
public:

  static vtkTrackerToolSensor *New();
  vtkTypeMacro(vtkTrackerToolSensor,vtkObject);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Get the tracker which owns this tool. 
  //vtkGetObjectMacro(Tracker,vtkTrackerSensor);
  vtkGetMacro(NumberOfSensors, int);

  vtkGetMacro(TapeLength, double);
  vtkSetMacro(TapeLength, double);

  vtkGetMacro(InterpolationInterval, int);
  vtkSetMacro(InterpolationInterval, int);

  // Description:
  // Get a reference to the transform associated with this tool.  The
  // transform will automatically update when Update() is called
  // on the tracking system.  You can connect this transform or its
  // matrix to a vtkActor.
  //vtkGetObjectMacro(Transform,vtkTransform);
  vtkTransform * GetTransform();
  vtkTransform * GetTransform(int num);
  // Description:
  // Get a running list of all the transforms received for this
  // tool.  See the vtkTrackerBuffer class for more information.
 // vtkGetObjectMacro(Buffer,vtkTrackerBuffer);
  void SetCalibrationMatrix(vtkMatrix4x4 *vmat);
  
  void SensorUpdate(int sensor, vtkMatrix4x4 *matrix, long flags, double timestamp);

  void SetTracker(vtkTrackerSensor *tracker);

void SetNumberOfSensors(int number);
vtkTrackerBuffer * GetBuffer(int sensor);


 ~vtkTrackerToolSensor();
 void Update();
protected:
  vtkTrackerToolSensor();

  vtkTransform *Transform[150];
  vtkMatrix4x4 *CalibrationMatrix;

  vtkAmoebaMinimizer *Minimizer;
  vtkDoubleArray *CalibrationArray;
  vtkTrackerBuffer *Buffer[150];
  vtkTrackerSensor * Tracker;

//BTX
  friend void vtkTrackerToolCalibrationFunction(void *userData);
//ETX

private:
  double TapeLength;
  int NumberOfSensors;            // number of sensors
  int InterpolationInterval;        // interpolation steps between sensor boundaries


  vtkTrackerToolSensor(const vtkTrackerToolSensor&);
  void operator=(const vtkTrackerToolSensor&);  

  vtkMatrix4x4 *TempMatrix;
  vtkMatrix4x4 *RawMatrix;
};

#endif







