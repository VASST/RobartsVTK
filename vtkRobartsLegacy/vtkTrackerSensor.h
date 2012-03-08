/*=========================================================================

  Program:   Shapetape for VTK
  Module:    $RCSfile: vtkTrackerSensor.h,v $
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
// .NAME vtkTrackerSensor - interfaces VTK with real-time 3D tracking systems
// .SECTION Description
// The vtkTrackerSensor is a generic VTK interface to real-time tracking
// systems.  Subclasses to this class implement this interface for
// the POLARIS (Northern Digital Inc., Waterloo, Canada), the
// Flock of Birds (Ascension Technology Corporation), and a few
// other systems.
// Derived classes should override the Probe(), InternalUpdate(),
// InternalStartTracking(), and InternalStopTracking() methods.
// The InternalUpdate() method is called from within a separate
// thread, therefore its contents must be thread safe.  Use the
// vtkPOLARISTracker as a framework for developing subclasses
// for new tracking systems.
// .SECTION see also
// vtkTrackerTool vtkPOLARISTracker vtkFlockTracker

#ifndef __vtkTrackerSensor_h
#define __vtkTrackerSensor_h

#include "vtkObject.h"
#include "vtkCriticalSection.h"
#include "vtkTrackerToolSensor.h"

class vtkMatrix4x4;
class vtkMultiThreader;
class vtkTrackerToolSensor;

class VTK_EXPORT vtkTrackerSensor : public vtkObject
{
public:
  static vtkTrackerSensor *New();
  vtkTypeMacro(vtkTrackerSensor,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Probe to see to see if the tracking system is connected to the
  // computer.  Returns 1 if the tracking system was found and is working.
  // Do not call this method while the system is Tracking.  This method
  // should be overridden in subclasses. 
  virtual int Probe();

  // Description:
  // Start the tracking system.  The tracking system is brought from
  // its ground state (i.e. on but not necessarily initialized) into
  // full tracking mode.  This method calls InternalStartTracking()
  // after doing a bit of housekeeping.
  virtual void StartTracking();

  // Description:
  // Stop the tracking system and bring it back to its ground state.
  // This method calls InternalStopTracking().
  virtual void StopTracking();

  // Description:
  // Test whether or not the system is tracking.
  virtual int IsTracking() { return this->Tracking; };

  // Description:
  // This method will sleep until the next transformation is sent
  // by the tracking system (up to a maximum sleep of 0.1 seconds)
  // and then it calls Update() on each of the tools.  Note that
  // this method does not call the InternalUpdate() method, which
  // is called by a separate thread.
  virtual void Update();
  
  // Description:
  // Get the internal update rate for this tracking system.  This is
  // the number of transformations sent by the tracking system per
  // second per tool.
  double GetInternalUpdateRate() { return this->InternalUpdateRate; };

  // Description:
  // Get the tool object for the specified port.  The first tool is
  // retrieved by GetTool(0).  See vtkTrackerSensorTool for more information.
  vtkTrackerToolSensor * GetTool(int i = 0);

  // Description:
  // Get the number of available tool ports.  This is the maxiumum that a
  // particular tracking system can support, not the number of tools
  // that are actually connected to the system.  In order to determine
  // how many tools are connected, you must call Update() and then
  // check IsMissing() for each tool between 0 and NumberOfTools-1.
  vtkGetMacro(NumberOfTools, int);

  // Description:
  // Get the timestamp for the last time that Update() was called, in
  // seconds since 1970 (i.e. the UNIX epoch).  This method is not a
  // good method of getting timestamps for tracking information,
  // you should use the vtkTrackerSensorTool GetTimeStamp() method to get
  // the timestamp associated with each transform.  This method is
  // only valuable for determining e.g. how old the transforms were
  // before the Update method was called.
  vtkGetMacro(UpdateTimeStamp,double);

  // Description:
  // Set one of the ports to be a reference, i.e. track other
  // tools relative to this one.  Set this to -1 (the default)
  // if a reference tool is not desired.
  vtkSetMacro(ReferenceTool, int);
  vtkGetMacro(ReferenceTool, int);

  // Description:
  // Set the transformation matrix between tracking-system coordinates
  // and the desired world coordinate system.  You can use 
  // vtkLandmarkTransform to create this matrix from a set of 
  // registration points.  Warning: the matrix is copied,
  // not referenced.
  void SetWorldCalibrationMatrix(vtkMatrix4x4* vmat);
  vtkMatrix4x4 *GetWorldCalibrationMatrix();

  // Description:
  // Make the unit emit a string of audible beeps.  This is
  // supported by the POLARIS.
  void Beep(int n);
  
  // Description:
  // Turn one of the LEDs on the specified tool on or off.  This
  // is supported by the POLARIS.
  void SetToolLED(int tool, int led, int state);

  // Description:
  // The subclass will do all the hardware-specific update stuff
  // in this function.   It should call ToolUpdate() for each tool.
  // Note that vtkTrackerSensor.cxx starts up a separate thread after
  // InternalStartTracking() is called, and that InternalUpdate() is
  // called repeatedly from within that thread.  Therefore, any code
  // within InternalUpdate() must be thread safe.  You can temporarily
  // pause the thread by locking this->UpdateMutex->Lock() e.g. if you
  // need to communicate with the device from outside of InternalUpdate().
  // A call to this->UpdateMutex->Unlock() will resume the thread.
  virtual void InternalUpdate() {};

//BTX
  // These are used by static functions in vtkTrackerSensor.cxx, and since
  // VTK doesn't generally use 'friend' functions they are public
  // instead of protected.  Do not use them anywhere except inside
  // vtkTrackerSensor.cxx.
  vtkCriticalSection *UpdateMutex;
  vtkTimeStamp UpdateTime;
  double InternalUpdateRate;  
//ETX

protected:
  vtkTrackerSensor();
  ~vtkTrackerSensor();

  // Description:
  // This function is called by InternalUpdate() so that the subclasses
  // can communicate information back to the vtkTrackerSensor base class, which
  // will in turn relay the information to the appropriate vtkTrackerTool.
  void ToolUpdate(int tool, vtkMatrix4x4 *matrix, long flags, double timestamp);

  // Description:
  // Set the number of tools for the tracker -- this method is
  // only called once within the constructor for derived classes.
  void SetNumberOfSensors(int sensors);

  // Description:
  // These methods should be overridden in derived classes: 
  // InternalStartTracking() should initialize the tracking device, and
  // InternalStopTracking() should free all resources associated with
  // the device.  These methods should return 1 if they are successful,
  // or 0 if they are not.
  virtual int InternalStartTracking() { return 1; };
  virtual int InternalStopTracking() { return 1; };

  // Description:
  // This method should be overridden in derived classes that can make
  // an audible beep.  The return value should be zero if an error
  // occurred while the request was being processed.
  virtual int InternalBeep(int n) { return 1; };

  // Description:
  // This method should be overridden for devices that have one or more LEDs
  // on the tracked tools. The return value should be zero if an error
  // occurred while the request was being processed.
  virtual int InternalSetToolLED(int tool, int led, int state) { return 1; };

  vtkMatrix4x4 *WorldCalibrationMatrix;
  int NumberOfTools;
  vtkTrackerToolSensor *Tool;
  int ReferenceTool;
  int Tracking;
  double UpdateTimeStamp;
  unsigned long LastUpdateTime;

  vtkMultiThreader *Threader;
  int ThreadId;

private:
  vtkTrackerSensor(const vtkTrackerSensor&);
  void operator=(const vtkTrackerSensor&);  
};

#endif

