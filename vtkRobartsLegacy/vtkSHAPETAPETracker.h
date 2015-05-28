/*=========================================================================

  Program:   ShapeTape for VTK
  Module:    $RCSfile: vtkSHAPETAPETracker.h,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2007/04/19 12:48:53 $
  Version:   $Revision: 1.1 $

==========================================================================

Copyright (c) 2003-2004
All rights reserved.

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
// .NAME vtkSHAPETAPETracker - VTK interface for Measurand's ShapeTape
// .SECTION Description
// The vtkSHAPETAPETracker class provides an  interface to the SHAPETAPE tracking system.
// .SECTION see also
// vtkTrackerTool vtkFlockTracker


#ifndef __vtkSHAPETAPETracker_h
#define __vtkSHAPETAPETracker_h

#include "vtkTrackerSensor.h"
#include ".\ShapeAPI\tapeAPI.h"
#include "vtkTrackerToolSensor.h"

class vtkFrameToTimeConverter;

// the number of tools the polaris can handle
#define VTK_SHAPETAPE_NSENSORS 181

class VTK_EXPORT vtkSHAPETAPETracker : public vtkTrackerSensor
{
public:
  static vtkSHAPETAPETracker *New();
  vtkTypeMacro(vtkSHAPETAPETracker,vtkTracker);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Get the a string (perhaps a long one) describing the type and version
  // of the device.
  vtkGetStringMacro(Version);

  vtkGetMacro(SerialPort, int);
  void SetSerialPort(int port);

  vtkGetMacro(BaudRate, int);
  void SetBaudRate(int baud);

  vtkGetMacro(NumberOfSensors, int);
  void SetNumberOfSensors(int num);

  vtkGetMacro(TapeLength, double);
  void SetTapeLength(double length);

//  vtkGetMacro(BendOnly, bool);
  void SetBendOnly(bool bend );

  vtkGetMacro(InterpolationInterval, int);
  void SetInterpolationInterval(int interval);

  int Probe();

  // Description:
  // Get an update from the tracking system and push the new transforms
  // to the tools.  This should only be used within vtkTracker.cxx.
  void InternalUpdate();

  int SaveAdvancedCalibration(char * filename, char * mstFile);

  int SaveNormalCalibration(char * filename);

  int SaveFlatCalibration(char * filename);

  int SaveHelicalCalibration(char * filename);

  int GetTapeSettings();

  int SetSettingsFile(char* settings_file);

  int SetCalibrationFile(char* settings_file);

  // I don't really think these do anything.  Atleast from my brief testing nothing changes by modifying these values
  int SetOrientation(double yaw, double pitch, double roll);
  int SetOrientation(double baseU[3], double baseB[3]);
  int SetPosition(float x, float y, float z);
  int SetPosition(double x, double y, double z);

  int Initialize(int type);
  void SetFlat();

protected:
  vtkSHAPETAPETracker();
  ~vtkSHAPETAPETracker();

  int BendAndTwistToCartesian();

  int GetQuaternionData(double * value[8]);
  // Description:
  // Start the tracking system.  The tracking system is brought from
  // its ground state into full tracking mode.  The POLARIS will
  // only be reset if communication cannot be established without
  // a reset.
  int InternalStartTracking();

  // Description:
  // Stop the tracking system and bring it back to its ground state:
  // Initialized, not tracking, at 9600 Baud.
  int InternalStopTracking();

  vtkFrameToTimeConverter *Timer;

  tapeAPI * ShapeTape;
  char *Version;

  vtkMatrix4x4 *SendMatrix;
  int SerialPort; 
  int BaudRate;
  int IsSHAPETAPETracking;

  int NumberOfSensors;            // number of sensors
  char CalibrationFile[150];          // calibration file name
  char SettingsFile[150];            // settings file
  double TapeLength;            // length of the tape
  int InterpolationInterval;        // interpolation steps between sensor boundaries
  bool BendOnly;              // true for bend oly takes, false for bend/twist tapes
  int NumberOfRegions;            // the number of sensor regions
  double RegionLength[VTK_SHAPETAPE_NSENSORS];
  int NumberOfTransforms;

private:
  void stTransformToMatrixd(const double trans[8], double matrix[16]);
  void stTransformToMatrixf(const double trans[8], float matrix[16]);

  vtkSHAPETAPETracker(const vtkSHAPETAPETracker&);
  void operator=(const vtkSHAPETAPETracker&);  

};

// type values for 'int Initialize(int type)'.
#define ST_LOADSETTINGFILE      0
#define ST_EVENSPACING_BENDTWIST  1
#define ST_UNEVENSPACING      2

#endif