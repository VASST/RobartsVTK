/*=========================================================================

  Program:   ShapeTape for VTK
  Module:    $RCSfile: vtkSHAPETAPETracker.cxx,v $
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

#include <limits.h>
#include <float.h>
#include <math.h>
#include <ctype.h>
#include <time.h>
#include "vtkMath.h"
#include "vtkTimerLog.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkTrackerTool.h"
#include "vtkObjectFactory.h"
#include "vtkTrackerBuffer.h"
#include "vtkSHAPETAPETracker.h"
#include "vtkFrameToTimeConverter.h"

//----------------------------------------------------------------------------
vtkSHAPETAPETracker* vtkSHAPETAPETracker::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkSHAPETAPETracker");
  if(ret)
    {
    return (vtkSHAPETAPETracker*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkSHAPETAPETracker;
}

//----------------------------------------------------------------------------
vtkSHAPETAPETracker::vtkSHAPETAPETracker()
{
  this->ShapeTape = NULL;              // Shapetape C++ object
  this->Version = NULL;                // do i need a version
  this->SendMatrix = vtkMatrix4x4::New();      // send matrix for the tool.
  this->IsSHAPETAPETracking = 0;          // is the shapetape tracking?  is it ever NOT tracking?
  this->SerialPort = -1; // default is to probe?  // serialPort the shapetape is on
  this->BaudRate = 115200;              // default baud rate
  this->NumberOfSensors = -1;            // number of sensors?
  this->CalibrationFile[0] = '\0';          // calibration file name
  this->TapeLength = 0.0;              // length of the tape
  this->InterpolationInterval = 0;          // interpolation steps between sensor boundaries
  this->BendOnly = true;              // true for bend oly takes, false for bend/twist tapes
  this->NumberOfRegions = 0;            // the number of sensor regions
  this->NumberOfTransforms = 0;
  for(int i=0; i < 48; i++)
  this->RegionLength[i] = 0.0;          // arror of sensor lengths.  Allows for variable sensor lengths.

  this->SettingsFile[0] = '\0';            // to load all information from the sensor

  this->Tool = vtkTrackerToolSensor::New();    // send matrix for the tool.
this->Tool->SetTracker(this);
  // for accurate timing
  this->Timer = vtkFrameToTimeConverter::New();
  this->Timer->SetNominalFrequency(60.0);



}

//----------------------------------------------------------------------------
vtkSHAPETAPETracker::~vtkSHAPETAPETracker() 
{
  if (this->Tracking)
    {
    this->StopTracking();
    }
  this->SendMatrix->Delete();

  if (this->Timer)
    {
    this->Timer->Delete();
    }
}

//----------------------------------------------------------------------------
void vtkSHAPETAPETracker::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkTrackerSensor::PrintSelf(os,indent);
  os << indent << "SendMatrix: " << this->SendMatrix << "\n";
  this->SendMatrix->PrintSelf(os,indent.GetNextIndent());
  os << indent << "IsSHAPETAPETracking: " << this->IsSHAPETAPETracking << "\n";
  os << indent << "SerialPort: " << this->SerialPort << "\n";
  os << indent << "BaudRate: " << this->BaudRate << "\n";
  os << indent << "NumberOfSensors: " << this->GetNumberOfSensors() << "\n";
  os << indent << "CalibrationFile: " << this->CalibrationFile << "\n";
  os << indent << "TapeLength: " << this->TapeLength << "\n";
  os << indent << "BendOnly: " << this->BendOnly << "\n";

}
 

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SetPosition(double x, double y, double z)
{
  if (this->ShapeTape) {
    this->UpdateMutex->Lock();
    this->ShapeTape->setPosition(x, y, z);
    this->UpdateMutex->Unlock();
    return 1;
  }
  else
    return 0;
}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SetPosition(float x, float y, float z)
{
  if (this->ShapeTape) {
    this->UpdateMutex->Lock();
    this->ShapeTape->setPosition((double)x, (double)y, (double)z);
    this->UpdateMutex->Unlock();
    return 1;
  }
  else
    return 0;
}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SetOrientation(double baseU[3], double baseB[3])
{
  if (this->ShapeTape) {
    this->UpdateMutex->Lock();
    this->ShapeTape->setOrientation(baseU, baseB);
    this->UpdateMutex->Unlock();
    return 1;
  }
  else
    return 0;
}    

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SetOrientation(double yaw, double pitch, double roll)
{
  if (this->ShapeTape) {
    this->UpdateMutex->Lock();
    this->ShapeTape->setOrientation(yaw, pitch, roll);
    this->UpdateMutex->Unlock();
    return 1;
  }
  else
    return 0;
}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::GetTapeSettings()
{
  if (this->ShapeTape) {
    this->UpdateMutex->Lock();
    this->ShapeTape->getTapeSettings(this->CalibrationFile, this->NumberOfSensors, this->InterpolationInterval, this->TapeLength,
                     this->NumberOfRegions, this->RegionLength);
    this->UpdateMutex->Unlock();
    return 0;
  }
  else
    return 1;
}


//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SetSettingsFile(char* settings_file)
{
  strcpy(this->SettingsFile, settings_file);
  return 1;
}


//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SetCalibrationFile(char* calibration_file)
{
  strcpy(this->CalibrationFile, calibration_file);
  return 1;
}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::GetQuaternionData(double * value[8])
{
  if (this->ShapeTape) {
//    for (int i = 0; i < this->NumberOfSensors; i++)
      this->ShapeTape->getCartesianData(value[4], value[5], value[6], value[0], value[1], value[2], value[3]);
    return 1;
  }
  else
    return 0;
}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SaveAdvancedCalibration(char * filename, char * mstFile)
{
  if (this->ShapeTape && filename && mstFile ) {
    this->UpdateMutex->Lock();
    this->ShapeTape->saveAdvancedCalibration(filename, mstFile);
      this->UpdateMutex->Unlock();
    return 1;
  }
  else
    return 0;
}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SaveNormalCalibration(char * filename)
{
  if (this->ShapeTape && filename) {
    this->UpdateMutex->Lock();
    this->ShapeTape->saveNormalCalibration(filename);
      this->UpdateMutex->Unlock();
    return 1;
  }
  else
    return 0;
}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SaveFlatCalibration(char * filename)
{
  if (this->ShapeTape && filename) {
    this->UpdateMutex->Lock();
    this->ShapeTape->saveFlatCalibration(filename);
      this->UpdateMutex->Unlock();
    return 1;
  }
  else
    return 0;
}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::SaveHelicalCalibration(char * filename)
{
  if (this->ShapeTape && filename) {
      this->UpdateMutex->Lock();
    this->ShapeTape->saveHelicalCalibration(filename);
      this->UpdateMutex->Unlock();

    return 1;
  }
  else
    return 0;
}

//----------------------------------------------------------------------------
void vtkSHAPETAPETracker::SetSerialPort(int port)
{
  if (this->IsSHAPETAPETracking == 0) 
    this->SerialPort = port;
  else {
    vtkErrorMacro( << "Cannot Set Serial Port When Tracking");
    return;
  }
}

//----------------------------------------------------------------------------
void vtkSHAPETAPETracker::SetBaudRate(int baud)
{
  if (this->IsSHAPETAPETracking == 0) 
    this->BaudRate = baud;
  else {
    vtkErrorMacro( << "Cannot Set Baud Rate When Tracking");
    return;
  }
}


//----------------------------------------------------------------------------
void vtkSHAPETAPETracker::SetTapeLength(double length )
{
  if (this->IsSHAPETAPETracking == 0) 
    this->TapeLength = length;
  else {
    vtkErrorMacro( << "Cannot Set Tape Length When Tracking");
    return;
  }
}

//----------------------------------------------------------------------------
void vtkSHAPETAPETracker::SetBendOnly(bool bend )
{
  if (this->IsSHAPETAPETracking == 0) 
    this->BendOnly = bend;
  else {
    vtkErrorMacro( << "Cannot Set Bend Only When Tracking");
    return;
  }
}

//----------------------------------------------------------------------------
void vtkSHAPETAPETracker::SetInterpolationInterval(int interval)
{
  if (this->IsSHAPETAPETracking == 0) 
    this->InterpolationInterval = interval;
  else {
    vtkErrorMacro( << "Cannot Set Interpolation Interval When Tracking");
    return;
  }
}

void vtkSHAPETAPETracker::SetNumberOfSensors(int num)
{
  if (this->IsSHAPETAPETracking == 0)  {
    this->NumberOfSensors = num;
    vtkTrackerSensor::SetNumberOfSensors(num);
  }

  else {
    vtkErrorMacro( << "Cannot Set Interpolation Interval When Tracking");
    return;
  }
}


//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::Initialize(int type) {
  if (this->IsSHAPETAPETracking == 0) {
    if (type == ST_EVENSPACING_BENDTWIST) {
      if (this->NumberOfSensors < 0 || this->NumberOfSensors > VTK_SHAPETAPE_NSENSORS) {
        vtkErrorMacro( << "Incorrect number of sensors.  Initialization failed");
        return 1;
      }
      if (this->CalibrationFile == NULL) {
        vtkErrorMacro( << "Calibration file is NULL");
        return 1;
      }
      if (this->TapeLength == 0) {
        vtkErrorMacro( << "Length of the tape is not specified");
        return 1;
      }
      if (this->ShapeTape)
        delete this->ShapeTape;
      this->ShapeTape = new tapeAPI(this->NumberOfSensors, this->BaudRate, this->CalibrationFile, this->TapeLength, this->InterpolationInterval);
      this->NumberOfTransforms = (this->NumberOfRegions*this->InterpolationInterval)+1;
      this->ShapeTape->flat();
      return 0;
    }
    else if (type == ST_UNEVENSPACING) {
      if (this->NumberOfSensors < 0 || this->NumberOfSensors > VTK_SHAPETAPE_NSENSORS) {
        vtkErrorMacro( << "Incorrect number of sensors.  Initialization failed");
        return 1;
      }
      if (this->CalibrationFile == NULL) {
        vtkErrorMacro( << "Calibration file is NULL");
        return 1;
      }
      if (this->TapeLength == 0) {
        vtkErrorMacro( << "Length of the tape is not specified");
        return 1;
      }
      for (int i = 0; i < this->NumberOfRegions; i++) {
        if (this->RegionLength[i] == 0){
          vtkErrorMacro( << "Region Length is invalid for the number of regions specified");
          return 1;
        }
      }
      if (this->ShapeTape)
        delete this->ShapeTape;
      this->ShapeTape = new tapeAPI(this->NumberOfSensors, this->BaudRate, this->CalibrationFile, this->TapeLength, this->InterpolationInterval, this->BendOnly, this->NumberOfRegions, this->RegionLength);
      this->NumberOfTransforms = (this->NumberOfRegions*this->InterpolationInterval)+1;
      this->ShapeTape->flat();
      return 0;
    }
    else if (type == ST_LOADSETTINGFILE) {
      if (this->SettingsFile == NULL){
        vtkErrorMacro( << "Settings file is null.  Unable to initalize");
        return 1;
      }

      if (this->ShapeTape)
        delete this->ShapeTape;
      this->ShapeTape = new tapeAPI(this->SettingsFile, "./");
      this->ShapeTape->flat();
      this->ShapeTape->getTapeSettings(this->CalibrationFile, this->NumberOfSensors, this->InterpolationInterval, this->TapeLength, this->NumberOfRegions, this->RegionLength);
      this->NumberOfTransforms = (this->NumberOfRegions*this->InterpolationInterval)+1;
      this->SetNumberOfSensors(this->NumberOfTransforms);
      this->Tool->SetTapeLength(this->TapeLength);
      this->Tool->SetInterpolationInterval(this->InterpolationInterval);
      return 0;
    }
    else {
      vtkErrorMacro( << "Invalid Initialization mode");
      return 1;
    }
  }
  else {
    vtkErrorMacro( << "Cannot Initialize Shapetape When Tracking");
    return 1;
  }

}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::InternalStartTracking()
{
 
  if (this->IsSHAPETAPETracking) {
    return 1;
    }
  if (!(this->ShapeTape)) {
   if (this->Initialize(ST_LOADSETTINGFILE))
    return 1;
  }

  // setup for number of breaks and matrix information

  // for accurate timing
  this->Timer->Initialize();

  this->IsSHAPETAPETracking = 1;

  return 1;
}

//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::InternalStopTracking()
{
  if (this->ShapeTape == 0)
    {
    return 0;
    }
  if (this->ShapeTape) {
  delete this->ShapeTape;
  this->IsSHAPETAPETracking = 0;
  }
//  this->ShapeTape = 0;

  return 1;
}

//----------------------------------------------------------------------------
// Important notes on the data collection rate of the POLARIS:
//
// The camera frame rate is 60Hz, and therefore the maximum data
// collection rate is also 60Hz.  The maximum data transfer rate
// to the computer is also 60Hz.
//
// Depending on the number of enabled tools, the data collection
// rate might be reduced.  Each of the active tools requires one
// camera frame, and all the passive tools (if any are enabled)
// collectively require one camera frame.
//
// Therefore if there are two enabled active tools, the data rate
// is reduced to 30Hz.  Ditto for an active tool and a passive tool.
// If all tools are passive, the data rate is 60Hz.  With 3 active
// tools and one or more passive tools, the data rate is 15Hz.
// With 3 active tools, or 2 active and one or more passive tools,
// the data rate is 20Hz.
//
// The data transfer rate to the computer is independent of the data
// collection rate, and there might be duplicated records.  The
// data tranfer rate is limited by the speed of the serial port
// and by the number of characters sent per data record.  If tools
// are marked as 'missing' then the number of characters that
// are sent will be reduced.

void vtkSHAPETAPETracker::InternalUpdate()
{
  int sensor;
  double transform[VTK_SHAPETAPE_NSENSORS][8];
  double * temp_transform[8];

  for (int size=0; size < 8; size++){
  temp_transform[size] = new double[this->NumberOfTransforms];
  }


  if (!this->IsSHAPETAPETracking)
    {
    vtkWarningMacro( << "called Update() when POLARIS was not tracking");
    return;
    }
  int data[1000];
  this->ShapeTape->pollTape(data);
  GetQuaternionData(temp_transform);

  for (int i=0; i < this->NumberOfTransforms; i++){
    for (int j=0; j < 7; j++){
    transform[i][j] = temp_transform[j][i];
    }
    transform[i][7] = 0.0;
  }
  for (int delete_loop=0; delete_loop < 8; delete_loop++){
  delete [] temp_transform[delete_loop];
  }

  unsigned long nextcount = this->Timer->GetLastFrame() + 1;

  // the timestamp is always created using the frame number of
  // the most recent transformation
  this->Timer->SetLastFrame(nextcount);

  double timestamp = this->Timer->GetTimeStampForFrame(nextcount);

  for (sensor = 0; sensor < this->NumberOfTransforms; sensor++) 
    {
    
    stTransformToMatrixd(transform[sensor],*this->SendMatrix->Element);
    this->SendMatrix->Transpose();

    // by default (if there is no camera frame number associated with
    // the tool transformation) the most recent timestamp is used.
    double tooltimestamp = timestamp;
    tooltimestamp = this->Timer->GetTimeStampForFrame(nextcount);
  // HOW TO DISPLAY INDIVIDUAL TOOLS
    this->Tool->SensorUpdate(sensor,this->SendMatrix,0,tooltimestamp);
    }

}


//----------------------------------------------------------------------------
void vtkSHAPETAPETracker::SetFlat()
{
    this->UpdateMutex->Lock();
  this->ShapeTape->flat();
  this->UpdateMutex->Unlock();
  return;
}


//----------------------------------------------------------------------------
int vtkSHAPETAPETracker::Probe()
{
   //vtkWarningMacro( << "No acceptable way to probe shapetape.");
   return 1;

}

inline void vtkSHAPETAPETracker::stTransformToMatrixf(const double trans[8], float matrix[16])
{
  double ww, xx, yy, zz, wx, wy, wz, xy, xz, yz, ss, rr, f;
  
  /* Determine some calculations done more than once. */
  ww = trans[0] * trans[0];
  xx = trans[1] * trans[1];
  yy = trans[2] * trans[2];
  zz = trans[3] * trans[3];
  wx = trans[0] * trans[1];
  wy = trans[0] * trans[2];
  wz = trans[0] * trans[3];
  xy = trans[1] * trans[2];
  xz = trans[1] * trans[3];
  yz = trans[2] * trans[3];

  rr = xx + yy + zz;
  ss = (ww - rr)*0.5f;
  /* Normalization factor */
  f = 2.0f/(ww + rr);
  
  /* Fill in the matrix. */
  matrix[0]  = (float)(( ss + xx)*f);
  matrix[1]  = (float)(( wz + xy)*f);
  matrix[2]  = (float)((-wy + xz)*f);
  matrix[3]  = 0;
  matrix[4]  = (float)((-wz + xy)*f);
  matrix[5]  = (float)(( ss + yy)*f);
  matrix[6]  = (float)(( wx + yz)*f);
  matrix[7]  = 0;
  matrix[8]  = (float)(( wy + xz)*f);
  matrix[9]  = (float)((-wx + yz)*f);
  matrix[10] = (float)(( ss + zz)*f);
  matrix[11] = 0;
  matrix[12] = (float)(trans[4]);
  matrix[13] = (float)(trans[5]);
  matrix[14] = (float)(trans[6]);
  matrix[15] = 1;
}

inline void vtkSHAPETAPETracker::stTransformToMatrixd(const double trans[8], double matrix[16])
{
  double ww, xx, yy, zz, wx, wy, wz, xy, xz, yz, ss, rr, f;
  
  /* Determine some calculations done more than once. */
  ww = trans[0] * trans[0];
  xx = trans[1] * trans[1];
  yy = trans[2] * trans[2];
  zz = trans[3] * trans[3];
  wx = trans[0] * trans[1];
  wy = trans[0] * trans[2];
  wz = trans[0] * trans[3];
  xy = trans[1] * trans[2];
  xz = trans[1] * trans[3];
  yz = trans[2] * trans[3];

  rr = xx + yy + zz;
  ss = (ww - rr)*0.5;
  /* Normalization factor */
  f = 2.0/(ww + rr);
  
  /* Fill in the matrix. */
  matrix[0]  = ( ss + xx)*f;
  matrix[1]  = ( wz + xy)*f;
  matrix[2]  = (-wy + xz)*f;
  matrix[3]  = 0;
  matrix[4]  = (-wz + xy)*f;
  matrix[5]  = ( ss + yy)*f;
  matrix[6]  = ( wx + yz)*f;
  matrix[7]  = 0;
  matrix[8]  = ( wy + xz)*f;
  matrix[9]  = (-wx + yz)*f;
  matrix[10] = ( ss + zz)*f;
  matrix[11] = 0;
  matrix[12] = trans[4];
  matrix[13] = trans[5];
  matrix[14] = trans[6];
  matrix[15] = 1;
}
