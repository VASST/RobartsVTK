/*=========================================================================

  Program:   Shapetape for VTK
  Module:    $RCSfile: vtkTrackerSensor.cxx,v $
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

#include <limits.h>
#include <float.h>
#include <math.h>
#include "vtkTrackerSensor.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkTimerLog.h"
#include "vtkTrackerToolSensor.h"
#include "vtkTrackerBuffer.h"
#include "vtkMultiThreader.h"
#include "vtkMutexLock.h"
#include "vtkCriticalSection.h"
#include "vtkObjectFactory.h"

//----------------------------------------------------------------------------
vtkTrackerSensor* vtkTrackerSensor::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkTrackerSensor");
  if(ret)
    {
    return (vtkTrackerSensor*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkTrackerSensor;
}

//----------------------------------------------------------------------------
vtkTrackerSensor::vtkTrackerSensor()
{
  this->Tracking = 0;
  this->WorldCalibrationMatrix = vtkMatrix4x4::New();
  this->NumberOfTools = 0;
  this->ReferenceTool = -1;
  this->UpdateTimeStamp = 0;
  this->Tool = 0;
  this->LastUpdateTime = 0;
  this->InternalUpdateRate = 0;

  // for threaded capture of transformations
  this->Threader = vtkMultiThreader::New();
  this->ThreadId = -1;
  this->UpdateMutex = vtkCriticalSection::New();
}

//----------------------------------------------------------------------------
vtkTrackerSensor::~vtkTrackerSensor()
{
  // The thread should have been stopped before the
  // subclass destructor was called, but just in case
  // se stop it here.
  if (this->ThreadId != -1)
  {
    this->Threader->TerminateThread(this->ThreadId);
    this->ThreadId = -1;
  }

  this->Tool->SetTracker(NULL);
  this->Tool->Delete();
  if (this->Tool)
  {
    delete this->Tool;
  }

  this->WorldCalibrationMatrix->Delete();

  this->Threader->Delete();
  this->UpdateMutex->Delete();
}
  
//----------------------------------------------------------------------------
void vtkTrackerSensor::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkObject::PrintSelf(os,indent);

  os << indent << "WorldCalibrationMatrix: " << this->WorldCalibrationMatrix << "\n";
  this->WorldCalibrationMatrix->PrintSelf(os,indent.GetNextIndent());
  os << indent << "Tracking: " << this->Tracking << "\n";
  os << indent << "ReferenceTool: " << this->ReferenceTool << "\n";
  os << indent << "NumberOfTools: " << this->NumberOfTools << "\n";
}
  
//----------------------------------------------------------------------------
// allocates a vtkTrackerSensorTool object for each of the tools.
void vtkTrackerSensor::SetNumberOfSensors(int sensors)
{

  if (this->NumberOfTools > 0) 
    {
    vtkErrorMacro( << "SetNumberOfTools() can only be called once");
    }
  this->NumberOfTools = 1;
 
//    this->Tool = vtkTrackerToolSensor::New();
//    this->Tool->SetTracker(this);
    this->Tool->SetNumberOfSensors(sensors);
}  

//----------------------------------------------------------------------------
vtkTrackerToolSensor *vtkTrackerSensor::GetTool(int i)
{
  return this->Tool;
}

//----------------------------------------------------------------------------
// this thread is run whenever the tracker is tracking
static void *vtkTrackerThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkTrackerSensor *self = (vtkTrackerSensor *)(data->UserData);

  double currtime[10];

  for (int i = 0;; i++)
    {
    // get current tracking rate over last 10 updates
    double newtime = vtkTimerLog::GetCurrentTime();
    double difftime = newtime - currtime[i%10];
    currtime[i%10] = newtime;
    if (i > 10 && difftime != 0)
      {
      self->InternalUpdateRate = (10.0/difftime);
      }

    // query the hardware tracker
    self->UpdateMutex->Lock();
    self->InternalUpdate();
    self->UpdateTime.Modified();
    self->UpdateMutex->Unlock();
    
    // check to see if we are being told to quit 
    data->ActiveFlagLock->Lock();
    int activeFlag = *(data->ActiveFlag);
    data->ActiveFlagLock->Unlock();

    if (activeFlag == 0)
      {
      return NULL;
      }
    }
}

//----------------------------------------------------------------------------
int vtkTrackerSensor::Probe()
{
  this->UpdateMutex->Lock();

  if (this->InternalStartTracking() == 0)
    {
    this->UpdateMutex->Unlock();
    return 0;
    }

  this->Tracking = 1;

  if (this->InternalStopTracking() == 0)
    {
    this->Tracking = 0;
    this->UpdateMutex->Unlock();
    return 0;
    }

  this->Tracking = 0;
  this->UpdateMutex->Unlock();
  return 1;
}

//----------------------------------------------------------------------------
void vtkTrackerSensor::StartTracking()
{

  int tracking = this->Tracking;

  this->LastUpdateTime = 0;

  this->Tracking = this->InternalStartTracking();

  this->UpdateMutex->Lock();

  if (this->Tracking && !tracking && this->ThreadId == -1)
    {
    this->ThreadId = this->Threader->SpawnThread((vtkThreadFunctionType)\
						 &vtkTrackerThread,this);
    }

  this->UpdateMutex->Unlock();
}

//----------------------------------------------------------------------------
void vtkTrackerSensor::StopTracking()
{
  if (this->Tracking && this->ThreadId != -1)
    {
    this->Threader->TerminateThread(this->ThreadId);
    this->ThreadId = -1;
    }

  this->InternalStopTracking();
  this->Tracking = 0;
}

//----------------------------------------------------------------------------
void vtkTrackerSensor::Update()
{
  if (!this->Tracking)
    { 
    return; 
    }

  if (this->LastUpdateTime == 0 ||
      this->LastUpdateTime == this->UpdateTime.GetMTime())
    {
    // wait at most 0.1s for the next transform to arrive,
    // which is marked by a change in the UpdateTime
    double waittime = vtkTimerLog::GetCurrentTime() + 0.1;
    if (this->LastUpdateTime == 0)
      {  // if this is the first transform, wait up to 5 seconds
      waittime += 5.0;
      }
    while (this->LastUpdateTime == this->UpdateTime.GetMTime() &&
	   vtkTimerLog::GetCurrentTime() < waittime) 
      {
#ifdef _WIN32
      Sleep(10);
#else
#ifdef unix
#ifdef linux
      usleep(10*1000);
#endif
#ifdef sgi
      struct timespec sleep_time, remaining_time;
      sleep_time.tv_sec = 10 / 1000;
      sleep_time.tv_nsec = 1000000*(10 % 1000);
      nanosleep(&sleep_time,&remaining_time);
#endif
#endif
#endif
      }
    }

  if (this->NumberOfTools > 0)
    {
    vtkTrackerToolSensor *trackerTool = this->Tool;
    trackerTool->Update();
    this->UpdateTimeStamp = trackerTool->GetTimeStamp();
    }

  this->LastUpdateTime = this->UpdateTime.GetMTime();
}

//----------------------------------------------------------------------------
void vtkTrackerSensor::SetWorldCalibrationMatrix(vtkMatrix4x4 *vmat)
{
  int i, j;

  for (i = 0; i < 4; i++) 
    {
    for (j = 0; j < 4; j++)
      {
      if (this->WorldCalibrationMatrix->GetElement(i,j) 
	  != vmat->GetElement(i,j))
	{
	break;
	}
      }
    if (j < 4)
      { 
      break;
      }
    }

  if (i < 4 || j < 4) // the matrix is different
    {
    this->WorldCalibrationMatrix->DeepCopy(vmat);
    this->Modified();
    }
}

//----------------------------------------------------------------------------
vtkMatrix4x4 *vtkTrackerSensor::GetWorldCalibrationMatrix()
{
  return this->WorldCalibrationMatrix;
}

//----------------------------------------------------------------------------
void vtkTrackerSensor::ToolUpdate(int bufferNumber, vtkMatrix4x4 *matrix, long flags, double timestamp) 
{
  vtkTrackerBuffer *buffer = this->Tool->GetBuffer(bufferNumber);

  buffer->Lock();
  buffer->AddItem(matrix, flags, timestamp);
  buffer->Unlock();
}
  
//----------------------------------------------------------------------------
void vtkTrackerSensor::Beep(int n)
{
  this->UpdateMutex->Lock();
  this->InternalBeep(n);
  this->UpdateMutex->Unlock();
}

//----------------------------------------------------------------------------
void vtkTrackerSensor::SetToolLED(int tool,  int led, int state)
{
  this->UpdateMutex->Lock();
  this->InternalSetToolLED(tool, led, state);
  this->UpdateMutex->Unlock();
}






