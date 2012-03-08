/*=========================================================================

  Program:   Lego Signal Box for VTK
  Module:    $RCSfile: vtkLegoSignalBox.cxx,v $
  Creator:   Danielle Pace <dpace@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: dpace $
  Date:      $Date: 2008/12/13 00:28:02 $
  Version:   $Revision: 1.6 $
==========================================================================

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

#include "vtkLegoSignalBox.h"
#include "vtkObjectFactory.h"

#include <limits.h>
#include <float.h>
#include <math.h>
#include "vtkMultiThreader.h"
#include "vtkMutexLock.h"
#include "vtkIntArray.h"
#include "vtkTimerLog.h"

//--------------------------------------------------------------------
vtkLegoSignalBox* vtkLegoSignalBox::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkLegoSignalBox");
  if(ret)
    {
    return (vtkLegoSignalBox*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkLegoSignalBox;
}

//--------------------------------------------------------------------
vtkLegoSignalBox::vtkLegoSignalBox()
{

  // init to -2 so it shows up as -1 in get methods
  this->MotorPort = -2;
  this->LightPort = -2;
  this->TouchPort = -2;
  this->SetMotorPower(10);
  this->MotorOn = 0;

  // empirically determined to go with trigger values
  //this->SleepInterval = 0.00899228451275;
  this->SleepInterval = 5;

  this->SignalBuffer = NULL;
  this->SignalBufferIndex = 0;
  this->WindowSize = 0;
  this->SpikeHeight = 0;
  this->MaxSignalInWindow = -1;
  this->MinSignalInWindow = -1;
  this->MinSpikeInterval = 0.6;
  this->OnFirstSignal = 0;

  this->PrintToFile = 0;
}

//--------------------------------------------------------------------
vtkLegoSignalBox::~vtkLegoSignalBox()
{
  if (this->IsStarted)
    {
    this->Stop();
    }
  else if (this->IsInitialized)
    {
    this->CloseConnection();
    }
  else
    {
    if (this->ThreadId != -1)
      {
      this->Threader->TerminateThread(this->ThreadId);
      this->ThreadId = -1;
      }
    }
  if (this->SignalBuffer)
    {
    this->SignalBuffer->Delete();
    }
  if (this->OutStream)
    {
    this->OutStream.close();
    }

  }
  
//--------------------------------------------------------------------
void vtkLegoSignalBox::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "WindowSize: " << this->WindowSize << "\n";
  os << indent << "SpikeHeight: " << this->SpikeHeight << "\n";
  os << indent << "MinSpikeInterval: " << this->MinSpikeInterval << "\n";
  os << indent << "MotorPort: " << this->GetMotorPort() << "\n";
  os << indent << "TouchPort: " << this->GetTouchPort() << "\n";
  os << indent << "LightPort: " << this->GetLightPort() << "\n";
  os << indent << "MotorPower: " << this->MotorPower << "\n";
  os << indent << "MotorOn: " << (this->MotorOn ? "On\n":"Off\n");
  os << indent << "SignalBuffer: " << this->SignalBuffer << "\n";
  if (this->SignalBuffer)
    {
    this->SignalBuffer->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "PrintToFile: " << (this->PrintToFile ? "On\n":"Off\n");

}

//--------------------------------------------------------------------
void vtkLegoSignalBox::SetPrintToFile(int toPrint)
  {
  if (toPrint == this->PrintToFile)
    {
    return;
    }

  this->PrintToFile = toPrint;

  if (toPrint)
    {
    char filename[100];
    sprintf(filename, "LegoSignal_%d.txt", this->MotorPower);
    this->OutStream.open(filename);
    OutStream.setf(ios::fixed);
    if (!this->OutStream)
      {
      this->PrintToFile = 0;
      }
    else
      {
      this->OutStream << "Trigger Timestamp LightSignal BMPrate ECGsignal" << std::endl;
      }
    }
  else
    {
    this->OutStream.close();
    }
  }

//--------------------------------------------------------------------
// Initialize the connection to the robot and set the touch and light sensors
void vtkLegoSignalBox::Initialize()
  {

  if (this->IsInitialized)
    {
    return;
    }

  // return if we have invalid port numbers
  if (this->MotorPort == -2 || this->LightPort == -2 || this->TouchPort == -2)
    {
    vtkWarningMacro( << "Invalid port numbers, could not initialize");
    return;
    }

  int open = this->Nxtusb.OpenLegoUSB();
  if (open == 0)
    {
    vtkWarningMacro( << "Failed opening");
    vtkWarningMacro( << this->Nxtusb.GetStatus());
    vtkWarningMacro( << "IMPORTANT!: Did you remember to run it as ROOT (or sudo)?");
    vtkWarningMacro( << "IMPORTANT!: Did you turned the LEGO NXT controller ON ?");
    return;
    }

  this->Nxtusb.SetSensorTouch(this->TouchPort);
  this->Nxtusb.SetSensorLight(this->LightPort, true);

  this->IsInitialized = 1;

  }

//--------------------------------------------------------------------
void vtkLegoSignalBox::CloseConnection()
  {
  if (!this->IsInitialized)
    {
    return;
    }

  this->Nxtusb.CloseLegoUSB();

  this->SignalBufferIndex = 0;
  this->MaxSignalInWindow = -1;
  this->MinSignalInWindow = -1;
  this->OnFirstSignal = 0;

  this->IsInitialized = 0;
  }

//-------------------------------------------------------------------------
void vtkLegoSignalBox::StartMotor()
  {
  if (!this->IsInitialized)
    {
    return;
    }

  if (!this->MotorOn)
    {
    this->Nxtusb.SetMotorOn(this->MotorPort, this->MotorPower);
    this->MotorOn = 1;

    // turn the light back on (could be turned off because it's a
    // battery drain
    this->Nxtusb.SetSensorLight(this->LightPort, true);
    }
  }

//-------------------------------------------------------------------------
void vtkLegoSignalBox::StopMotor()
  {
  if (!this->IsInitialized)
    {
    return;
    }

  if (this->MotorOn)
    {
    this->Nxtusb.StopMotor(this->MotorPort, false);
    this->MotorOn = 0;
    this->MaxSignalInWindow = 0;
    this->MinSignalInWindow = 0;
    this->IndicesSinceMaxSignalInWindow = -1;
    this->IndicesSinceMinSignalInWindow = -1;

    // turn off the active light since it is a battery drain
    this->Nxtusb.SetSensorLight(this->LightPort, false);
    }

  }


//-------------------------------------------------------------------------
void vtkLegoSignalBox::CheckForTouch()
  {
  if (!this->IsStarted || !this->UseTouchSensor)
    {
    return;
    }

  if (this->Nxtusb.GetTouchSensor(this->TouchPort))
    {

    // give the user time to release the button
		#ifdef _WIN32
      Sleep(1000);
		#else
		#ifdef unix
		#ifdef linux
			usleep(1000000);
		#endif
		#endif
		#endif

    // turning motor on
    if (!this->MotorOn)
      {
      this->StartMotor();
      }
    // turning motor off
    else
      {
      this->StopMotor();
      }
    }

  }

//-------------------------------------------------------------------------
static void *vtkLegoSignalBoxThread(vtkMultiThreader::ThreadInfo *data)
  {

	vtkLegoSignalBox *self = (vtkLegoSignalBox *)(data->UserData);
  int useTouch = self->GetUseTouchSensor();

	for (int i = 0;; i++)
    {

	  #ifdef _WIN32
		  Sleep(self->GetSleepInterval());
	  #else
	  #ifdef unix
	  #ifdef linux
		  usleep(self->GetSleepInterval() * 1000);
	  #endif
	  #endif
	  #endif

	  // query the hardware tracker
	  self->UpdateMutex->Lock();
    if (useTouch)
      {
      self->CheckForTouch();
      }
	  self->Update();
	  self->UpdateTime.Modified();
	  self->UpdateMutex->Unlock();

    // check to see if we are being told to quit 
    data->ActiveFlagLock->Lock();
    int activeFlag = *(data->ActiveFlag);
    data->ActiveFlagLock->Unlock();
    if (activeFlag == 0) {
	    return NULL;
    }
  }
    
}

//-------------------------------------------------------------------------
void vtkLegoSignalBox::Start()
  {

  if (this->IsStarted)
    {
		return;
    }

  if (!this->IsInitialized)
    {
    this->Initialize();
    }

  // if initialization fails
  if (!this->IsInitialized)
    {
    return;
    }
  
  this->SignalBuffer = vtkIntArray::New();
  this->SignalBuffer->SetNumberOfValues(this->WindowSize);
  this->SignalBufferIndex = 0;
  this->MaxSignalInWindow = -1;
  this->MinSignalInWindow = -1;
  this->OnFirstSignal = 0;

  if (this->PrintToFile)
    {
    this->OutStream << "Power:" << this->MotorPower << std::endl;
    }

  if (!this->UseTouchSensor)
    {
    this->StartMotor();
    }

	this->UpdateMutex->Lock();

	if (this->ThreadId == -1)
    {
		this->ThreadId = this->Threader->SpawnThread((vtkThreadFunctionType)&vtkLegoSignalBoxThread,this);
	  }
	this->UpdateMutex->Unlock();

	this->IsStarted=1;
  }

//-------------------------------------------------------------------------
void vtkLegoSignalBox::Stop()
  {

  if (this->ThreadId != -1)
    {
    this->Threader->TerminateThread(this->ThreadId);
    this->ThreadId = -1;
    }

  this->StopMotor();

  if (this->IsInitialized)
    {
    this->CloseConnection();
    }

  // zero out the signal buffer
  if (this->SignalBuffer)
    {
    for (int i = 0; i < this->WindowSize; i++)
      {
      this->SignalBuffer->SetValue(i, 0);
      }
    }

  this->IsStarted = 0;

  }

//-------------------------------------------------------------------------
int vtkLegoSignalBox::GetECG(void) {

  if (this->ThreadId == -1)
    {
    return -1;
    }

  this->UpdateTimestamp();

  // update the buffer holding the signal within a specific window
  int signal = this->Nxtusb.GetLightSensor(this->LightPort);
  this->SignalBufferIndex = (this->SignalBufferIndex + 1) % this->WindowSize;
  this->SignalBuffer->SetValue(this->SignalBufferIndex, signal);

  if (this->MotorOn)
    {

    // set max and min signals on first value
    if (this->MinSignalInWindow == -1 || this->MaxSignalInWindow == -1)
      {
      this->MinSignalInWindow = signal;
      this->MaxSignalInWindow = signal;
      this->IndicesSinceMaxSignalInWindow = 0;
      this->IndicesSinceMinSignalInWindow = 0;
      }
    // update the max and min buffer values if applicable
    else
      {
      if (signal <= this->MinSignalInWindow)
        {
        this->MinSignalInWindow = signal;
        this->IndicesSinceMinSignalInWindow = 0;
        }
      else
        {
        this->IndicesSinceMinSignalInWindow++;
        }

      if (signal >= this->MaxSignalInWindow)
        {
        this->MaxSignalInWindow = signal;
        this->IndicesSinceMaxSignalInWindow = 0;
        }
      else
        {
        this->IndicesSinceMaxSignalInWindow++;
        }
      }

    // if the min or max signals have fallen off the buffer, then re-search
    int max, min, curr, currIndex, indexCounter;
    if (this->IndicesSinceMaxSignalInWindow >= this->WindowSize)
      {
      // initialize
      max = this->SignalBuffer->GetValue(this->SignalBufferIndex);
      indexCounter = 0;
      // loop through the rest of the values
      for (int j = 1; j < this->WindowSize; j++)
        {
        currIndex = (this->SignalBufferIndex - j) % this->WindowSize;
        if (currIndex < 0)
          {
          currIndex = currIndex + this->WindowSize;
          }
        curr = this->SignalBuffer->GetValue(currIndex);
        if (curr >= max)
          {
          max = curr;
          indexCounter = j;
          }
        }
      this->MaxSignalInWindow = max;
      this->IndicesSinceMaxSignalInWindow = indexCounter;
      }

    if (this->IndicesSinceMinSignalInWindow >= this->WindowSize)
      {
      // initialize
      min = this->SignalBuffer->GetValue(this->SignalBufferIndex);
      indexCounter = 0;
      // loop through the rest of the values
      for (int j = 1; j < this->WindowSize; j++)
        {
        currIndex = (this->SignalBufferIndex - j) % this->WindowSize;
        if (currIndex < 0)
          {
          currIndex = currIndex + this->WindowSize;
          }
        curr = this->SignalBuffer->GetValue(currIndex);
        if (curr <= min)
          {
          min = curr;
          indexCounter = j;
          }
        }
      this->MinSignalInWindow = min;
      this->IndicesSinceMinSignalInWindow = indexCounter;
      }

    int signalDiff;
    int timeDiff;
    int trigger; // default no trigger

    signalDiff = this->MaxSignalInWindow - this->MinSignalInWindow;

    // we are starting a new cycle
    if (signalDiff > this->SpikeHeight)
      {
      timeDiff = this->Timestamp - this->StartSignalTimeStamp;

      if (this->OnFirstSignal || timeDiff > this->MinSpikeInterval)
        {
        if (this->AudibleBeep)
          {
          std::cout << "\a";
          }
        trigger = 1;
		    this->CalculateECGRate(this->StartSignalTimeStamp, this->Timestamp);
		    this->StartSignalTimeStamp = this->Timestamp;
        }
      }

    // the beating rate is invalid if if we are not starting a new cycle but we have
    // waited for more than two cycles without getting a new cycle
	  else if ( (this->Timestamp - this->StartSignalTimeStamp) >= ((60*2)/this->ECGRateBPM) ) 
      {
		  this->ECGRateBPM = -1.0;
      trigger = 0;
	    }

    // we are not starting a new cycle but we are still valid
    else
      {
      trigger = 0;
      }

    // return the signal, and update the phase if the beating rate is valid
	  if (this->ECGRateBPM < 0)
      {
      if (this->PrintToFile)
        {
        this->OutStream << trigger << " " << this->Timestamp << " " << signal << " " << this->ECGRateBPM << " " << this->ECGPhase << std::endl;
        }
		  return signal;
	    }
	  else
      {
		  this->CalculatePhase(this->StartSignalTimeStamp, this->Timestamp);
      if (this->PrintToFile)
        {
        this->OutStream << trigger << " " << this->Timestamp << " " << signal << " " << this->ECGRateBPM << " " << this->ECGPhase << std::endl;
        }
		  return signal;
	    }
    }

  // motor is not on
  else
    {
    this->ECGRateBPM = -1.0;
    return signal;
    }
}


//--------------------------------------------------------------------
void vtkLegoSignalBox::SetMotorPort(int port)
  {
  if (port > 0 & port < 4)
    {
    this->MotorPort = port - 1;
    }
  }

//--------------------------------------------------------------------
int vtkLegoSignalBox::GetMotorPort()
  {
  return this->MotorPort + 1;
  }

//--------------------------------------------------------------------
void vtkLegoSignalBox::SetTouchPort(int port)
  {
  if (port > 0 & port < 5)
    {
    this->TouchPort = port - 1;
    }
  }

//--------------------------------------------------------------------
int vtkLegoSignalBox::GetTouchPort()
  {
  return this->TouchPort + 1;
  }

//--------------------------------------------------------------------
void vtkLegoSignalBox::SetLightPort(int port)
  {
  if (port > 0 & port < 5)
    {
    this->LightPort = port - 1;
    }
  }

//--------------------------------------------------------------------
int vtkLegoSignalBox::GetLightPort()
  {
  return this->LightPort + 1;
  }

//--------------------------------------------------------------------
void vtkLegoSignalBox::SetMotorPower(int power)
  {

  if (power < 10 || power > 100 || power % 10 != 0)
    {
    return;
    }

  this->MotorPower = power;

  if (power >= 30)
    {
    this->WindowSize = 15;
    this->SpikeHeight = 14;
    }
  else if (power == 20)
    {
    this->WindowSize = 25;
    this->SpikeHeight = 15;
    }
  else //if (power == 10)
    {
    this->WindowSize = 45;
    this->SpikeHeight = 15;
    }
  }