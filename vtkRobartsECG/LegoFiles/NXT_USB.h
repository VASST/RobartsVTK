/*
Modified from NXT++
http://nxtpp.sourceforge.net/

And Device::USB
http://search.cpan.org/~gwadej/Device-USB-0.21/lib/Device/USB.pm
*/

/*=========================================================================
  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: NXT_USB.h,v $
  Language:  C++
  Date:      $Date: 2008/12/16 15:09:13 $
  Version:   $Revision: 1.1 $
Copyright (c) 2003 Insight Software Consortium
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * The name of the Insight Consortium, nor the names of any consortium members,
   nor of any contributors, may be used to endorse or promote products derived
   from this software without specific prior written permission.
 * Modified source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ‘‘AS IS’’
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

#ifndef __NXT_USB_h
#define __NXT_USB_h

#include "NXT_USB_linux.h"

class NXT_USB
{

public:

  NXT_USB();
  ~NXT_USB();

  // functions

  int OpenLegoUSB();
  int CloseLegoUSB();

  void SetSensorLight(int port, bool active);
  void SetSensorTouch(int port);
  void SetSensorSound(int port, bool dba);
  void SetSensorUS(int port);
  void SetUSOff(int port);
  void SetUSSingleShot(int port);
  void SetUSContinuous(int port);
  void SetUSEventCapture(int port);
  void SetUSContinuousInterval(int port, int interval);

  int GetLightSensor(int port);
  bool GetTouchSensor(int port);
  int GetSoundSensor(int port);
  int GetUSSensor(int port);

  void SetMotorOn(int port, int power);
  void SetMotorOn(int port, int power, int tachoCount);
  void MoveMotor(int port, int power, int tachoCount);
  void StopMotor(int port, bool brake);
    
  int GetMotorRotation(int port, bool relative);
  void ResetMotorPosition(int port, bool relative);

  void PlayTone(int frequency, int duration);

  char * GetDeviceFilename();
  int GetIDVendor();
  int GetIDProduct();
  char * GetStatus();

  void SendCommand (char * outbuf, int outbufSize, char * inbuf, int inbufSize);
  int LSGetStatus(int port);

  //attributes:
  private:
    NXT_USB_linux *usbConn;

};

#endif

