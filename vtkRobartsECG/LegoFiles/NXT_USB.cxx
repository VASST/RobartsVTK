/*
Modified from NXT++
http://nxtpp.sourceforge.net/

And Device::USB
http://search.cpan.org/~gwadej/Device-USB-0.21/lib/Device/USB.pm
*/

/*=========================================================================
  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: NXT_USB.cxx,v $
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

/** \class NXT_USB
 * \brief Controls a LEGO Mindstorms NXT robot via a USB connection
 *
 * NXT_USB controls a LEGO Mindstorms NXT robot over a USB connection.  Namely
 * one can read from the sensors, move the motors and read rotational information
 * from the motors.
 */

#include "NXT_USB.h"
#include <cstring>
#include <iostream>

// in and out ports
const int IN_1 = 0;
const int IN_2 = 1;
const int IN_3 = 2;
const int IN_4 = 3;
const int OUT_A = 0;
const int OUT_B = 1;
const int OUT_C = 2;

// response codes for direct commands
static const char RESPONSE = 0x00;
static const char NO_RESPONSE = 0x80;

// some command enumerations
static const char PLAYTONE = 0x03;
static const char SETOUTPUTSTATE = 0x04;
static const char SETINPUTMODE = 0x05;
static const char GETOUTPUTSTATE = 0x06;
static const char GETINPUTVALUES = 0x07;
static const char RESETMOTORPOSITION = 0x0A;
static const char LSGETSTATUS = 0x0E;
static const char LSWRITE = 0x0F;
static const char LSREAD = 0x10;

// some enumerations for "Sensor Type"
static const char SWITCH = 0x01; // touch sensor
static const char LIGHT_ACTIVE = 0x05;
static const char LIGHT_INACTIVE = 0x06;
static const char SOUND_DB = 0x07;
static const char SOUND_DBA = 0x08;
static const char LOWSPEED_9V = 0x0B;

// some enumerations for "Sensor Mode"
static const char RAWMODE = 0x00;
static const char BOOLEANMODE = 0x20;
static const char PCTFULLSCALEMODE = 0x80;

// some enumerations for "Mode"
static const char MFLOAT = 0x00; // not an official enumation
static const char MOTORON = 0x01;
static const char BRAKE = 0x02;
static const char REGULATED = 0x04;

// some enumerations for "Regulation Mode"
static const char REGULATION_MODE_IDLE = 0x00;
static const char REGULATION_MODE_MOTOR_SPEED = 0x01;

// some enumerations for "RunState"
static const char MOTOR_RUN_STATE_IDLE = 0x00;
static const char MOTOR_RUN_STATE_RUNNING = 0x20;

// some enumerations for LS commands
static const char US_ADDRESS = 0x02;

// some enumerations for the US
static const char SET_US_MODE = 0x41;
static const char READ_US_BYTE0 = 0x42;

static const char SET_US_CONTINUOUSINTERVAL = 0x40;
static const char US_MODE_OFF = 0x00;
static const char US_MODE_SINGLESHOT = 0x01;
static const char US_MODE_CONTINUOUS = 0x02;
static const char US_MODE_EVENTCAPTURE = 0x03;

/** Constructor */
NXT_USB::NXT_USB()
{
  usbConn = new NXT_USB_linux();

}
/** Destructor */
NXT_USB::~NXT_USB()
{
  delete usbConn;
}

/** Open the USB connection for the LEGO NXT device */

int NXT_USB::OpenLegoUSB()
{
  return this->usbConn->OpenLegoUSB();
}

/** Closes the USB connection for the LEGO NXT device */
int NXT_USB::CloseLegoUSB()
{
  return this->usbConn->CloseLegoUSB();
}

/** Set the sensor type of the input port defined by the "port" parameter
 * to a light sensor.  Use active = true if the light sensor should be active
 * (reflected light) and active = false if it should be passive (ambient light) */
void NXT_USB::SetSensorLight(int port, bool active)
{

  char outbuf[] = {NO_RESPONSE, SETINPUTMODE, port, 0, PCTFULLSCALEMODE};
  char inbuf[3];
  if (active)
  {
    outbuf[3] = LIGHT_ACTIVE;
  }
  else
  {
    outbuf[3] = LIGHT_INACTIVE;
  }
  this->usbConn->SendCommand(outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Set the sensor type of the input port defined by the "port" parameter to a
 * touch sensor. */
void NXT_USB::SetSensorTouch(int port)
{
  char outbuf[] = {NO_RESPONSE, SETINPUTMODE, port, SWITCH, BOOLEANMODE};
  char inbuf[3];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Set the sensor type of the input port defined by the "port" parameter to a
 * sound sensor.  Use dba = true for the DBA reading, and dba = false for the
 * DB reading */
void NXT_USB::SetSensorSound(int port, bool dba)
{
  char outbuf[] = {NO_RESPONSE, SETINPUTMODE, port, 0, PCTFULLSCALEMODE};

  if (dba)
  {
  outbuf[3] = SOUND_DBA;
  }
  else
  {
  outbuf[3] = SOUND_DB;
  }    
  char inbuf[3];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Set the sensor type of the input port defined by the "port" parameter to a
 * ultrasonic sensor. */
void NXT_USB::SetSensorUS(int port)
{
  char outbuf[] = {RESPONSE, SETINPUTMODE, port, LOWSPEED_9V, RAWMODE};
  char inbuf[3];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Turn the ultrasonic sensor off. */
void NXT_USB::SetUSOff(int port)
{
  char outbuf[] = {RESPONSE, LSWRITE, port, 3, 0, US_ADDRESS, SET_US_MODE, US_MODE_OFF};
  char inbuf[3];
  this->usbConn->SendCommand(outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Set the ultrasonic sensor to single shot mode. */
void NXT_USB::SetUSSingleShot(int port)
{
  char outbuf[] = {RESPONSE, LSWRITE, port, 3, 0, US_ADDRESS, SET_US_MODE, US_MODE_SINGLESHOT};
  char inbuf[3];
  this->usbConn->SendCommand(outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Set the ultrasonic sensor to continuous mode. */
void NXT_USB::SetUSContinuous(int port)
{
  char outbuf[] = {RESPONSE, LSWRITE, port, 3, 0, US_ADDRESS, SET_US_MODE, US_MODE_CONTINUOUS};
  char inbuf[3];
  this->usbConn->SendCommand(outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Set the ultrasonic sensor to event capture mode. */
void NXT_USB::SetUSEventCapture(int port)
{
  char outbuf[] = {RESPONSE, LSWRITE, port, 3, 0, US_ADDRESS, SET_US_MODE, US_MODE_EVENTCAPTURE};
  char inbuf[3];
  this->usbConn->SendCommand(outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Set the ultrasonic sensor to continuous mode, with a defined interval. */
void NXT_USB::SetUSContinuousInterval(int port, int interval)
{
  char outbuf[] = {RESPONSE, LSWRITE, port, 3, 0, US_ADDRESS, SET_US_CONTINUOUSINTERVAL, interval};
  char inbuf[3];
  this->usbConn->SendCommand(outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Read from the light sensor (returns -1 on error). */
int NXT_USB::GetLightSensor(int port)
{
  char outbuf[] = {RESPONSE, GETINPUTVALUES, port};
  char inbuf[16];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
  return (((int) inbuf[13])*256) + ((int) inbuf[12]);
}

/** Read from the touch sensor (returns -1 on error). */
bool NXT_USB::GetTouchSensor(int port)
{
  char outbuf[] = {RESPONSE, GETINPUTVALUES, port};
  char inbuf[16];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
  return (bool) (((int) inbuf[13])*256) + ((int) inbuf[12]);
}

/** Read from the sound sensor (returns -1 on error). */
int NXT_USB::GetSoundSensor(int port)
{
  char outbuf[] = {RESPONSE, GETINPUTVALUES, port};
  char inbuf[16];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
  return (((int) inbuf[13])*256) + ((int) inbuf[12]);
}

/** Read from the ultrasonic sensor (returns -1 on error). */
int NXT_USB::GetUSSensor(int port)
{
  //command the sensor to read one byte
  char outbuf[] = {RESPONSE, LSWRITE, port, 2, 1, US_ADDRESS, READ_US_BYTE0};
  char inbuf[3];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
    
  // wait until there are bytes to be read
  int lsStatus = 0;
  int i = 0; // timeout if we wait too long
  do
  {
    lsStatus = LSGetStatus(port);
    i++;
  } while (lsStatus == 0 && i < 10);

  // timed out
  if (lsStatus == 0)
  {
    return -1;
  }

  // perform read
  char outbuf2[] = {RESPONSE, LSREAD, port};
  char inbuf2[20];
  this->usbConn->SendCommand(outbuf2, sizeof(outbuf2), inbuf2, sizeof(inbuf2));    
  return (int) inbuf2[4];
}

/** Turn on the motor connected to the output port defined by the "port" parameter
 * at the power level defined by "power" (Power must be between -100 (100%
 * power reverse) and 100 (100% power forward).  The motor will turn indefinitely
 * until it is turned off */
void NXT_USB::SetMotorOn(int port, int power)
{
  if (power < -100 || power > 100)
  {
    return;
  }

  char outbuf[] = {NO_RESPONSE, SETOUTPUTSTATE, port, power, MOTORON | BRAKE | REGULATED, REGULATION_MODE_MOTOR_SPEED, 0, MOTOR_RUN_STATE_RUNNING, 0, 0, 0, 0, 0};
  char inbuf[3];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Turn on the motor connected to the output port defined by the "port" parameter
 * at the power level defined by "power" for a specified number of degrees
 *   Power must be between -100 (100% power reverse) and 100 (100% power forward).
 *   TachoCount is the absolute number of degrees to turn and must be greater than
 *   or equal to zero
 *   Use this function instead of MoveMotor if you want to be able to run other
 *   pieces of code while the motor is turning */
void NXT_USB::SetMotorOn(int port, int power, int tachoCount)
{
  if (power < -100 || power > 100 || tachoCount < 0)
  {
    return;
  }

  char byte8 = 0;
  char byte9 = 0;

  if (tachoCount > 256)
  {
    byte9 = tachoCount / 256;
    tachoCount = tachoCount - (byte9 * 256);    
  }
  if (tachoCount > 0)
  {
    byte8 = tachoCount;
  }

  char outbuf[] = {NO_RESPONSE, SETOUTPUTSTATE, port, power, MOTORON | BRAKE | REGULATED, REGULATION_MODE_MOTOR_SPEED, 0, MOTOR_RUN_STATE_RUNNING, byte8, byte9, 0, 0, 0};
  char inbuf[3];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Turn on the motor connected to the output port defined by the "port" parameter
 * at the power level defined by "power" for a specified number of degrees
 *   Power must be between -100 (100% power reverse) and 100 (100% power forward).
 *   TachoCount is the absolute number of degrees to turn and must be greater than
 *   or equal to zero
 *   Use this function instead of SetMotorOn if you want your program to stall until
 *   the motor has finished turning */
void NXT_USB::MoveMotor(int port, int power, int tachoCount)
{
  if (power < -100 || power > 100 || tachoCount <= 0)
  {
    return;
  }

    // figure out where we want to be
  int goalDegrees;
  if (power > 0)
  {
    goalDegrees = this->GetMotorRotation(port, false) + tachoCount;
  }
  else
  {
    goalDegrees = this->GetMotorRotation(port, false) - tachoCount;
  }

    //peform the movement
  this->SetMotorOn(port, power);
    int i = 0; // new
  if (power > 0)
  {
    while (this->GetMotorRotation(port, false) < goalDegrees) {}
    
  }
  else
  {
    while (this->GetMotorRotation(port, false) > goalDegrees) {}

  }    
  this->StopMotor(port, true);
}

/** Stop the motor connected to the output port defined by the "port" variable.
 * Use brake = true for an active brake, and brake = false if you just want to
 * stop the motor from continuing to turn */
void NXT_USB::StopMotor(int port, bool brake)
{
  char outbuf[] = {NO_RESPONSE, SETOUTPUTSTATE, port, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
  if (brake)
  {
    outbuf[4] = MOTORON | BRAKE | REGULATED;
    outbuf[5] = REGULATION_MODE_MOTOR_SPEED;
    outbuf[7] = MOTOR_RUN_STATE_RUNNING;
  }
  else
  {
    outbuf[4] = MFLOAT;
    outbuf[5] = REGULATION_MODE_IDLE;
    outbuf[7] = MOTOR_RUN_STATE_IDLE;
  } 
  char inbuf[3];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Get the current rotation of the motor.  Use relative = true to get the current
 *  number of degrees that the motor has rotated through since the last programmed
 *  movement, and relative = false to get the current number of degrees that the
 *  motor has rotated through since to the last reset */
int NXT_USB::GetMotorRotation(int port, bool relative)
{
  char outbuf[] = {RESPONSE, GETOUTPUTSTATE, port};
  char inbuf[25];
  this->usbConn->SendCommand(outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));

  if (relative)
  {
    return (((int) inbuf[20])*16777216) + (((int) inbuf[19])*65536) + (((int) inbuf[18])*256) + ((int) inbuf[17]);
  }
  else
  {
    return (((int) inbuf[24])*16777216) + (((int) inbuf[23])*65536) + (((int) inbuf[22])*256) + ((int) inbuf[21]);
  }
}

/** Reset the rotation sensor of the motor.  Use relative = true to reset the relative
 * number of degrees that the motor has rotated through since the last programmed
 * movement, and relative = false to get the current number of degrees that the
 * motor has rotated through since the last reset */
void NXT_USB::ResetMotorPosition(int port, bool relative)
{
  char outbuf[] = {NO_RESPONSE, RESETMOTORPOSITION, port, relative};
  char inbuf[3];
  this->usbConn->SendCommand(outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Play a tone on the LEGO Mindstorms NXT.  Frequency must be in the range
 * 200 - 14000 and duration (in ms) must be in the range 0-65535. */
void NXT_USB::PlayTone(int frequency, int duration)
{
  if (frequency < 200 || frequency > 14000 || duration < 0 || duration > 65535)
  {
    return;
  }

  char byte2 = 0;
  char byte3 = 0;
  char byte4 = 0;
  char byte5 = 0;

  if (frequency > 256)
  {
    byte3 = frequency / 256;
    frequency = frequency - (byte3 * 256);
  }
  if (frequency > 0)
  {
    byte2 = frequency;
  }

  if (duration> 256)
  {
    byte5 = duration / 256;
    duration = duration - (byte5 * 256);
  }
  if (duration > 0)
  {
    byte4 = duration;
  }
  char outbuf[] = {NO_RESPONSE, PLAYTONE, byte2, byte3, byte4, byte5};
  char inbuf[3];
  this->usbConn->SendCommand(outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
}

/** Get the device filename for the LEGO Mindstorms NXT device */
char * NXT_USB::GetDeviceFilename()
{
  return this->usbConn->GetDeviceFilename();;
}

/** Get the vendor ID number for the LEGO Mindstorms NXT device */
int NXT_USB::GetIDVendor()
{
  return this->usbConn->GetIDVendor();
}

/** Get the product ID number for the LEGO Mindstorms NXT device */
int NXT_USB::GetIDProduct()
{
  return this->usbConn->GetIDProduct();
}

/** Get a textual version of the status of the USB connection */
char * NXT_USB::GetStatus()
{
  return this->usbConn->GetStatus();
}

/** Sends a direct command to the Lego NXT over USB.  This function should be used by only advanced users.
    -outbuf - the direct command (see the official LEGO Mindstorms documentation for direct commands)
    -outbufSize - the size of the outbuf array
    -inbuf - the response message, NULL if no response is needed
    -inbufSize - the size of the response message, 0 if no response is needed
*/
void NXT_USB::SendCommand (char * outbuf, int outbufSize, char * inbuf, int inbufSize)
{
  return this->usbConn->SendCommand(outbuf, outbufSize, inbuf, inbufSize);
}

/** Get the LS status of the Lego Mindstorms NXT - should only be used by advanced
* users */
int NXT_USB::LSGetStatus(int port)
{
  char outbuf[] = {RESPONSE, LSGETSTATUS, port};
  char inbuf[4];
  this->usbConn->SendCommand (outbuf, sizeof(outbuf), inbuf, sizeof(inbuf));
  return (int) inbuf[3];
}
