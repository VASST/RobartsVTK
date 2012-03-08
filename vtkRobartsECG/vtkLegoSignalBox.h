/*=========================================================================

  Program:   Lego Signal Box for VTK
  Module:    $RCSfile: vtkLegoSignalBox.h,v $
  Creator:   Danielle Pace <dpace@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: dpace $
  Date:      $Date: 2008/12/12 00:18:17 $
  Version:   $Revision: 1.4 $

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
vBE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.

=========================================================================*/
// .NAME vtkLegoSignalBox - trigger switch using Lego light sensor

#ifndef __vtkLegoSignalBox_h
#define __vtkLegoSignalBox_h

#include "vtkSignalBox.h"
#include "NXT_USB.h"
#include <fstream> // for writing to file

class vtkDoubleArray;
class vtkIntArray;
class vtkMultiThreader;

class vtkLegoSignalBox : public vtkSignalBox
{
public:
  static vtkLegoSignalBox *New();
  vtkTypeMacro(vtkLegoSignalBox,vtkSignalBox);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Initialize the connection to the LEGO robot
  void Initialize();

  // Description
  // Wait until button touch to start
  void Start();

  // Description:
  // Stop triggering
  void Stop();

  // Description:
  // Port numbers for the motor, touch sensor and light sensor
  // These are the ports on the NXT (1-3 for A-C, 1-4 for 1-4);
  void SetMotorPort(int port);
  int GetMotorPort();
  void SetTouchPort(int port);
  int GetTouchPort();
  void SetLightPort(int port);
  int GetLightPort();

  // Description:
  // Set the motor's power as a percentage from 10 to 100
  // (should be a multiple of 10)
  void SetMotorPower(int power);
  vtkGetMacro(MotorPower, int);

  // Description:
  // Prints the 
  void SetPrintToFile(int toPrint);
  vtkGetMacro(PrintToFile, int);

  // Description:
  // Use the touch sensor to toggle the motor on and off
  // (default off)
  vtkSetMacro(UseTouchSensor, int);
  vtkGetMacro(UseTouchSensor, int);
  vtkBooleanMacro(UseTouchSensor, int);

  // Description:
  // Checks to see if the touch sensor has been hit, and turns
  // the motor on and off alternately
  void CheckForTouch();

protected:

  vtkLegoSignalBox();
  ~vtkLegoSignalBox();

  // Description:
  // Close the connection to the LEGO robot
  void CloseConnection();

  // Description:
  // Get the ECG signal and update phase and beating rate
  int GetECG(void);

  // Description:
  // Turn the motor on or off
  void StartMotor();
  void StopMotor();

  // both of these are dependent on the motor power
  int WindowSize; // the size of the window we are looking back on
  double SpikeHeight; // the size of the spike we are looking for
  double MinSpikeInterval; // minimum period

  NXT_USB Nxtusb;  // handle the connection to the lego robot

  int MotorPort;
  int TouchPort;
  int LightPort;
  int MotorPower;
  int MotorOn;

  // holds the previous light signal values so that we can look back
  // to see if we are at a new trigger point
  // size is window size, which is dependent on the motor power
  vtkIntArray* SignalBuffer;
  int SignalBufferIndex;

  int MaxSignalInWindow;
  int MinSignalInWindow;
  int IndicesSinceMaxSignalInWindow;
  int IndicesSinceMinSignalInWindow;

  int UseTouchSensor;

  int PrintToFile;
  ofstream OutStream; // for printing to file

  int OnFirstSignal;

private:
  vtkLegoSignalBox(const vtkLegoSignalBox&);
  void operator=(const vtkLegoSignalBox&);  
};

#endif

