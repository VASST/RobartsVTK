/*
Modified from NXT++
http://nxtpp.sourceforge.net/

And Device::USB
http://search.cpan.org/~gwadej/Device-USB-0.21/lib/Device/USB.pm
*/

/*=========================================================================
  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: NXT_USB_linux.h,v $
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

/** \class NXT_USB_linux
 * \brief Interface to USB devices for Linux machines, using libusb
 *
 * NXT_USB_linux is used by the class NXT_USB to open and close a USB connection
 * between a Linux computer and a LEGO Mindstorms NXT robot, and also to send
 * commands to the robot.
 */


#ifndef __NXT_USB_linux_h
#define __NXT_USB_linux_h

#include "usb.h"

class NXT_USB_linux
{
public:
  NXT_USB_linux();
  ~NXT_USB_linux();

  // functions
  int OpenLegoUSB();
  int CloseLegoUSB();
  char * GetDeviceFilename();
  int GetIDVendor();
  int GetIDProduct();
  char * GetStatus();
  void SendCommand (char * outbuf, int outbufSize, char * inbuf, int inbufSize);

  // attributes
  private:
    struct usb_dev_handle *pUSBHandleLego;
    struct usb_device *devLego;
    int status;

};

#endif



