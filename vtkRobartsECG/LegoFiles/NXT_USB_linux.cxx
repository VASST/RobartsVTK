/*
Modified from NXT++
http://nxtpp.sourceforge.net/

And Device::USB
http://search.cpan.org/~gwadej/Device-USB-0.21/lib/Device/USB.pm
*/

/*=========================================================================
  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: NXT_USB_linux.cxx,v $
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

#include <cstring>
#include <iostream>
#include "NXT_USB_linux.h"

// response codes for direct commands
static const char RESPONSE = 0x00;
static const char NO_RESPONSE = 0x80;

//connection constants
static const int USB_ID_VENDOR_LEGO = 0x0694;
static const int USB_ID_PRODUCT_NXT = 0x0002;
static const int USB_INTERFACE = 0;
static const int USB_OUT_ENDPOINT = 0x01;
static const int USB_IN_ENDPOINT = 0x82;
static const int USB_TIMEOUT = 1000;

// status
static const int NOT_CONNECTED = 0;
static const int NOT_FOUND = 1;
static const int NOT_CLAIMED = 2;
static const int NOT_CLAIMED_AFTER_RESET = 3;
static const int INTERFACE_NOT_CLAIMED = 4;
static const int CONNECTED = 5;
static const int INTERFACE_NOT_RELEASED = 6;
static const int NOT_RELEASED = 7;


/** Constructor */
NXT_USB_linux::NXT_USB_linux()
{
  this->devLego = 0;
  this->pUSBHandleLego = 0;
  this->status = NOT_CONNECTED;
}

/** Destructor */
NXT_USB_linux::~NXT_USB_linux()
{
  if (this->status == CONNECTED)
  {
    this->CloseLegoUSB();
  }
}

/** Open the USB connection for the LEGO NXT device */
int NXT_USB_linux::OpenLegoUSB()
{
  usb_init();
  usb_find_busses();
  usb_find_devices();
    
  struct usb_bus *bus = 0;
  bool found = false;

  for (bus = usb_busses; 0 != bus && !found; bus = bus->next)
  {
    struct usb_device *temp_dev = 0;
    for (temp_dev = bus->devices; 0 != temp_dev && !found; temp_dev = temp_dev->next)
    {
      if ((temp_dev->descriptor.idVendor == USB_ID_VENDOR_LEGO) &&
      (temp_dev->descriptor.idProduct == USB_ID_PRODUCT_NXT))
      {
        this->devLego = temp_dev;
        found = true;
      }
    }
  }

  if (!found)
  {
    std::cerr << GetStatus() << std::endl;
    this->status = NOT_FOUND;
    return 0;
  }

  this->pUSBHandleLego = usb_open(this->devLego);

  if (this->pUSBHandleLego == 0) {
    std::cerr << GetStatus() << std::endl;
    this->status = NOT_CLAIMED;
    return 0;
  }

// libusb-win32 looses everythin after reset (not needed)
#ifndef WIN32
  usb_reset(this->pUSBHandleLego);
  this->pUSBHandleLego = usb_open(this->devLego);

  if (this->pUSBHandleLego == 0) {
    std::cerr << GetStatus() << std::endl;
    this->status = NOT_CLAIMED_AFTER_RESET;
    return 0;
  }
#endif

#ifdef WIN32
  usb_set_configuration(this->pUSBHandleLego,1);
#endif

  int interfaceReturn = -1;
  if (this->devLego->config)
  {
    if (this->devLego->config->interface)
    {
      if (this->devLego->config->interface->altsetting)
      {
      interfaceReturn = usb_claim_interface(this->pUSBHandleLego, this->devLego->config->interface->altsetting->bInterfaceNumber);
      }
    }
  }

  if (interfaceReturn < 0)
  {
    std::cerr << GetStatus() << std::endl;
    this->status = INTERFACE_NOT_CLAIMED;
    return 0;
  }

  this->status = CONNECTED;
  return 1;

} // end Open

/** Closes the USB connection for the Lego NXT device*/
int NXT_USB_linux::CloseLegoUSB()
{
  int interfaceRelease = -1;
  if (this->devLego->config)
  {
    if (this->devLego->config->interface)
    {
      if (this->devLego->config->interface->altsetting)
      {
        interfaceRelease = usb_release_interface(this->pUSBHandleLego, devLego->config->interface->altsetting->bInterfaceNumber);
      }
    }
  }
  if (interfaceRelease < 0)
  {
    std::cerr << GetStatus() << std::endl;
    this->status = INTERFACE_NOT_RELEASED;
    return 0;
  }

  int close = usb_close(this->pUSBHandleLego);
  if (close < 0) {
    std::cerr << GetStatus() << std::endl;
    this->status = NOT_RELEASED;
    return 0;
  }

  this->status = NOT_CONNECTED;
  return 1;
} // end Close

/** Get the device filename for the LEGO Mindstorms NXT device */
char * NXT_USB_linux::GetDeviceFilename()
{
  return this->devLego->filename;
}

/** Get the vendor ID number for the LEGO Mindstorms NXT device */
int NXT_USB_linux::GetIDVendor()
{
  return (int) this->devLego->descriptor.idVendor;
}

/** Get the product ID numberfor the LEGO Mindstorms NXT device */
int NXT_USB_linux::GetIDProduct()
{
  return (int) this->devLego->descriptor.idProduct;
}
/** Get a textual version of the status of the USB connection */
char * NXT_USB_linux::GetStatus()
{
  if (this->status == NOT_CONNECTED)
    return "Not connected";
  if (this->status == NOT_FOUND)
    return "Lego NXT not found";
  if (this->status == NOT_CLAIMED)
    return "Not able to claim the Lego NXT device";
  if (this->status == NOT_CLAIMED_AFTER_RESET)
    return "Not able to claim the Lego NXT device after reset";
  if (this->status == INTERFACE_NOT_CLAIMED)
    return "Not able to claim the Lego NXT device";
  if (this->status == CONNECTED)
    return "Connected";
  if (this->status == INTERFACE_NOT_RELEASED)
    return "Not able to release the Lego NXT interface";
  //if (this->status == NOT_RELEASED)
  return "Not able to release the Lego NXT device";
}

/** Sends a direct command to the Lego NXT over USB.  This function should be used by
 * only advanced users.
 *  outbuf - the direct command (see the official LEGO Mindstorms documentation 
 *  for direct commands)
 *  outbufSize - the size of the outbuf array
 *  inbuf - the response message, NULL if no response is needed
 *  inbufSize - the size of the response message, 0 if no response is needed */
void NXT_USB_linux::SendCommand (char * outbuf, int outbufSize, char * inbuf, int inbufSize)
{
  usb_bulk_write(this->pUSBHandleLego, USB_OUT_ENDPOINT, outbuf, outbufSize, USB_TIMEOUT);

  usb_bulk_read(this->pUSBHandleLego, USB_IN_ENDPOINT, inbuf, inbufSize, USB_TIMEOUT);

} // end SendCommand
