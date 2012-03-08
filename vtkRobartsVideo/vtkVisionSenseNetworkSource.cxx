/*=========================================================================

  File: vtkVisionSenseNetworkSource.cxx
  Author: Kyle Charbonneau <kcharbon@imaging.robarts.ca>
  Language: C++
  Description: 
     A vtk class to pull network images from the VisionSense
     stereo endoscope.
  Notes:
     The endoscope works as such:
     1. Start a TCP connection with it.
     2. It first sends a single frame header with information on the size
        of the image.
              struct FRAME_HEADER {
                int width;
                int height;
                int bpp;
                short channels;   
              };
     3. It then constantly sends out a frame header + 2 images until the
        connection is closed.
     
=========================================================================*/

#include "vtkObjectFactory.h"
#include "vtkMultiThreader.h"
#include "vtkTimerLog.h"
#include "vtkMutexLock.h"

#include "vtkVisionSenseNetworkSource.h"

vtkVisionSenseNetworkSource* vtkVisionSenseNetworkSource::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkVisionSenseNetworkSource");
  if (ret)
    {
    return (vtkVisionSenseNetworkSource*) ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkVisionSenseNetworkSource;
}

vtkVisionSenseNetworkSource::vtkVisionSenseNetworkSource()
{
  this->RemotePort = 0;
  this->RemoteAddress = NULL;
  
  this->CurrentFrame = 0;
  
  this->CurrentBuffer = 0;
  this->MaxFrameRate = 15;
  
  this->Connected = 0;
  this->Streaming = 0;
  
  this->ThreadId = -1;
  this->Threader = vtkMultiThreader::New();
  
  this->SwitchLeft = 0;
  this->SwitchRight = 0;
}

vtkVisionSenseNetworkSource::~vtkVisionSenseNetworkSource()
{
  this->Disconnect();
}

//----------------------------------------------------------------------------
void vtkVisionSenseNetworkSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

int vtkVisionSenseNetworkSource::Connect()
{
 if (this->Connected) 
  {
    return -1;
  }

  if (this->RemoteAddress == NULL) 
  {
    vtkErrorMacro(<<"Initialize: Remote Address was not set!");
    return -1;
  }

  if (this->RemotePort == 0) 
  {
    vtkErrorMacro(<<"Initialize: Remote Port was not set!");
    return -1;
  }

  #ifdef _WIN32 // Windows socket code

    // Create and bind a local socket
    unsigned short ver = MAKEWORD(2, 2);
    WSADATA wsad;
    WSAStartup(ver, &wsad);
    this->DataSocket = socket(AF_INET, SOCK_STREAM, 0);

    // Fill in the remote address structure
    SOCKADDR_IN saddr;
    saddr.sin_addr.S_un.S_addr = inet_addr(this->RemoteAddress);
    saddr.sin_family = AF_INET;
    saddr.sin_port = htons(this->RemotePort);

    // Try to connect the socket to the endoscope
    if (connect(this->DataSocket, (sockaddr *) &saddr, sizeof(saddr)) == SOCKET_ERROR) 
    {
      vtkErrorMacro(<<"Initialize: Could not connect to camera!");
      return -1;
    }

  #elif defined(__FreeBSD__) || defined(__linux__) || defined(sgi) // Linux socket code

    this->DataSocket = socket(PF_INET, SOCK_STREAM, 0);

    struct sockaddr_in dest_addr;

    dest_addr.sin_family = AF_INET;          
    dest_addr.sin_port = htons(this->RemotePort);
    dest_addr.sin_addr.s_addr = inet_addr(this->RemoteAddress);
    memset(dest_addr.sin_zero, '\0', sizeof dest_addr.sin_zero);

    // Try to connect the socket to the endoscope
    if (connect(this->DataSocket, (struct sockaddr *) &dest_addr, sizeof(dest_addr)) == SOCKET_ERROR) 
    {
      vtkErrorMacro(<<"Initialize: Could not connect to camera!");
      return -1;
    }

  #endif

  this->Connected = 1;

  // Receive the initial header with frame information
  if (this->RecieveIntoBuffer((char *) &this->ImageHeader, sizeof(FRAME_HEADER)) == -1)
  {
    vtkErrorMacro(<<"Initialize: Could not retreive header from camera!");
    this->Disconnect();
    return -1;
  }

  // Calculate image requirements and allocate buffers
  this->ImageSize = this->ImageHeader.width*this->ImageHeader.height*this->ImageHeader.bpp;
  this->ImageBuffer1 = new char[this->ImageSize*2+sizeof(FRAME_HEADER)];
  this->ImageBuffer2 = new char[this->ImageSize*2+sizeof(FRAME_HEADER)];

  return 0;
}

void vtkVisionSenseNetworkSource::Disconnect()
{
  if (this->Streaming) 
  {
    this->Stop();
  }
  
  if (this->Connected)
  {
    // Close remote connection
    #ifdef _WIN32
      closesocket(this->DataSocket);
    #elif defined(__FreeBSD__) || defined(__linux__) || defined(sgi)
      close(this->DataSocket);
    #endif
    
    // Free memory
    delete this->ImageBuffer1;
    delete this->ImageBuffer2;
    
    this->Connected = 0;
  }
}

int vtkVisionSenseNetworkSource::GetWidth()
{
  if (!this->Connected)
  {
    vtkErrorMacro(<<"GetWidth: Not connected yet!");
    return -1;
  }
  return this->ImageHeader.width;
}

int vtkVisionSenseNetworkSource::GetHeight()
{
  if (!this->Connected)
  {
    vtkErrorMacro(<<"GetHeight: Not connected yet!");
    return -1;
  }
  return this->ImageHeader.height;
}

void vtkVisionSenseNetworkSource::SetRemoteAddress(char *address)
{
  this->RemoteAddress = address;
}

/* This gets a pointer to the current location of the right buffer
 * with an image in it, it constantly changes as new images are
 * donwloaded
 */
char *vtkVisionSenseNetworkSource::GetRightBuffer() 
{
  if (this->SwitchRight)
    this->SwitchRight = 0;
  
  if (this->CurrentBuffer) 
  {
    return this->ImageBuffer2+sizeof(FRAME_HEADER)+this->ImageSize;
  } 
  else 
  {
    return this->ImageBuffer1+sizeof(FRAME_HEADER)+this->ImageSize;
  }

}
/* This gets a pointer to the current location of the left buffer
 * with an image in it, it constantly changes as new images are
 * donwloaded
 */
char *vtkVisionSenseNetworkSource::GetLeftBuffer() 
{
  if (this->SwitchLeft)
    this->SwitchLeft = 0;
  
  if (this->CurrentBuffer) 
  {
    return this->ImageBuffer2+sizeof(FRAME_HEADER);
  } 
  else 
  {
    return this->ImageBuffer1+sizeof(FRAME_HEADER);
  }
}

// Download length bytes into buffer
int vtkVisionSenseNetworkSource::RecieveIntoBuffer(char *buffer, int length)
{
  int received = 0;
  int val;

  do {
    val = recv(this->DataSocket, buffer+received, length-received, 0);
    if (val == SOCKET_ERROR) 
    {
      return -1;
    }
    
    received = received + val;
  } while (received < length);

  return 0;
}

// Switch between buffers
inline void vtkVisionSenseNetworkSource::SwitchBuffers() 
{
  if (this->CurrentBuffer) 
  {
    this->CurrentBuffer = 0;
  }
  else
  {
    this->CurrentBuffer = 1;
  }
  this->SwitchLeft = 1;
  this->SwitchRight = 1;
}

void vtkVisionSenseNetworkSource::GetFrame()
{
  // Make sure buffers have been switched before starting to
  // avoid tearing of images
  if (this->SwitchLeft || this->SwitchRight)
  {
    return;
  }
  
  // Read into correct buffer
  char *buffer;
  if (this->CurrentBuffer) 
  {
    buffer = this->ImageBuffer1;
  } else {
    buffer = this->ImageBuffer2;
  }

  if (this->RecieveIntoBuffer(buffer, sizeof(FRAME_HEADER)+this->ImageSize*2) == -1) 
  {
    return;
  }
  
  this->CurrentFrame++;
  this->SwitchBuffers();
}

static inline void vtkSleep(double duration)
{
  duration = duration; // avoid warnings
  // sleep according to OS preference
  #ifdef _WIN32
    Sleep((int)(1000*duration));
  #elif defined(__FreeBSD__) || defined(__linux__) || defined(sgi)
    struct timespec sleep_time, dummy;
    sleep_time.tv_sec = (int)duration;
    sleep_time.tv_nsec = (int)(1000000000*(duration-sleep_time.tv_sec));
    nanosleep(&sleep_time,&dummy);
  #endif
}

static int vtkThreadSleep(vtkMultiThreader::ThreadInfo *data, double time)
{
  // loop either until the time has arrived or until the thread is ended
  for (int i = 0;; i++)
  {
    double remaining = time - vtkTimerLog::GetUniversalTime();

    // check to see if we have reached the specified time
    if (remaining <= 0)
    {
      return 1;
    }
    
    // check the ActiveFlag at least every 0.1 seconds
    if (remaining > 0.1)
    {
      remaining = 0.1;
    }

    // check to see if we are being told to quit 
    data->ActiveFlagLock->Lock();
    int activeFlag = *(data->ActiveFlag);
    data->ActiveFlagLock->Unlock();

    if (activeFlag == 0)
    {
      break;
    }

    vtkSleep(remaining);
  }

  return 0;
}

static void *vtkVisionSenseNetworkThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkVisionSenseNetworkSource *self = (vtkVisionSenseNetworkSource *)(data->UserData);
  
  double startTime = vtkTimerLog::GetUniversalTime();
  double rate = self->GetMaxFrameRate();
  int frame = 0;

  do
  {
    frame++;
    self->GetFrame();
  }
  while (vtkThreadSleep(data, startTime + frame/rate));
  
  return NULL;
}

// Stops the thread that is downloading images
void vtkVisionSenseNetworkSource::Stop()
{
  if (this->Streaming)
  {
    this->Threader->TerminateThread(this->ThreadId);
    this->ThreadId = -1;
    this->Streaming = 0;
    this->Modified();
  }
}

// Starts a thread that constantly downloads new images for a video source to use
void vtkVisionSenseNetworkSource::Start()
{

    if (!this->Connected)
    {
      vtkErrorMacro(<<"Start: vtkVisionSenseNetworkSource must be properly initialized before recording!"); 
      return;
    }
    
    if (!this->Streaming) 
    {
      this->Streaming = 1;
      this->Modified();
      this->CurrentFrame = 0;
      this->ThreadId = this->Threader->SpawnThread((vtkThreadFunctionType) &vtkVisionSenseNetworkThread, this);
    }
    
}
