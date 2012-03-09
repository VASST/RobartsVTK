/*=========================================================================

  File: vtkEpiphanVideoSource.cxx
  Author: Chris Wedlake <cwedlake@robarts.ca>

  Language: C++
  Description: 
     
=========================================================================

  Copyright (c) Chris Wedlake, cwedlake@robarts.ca

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

#include "vtkEpiphanVideoSource.h"
#include "vtkTimerLog.h"
#include "vtkObjectFactory.h"
#include "vtkCriticalSection.h"
#include "vtkUnsignedCharArray.h"
#include "vtkMutexLock.h"
#include "vtkVisionSenseNetworkSource.h"

#include <vtkstd/string> 

vtkEpiphanVideoSource* vtkEpiphanVideoSource::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkEpiphanVideoSource");
  if(ret)
    {
    return (vtkEpiphanVideoSource*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkEpiphanVideoSource;
}

//----------------------------------------------------------------------------
vtkEpiphanVideoSource::vtkEpiphanVideoSource()
{
  int i;
  
  this->Initialized = 0;

  this->AutoAdvance = 1;

  this->FrameSize[0] = 320;
  this->FrameSize[1] = 240;
  this->FrameSize[2] = 1;

  for (i = 0; i < 6; i++)
    {
    this->FrameBufferExtent[i] = 0;
    }
  
  this->Playing = 0;
  this->Recording = 0;

  this->SetFrameRate(30);

  this->FrameCount = 0;
  this->FrameIndex = -1;

  this->StartTimeStamp = 0;
  this->FrameTimeStamp = 0;

  this->OutputNeedsInitialization = 1;

  this->OutputFormat = VTK_LUMINANCE;
  this->NumberOfScalarComponents = 1;

  this->NumberOfOutputFrames = 1;

  this->Opacity = 1.0;

  for (i = 0; i < 3; i++)
    {
    this->ClipRegion[i*2] = 0;
    this->ClipRegion[i*2+1] = VTK_INT_MAX;
    this->OutputWholeExtent[i*2] = 0;
    this->OutputWholeExtent[i*2+1] = -1;
    this->DataSpacing[i] = 1.0;
    this->DataOrigin[i] = 0.0;
    }

  for (i = 0; i < 6; i++)
    {
    this->LastOutputExtent[i] = 0;
    }
  this->LastNumberOfScalarComponents = 0;

  this->FlipFrames = 0;

  this->PlayerThreader = vtkMultiThreader::New();
  //this->PlayerThreader->SingleMethodExecute();
  this->PlayerThreadId = -1;

  this->FrameBufferMutex = vtkCriticalSection::New();

  this->FrameBufferSize = 0;
  this->FrameBuffer = NULL;
  this->FrameBufferTimeStamps = NULL;
  this->FrameBufferIndex = 0;
  this->SetFrameBufferSize(1);

  this->FrameBufferBitsPerPixel = 8;
  this->FrameBufferRowAlignment = 1;

  this->SetNumberOfInputPorts(0);


  this->Initialized = 0;
  this->pauseFeed = 0;
  this->status = V2U_GRABFRAME_STATUS_OK;
  this->fg = NULL;
  this->cropRect = new V2URect;

  for (unsigned int i =0; i < 15; i++){
	  this->serialNumber[i] ='\0';
  }
}

//----------------------------------------------------------------------------
vtkEpiphanVideoSource::~vtkEpiphanVideoSource()
{
  this->vtkEpiphanVideoSource::ReleaseSystemResources();
}  

//----------------------------------------------------------------------------
void vtkEpiphanVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent); 
}

//----------------------------------------------------------------------------
void vtkEpiphanVideoSource::Initialize()
{
  if (this->Initialized) 
  {
    return;
  }

  // Setup some needed values
  vtkVideoSource::SetOutputFormat(VTK_RGB);

  FrmGrabNet_Init();

  char input[15];
  strncpy_s(input, "sn:", 15);
  strncat_s(input, this->serialNumber, 15);

  this->fg = FrmGrab_Open(this->serialNumber);
  
  if (this->fg) {
	this->fg = FrmGrabLocal_Open();
	vtkErrorMacro(<<"Epiphan Device with set serial number not found, looking for any available device instead");
  }

  if (this->fg != NULL) {
	  this->Initialized = 1;
  } else {
	  vtkErrorMacro(<<"Epiphan Device Not found");
	  return;
  }
  FrmGrab_SetMaxFps(this->fg, 25.0);

  // Initialization worked
  this->Initialized = 1;

  // Update frame buffer  to reflect any changes
  this->UpdateFrameBuffer();
}  

//----------------------------------------------------------------------------
void vtkEpiphanVideoSource::ReleaseSystemResources()
{
  this->Initialized = 0;
  if (this->fg != NULL) {
	FrmGrab_Close(this->fg);
  }
}

void vtkEpiphanVideoSource::InternalGrab()
{

  // get a thread lock on the frame buffer
  this->FrameBufferMutex->Lock();

  // Get pointer to data from the network source

  this->cropRect->x = this->ClipRegion[0];
  this->cropRect->width = this->ClipRegion[1]-this->ClipRegion[0];
  this->cropRect->y = this->ClipRegion[2];
  this->cropRect->height = this->ClipRegion[3]-this->ClipRegion[2];

  V2U_GrabFrame2 * frame = FrmGrab_Frame(this->fg, V2U_GRABFRAME_FORMAT_RGB24, cropRect);
  if (frame == NULL) {
	  this->FrameBufferMutex->Unlock();
	  return;
  }
  if (frame->retcode != V2UERROR_OK) { 
	  cout << "Error: " << frame->retcode << endl;
	  this->FrameBufferMutex->Unlock();
	  return;
  } 


  if (this->AutoAdvance)
    {
    this->AdvanceFrameBuffer(1);
    if (this->FrameIndex + 1 < this->FrameBufferSize)
      {
      this->FrameIndex++;
      }
    }

  int index = this->FrameBufferIndex % this->FrameBufferSize;
  while (index < 0)
    {
    index += this->FrameBufferSize;
    }

  char *buffer = (char *)frame->pixbuf;
  
  // Get a pointer to the location of the frame buffer
  char *ptr = (char *) reinterpret_cast<vtkUnsignedCharArray *>(this->FrameBuffer[index])->GetPointer(0);
  
  // Copy image into frame buffer
  memcpy(ptr, buffer, frame->pixbuflen);

  FrmGrab_Release(this->fg, frame);
  this->FrameBufferTimeStamps[index] = vtkTimerLog::GetUniversalTime();

  if (this->FrameCount++ == 0)
    {
    this->StartTimeStamp = this->FrameBufferTimeStamps[index];
    }

  this->Modified();

  this->FrameBufferMutex->Unlock();
}

void vtkEpiphanVideoSource::UpdateFrameBuffer()
{
  int i, oldExt;
  int ext[3];
  vtkDataArray *buffer;

  // clip the ClipRegion with the FrameSize
  for (i = 0; i < 3; i++)
    {
    oldExt = this->FrameBufferExtent[2*i+1] - this->FrameBufferExtent[2*i] + 1;
    this->FrameBufferExtent[2*i] = ((this->ClipRegion[2*i] > 0) 
                             ? this->ClipRegion[2*i] : 0);  
    this->FrameBufferExtent[2*i+1] = ((this->ClipRegion[2*i+1] < 
                                       this->FrameSize[i]-1) 
                             ? this->ClipRegion[2*i+1] : this->FrameSize[i]-1);

    ext[i] = this->FrameBufferExtent[2*i+1] - this->FrameBufferExtent[2*i] + 1;
    if (ext[i] < 0)
      {
      this->FrameBufferExtent[2*i] = 0;
      this->FrameBufferExtent[2*i+1] = -1;
      ext[i] = 0;
      }

    if (oldExt > ext[i])
      { // dimensions of framebuffer changed
      this->OutputNeedsInitialization = 1;
      }
    }

  // total number of bytes required for the framebuffer
  int bytesPerRow = ext[0]*(this->FrameBufferBitsPerPixel/8);
  bytesPerRow = ((bytesPerRow + this->FrameBufferRowAlignment - 1) /
                 this->FrameBufferRowAlignment)*this->FrameBufferRowAlignment;
  int totalSize = bytesPerRow * ext[1] * ext[2];
  i = this->FrameBufferSize;

  while (--i >= 0)
    {
    buffer = reinterpret_cast<vtkDataArray *>(this->FrameBuffer[i]);
    if (buffer->GetDataType() != VTK_UNSIGNED_CHAR ||
        buffer->GetNumberOfComponents() != 1 ||
        buffer->GetNumberOfTuples() != totalSize)
      {
      buffer->Delete();
      buffer = vtkUnsignedCharArray::New();
      this->FrameBuffer[i] = buffer;
      buffer->SetNumberOfComponents(1);
      buffer->SetNumberOfTuples(totalSize);
      }
    }
}

void vtkEpiphanVideoSource::SetSerialNumber(char * serial) {
	strncpy_s(this->serialNumber, serial, 15);
}

//----------------------------------------------------------------------------
// platform-independent sleep function
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

//----------------------------------------------------------------------------
// Sleep until the specified absolute time has arrived.
// You must pass a handle to the current thread.  
// If '0' is returned, then the thread was aborted before or during the wait.
static int vtkThreadSleep(vtkMultiThreader::ThreadInfo *data, double time)
{
  // loop either until the time has arrived or until the thread is ended
  for (int i = 0;; i++)
    {
    double remaining = time - vtkTimerLog::GetUniversalTime();

    // check to see if we have reached the specified time
    if (remaining <= 0)
      {
      if (i == 0)
        {
        vtkGenericWarningMacro("Dropped a video frame.");
        }
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

//----------------------------------------------------------------------------
// this function runs in an alternate thread to asyncronously grab frames
static void *vtkEpiphanVideoSourceRecordThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkEpiphanVideoSource *self = (vtkEpiphanVideoSource *)(data->UserData);
  
  double startTime = vtkTimerLog::GetUniversalTime();
  double rate = self->GetFrameRate();
  int frame = 0;

  do
    {
    self->InternalGrab();
    frame++;
    }
  while (vtkThreadSleep(data, startTime + frame/rate));

  return NULL;
}

//----------------------------------------------------------------------------
// Set the source to grab frames continuously.
// You should override this as appropriate for your device.  
void vtkEpiphanVideoSource::Record()
{
  if (this->Playing)
    {
    this->Stop();
    }

  if (!this->Recording)
    {
    this->Initialize();

    this->Recording = 1;
    this->FrameCount = 0;
	this->pauseFeed = 0;
    this->Modified();
    this->PlayerThreadId = 
      this->PlayerThreader->SpawnThread((vtkThreadFunctionType)\
                                &vtkEpiphanVideoSourceRecordThread,this);
    }
}

//----------------------------------------------------------------------------
// this function runs in an alternate thread to 'play the tape' at the
// specified frame rate.
static void *vtkEpiphanVideoSourcePlayThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkVideoSource *self = (vtkVideoSource *)(data->UserData);
 
  double startTime = vtkTimerLog::GetUniversalTime();
  double rate = self->GetFrameRate();
  int frame = 0;

  do
    {
    self->Seek(1);
    frame++;
    }
  while (vtkThreadSleep(data, startTime + frame/rate));

  return NULL;
}
 
//----------------------------------------------------------------------------
// Set the source to play back recorded frames.
// You should override this as appropriate for your device.  
void vtkEpiphanVideoSource::Play()
{
  if (this->Recording)
    {
    this->Stop();
    }

  if (!this->Playing)
    {
    this->Initialize();

    this->Playing = 1;
    this->Modified();
    this->PlayerThreadId = 
      this->PlayerThreader->SpawnThread((vtkThreadFunctionType)\
                                        &vtkEpiphanVideoSourcePlayThread,this);
    }
}

//----------------------------------------------------------------------------
// Stop continuous grabbing or playback.  You will have to override this
// if your class overrides Play() and Record()
void vtkEpiphanVideoSource::Stop()
{
  if (this->Playing || this->Recording)
    {
    this->PlayerThreader->TerminateThread(this->PlayerThreadId);
    this->PlayerThreadId = -1;
    this->Playing = 0;
    this->Recording = 0;
    this->Modified();
    }
} 

void vtkEpiphanVideoSource::Pause() {
	this->pauseFeed = 1;
}
void vtkEpiphanVideoSource::UnPause() {
	this->pauseFeed = 0;
}