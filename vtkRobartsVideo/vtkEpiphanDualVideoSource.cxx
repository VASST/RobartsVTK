/*=========================================================================

  File: vtkEpiphanDualVideoSource.cxx
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

#include "vtkEpiphanDualVideoSource.h"
#include "vtkTimerLog.h"
#include "vtkObjectFactory.h"
#include "vtkCriticalSection.h"
#include "vtkUnsignedCharArray.h"
#include "vtkMutexLock.h"
#include "vtkVisionSenseNetworkSource.h"

#include <vtkstd/string> 

vtkEpiphanDualVideoSource* vtkEpiphanDualVideoSource::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkEpiphanDualVideoSource");
  if(ret)
    {
    return (vtkEpiphanDualVideoSource*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkEpiphanDualVideoSource;
}

//----------------------------------------------------------------------------
vtkEpiphanDualVideoSource::vtkEpiphanDualVideoSource()
{

  this->Initialized = 0;
  this->pauseFeed = 0;
  this->status = V2U_GRABFRAME_STATUS_OK;
  this->fg = NULL;
  this->cropRect = new V2URect;

  this->FrameBufferBitsPerPixel = 24;
  this->vtkVideoSource::SetOutputFormat(VTK_RGB);
  this->vtkVideoSource::SetFrameBufferSize( 100 );
  this->vtkVideoSource::SetFrameRate( 25.0f );

  for (unsigned int i =0; i < 15; i++){
	  this->serialNumber[i] ='\0';
  }
  this->SetNumberOfOutputPorts(3);

}

//----------------------------------------------------------------------------
vtkEpiphanDualVideoSource::~vtkEpiphanDualVideoSource()
{
  this->vtkEpiphanDualVideoSource::ReleaseSystemResources();
  if (this->fg) {
	  FrmGrab_Deinit(); // not sure if this stops all devices??
  }
}  

//----------------------------------------------------------------------------
void vtkEpiphanDualVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent); 
}

//----------------------------------------------------------------------------
void vtkEpiphanDualVideoSource::Initialize()
{
  if (this->Initialized) 
  {
    return;
  }
  
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

  V2U_VideoMode vm;
  if (FrmGrab_DetectVideoMode(this->fg,&vm) && vm.width && vm.height) {
	  this->SetFrameSize(vm.width,vm.height,1);
	  //this->SetFrameRate((vm.vfreq+50)/1000);
  } else {
	vtkErrorMacro(<<"No signal detected");
  }

  FrmGrab_SetMaxFps(this->fg, 25.0);

  // Initialization worked
  this->Initialized = 1;
  
  // Update frame buffer  to reflect any changes
  this->UpdateFrameBuffer();
}  

//----------------------------------------------------------------------------
void vtkEpiphanDualVideoSource::ReleaseSystemResources()
{
  this->Initialized = 0;
  if (this->fg != NULL) {
	FrmGrab_Close(this->fg);
  }
}

void vtkEpiphanDualVideoSource::InternalGrab()
{

  // get a thread lock on the frame buffer
  this->FrameBufferMutex->Lock();

  this->cropRect->x = this->FrameBufferExtent[0];
  this->cropRect->width = this->FrameBufferExtent[1]-this->FrameBufferExtent[0]+1;
  this->cropRect->y = this->FrameBufferExtent[2];
  this->cropRect->height = this->FrameBufferExtent[3]-this->FrameBufferExtent[2]+1;
  
  //imgu *IA;
  V2U_GrabFrame2 * frame = NULL;

  V2U_UINT32 format = V2U_GRABFRAME_BOTTOM_UP_FLAG; // seems to be needed to orientate correctly.

  if (this->OutputFormat == VTK_LUMINANCE) {
	format |= V2U_GRABFRAME_FORMAT_YUY2;
  } else if (this->OutputFormat == VTK_RGB) {
	format |= V2U_GRABFRAME_FORMAT_RGB24;
  } else if (this->OutputFormat == VTK_RGBA) {
	format |= V2U_GRABFRAME_FORMAT_ARGB32;
  } else {
	  // no clue what format to grab, you can add more.
	  return;
  }

  frame= FrmGrab_Frame(this->fg, format, cropRect);

  if (frame == NULL || frame->imagelen <= 0) {
	  this->FrameBufferMutex->Unlock();
	  this->Stop();
	  return;
  }
  if (frame->retcode != V2UERROR_OK) { 
	  cout << "Error: " << frame->retcode << endl;
	  this->FrameBufferMutex->Unlock();
	  this->Stop();
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

  //imguAllocate(&IA, frame->mode.width, frame->mode.height,3);

  char *buffer = (char *)frame->pixbuf;
  
  // Get a pointer to the location of the frame buffer
  char *ptr = (char *) reinterpret_cast<vtkUnsignedCharArray *>(this->FrameBuffer[index])->GetPointer(0);

  char *ptrLeft = (char *) reinterpret_cast<vtkUnsignedCharArray *>(this->FrameBufferLeft[index])->GetPointer(0);
  char *ptrRight = (char *) reinterpret_cast<vtkUnsignedCharArray *>(this->FrameBufferLeft[index])->GetPointer(0);
  
  // Copy Full Buffer
  memcpy(ptr, buffer, frame->imagelen);

  // Copy Left Frame
  for (int height=0; height < this->ClipRegionLeft[3]-this->ClipRegionLeft[2]; height++) {
	  int padding = (this->ClipRegionLeft[1]-this->ClipRegionLeft[0]);
	  memcpy(ptrLeft+(padding*height), buffer+(this->cropRect->width*height), padding);
  }

  // Copy Right Frame
  for (int height=0; height < this->ClipRegionRight[3]-this->ClipRegionRight[2]; height++) {
	  int padding = (this->ClipRegionRight[1]-this->ClipRegionRight[0]);
	  memcpy(ptrRight+(padding*height), buffer+(this->cropRect->width*height), padding);
  }

  FrmGrab_Release(this->fg, frame);
  this->FrameBufferTimeStamps[index] = vtkTimerLog::GetUniversalTime();

  if (this->FrameCount++ == 0)
    {
    this->StartTimeStamp = this->FrameBufferTimeStamps[index];
    }

  this->Modified();

  this->FrameBufferMutex->Unlock();
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
static void *vtkEpiphanDualVideoSourceRecordThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkEpiphanDualVideoSource *self = (vtkEpiphanDualVideoSource *)(data->UserData);
  
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
// this function runs in an alternate thread to 'play the tape' at the
// specified frame rate.
static void *vtkEpiphanDualVideoSourcePlayThread(vtkMultiThreader::ThreadInfo *data)
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

void vtkEpiphanDualVideoSource::SetClipRegion(int x0, int x1, int y0, int y1, int z0, int z1) {
	vtkVideoSource::SetClipRegion(x0,x1,y0,y1,z0,z1);
}

void vtkEpiphanDualVideoSource::SetClipRegionLeft(int x0, int x1, int y0, int y1, int z0, int z1)
{
  if (this->ClipRegionLeft[0] != x0 || this->ClipRegionLeft[1] != x1 ||
	  this->ClipRegionLeft[2] != y0 || this->ClipRegionLeft[3] != y1 ||
	  this->ClipRegionLeft[4] != z0 || this->ClipRegionLeft[5] != z1)
	{
	this->Modified();
	if (this->Initialized) 
		{ // modify the FrameBufferExtent
		this->FrameBufferMutex->Lock();
		this->ClipRegionLeft[0] = x0; this->ClipRegionLeft[1] = x1;
		this->ClipRegionLeft[2] = y0; this->ClipRegionLeft[3] = y1;
		this->ClipRegionLeft[4] = z0; this->ClipRegionLeft[5] = z1;
		this->UpdateFrameBufferLeft();
		this->FrameBufferMutex->Unlock();
		}
	else
		{
		this->ClipRegionLeft[0] = x0; this->ClipRegionLeft[1] = x1;
		this->ClipRegionLeft[2] = y0; this->ClipRegionLeft[3] = y1;
		this->ClipRegionLeft[4] = z0; this->ClipRegionLeft[5] = z1;
		}
  }
}

void vtkEpiphanDualVideoSource::SetClipRegionRight(int x0, int x1, int y0, int y1, int z0, int z1)
{
  if (this->ClipRegionRight[0] != x0 || this->ClipRegionRight[1] != x1 ||
	  this->ClipRegionRight[2] != y0 || this->ClipRegionRight[3] != y1 ||
	  this->ClipRegionRight[4] != z0 || this->ClipRegionRight[5] != z1)
	{
	this->Modified();
	if (this->Initialized) 
		{ // modify the FrameBufferExtent
		this->FrameBufferMutex->Lock();
		this->ClipRegionRight[0] = x0; this->ClipRegionRight[1] = x1;
		this->ClipRegionRight[2] = y0; this->ClipRegionRight[3] = y1;
		this->ClipRegionRight[4] = z0; this->ClipRegionRight[5] = z1;
		this->UpdateFrameBuffer();
		this->FrameBufferMutex->Unlock();
		}
	else
		{
		this->ClipRegionRight[0] = x0; this->ClipRegionRight[1] = x1;
		this->ClipRegionRight[2] = y0; this->ClipRegionRight[3] = y1;
		this->ClipRegionRight[4] = z0; this->ClipRegionRight[5] = z1;
		}
  }
}

//----------------------------------------------------------------------------
// set or change the circular buffer size
// you will have to override this if you want the buffers 
// to be device-specific (i.e. something other than vtkDataArray)
void vtkEpiphanDualVideoSource::SetFrameBufferSize(int bufsize)
{
  int i;
  void **framebuffer;
  void **framebufferLeft;
  void **framebufferRight;
  double *timestamps;

  if (bufsize < 0)
    {
    vtkErrorMacro(<< "SetFrameBufferSize: There must be at least one framebuffer");
    }

  if (bufsize == this->FrameBufferSize && bufsize != 0)
    {
    return;
    }

  this->FrameBufferMutex->Lock();

  if (this->FrameBuffer == 0)
    {
    if (bufsize > 0)
      {
      this->FrameBufferIndex = 0;
      this->FrameIndex = -1;
      this->FrameBuffer = new void *[bufsize];
	  this->FrameBufferLeft = new void *[bufsize];
	  this->FrameBufferRight = new void *[bufsize];
      this->FrameBufferTimeStamps = new double[bufsize];
      for (i = 0; i < bufsize; i++)
        {
        this->FrameBuffer[i] = vtkUnsignedCharArray::New();
		this->FrameBufferLeft[i] = vtkUnsignedCharArray::New();
		this->FrameBufferRight[i] = vtkUnsignedCharArray::New();
        this->FrameBufferTimeStamps[i] = 0.0;
        } 
      this->FrameBufferSize = bufsize;
      this->Modified();
      }
    }
  else 
    {
    if (bufsize > 0)
      {
      framebuffer = new void *[bufsize];
	  framebufferLeft = new void *[bufsize];
	  framebufferRight = new void *[bufsize];
      timestamps = new double[bufsize];
      }
    else
      {
      framebuffer = NULL;
	  framebufferLeft = NULL;
	  framebufferRight = NULL;
      timestamps = NULL;
      }

    // create new image buffers if necessary
    for (i = 0; i < bufsize - this->FrameBufferSize; i++)
      {
      framebuffer[i] = vtkUnsignedCharArray::New();
	  framebufferLeft[i] = vtkUnsignedCharArray::New();
	  framebufferRight[i] = vtkUnsignedCharArray::New();
      timestamps[i] = 0.0;
      }
    // copy over old image buffers
    for (; i < bufsize; i++)
      {
      framebuffer[i] = this->FrameBuffer[i-(bufsize-this->FrameBufferSize)];
	  framebufferLeft[i] = this->FrameBufferLeft[i-(bufsize-this->FrameBufferSize)];
	  framebufferRight[i] = this->FrameBufferRight[i-(bufsize-this->FrameBufferSize)];
      }

    // delete image buffers we no longer need
    for (i = 0; i < this->FrameBufferSize-bufsize; i++)
      {
      reinterpret_cast<vtkDataArray *>(this->FrameBuffer[i])->Delete();
	  reinterpret_cast<vtkDataArray *>(this->FrameBufferLeft[i])->Delete();
	  reinterpret_cast<vtkDataArray *>(this->FrameBufferRight[i])->Delete();
      }

    if (this->FrameBuffer)
      {
      delete [] this->FrameBuffer;
	  delete [] this->FrameBufferLeft;
	  delete [] this->FrameBufferRight;
      }
    this->FrameBuffer = framebuffer;
	this->FrameBufferLeft = framebufferLeft;
	this->FrameBufferRight = framebufferRight;

    if (this->FrameBufferTimeStamps)
      {
      delete [] this->FrameBufferTimeStamps;
      }
    this->FrameBufferTimeStamps = timestamps;

    // make sure that frame buffer index is within the buffer
    if (bufsize > 0)
      {
      this->FrameBufferIndex = this->FrameBufferIndex % bufsize;
      if (this->FrameIndex >= bufsize)
        {
        this->FrameIndex = bufsize - 1;
        }
      }
    else
      {
      this->FrameBufferIndex = 0;
      this->FrameIndex = -1;
      }

    this->FrameBufferSize = bufsize;
    this->Modified();
    }

  if (this->Initialized)
    {
    this->UpdateFrameBuffer();
    }

  this->FrameBufferMutex->Unlock();
}

//----------------------------------------------------------------------------
// Update the FrameBuffers according to any changes in the FrameBuffer*
// information. 
// This function should always be called from within a FrameBufferMutex lock
// and should never be called asynchronously.
// It sets up the FrameBufferExtent
void vtkEpiphanDualVideoSource::UpdateFrameBufferLeft()
{
  int i, oldExt;
  int ext[3];
  vtkDataArray *buffer;

  // clip the ClipRegion with the FrameSize
  for (i = 0; i < 3; i++)
    {
    oldExt = this->FrameBufferExtent[2*i+1] - this->FrameBufferExtent[2*i] + 1;
    this->FrameBufferExtent[2*i] = ((this->ClipRegionLeft[2*i] > 0) 
                             ? this->ClipRegionLeft[2*i] : 0);  
    this->FrameBufferExtent[2*i+1] = ((this->ClipRegionLeft[2*i+1] < 
                                       this->FrameSize[i]-1) 
                             ? this->ClipRegionLeft[2*i+1] : this->FrameSize[i]-1);

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
  int bytesPerRow = (ext[0]*this->FrameBufferBitsPerPixel+7)/8;
  bytesPerRow = ((bytesPerRow + this->FrameBufferRowAlignment - 1) /
                 this->FrameBufferRowAlignment)*this->FrameBufferRowAlignment;
  int totalSize = bytesPerRow * ext[1] * ext[2];

  i = this->FrameBufferSize;

  while (--i >= 0)
    {
    buffer = reinterpret_cast<vtkDataArray *>(this->FrameBufferLeft[i]);
    if (buffer->GetDataType() != VTK_UNSIGNED_CHAR ||
        buffer->GetNumberOfComponents() != 1 ||
        buffer->GetNumberOfTuples() != totalSize)
      {
      buffer->Delete();
      buffer = vtkUnsignedCharArray::New();
      this->FrameBufferLeft[i] = buffer;
      buffer->SetNumberOfComponents(1);
      buffer->SetNumberOfTuples(totalSize);
      }
    }
}


//----------------------------------------------------------------------------
// Update the FrameBuffers according to any changes in the FrameBuffer*
// information. 
// This function should always be called from within a FrameBufferMutex lock
// and should never be called asynchronously.
// It sets up the FrameBufferExtent
void vtkEpiphanDualVideoSource::UpdateFrameBufferRight()
{
  int i, oldExt;
  int ext[3];
  vtkDataArray *buffer;

  // clip the ClipRegion with the FrameSize
  for (i = 0; i < 3; i++)
    {
    oldExt = this->FrameBufferExtent[2*i+1] - this->FrameBufferExtent[2*i] + 1;
    this->FrameBufferExtent[2*i] = ((this->ClipRegionRight[2*i] > 0) 
                             ? this->ClipRegionRight[2*i] : 0);  
    this->FrameBufferExtent[2*i+1] = ((this->ClipRegionRight[2*i+1] < 
                                       this->FrameSize[i]-1) 
                             ? this->ClipRegionRight[2*i+1] : this->FrameSize[i]-1);

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
  int bytesPerRow = (ext[0]*this->FrameBufferBitsPerPixel+7)/8;
  bytesPerRow = ((bytesPerRow + this->FrameBufferRowAlignment - 1) /
                 this->FrameBufferRowAlignment)*this->FrameBufferRowAlignment;
  int totalSize = bytesPerRow * ext[1] * ext[2];

  i = this->FrameBufferSize;

  while (--i >= 0)
    {
    buffer = reinterpret_cast<vtkDataArray *>(this->FrameBufferRight[i]);
    if (buffer->GetDataType() != VTK_UNSIGNED_CHAR ||
        buffer->GetNumberOfComponents() != 1 ||
        buffer->GetNumberOfTuples() != totalSize)
      {
      buffer->Delete();
      buffer = vtkUnsignedCharArray::New();
      this->FrameBufferRight[i] = buffer;
      buffer->SetNumberOfComponents(1);
      buffer->SetNumberOfTuples(totalSize);
      }
    }
}

//----------------------------------------------------------------------------
vtkImageData* vtkEpiphanDualVideoSource::GetOutputLeft()
{
  return this->GetOutput(1);
}

vtkImageData* vtkEpiphanDualVideoSource::GetOutputRight()
{
  return this->GetOutput(2);
}
