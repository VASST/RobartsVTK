/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkTestECGVideoSource.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkTestECGVideoSource.h"
#include "vtkTimerLog.h"
#include "vtkObjectFactory.h"
#include "vtkCriticalSection.h"
#include "vtkDataArray.h"
#include "vtkMultiThreader.h"
#include "vtkMutexLock.h"

#include <ctype.h>
#include <string.h>
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

#if (VTK_MAJOR_VERSION <= 5)
vtkCxxRevisionMacro(vtkTestECGVideoSource, "$Revision: 1.00 $");
#endif

vtkStandardNewMacro(vtkTestECGVideoSource);

//----------------------------------------------------------------------------
vtkTestECGVideoSource::vtkTestECGVideoSource()
{
  this->Initialized = 0;
  this->pauseFeed = -1;
  this->phase = 0;
  this->totalPhases = 1;
}

//----------------------------------------------------------------------------
vtkTestECGVideoSource::~vtkTestECGVideoSource()
{
  ReleaseSystemResources();
}

//----------------------------------------------------------------------------
void vtkTestECGVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

}

void vtkTestECGVideoSource::SetECGPhase(int newPhase) {
  this->phase = newPhase;
}

int vtkTestECGVideoSource::GetECGPhase() {
  return this->phase;
}

void vtkTestECGVideoSource::SetNumberOfECGPhases(int newTotal) {
  this->totalPhases = newTotal;
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
static void *vtkTestECGVideoSourceRecordThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkTestECGVideoSource *self = (vtkTestECGVideoSource *)(data->UserData);

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
void vtkTestECGVideoSource::Record()
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
                                &vtkTestECGVideoSourceRecordThread,this);
    }
}

void vtkTestECGVideoSource::Pause() {
  this->pauseFeed = 1;
}
void vtkTestECGVideoSource::UnPause() {
  this->pauseFeed = 0;
}



//----------------------------------------------------------------------------
// Copy pseudo-random noise into the frames.  This function may be called
// asynchronously.
void vtkTestECGVideoSource::InternalGrab()
{
  int i,index;
  //static int randsave = 0;
  //int randNum;
  int phaseCounter = -5;
  unsigned char *ptr;
  int *lptr;

  if (this->pauseFeed == 1) {
      return;
  }

  // get a thread lock on the frame buffer
  this->FrameBufferMutex->Lock();

  if (this->AutoAdvance)
    {
    this->AdvanceFrameBuffer(1);
    if (this->FrameIndex + 1 < this->FrameBufferSize)
      {
      this->FrameIndex++;
      }
    }

  index = this->FrameBufferIndex % this->FrameBufferSize;
  while (index < 0)
    {
    index += this->FrameBufferSize;
    }

  int bytesPerRow = ((this->FrameBufferExtent[1]-this->FrameBufferExtent[0]+1)*
                     this->FrameBufferBitsPerPixel + 7)/8;
  bytesPerRow = ((bytesPerRow + this->FrameBufferRowAlignment - 1) /
                 this->FrameBufferRowAlignment)*this->FrameBufferRowAlignment;
  int totalSize = bytesPerRow *
                   (this->FrameBufferExtent[3]-this->FrameBufferExtent[2]+1) *
                   (this->FrameBufferExtent[5]-this->FrameBufferExtent[4]+1);

  //randNum = randsave;

  // copy 'noise' into the frame buffer
  ptr = reinterpret_cast<vtkUnsignedCharArray *>(this->FrameBuffer[index])->GetPointer(0);

  // Somebody should check this:
  lptr = (int *)(((((long)ptr) + 3)/4)*4);
  i = totalSize/4;

  int width = (this->FrameBufferExtent[3]-this->FrameBufferExtent[2]+1);
  int height = (this->FrameBufferExtent[5]-this->FrameBufferExtent[4]+1);

  //int colour =

  while (--i >= 0)
    {
    //randNum = 1664525*randNum + 1013904223;
    //*lptr++ = randNum;
    if (i % (width)  == this->phase) {
      *lptr++ = 1013904223;
    }
    else
      *lptr++ = 0;
    }
/*
  unsigned char *ptr1 = ptr + 4;
  i = (totalSize-4)/16;

  while (--i >= 0)
    {
    //randNum = 1664525*randNum + 1013904223;
    //*ptr1 = randNum;
    *ptr1 = 1013904223;

    ptr1 += 16;
    }
  //randsave = randNum;
*/


  this->FrameBufferTimeStamps[index] = vtkTimerLog::GetUniversalTime();

  if (this->FrameCount++ == 0)
    {
    this->StartTimeStamp = this->FrameBufferTimeStamps[index];
    }

  this->Modified();

  this->FrameBufferMutex->Unlock();
}
/*
//----------------------------------------------------------------------------
// for accurate timing of the transformation: this solves a differential
// equation that works to smooth out the jitter in the times that
// are returned by vtkTimerLog::GetUniversalTime() i.e. the system clock.
double vtkTestECGVideoSource::CreateTimeStampForFrame(unsigned long framecount)
{
  double timestamp = vtkTimerLog::GetUniversalTime();

  double frameperiod = ((timestamp - this->LastTimeStamp)/
                        (framecount - this->LastFrameCount));
  double deltaperiod = (frameperiod - this->EstimatedFramePeriod)*0.01;

  this->EstimatedFramePeriod += deltaperiod;
  this->LastTimeStamp += ((framecount - this->LastFrameCount)*
                          this->NextFramePeriod);
  this->LastFrameCount = framecount;

  double diffperiod = (timestamp - this->LastTimeStamp);

  if (diffperiod < -0.2 || diffperiod > 0.2)
    { // time is off by more than 0.2 seconds: reset the clock
    this->EstimatedFramePeriod -= deltaperiod;
    this->NextFramePeriod = this->EstimatedFramePeriod;
    this->LastTimeStamp = timestamp;
    return timestamp;
    }

  diffperiod *= 0.1;
  double maxdiff = 0.001;
  if (diffperiod < -maxdiff)
    {
    diffperiod = -maxdiff;
    }
  else if (diffperiod > maxdiff)
    {
    diffperiod = maxdiff;
    }

  this->NextFramePeriod = this->EstimatedFramePeriod + diffperiod;

  return this->LastTimeStamp;
}
*/
//----------------------------------------------------------------------------
// Circulate the buffer and grab a frame.
// This particular implementation just copies random noise into the frames,
// you will certainly want to override this method (also note that this
// is the only method which you really have to override)
void vtkTestECGVideoSource::Grab()
{
  // ensure that the hardware is initialized.
  this->Initialize();

  this->InternalGrab();
}

//----------------------------------------------------------------------------
// Stop continuous grabbing or playback.  You will have to override this
// if your class overrides Play() and Record()
void vtkTestECGVideoSource::Stop()
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
