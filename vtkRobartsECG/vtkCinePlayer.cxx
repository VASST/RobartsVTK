/*=========================================================================

  Program:   CinePlayer for AtamaiViewer/Vasst Project
  Module:    $RCSfile: vtkCinePlayer.cxx,v $
  Creator:   Chris Wedlake <cweldake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2007/04/19 12:48:52 $
  Version:   $Revision: 1.1 $

==========================================================================

Copyright (c) 2000-2007 Robarts Research Institute

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

#include <limits.h>
#include <float.h>
#include <math.h>
#include "vtkTimerLog.h"
#include "vtkMultiThreader.h"
#include "vtkMutexLock.h"
#include "vtkCriticalSection.h"
#include "vtkObjectFactory.h"
#include "vtkCinePlayer.h"

//----------------------------------------------------------------------------
vtkCinePlayer* vtkCinePlayer::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkCinePlayer");
  if(ret)
    {
    return (vtkCinePlayer*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkCinePlayer;
}

//----------------------------------------------------------------------------
vtkCinePlayer::vtkCinePlayer()
{
  this->Playing = 0;

  // for threaded capture of transformations
  this->Threader = vtkMultiThreader::New();
  this->ThreadId = -1;
  this->UpdateMutex = vtkCriticalSection::New();
  this->RequestUpdateMutex = vtkCriticalSection::New();
}

//----------------------------------------------------------------------------
vtkCinePlayer::~vtkCinePlayer()
{
  // The thread should have been stopped before the
  // subclass destructor was called, but just in case
  // se stop it here.
  if (this->ThreadId != -1)
    {
    this->Threader->TerminateThread(this->ThreadId);
    this->ThreadId = -1;
    }

  this->Threader->Delete();
  this->UpdateMutex->Delete();
  this->RequestUpdateMutex->Delete();
}
  
//----------------------------------------------------------------------------
void vtkCinePlayer::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkObject::PrintSelf(os,indent);

}
 
//----------------------------------------------------------------------------
// this thread is run whenever the tracker is tracking
static void *vtkCinePlayerThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkCinePlayer *self = (vtkCinePlayer *)(data->UserData);

  double currtime[10];

  // loop until cancelled
  for (int i = 0;; i++)
    {
    // get current tracking rate over last 10 updates
#if (VTK_MAJOR_VERSION <= 4)
    double newtime = vtkTimerLog::GetCurrentTime();
#else
    double newtime = vtkTimerLog::GetUniversalTime();
#endif
    double difftime = newtime - currtime[i%10];
    currtime[i%10] = newtime;
    if (i > 10 && difftime != 0)
      {
      self->InternalUpdateRate = (10.0/difftime);
      }

    // query the hardware tracker
    self->UpdateMutex->Lock();
    self->InternalUpdate();
    self->UpdateTime.Modified();
    self->UpdateMutex->Unlock();

    // check to see if main thread wants to lock the UpdateMutex
    self->RequestUpdateMutex->Lock();
    self->RequestUpdateMutex->Unlock();
    
    // check to see if we are being told to quit 
    data->ActiveFlagLock->Lock();
    int activeFlag = *(data->ActiveFlag);
    data->ActiveFlagLock->Unlock();

    if (activeFlag == 0)
      {
      return NULL;
      }
    }
}

//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void vtkCinePlayer::StartPlaying()
{
  // start the tracking thread
  if (this->ThreadId == -1)
    {
    return;
    }

  // this will block the tracking thread until we're ready
  this->UpdateMutex->Lock();

  // start the tracking thread
  this->ThreadId = this->Threader->SpawnThread((vtkThreadFunctionType)\
                                               &vtkCinePlayerThread,this);
  this->LastUpdateTime = this->UpdateTime.GetMTime();

  // allow the tracking thread to proceed
  this->UpdateMutex->Unlock();

  // wait until the first update has occurred before returning
  int timechanged = 0;
  while (!timechanged)
    {
    this->RequestUpdateMutex->Lock();
    this->UpdateMutex->Lock();
    this->RequestUpdateMutex->Unlock();
    timechanged = (this->LastUpdateTime != this->UpdateTime.GetMTime());
    this->UpdateMutex->Unlock();
#ifdef _WIN32
    Sleep((int)(100));
#elif defined(__FreeBSD__) || defined(__linux__) || defined(sgi) || defined(__APPLE__)
    struct timespec sleep_time, dummy;
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 100000000;
    nanosleep(&sleep_time,&dummy);
#endif
    }
}


void vtkCinePlayer::InternalUpdate() {
	int imageLen, surfaceLen,maxLength;
	imageLen = this->MaxSurfaceLength;
	surfaceLen = this->MaxSurfaceLength;
	maxLength = max(imageLen, surfaceLen);
}



//----------------------------------------------------------------------------
void vtkCinePlayer::StopPlaying()
{
  if (this->Playing && this->ThreadId != -1)
    {
    this->Threader->TerminateThread(this->ThreadId);
    this->ThreadId = -1;
    }

  this->Playing = 0;
}

//----------------------------------------------------------------------------
void vtkCinePlayer::Update()
{
/*  if (!this->Tracking)
    { 
    return; 
    }

  for (int tool = 0; tool < this->NumberOfTools; tool++)
    {
    vtkCinePlayerTool *trackerTool = this->Tools[tool];

    trackerTool->Update();

    this->UpdateTimeStamp = trackerTool->GetTimeStamp();
    }

  this->LastUpdateTime = this->UpdateTime.GetMTime();
  */
}





/*
void vtkCinePlayer::AdvanceFrame(int value) {
	if (value > 0)
		this->FrameChange(value);
	else
		return;

}

void vktCinePlayer::RetreatFrame(int value) {
	if (value > 0)
		this->FrameChange(-1*value);
	else
		return;
}

void vtkCinePlayer::FrameChange(int value) {

}

void vtkCinePlayer::GoToFrame(int value) {

}
*/

int vtkCinePlayer::CreateNewImageGroup() {
	Grouping newGroup;
	this->ImageGroups.push_back(newGroup);
	return (this->ImageGroups.size()-1);
}

int vtkCinePlayer::CreateNewSurfaceGroup() {
	Grouping newGroup ;
	this->SurfaceGroups.push_back(newGroup);
	return (this->SurfaceGroups.size()-1);
}

int vtkCinePlayer::AddActorToImageGroup(vtkActor * actor, int ImageIndex, int GroupIndex) {
	if (ImageIndex > this->ImageGroups.size()-1 || ImageIndex < 0 || this->ImageGroups.size() == 0) {
		return -1;
	}

	if (GroupIndex <= this->ImageGroups[ImageIndex].size()) {
		this->ImageGroups[ImageIndex][GroupIndex].push_back(actor);
		return GroupIndex;
	}
	else if (GroupIndex == this->ImageGroups[ImageIndex].size()) {
		ActorGroup actorGroup;
		this->ImageGroups[ImageIndex].push_back(actorGroup);	
		this->ImageGroups[ImageIndex][GroupIndex].push_back(actor);
		return GroupIndex;
	}
	else {
		return -1;
	}
}

int vtkCinePlayer::AddActorToSurfaceGroup(vtkActor * actor, int SurfaceIndex, int GroupIndex) {
	if (SurfaceIndex > this->SurfaceGroups.size()-1 || SurfaceIndex < 0 || this->SurfaceGroups.size() == 0) {
		return -1;
	}

	if (GroupIndex <= this->SurfaceGroups[SurfaceIndex].size()) {
		this->SurfaceGroups[SurfaceIndex][GroupIndex].push_back(actor);
		return GroupIndex;
	}
	else if (GroupIndex == this->SurfaceGroups[SurfaceIndex].size()) {
		ActorGroup actorGroup;
		this->SurfaceGroups[SurfaceIndex].push_back(actorGroup);	
		this->SurfaceGroups[SurfaceIndex][GroupIndex].push_back(actor);
		return GroupIndex;
	}
	else {
		return -1;
	}
}