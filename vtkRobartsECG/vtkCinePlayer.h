/*=========================================================================

  Program:   CinePlayer for AtamaiViewer/Vasst Project
  Module:    $RCSfile: vtkCinePlayer.h,v $
  Creator:   Chris Wedlake <cweldake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2007/04/19 12:48:52 $
  Version:   $Revision: 1.1 $

==========================================================================

Copyright (c) 2000-2007 Robarts, Inc.

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
#ifndef __vtkCinePlayer_h
#define __vtkCinePlayer_h

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "vtkObject.h"
#include "vtkCriticalSection.h"
#include "vtkImageData.h"
#include "vtkActor.h"

class vtkMatrix4x4;
class vtkMultiThreader;

class VTK_EXPORT vtkCinePlayer : public vtkObject
{
public:
  static vtkCinePlayer *New();
  vtkTypeMacro(vtkCinePlayer,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Start the tracking system.  The tracking system is brought from
  // its ground state (i.e. on but not necessarily initialized) into
  // full tracking mode.  This method calls InternalStartTracking()
  // after doing a bit of housekeeping.
  void StartPlaying();

  // Description:
  // Stop the tracking system and bring it back to its ground state.
  // This method calls InternalStopTracking().
  void StopPlaying();

  // Description:
  // Test whether or not the system is tracking.
  int IsPlaying() { return this->Playing; };

  // Description:
  // This method will call Update() on each of the tools.  Note that
  // this method does not call the InternalUpdate() method, which
  // is called by a separate thread.
  void Update();
  
  // Description:
  // Get the internal update rate for this tracking system.  This is
  // the number of transformations sent by the tracking system per
  // second per tool.
  double GetInternalUpdateRate() { return this->InternalUpdateRate; };

  vtkImageData * GetImageGroup(int image);

  vtkImageData * GetSurfaceGroup(int image);

  // Description:
  // Get the timestamp for the last time that Update() was called, in
  // seconds since 1970 (i.e. the UNIX epoch).  This method is not a
  // good method of getting timestamps for tracking information,
  // you should use the vtkCinePlayerTool GetTimeStamp() method to get
  // the timestamp associated with each transform.  This method is
  // only valuable for determining e.g. how old the transforms were
  // before the Update method was called.
  vtkGetMacro(UpdateTimeStamp,double);

  void SetCallbackMethods(void * function);


  // Description:
  // The subclass will do all the hardware-specific update stuff
  // in this function.   It should call ToolUpdate() for each tool.
  // Note that vtkCinePlayer.cxx starts up a separate thread after
  // InternalStartTracking() is called, and that InternalUpdate() is
  // called repeatedly from within that thread.  Therefore, any code
  // within InternalUpdate() must be thread safe.  You can temporarily
  // pause the thread by locking this->UpdateMutex->Lock() e.g. if you
  // need to communicate with the device from outside of InternalUpdate().
  // A call to this->UpdateMutex->Unlock() will resume the thread.
  void InternalUpdate();

//BTX
  // These are used by static functions in vtkCinePlayer.cxx, and since
  // VTK doesn't generally use 'friend' functions they are public
  // instead of protected.  Do not use them anywhere except inside
  // vtkCinePlayer.cxx.
  vtkCriticalSection *UpdateMutex;
  vtkCriticalSection *RequestUpdateMutex;
  vtkTimeStamp UpdateTime;
  double InternalUpdateRate;  
//ETX
  int AddActorToSurfaceGroup(vtkActor * actor, int SurfaceIndex, int GroupIndex);
  
  int AddActorToImageGroup(vtkActor * actor, int ImageIndex, int GroupIndex);
  int CreateNewImageGroup();
  int CreateNewSurfaceGroup();

protected:
  vtkCinePlayer();
  ~vtkCinePlayer();

  double Rate;
  //vector ImageDataSets;
  //vector SurfaceDataSets;
  int AnimationIndex;
  int NumberOfSurfaceGroups;
  int NumberOfImageGroups;
  int Playing;


  int MaxImageLength;
  int MaxSurfaceLength;

  double UpdateTimeStamp;
  unsigned long LastUpdateTime;

  vtkMultiThreader *Threader;
  int ThreadId;

  //BTX
  //std::vector<vtkForceFeedback *>  forceModel;
  typedef std::vector<vtkActor *> ActorGroup; // Each ActorGroup can is one actor that can be inside multiple renders
  typedef std::vector<ActorGroup> Grouping;   // Each Grouping is a series of images (Example, 20 phases of the heart beating)
  std::vector<Grouping> ImageGroups;      // Group of ALL Images that will be involved in the CinePlayer
  std::vector<Grouping> SurfaceGroups;      // Group of ALL Surfaces that will be involved in the CinePlayer
  int NumberOfFrames;
  //ETX

private:
  vtkCinePlayer(const vtkCinePlayer&);
  void operator=(const vtkCinePlayer&);  
};

#endif

