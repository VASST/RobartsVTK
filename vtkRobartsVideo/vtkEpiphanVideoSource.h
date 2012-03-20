/*=========================================================================

  File: vtkEpiphanVideoSource.h
  Author: Kyle Charbonneau <kcharbon@imaging.robarts.ca>
  Language: C++

=========================================================================*/

#ifndef __vtkEpiphanVideoSource_h
#define __vtkEpiphanVideoSource_h

#include <windows.h>

#include "vtkVideoSource.h"  


#include "vtkMultiThreader.h"

#include "epiphan/frmgrab.h"

/* exit codes */
#define V2U_GRABFRAME_STATUS_OK       0  /* successful completion */
#define V2U_GRABFRAME_STATUS_NODEV    1  /* VGA2USB device not found */
#define V2U_GRABFRAME_STATUS_VMERR    2  /* Video mode detection failure */
#define V2U_GRABFRAME_STATUS_NOSIGNAL 3  /* No signal detected */
#define V2U_GRABFRAME_STATUS_GRABERR  4  /* Capture error */
#define V2U_GRABFRAME_STATUS_IOERR    5  /* File save error */
#define V2U_GRABFRAME_STATUS_CMDLINE  6  /* Command line syntax error */


class VTK_EXPORT vtkEpiphanVideoSource : public vtkVideoSource
{
public:

  vtkTypeMacro(vtkEpiphanVideoSource,vtkVideoSource);
  static vtkEpiphanVideoSource *New();
  void PrintSelf(ostream& os, vtkIndent indent);   
  
  // Initialize (source must be set first)
  void Initialize();
  
  // Internal use only
  void ReleaseSystemResources();
  void UpdateFrameBuffer();
  void InternalGrab();

  void SetSerialNumber(char * serial);

  void Record();
  void Play();
  void Stop();

  void SetFrameRate(float rate);
  void SetOutputFormat(int format);
  void SetClipRegion(int x0, int x1, int y0, int y1, int z0, int z1);

  void Pause();
  void UnPause();
  // Empty methods to make sure certain paramaters aren't chagned
  //void SetFrameSize(int, int, int);
  //void SetOutputFormat(int);
  
protected:
  vtkEpiphanVideoSource();
  ~vtkEpiphanVideoSource();
  int status;
  FrmGrabber* fg;
  V2URect * cropRect;
  char serialNumber[15];
  int pauseFeed;
};

#endif
