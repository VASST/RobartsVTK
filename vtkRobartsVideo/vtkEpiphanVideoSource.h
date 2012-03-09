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
//#include "epiphan/v2u_lib.h"

typedef struct _V2UCaptureFormatInfo {
    V2U_UINT32 format;
    const char* opt;
    const char* fourcc;
    V2U_BOOL flip;
} V2UCaptureFormatInfo;

static const V2UCaptureFormatInfo v2uCaptureFormatInfo[] = {
      /* format */                /* opt */ /* fourcc */  /* flip */
    { V2U_GRABFRAME_FORMAT_RGB4   ,"rgb4"   ,NULL        ,V2U_TRUE  },
    { V2U_GRABFRAME_FORMAT_RGB8   ,"rgb8"   ,"\0\0\0\0"  ,V2U_TRUE  },
    { V2U_GRABFRAME_FORMAT_RGB16  ,"rgb16"  ,"\0\0\0\0"  ,V2U_TRUE  },
    { V2U_GRABFRAME_FORMAT_BGR16  ,"bgr16"  ,NULL        ,V2U_TRUE  },
    { V2U_GRABFRAME_FORMAT_RGB24  ,"rgb24"  ,NULL        ,V2U_TRUE  },
    { V2U_GRABFRAME_FORMAT_BGR24  ,"bgr24"  ,"\0\0\0\0"  ,V2U_TRUE  },
    { V2U_GRABFRAME_FORMAT_ARGB32 ,"argb32" ,NULL        ,V2U_TRUE  },
    { V2U_GRABFRAME_FORMAT_CRGB24 ,"crgb24" ,"V2UV"      ,V2U_FALSE },
    { V2U_GRABFRAME_FORMAT_CBGR24 ,"cbgr24" ,"V2UV"      ,V2U_FALSE },
    { V2U_GRABFRAME_FORMAT_CYUY2  ,"cyuy2"  ,"V2UV"      ,V2U_FALSE },
    { V2U_GRABFRAME_FORMAT_Y8     ,"y8"     ,NULL        ,V2U_FALSE },
    { V2U_GRABFRAME_FORMAT_YUY2   ,"yuy2"   ,"YUY2"      ,V2U_FALSE },
    { V2U_GRABFRAME_FORMAT_2VUY   ,"uyvy"   ,"UYVY"      ,V2U_FALSE },
    { V2U_GRABFRAME_FORMAT_YV12   ,"yv12"   ,"YV12"      ,V2U_FALSE },
    { V2U_GRABFRAME_FORMAT_I420   ,"i420"   ,"IYUV"      ,V2U_FALSE },
    { V2U_GRABFRAME_FORMAT_NV12   ,"nv12"   ,NULL      ,V2U_FALSE }
};

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
