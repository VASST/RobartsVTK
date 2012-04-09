/*=========================================================================

  File: vtkEpiphanDualVideoSource.h
  Author: Kyle Charbonneau <kcharbon@imaging.robarts.ca>
  Language: C++

=========================================================================*/

#ifndef __vtkEpiphanDualVideoSource_h
#define __vtkEpiphanDualVideoSource_h

#include <windows.h>

#include "vtkEpiphanVideoSource.h"  

class VTK_EXPORT vtkEpiphanDualVideoSource : public vtkEpiphanVideoSource
{
public:

  vtkTypeMacro(vtkEpiphanDualVideoSource,vtkVideoSource);
  static vtkEpiphanDualVideoSource *New();
  void PrintSelf(ostream& os, vtkIndent indent);   
  
  // Initialize (source must be set first)
  void Initialize();
  
  // Internal use only
  void ReleaseSystemResources();
  void InternalGrab();
  void SetClipRegion(int x0, int x1, int y0, int y1, int z0, int z1);
  void SetClipRegionLeft(int x0, int x1, int y0, int y1, int z0, int z1);
  void SetClipRegionRight(int x0, int x1, int y0, int y1, int z0, int z1);

  void SetFrameBufferSize(int bufsize);

  vtkImageData* GetOutputLeft();
  vtkImageData* GetOutputRight();

protected:
  vtkEpiphanDualVideoSource();
  ~vtkEpiphanDualVideoSource();

  void UpdateFrameBufferLeft();
  void UpdateFrameBufferRight();

  int ClipRegionLeft[6];
  int ClipRegionRight[6];
  void **FrameBufferLeft;
  void **FrameBufferRight;

};

#endif
