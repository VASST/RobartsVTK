/*=========================================================================

  File: vtkReplayImageVideoSource.h
  Author: Kyle Charbonneau <kcharbon@imaging.robarts.ca>
  Language: C++

=========================================================================*/

#ifndef __vtkReplayImageVideoSource_h
#define __vtkReplayImageVideoSource_h

//#include <windows.h>
#include <vector>

#include "vtkVideoSource.h"  

#include "vtkMultiThreader.h"

class VTK_EXPORT vtkReplayImageVideoSource : public vtkVideoSource
{
public:

  vtkTypeMacro(vtkReplayImageVideoSource,vtkVideoSource);
  static vtkReplayImageVideoSource *New();
  void PrintSelf(ostream& os, vtkIndent indent);   
  
  // Initialize (source must be set first)
  void Initialize();
  
  // Internal use only
  void ReleaseSystemResources();
  //void UpdateFrameBuffer();
  void InternalGrab();
  
  void Record();
  void Play();
  void Stop();

  int GetRecording() { return this->Playing;}

  void Pause();
  void UnPause();
  
  void LoadFile(char * filename);
  int LoadFolder(char * folder, char * filetype);
  //  int LoadFolder2(char * folder, char * filetype);
  void Clear();
  void SetClipRegion(int x0, int x1, int y0, int y1, int z0, int z1);

protected:
  vtkReplayImageVideoSource();
  ~vtkReplayImageVideoSource();

  int status;

  std::vector<vtkImageData *> loadedData;
  int imageIndex;

  bool SetFrameSizeAutomatically;

  int pauseFeed;
  int currentLength;
};

#endif
