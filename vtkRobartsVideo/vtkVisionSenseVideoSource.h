/*=========================================================================

  File: vtkVisionSenseVideoSource.h
  Author: Kyle Charbonneau <kcharbon@imaging.robarts.ca>
  Language: C++

=========================================================================*/

#ifndef __vtkVisionSenseVideoSource_h
#define __vtkVisionSenseVideoSource_h

#include "vtkVideoSource.h"   
 
class vtkVisionSenseNetworkSource;

class VTK_EXPORT vtkVisionSenseVideoSource : public vtkVideoSource
{
public:

  vtkTypeMacro(vtkVisionSenseVideoSource,vtkVideoSource);
  static vtkVisionSenseVideoSource *New();
  void PrintSelf(ostream& os, vtkIndent indent);   
  
  // Initialize (source must be set first)
  void Initialize();
  
  // Internal use only
  void ReleaseSystemResources();
  void UpdateFrameBuffer();
  void InternalGrab();

  // Empty methods to make sure certain paramaters aren't chagned
  void SetFrameSize(int, int, int);
  void SetOutputFormat(int);

  // Set camera
  void SetRightCamera();
  void SetLeftCamera();
  
  // Set/Get the source
  void SetSource(vtkVisionSenseNetworkSource *);
  vtkVisionSenseNetworkSource *GetSource() {return this->Source;};
  
protected:
  vtkVisionSenseVideoSource();
  ~vtkVisionSenseVideoSource();

  // Which camera (left or right) this should request data from
  int Camera;
  
  // Source of data
  vtkVisionSenseNetworkSource *Source;
  
  // Number of bytes in one image (left or right)
  int ImageSize;

};

#endif
