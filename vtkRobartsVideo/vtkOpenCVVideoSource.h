/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkOpenCVVideoSource.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkOpenCVVideoSource - Open Computer Vision video digitizer
// .SECTION Description
// vtkOpenCVVideoSource grabs frames or streaming video from an OpenCV compatible
// camera. 
// .SECTION Caveats
// With some capture cards, if this class is leaked and ReleaseSystemResources 
// is not called, you may have to reboot before you can capture again.
// vtkVideoSource used to keep a global list and delete the video sources
// if your program leaked, due to exit crashes that was removed.
//
// .SECTION See Also
// vtkVideoSource vtkMILVideoSource

#ifndef __vtkOpenCVVideoSource_h
#define __vtkOpenCVVideoSource_h

#include "vtkVideoSource.h"
#include "vtkImageData.h"
#include "vtkMutexLock.h"

class vtkOpenCVVideoSourceInternal;

class VTK_EXPORT vtkOpenCVVideoSource : public vtkVideoSource
{
friend class vtkOpenCVVideoSourceInternal;
public:
  static vtkOpenCVVideoSource *New();
  vtkTypeMacro(vtkOpenCVVideoSource,vtkVideoSource);
  void PrintSelf(ostream& os, vtkIndent indent);   
 
  // Description:
  // Request a particular frame size (set the third value to 1).
  void SetFrameSize(int x, int y, int z);
  virtual void SetFrameSize(int dim[3]) { 
    this->SetFrameSize(dim[0], dim[1], dim[2]); };

  //
  void UpdateFrameBuffer();
  void InternalGrab();

  // Description:
  // Request a particular output format (default: VTK_RGB).
  void SetOutputFormat(int format);

  // Description:
  // Set the video input.
  void SetVideoSourceNumber(unsigned int n);
  unsigned int GetVideoSourceNumber();

  // Description:
  // Initialize the driver (this is called automatically when the
  // first grab is done).
  void Initialize();

  // Description:
  // Free the driver (this is called automatically inside the
  // destructor).
  void ReleaseSystemResources();

protected:
  vtkOpenCVVideoSource();
  ~vtkOpenCVVideoSource();

  vtkOpenCVVideoSourceInternal *Internal;
  
  vtkMutexLock* OpenCVFirstBufferLock;
  vtkMutexLock* OpenCVSecondBufferLock;

  unsigned int videoSourceNumber;
  unsigned int ImageSize;

private:
  vtkOpenCVVideoSource(const vtkOpenCVVideoSource&);  // Not implemented.
  void operator=(const vtkOpenCVVideoSource&);  // Not implemented.
};

#endif





