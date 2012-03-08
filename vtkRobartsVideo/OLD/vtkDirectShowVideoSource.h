/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkDirectShowVideoSource.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkDirectShowVideoSource - Video-for-Windows video digitizer
// .SECTION Description
// vtkDirectShowVideoSource grabs frames or streaming video from a
// Video for Windows compatible device on the DirectShow platform. 
// .SECTION Caveats
// With some capture cards, if this class is leaked and ReleaseSystemResources 
// is not called, you may have to reboot before you can capture again.
// vtkVideoSource used to keep a global list and delete the video sources
// if your program leaked, due to exit crashes that was removed.
//
// .SECTION See Also
// vtkVideoSource vtkMILVideoSource

#ifndef __vtkDirectShowVideoSource_h
#define __vtkDirectShowVideoSource_h

#include "vtkVideoSource.h"
#include "vtkImageData.h"

class vtkDirectShowVideoSourceInternal;

class VTK_EXPORT vtkDirectShowVideoSource : public vtkVideoSource
{
public:
  static vtkDirectShowVideoSource *New();
  vtkTypeMacro(vtkDirectShowVideoSource,vtkVideoSource);
  void PrintSelf(ostream& os, vtkIndent indent);   
 
  // Description:
  // Request a particular frame size (set the third value to 1).
  void SetFrameSize(int x, int y, int z);
  void SetFrameSize(int dim[3]) { 
    this->SetFrameSize(dim[0], dim[1], dim[2]); };
  
  // Description:
  // Request a particular frame rate (default 30 frames per second).
  void SetFrameRate(float rate);
  float GetFrameRate();

  //
  //vtkImageData* GetOutput();
  void			Update();

  // Description:
  // Request a particular output format (default: VTK_RGB).
  void SetOutputFormat(int format);

  // Description:
  // Bring up a modal dialog box for video format selection.
  void VideoFormatDialog();

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
  vtkDirectShowVideoSource();
  ~vtkDirectShowVideoSource();
  void BuildDeviceList();

  // BTX
  vtkDirectShowVideoSourceInternal *Internal;
  vtkImageData* output;
  char videoMenuName[10][255];
  bool deviceListPopulated;
  int numVideoCapDevices;
  // ETX

  unsigned int videoSourceNumber;

private:
  vtkDirectShowVideoSource(const vtkDirectShowVideoSource&);  // Not implemented.
  void operator=(const vtkDirectShowVideoSource&);  // Not implemented.
};

#endif





