/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkVideoSource2.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkVideoSource2 - Superclass of video input devices for VTK
// .SECTION Description
// vtkVideoSource2 is a superclass for video input interfaces for VTK.
// The goal is to provide an interface which is very similar to the
// interface of a VCR, where the 'tape' is an internal frame buffer
// capable of holding a preset number of video frames.  Specialized
// versions of this class record input from various video input sources.
// This base class records input from a noise source.  vtkVideoSource2 uses
// vtkVideoBuffer2 and vtkVideoFrame2 to hold its image data, and should be
// used instead of vtkVideoSource if you want to be able to access data while
// recording or playing.
// .SECTION Caveats
// You must call the ReleaseSystemResources() method before the application
// exits.  Otherwise the application might hang while trying to exit.
// .SECTION See Also
// vtkWin32VideoSource2 vtkMILVideoSource2 vtkVideoBuffer2 vtkVideoFrame2

#ifndef __vtkVideoSource2_h
#define __vtkVideoSource2_h

#include "vtkImageAlgorithm.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

class vtkTimerLog;
class vtkCriticalSection;
class vtkMultiThreader;
class vtkScalarsToColors;
class vtkVideoBuffer2;
class vtkImageWriter;
class vtkImageReader2;

#define FILETYPE_BMP  1
//#define FILETYPE_MINC 2
#define FILETYPE_PNG  3
#define FILETYPE_TIFF 4

// Frame grabber type is used by UnpackRasterLine in vtkVideoFrame2
#define FG_BASE    1
#define FG_MIL    2
#define FG_WIN32  3

class VTK_EXPORT vtkVideoSource2 : public vtkImageAlgorithm
{
public:
  static vtkVideoSource2 *New();
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkVideoSource2,vtkImageAlgorithm);
#else
  vtkTypeMacro(vtkVideoSource2, vtkImageAlgorithm);
#endif
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Record incoming video at the specified FrameRate.  The recording
  // continues indefinitely until Stop() is called.
  virtual void Record();

  // Description:
  // Play through the 'tape' sequentially at the specified frame rate.
  // If you have just finished Recording, you should call Rewind() first.
  virtual void Play();

  // Description:
  // Stop recording or playing.
  virtual void Stop();

  // Description:
  // Rewind to the frame with the earliest timestamp.  Record operations
  // will start on the following frame, therefore if you want to re-record
  // over this frame you must call Seek(-1) before calling Grab() or Record().
  virtual void Rewind();

  // Description:
  // FastForward to the last frame that was recorded (i.e. to the frame
  // that has the most recent timestamp).
  virtual void FastForward();

  // Description:
  // Seek forwards or backwards by the specified number of frames
  // (positive is forward, negative is backward).
  virtual void Seek(int n);

  // Description:
  // Grab a single video frame.
  virtual void Grab();

  // Description:
  // Are we in record mode? (record mode and play mode are mutually
  // exclusive).
  vtkGetMacro(Recording,int);

  // Description:
  // Are we in play mode? (record mode and play mode are mutually
  // exclusive).
  vtkGetMacro(Playing,int);

  // Description:
  // Set/Get the full-frame size.  This must be an allowed size for the device,
  // the device may either refuse a request for an illegal frame size or
  // automatically choose a new frame size.
  // The default is usually 320x240x1, but can be device specific.
  // The 'depth' should always be 1 (unless you have a device that
  // can handle 3D acquisition).
  virtual void SetFrameSize(int x, int y, int z);
  virtual void SetFrameSize(int dim[3]) {
    this->SetFrameSize(dim[0], dim[1], dim[2]); };
  virtual int* GetFrameSize();
  virtual void GetFrameSize(int &x, int &y, int &z);
  virtual void GetFrameSize(int dim[3]);

  // Description:
  // Set/Get a particular frame rate (default 30 frames per second).
  virtual void SetFrameRate(float rate);
  vtkGetMacro(FrameRate,float);

  // Description:
  // Set/Get the output format.  This must be appropriate for device,
  // usually only VTK_LUMINANCE, VTK_RGB, and VTK_RGBA are supported.
  virtual void SetOutputFormat(int format);
  void SetOutputFormatToLuminance() { this->SetOutputFormat(VTK_LUMINANCE); };
  void SetOutputFormatToRGB() { this->SetOutputFormat(VTK_RGB); };
  void SetOutputFormatToRGBA() { this->SetOutputFormat(VTK_RGBA); };
  virtual int GetOutputFormat();

  // Description:
  // Set/Get size of the frame buffer, i.e. the number of frames that
  // the 'tape' can store.
  virtual void SetFrameBufferSize(int FrameBufferSize);
  virtual int GetFrameBufferSize();

  // Description:
  // Set/Get the number of frames to copy to the output on each execute.
  // The frames will be concatenated along the Z dimension, with the
  // most recent frame first.  The default is 1.
  vtkSetMacro(NumberOfOutputFrames,int);
  vtkGetMacro(NumberOfOutputFrames,int);

  // Description:
  // Set/Get whether to automatically advance the buffer before each grab.
  // Default: on
  vtkBooleanMacro(AutoAdvance,int);
  vtkSetMacro(AutoAdvance,int)
  vtkGetMacro(AutoAdvance,int);

  // Description:
  // Set/Get the clip rectangle for the frames.  The video will be clipped
  // before it is copied into the framebuffer.  Changing the ClipRegion
  // will destroy the current contents of the framebuffer.
  // The default ClipRegion is (0,VTK_INT_MAX,0,VTK_INT_MAX,0,VTK_INT_MAX).
  virtual void SetClipRegion(int r[6]) {
    this->SetClipRegion(r[0],r[1],r[2],r[3],r[4],r[5]); };
  virtual void SetClipRegion(int x0, int x1, int y0, int y1, int z0, int z1);
  vtkGetVector6Macro(ClipRegion,int);

  // Description:
  // Set/Get the WholeExtent of the output.  This can be used to either
  // clip or pad the video frame.  This clipping/padding is done when
  // the frame is copied to the output, and does not change the contents
  // of the framebuffer.  This is useful e.g. for expanding
  // the output size to a power of two for texture mapping.  The
  // default is (0,-1,0,-1,0,-1) which causes the entire frame to be
  // copied to the output.
  vtkSetVector6Macro(OutputWholeExtent,int);
  vtkGetVector6Macro(OutputWholeExtent,int);

  // Description:
  // Set/Get the pixel spacing.
  // Default: (1.0,1.0,1.0)
  vtkSetVector3Macro(DataSpacing,double);
  vtkGetVector3Macro(DataSpacing,double);

  // Description:
  // Set/Get the coordinates of the lower, left corner of the frame.
  // Default: (0.0,0.0,0.0)
  vtkSetVector3Macro(DataOrigin,double);
  vtkGetVector3Macro(DataOrigin,double);

  // Description:
  // For RGBA output only (4 scalar components), set/get the opacity.  This
  // will not modify the existing contents of the framebuffer, only
  // subsequently grabbed frames.  The default is 1.0.
  virtual void SetOpacity(float alpha);
  virtual float GetOpacity();

  // Description:
  // This value is incremented each time a frame is grabbed.
  // reset it to zero (or any other value) at any time.
  vtkGetMacro(FrameCount, int);
  vtkSetMacro(FrameCount, int);

  // Description:
  // Get the frame index relative to the 'beginning of the tape'.  The
  // maximum value is the frame buffer's size - 1.
  vtkGetMacro(FrameIndex, int);

  // Description:
  // Get a time stamp in seconds (resolution of milliseconds) for
  // a video frame.   Time began on Jan 1, 1970.  You can specify
  // a number (negative or positive) to specify the position of the
  // video frame relative to the current frame (positive is forwards,
  // negative is backwards).
  virtual double GetFrameTimeStamp(int frame);

  // Description:
  // Get a time stamp in seconds (resolution of milliseconds) for
  // the Output.  Time began on Jan 1, 1970.  This timestamp is only
  // valid after the Output has been Updated.
  virtual double GetFrameTimeStamp() { return this->FrameTimeStamp; };

  // Description:
  // Get the buffer that is used to hold the video frames.
  virtual vtkVideoBuffer2 *GetBuffer() { return this->Buffer; };

  // Description:
  // Initialize the hardware.  This is called automatically
  // on the first Update or Grab.
  virtual void Initialize();
  virtual int GetInitialized() { return this->Initialized; };

  // Description:
  // Release the video driver.  This method must be called before
  // application exit, or else the application might hang during
  // exit.
  virtual void ReleaseSystemResources();

  // Description:
  // The internal function which actually does the grab.  You will
  // definitely want to override this if you develop a vtkVideoSource2
  // subclass.
  virtual void InternalGrab();

  // Description:
  // And internal variable which marks the beginning of a Record session.
  // These methods are for internal use only.
  void SetStartTimeStamp(double t) { this->StartTimeStamp = t; };
  virtual double GetStartTimeStamp() { return this->StartTimeStamp; };

  // Description:
  // Writes buffer contents to file: the summary file lists the image file names and their
  // associated timestamps, and a new image file is created for each frame in the buffer.
  // Currently only 2D images in VTK_LUMINANCE, VTK_RGB or VTK_RGBA format are supported.
  virtual void WriteFramesAsBMP(const char *summaryFileName, const char *filePrefix);
  virtual void WriteFramesAsBMP(const char *summaryFileName, const char *filePrefix, int numFrames);
  virtual void WriteFramesAsPNG(const char *summaryFileName, const char *filePrefix);
  virtual void WriteFramesAsPNG(const char *summaryFileName, const char *filePrefix, int numFrames);
  virtual void WriteFramesAsTIFF(const char *summaryFileName, const char *filePrefix, int compression);
  virtual void WriteFramesAsTIFF(const char *summaryFileName, const char *filePrefix, int compression, int numFrames);
  //virtual void WriteFramesAsMINCImage(const char *summaryFileName, const char *filePrefix);
  //virtual void WriteFramesAsMINCImage(const char *summaryFileName, const char *filePrefix, numFrames);

  // Description:
  // Reads buffer contents from file:  the summary file should have been generated by one of
  // vtkVideoSource2's WriteFramesAs functions.  Currently only 2D images in VTK_LUMINANCE,
  // VTK_RGB or VTK_RGBA format are supported.
  virtual void ReadFramesAsBMP(const char *summaryFileName);
  virtual void ReadFramesAsPNG(const char *summaryFileName);
  virtual void ReadFramesAsTIFF(const char *summaryFileName);
  //virtual void ReadFramesAsMINCImage(const char *summaryFileName);

protected:
  vtkVideoSource2();
  ~vtkVideoSource2();
  virtual int RequestInformation(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
  virtual void WriteFramesToFile(vtkImageWriter *writer, const char *summaryFileName, const char *filePrefix, const char *filePattern, int numFrames);
  virtual void ReadFramesFromFile(vtkImageReader2 *reader, const char *summaryFileName, int fileType);
  virtual void CopyImageToFrame(unsigned char *outPtr, unsigned char *inPtr, int bytesInFrame, int outputFormat, int *dimensions, float opacity, int fileType);

  int Initialized;

  int ClipRegion[6];
  int OutputWholeExtent[6];
  double DataSpacing[3];
  double DataOrigin[3];
  // set according to the OutputFormat
  int NumberOfScalarComponents;
  // The FrameOutputExtent is the WholeExtent for a single output frame.
  // It is initialized in ExecuteInformation.
  int FrameOutputExtent[6];

  // save this information from the output so that we can see if the
  // output scalars have changed
  int LastNumberOfScalarComponents;
  int LastOutputExtent[6];

  int Recording;
  int Playing;
  float FrameRate;
  int FrameCount;
  int FrameIndex;
  double StartTimeStamp;
  double FrameTimeStamp;

  int AutoAdvance;
  int NumberOfOutputFrames;

  // set if output needs to be cleared to be cleared before being written
  int OutputNeedsInitialization;

  // An example of asynchrony
  vtkMultiThreader *PlayerThreader;
  int PlayerThreadId;

  // The buffer used to hold the frames
  vtkVideoBuffer2 *Buffer;

  // Description:
  // These methods can be overridden in subclasses
  virtual void UpdateFrameBuffer();
  virtual void AdvanceFrameBuffer(int n);
  virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

private:
  vtkVideoSource2(const vtkVideoSource2&);  // Not implemented.
  void operator=(const vtkVideoSource2&);  // Not implemented.
};

#endif
