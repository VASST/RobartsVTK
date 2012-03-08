/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkFreehandUltrasound2.h,v $
  Language:  C++
  Date:      $Date: 2008/12/15 01:31:46 $
  Version:   $Revision: 1.20 $
  Thanks:    Thanks to David G. Gobbi who developed this class.

==========================================================================

Copyright (c) 2000-2007 Atamai, Inc.

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
// .NAME vtkFreehandUltrasound2 - real-time freehand ultrasound reconstruction
// .SECTION Description
// vtkFreehandUltrasound2 will incrementally compound ultrasound images into a
// reconstruction volume, given a transform which specifies the location of
// each ultrasound slice.  An alpha component is appended to the output to
// specify the coverage of each pixel in the output volume (i.e. whether or
// not a voxel has been touched by the reconstruction)
// .SECTION see also
// vtkVideoSource2, vtkTracker, vtkTrackerTool


#ifndef __vtkFreehandUltrasound2_h
#define __vtkFreehandUltrasound2_h

#include "vtkImageAlgorithm.h"

class vtkLinearTransform;
class vtkMatrix4x4;
class vtkMultiThreader;
class vtkVideoSource2;
class vtkTrackerTool;
class vtkTrackerBuffer;
class vtkCriticalSection;
class vtkImageData;
class vtkImageThreshold;
class vtkImageClip;
class vtkTransform;
class vtkSignalBox;

#define VTK_FREEHAND_NEAREST 0
#define VTK_FREEHAND_LINEAR 1

class VTK_EXPORT vtkFreehandUltrasound2 : public vtkImageAlgorithm
{
public:
  
  static vtkFreehandUltrasound2 *New();
  vtkTypeRevisionMacro(vtkFreehandUltrasound2, vtkImageAlgorithm);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description: 
  // Set/Get the 2D image slice to insert into the reconstruction volume
  // the slice is the vtkImageData 'slice' (kind of like an input)
  // that is inserted into the reconstructed 3D volume (the output)
  virtual void SetSlice(vtkImageData *);
  virtual vtkImageData* GetSlice();

  // Description:
  // Get the accumulation buffer
  // accumulation buffer is for compounding, there is a voxel in
  // the accumulation buffer for each voxel in the output
  virtual vtkImageData *GetAccumulationBuffer();

  // Description:
  // Get the accumulation buffer, for use with triggering
  // accumulation buffer is for compounding, there is a voxel in
  // the accumulation buffer for each voxel in the output
  virtual vtkImageData *GetAccumulationBuffer(int port);

  // Description:
  // Get the output reconstructed 3D ultrasound volume
  // (the output is the reconstruction volume, the second component
  // is the alpha component that stores whether or not a voxel has
  // been touched by the reconstruction)
  virtual vtkImageData *GetOutput();

  // Description:
  // Get the output reconstructed 3D ultrasound volume, for use with
  // triggering
  // (the output is the reconstruction volume, the second component
  // is the alpha component that stores whether or not a voxel has
  // been touched by the reconstruction)
  virtual vtkImageData *GetOutput(int port);

  // Description:
  // Set/Get the video source to input the slices from
  virtual void SetVideoSource(vtkVideoSource2 *);
  vtkGetObjectMacro(VideoSource,vtkVideoSource2);

  // Description:
  // Set/Get the tracker tool to input transforms from
  virtual void SetTrackerTool(vtkTrackerTool *);
  vtkGetObjectMacro(TrackerTool,vtkTrackerTool);

  // Description:
  // Start doing a reconstruction from the video frames stored
  // in the VideoSource buffer.  You should first use 'Seek'
  // on the VideoSource to rewind first.  Then the reconstruction
  // will advance through n frames one by one until the
  // reconstruction is complete.  The reconstruction
  // is performed in the background.
  void StartReconstruction(int n);
  
  // Description:
  // Stop the reconstruction.  The number of frames remaining to
  // be reconstructed is returned.
  int StopReconstruction();

  // Description:
  // Start doing real-time reconstruction from the video source.
  // This will spawn a thread that does the reconstruction in the
  // background.
  void StartRealTimeReconstruction();
  
  // Description:
  // Stop the real-time reconstruction.
  void StopRealTimeReconstruction();

  // Description:
  // Get the reconstruction rate.
  double GetReconstructionRate() { return this->ReconstructionRate; };

  // Description:
  // Fill holes in the output by using the weighted average of the
  // surrounding voxels.  If Compounding is off, then all hit voxels
  // are weighted equally. 
  void FillHolesInOutput();

  // Description:
  // Save the raw data in the (relative!) directory specified.  The directory will
  // be created if it doesn't exist, and the following files will be
  // written inside it:
  // freehand.txt - a file with the freehand parameters within it
  // track.txt - a file with timestamped tracking information;
  // video.txt - a file with timestamps for each video image;
  // zXXXX.png - all of the video images, in sequential order.
  // You should first use 'Seek' on the VideoSource to rewind it.
  // Then the vtkVideoSource2 will be advanced one frame at a time
  // until n frames have been saved.
  void SaveRawData(const char *directory, int n);

  // Description:
  // Read the raw data from the specified directory and use it for the
  // following reconstructions.
  void ReadRawData(const char *directory);

  // Description:
  // Set the time by which the video lags behind the tracking information,
  // in seconds.  This value may be negative.  Default: 0.
  vtkSetMacro(VideoLag,double);
  vtkGetMacro(VideoLag,double);

  // Description:
  // Cause the slice to be inserted into the first reconstruction volume
  void InsertSlice();

  // Description:
  // Cause the slice to be inserted into the ith reconstruction volume
  void InsertSlice(int i);

  // Description:
  // Clear the data volume.
  void ClearOutput();

  // Description:
  // Set the clip rectangle (x0,y0,x1,y1) to apply to the image. 
  // Specify the rectange in millimeter coords, not pixel indices.
  vtkSetVector4Macro(ClipRectangle,double);
  vtkGetVector4Macro(ClipRectangle,double);

  // Description:
  // Get the clip rectangle as an extent, given a specific origin
  // spacing, and max possible extent.
  void GetClipExtent(int clipExtent[6],
		     vtkFloatingPointType origin[3],
		     vtkFloatingPointType spacing[3],
		     const int extent[6]);

  // Description:
  // If the ultrasound probe collects a fan of data, specify the position and
  // dimensions of the fan.
  vtkSetVector2Macro(FanAngles,double);
  vtkGetVector2Macro(FanAngles,double);
  vtkSetVector2Macro(FanOrigin,double);
  vtkGetVector2Macro(FanOrigin,double);
  vtkSetMacro(FanDepth,double);
  vtkGetMacro(FanDepth,double);

  // Description:
  // Set the axes of the slice to insert into the reconstruction volume,
  // relative the (x,y,z) axes of the reconstruction volume itself.
  // The axes are extracted from the 4x4 matrix:  The x-axis is the 
  // first column, the y-axis is the second column, the z-axis is the 
  // third column, and the origin is the final column.  The bottom
  // row of the matrix should always be (0,0,0,1).
  // If you don't set the axes, the axes will default to 
  // (1,0,0), (0,1,0), (0,0,1) and their origin will be (0,0,0)
  virtual void SetSliceAxes(vtkMatrix4x4 *);
  vtkGetObjectMacro(SliceAxes,vtkMatrix4x4);

  // Description:
  // Set a transform to be applied to the SliceAxes.
  // If you don't set this, it will be treated as the identity transform.
  // the slice axes matrix and slice transform together give the
  // coordinate transformation from the local coordinate system
  // of the Slice to the coordinate system of the Output.
  virtual void SetSliceTransform(vtkLinearTransform *);
  vtkGetObjectMacro(SliceTransform,vtkLinearTransform);

  // Description:
  // Decide whether or not to use a rotating probe (default off)
  void SetRotatingProbe(int probe);
  vtkGetMacro(RotatingProbe,int);

  // Description:
  // Set the fan rotation (rotating probe) - If you don't set this, it will be treated as 0.
  void SetFanRotation(int rot);
  vtkGetMacro(FanRotation,int);

  // Description:
  // Keeps track of the old fan rotation (rotating probe), so that it can be reversed when
  // we get the new rotation
  void SetPreviousFanRotation(int rot);
  vtkGetMacro(PreviousFanRotation,int);

  // Description:
  // Threshold the image to pick out the screen pixels showing the
  // rotation (min intensity is FanRotationImageThreshold1, max
  // intensity is FanRotationImageThreshold2)
  // TODO should read from file... for parameters like fan depth too
  void SetFanRotationImageThreshold1(int thresh);
  vtkGetMacro(FanRotationImageThreshold1,int);
  void SetFanRotationImageThreshold2(int thresh);
  vtkGetMacro(FanRotationImageThreshold2,int);
  vtkGetObjectMacro(RotationThresholder, vtkImageThreshold);

  // Description:
  // Parameters for isolating the region on the screen holding the rotation
  // value
  vtkSetMacro(FanRotationXShift,int);
  vtkGetMacro(FanRotationXShift,int);
  vtkSetMacro(FanRotationYShift,int);
  vtkGetMacro(FanRotationYShift,int);

  // Description:
  // Fan depth in centimeters
  vtkSetMacro(FanDepthCm,int);
  vtkGetMacro(FanDepthCm,int);

  // Description:
  // Clipper to isolate the part of the image containing the rotation data
  vtkGetObjectMacro(RotationClipper, vtkImageClip);

  // Description:
  // Code to deal with whether the image is "flipped" (image is flipped if
  // the point of the fan is at the top of screen)
  vtkSetMacro(ImageIsFlipped,int);
  vtkGetMacro(ImageIsFlipped,int);

  // Description:
  // Get the current rotation of the fan (rotating probe)
  int CalculateFanRotationValue(vtkImageThreshold *);

//BTX

  // Description:
  // Get a coded representation of the pixels showing the
  // current rotation of the fan (rotating probe)
  int GetFanRepresentation (vtkImageThreshold *, int[12]);

//ETX

  // Description:
  // Turn on and off optimizations (default on, turn them off only if
  // they are not stable on your architecture).
  //   0 means no optimization (almost never used)
  //   1 means break transformation into x, y and z components, and
  //      don't do bounds checking for nearest-neighbor interpolation
  //   2 means used fixed-point (i.e. integer) math instead of float math
  vtkSetMacro(Optimization,int);
  vtkGetMacro(Optimization,int);

  // Description:
  // Set/Get the interpolation mode, default is nearest neighbor. 
  vtkSetMacro(InterpolationMode,int);
  vtkGetMacro(InterpolationMode,int);
  void SetInterpolationModeToNearestNeighbor()
    { this->SetInterpolationMode(VTK_FREEHAND_NEAREST); };
  void SetInterpolationModeToLinear()
    { this->SetInterpolationMode(VTK_FREEHAND_LINEAR); };
  char *GetInterpolationModeAsString();

  // Description:
  // Turn on or off the compounding (default on, which means
  // that scans will be compounded where they overlap instead of the
  vtkGetMacro(Compounding,int);
  void SetCompounding(int);
  vtkBooleanMacro(Compounding,int);

  // Description:
  // Spacing, origin, and extent of output data
  // You MUST set this information.
  vtkSetVector3Macro(OutputSpacing, vtkFloatingPointType);
  vtkGetVector3Macro(OutputSpacing, vtkFloatingPointType);
  vtkSetVector3Macro(OutputOrigin, vtkFloatingPointType);
  vtkGetVector3Macro(OutputOrigin, vtkFloatingPointType);
  vtkSetVector6Macro(OutputExtent, int);
  vtkGetVector6Macro(OutputExtent, int);

  // Description:
  // If true, flips image data along the X/Y axis when copying from the
  // input frame to the output image
  vtkSetMacro(FlipHorizontalOnOutput, int);
  vtkGetMacro(FlipHorizontalOnOutput, int);
  vtkBooleanMacro(FlipHorizontalOnOutput,int);
  vtkSetMacro(FlipVerticalOnOutput, int);
  vtkGetMacro(FlipVerticalOnOutput, int);
  vtkBooleanMacro(FlipVerticalOnOutput, int);

  // Description:
  // When determining the modified time of the source. 
  unsigned long int GetMTime();

  //Description:
  //Number of pixels from the tip of the fan to the bottom of screen,only
  //used if flipping horizontally
  vtkSetMacro(NumberOfPixelsFromTipOfFanToBottomOfScreen,int)
  vtkGetMacro(NumberOfPixelsFromTipOfFanToBottomOfScreen,int)

  // Description:
  // Option to turn triggering for ECG-gated acquisition on or off (default off)
  vtkSetMacro(Triggering,int);
  vtkGetMacro(Triggering,int);
  vtkBooleanMacro(Triggering, int);

  // Description:
  // Signal box for the triggering (must have at least two phases)
  virtual void SetSignalBox(vtkSignalBox *);
  vtkGetObjectMacro(SignalBox, vtkSignalBox);

  // Description:
  // The current phase for triggering
  vtkSetMacro(CurrentPhase,int);
  vtkGetMacro(CurrentPhase,int);

  // Description:
  // The previous phase for triggering
  vtkSetMacro(PreviousPhase,int);
  vtkGetMacro(PreviousPhase,int);

  // Description:
  // The number of output volumes, set to one by default
  // (if you are triggering, this must be equal to
  // the number of phases in the signal box, and must be
  // at least two)
  void SetNumberOfOutputVolumes(int num);
  vtkGetMacro(NumberOfOutputVolumes,int);

  // Description:
  // Whether or not to discard image slices that occur during
  // outlier heart rates (default off)
  void SetDiscardOutlierHeartRates(int discard);
  vtkGetMacro(DiscardOutlierHeartRates, int);
  vtkBooleanMacro(DiscardOutlierHeartRates, int);

  // Description:
  // Set/Get the image slice for a particular phase in the buffer, when
  // discarding outlier heart rates
  void SetSliceBuffer(int phase, vtkImageData* inData);
  vtkImageData* GetSliceBuffer(int phase);

  // Description:
  // Set/Get the slice axes matrix for a particular phase in the buffer,
  // when discarding outlier heart rates
  void SetSliceAxesBuffer(int phase, vtkMatrix4x4* matrix);
  vtkMatrix4x4* GetSliceAxesBuffer(int phase);

  // Description:
  // Set/Get the slice transform for a particular phase
  // in the buffer, when discarding outlier heart rates
  void SetSliceTransformBuffer(int phase, vtkLinearTransform* transform);
  vtkLinearTransform* GetSliceTransformBuffer(int phase);

  // Description:
  // Set the amount of time (in seconds) that we will monitor the heart
  // rate for before starting the reconstruction - the mean heart rate
  // will be used as the baseline heart rate (default 10 seconds)
  vtkSetMacro(ECGMonitoringTime, double);
  vtkGetMacro(ECGMonitoringTime, double);

  // Description:
  // Set the maximum number of trials to find the heart rate measurements
  // (default 5)
  vtkSetMacro(NumECGMeasurementTrials, int);
  vtkGetMacro(NumECGMeasurementTrials, int);

  // Description:
  // Percentage of the heart rate that the heart rate can increase
  // by and still be considered valid (i.e. specify 10 for 10%)
  // (default 20%)
  vtkSetMacro(PercentageIncreasedHeartRateAllowed, double);
  vtkGetMacro(PercentageIncreasedHeartRateAllowed, double);

  // Description:
  // Percentage of the heart rate that the heart rate can decrease
  // by and still be considered valid (i.e. specify 10 for 10%)
  // (default 20%)
  vtkSetMacro(PercentageDecreasedHeartRateAllowed, double);
  vtkGetMacro(PercentageDecreasedHeartRateAllowed, double);

  // Description:
  // The mean heart rate over the time specified by ECGMonitoringTime,
  // as calculated at the beginning of the reconstruction
  vtkGetMacro(MeanHeartRate, double);

  // Description:
  // The maximum heart rate that is still valid
  vtkGetMacro(MaxAllowedHeartRate, double);

  // Description:
  // The minimum heart rate that is still valid
  vtkGetMacro(MinAllowedHeartRate, double);

  // Description:
  // Stores the timestamps used to insert the slices if turned on (default off)
  void SetSaveInsertedTimestamps(int insert);
  vtkGetMacro(SaveInsertedTimestamps, int);
  vtkBooleanMacro(SaveInsertedTimestamps, int);

  // Description:
  // Give an estimation of the number of slices that will be inserted per
  // phase, in order to initialize the array storing the inserted timestamps
  // (this is an estimated maximum, will stop inserting timestamps if it is
  // inserted with a warning message) (default 250)
  void SetMaximumNumberOfInsertionsPerPhase(int num);
  vtkGetMacro(MaximumNumberOfInsertionsPerPhase, int);

  // Description:
  // Get/Insert the most recent timestamp into the timestamp buffer
  void InsertSliceTimestamp(int phase, double timestamp);
  double GetSliceTimestamp(int phase);

  // Description:
  // Increment the counter for the timestamp buffer
  void IncrementSliceTimestampCounter(int phase);

  // Description:
  // Print out the summary file for the scan
  void SaveSummaryFile(const char *directory);

  // Description:
  // Execute the reconstruction thread
  void ThreadedSliceExecute(vtkImageData *inData, vtkImageData *outData,
			    int extent[6], int threadId, int phase);
  
  // Description:
  // To split the extent over meany threads
  int SplitSliceExtent(int splitExt[6], int startExt[6], int num, int total);

  // Description:
  // For filling holes
  void ThreadedFillExecute(vtkImageData *outData,	
			   int outExt[6], int threadId, int phase);

  // Attributes - not protected to be accessible from reconstruction thread

//BTX
  double ReconstructionRate;
  int RealTimeReconstruction; // # real-time or buffered
  int ReconstructionFrameCount; // # of frames to reconstruct
  // used for non-realtime reconstruction, and to store the tracker buffer's
  // information after we stop tracking
  vtkTrackerBuffer *TrackerBuffer; // used for non-realtime reconstruction,
//ETX

  int PixelCount[4];

 // Description:
  // Set/Get the number of pixels inserted (by a particular thread)
  int GetPixelCount();
  void SetPixelCount(int threadId, int val);
  void IncrementPixelCount(int threadId, int increment);

  // Description:
  // Get the thread ID for the reconstruction thread
  int GetReconstructionThreadId(){ return this->ReconstructionThreadId; };

protected:
  vtkFreehandUltrasound2();
  ~vtkFreehandUltrasound2();

  double VideoLag;
  vtkImageData *Slice;

  int RotatingProbe;
  int FanRotation;
  int PreviousFanRotation;
  int FanRotationImageThreshold1;
  int FanRotationImageThreshold2;
  int FanRotationXShift;
  int FanRotationYShift;
  int FanDepthCm;
  vtkImageClip *RotationClipper;
  vtkImageThreshold* RotationThresholder;
  int ImageIsFlipped; // 0 means no (good pizza), 1 means yes (bad pizza)
  int FlipHorizontalOnOutput;
  int FlipVerticalOnOutput;

  vtkMatrix4x4 *SliceAxes;
  vtkLinearTransform *SliceTransform;

  int InterpolationMode;
  int Optimization;
  int Compounding;
  vtkFloatingPointType OutputOrigin[3];
  vtkFloatingPointType OutputSpacing[3];
  int OutputExtent[6];

  //TODO put back?
  /*vtkFloatingPointType OldOutputOrigin[3];
  vtkFloatingPointType OldOutputSpacing[3];
  int OldOutputExtent[6];
  int OldScalarType;
  int OldNComponents;*/

  double ClipRectangle[4];
  double FanAngles[2];
  double FanOrigin[2];
  double FanDepth;
  int NumberOfPixelsFromTipOfFanToBottomOfScreen;

  vtkMatrix4x4 *IndexMatrix;
  vtkMatrix4x4 *LastIndexMatrix;

  vtkImageData **AccumulationBuffers;
  int NeedsClear;

  vtkCriticalSection *ActiveFlagLock;
  vtkMultiThreader *Threader;
  int NumberOfThreads;

  vtkVideoSource2 *VideoSource;
  vtkTrackerTool *TrackerTool;

  int ReconstructionThreadId;

  int Triggering;
  vtkSignalBox *SignalBox;
  int CurrentPhase;
  int PreviousPhase;
  int NumberOfOutputVolumes;

  int DiscardOutlierHeartRates;
  double ECGMonitoringTime;
  int NumECGMeasurementTrials;
  double PercentageIncreasedHeartRateAllowed;
  double PercentageDecreasedHeartRateAllowed;
  double MeanHeartRate;
  double MaxAllowedHeartRate;
  double MinAllowedHeartRate;

  // For discarding ECGs - put slice and transform into these buffers
  // (size = num output volumes), wait one cycle, check ECG and put them
  // in if ECG is good
  vtkImageData **SliceBuffer;
  vtkMatrix4x4 **SliceAxesBuffer;
  vtkLinearTransform **SliceTransformBuffer;

  // For saving the timestamps of the inserted slices
  int SaveInsertedTimestamps;
  int MaximumNumberOfInsertionsPerPhase;
  double **InsertedTimestampsBuffer;
  int *InsertedTimestampsCounter;

  // Description:
  // Setup the threader and execute
  void MultiThread(vtkImageData *inData, vtkImageData *outData, int phase);

  // Description:
  // Setup the threader and execute
  void MultiThreadFill(vtkImageData *outData, int phase);

  // Description:
  // The IndexMatrix gives the coordinate transformation from (i,j,k)
  // voxel indices in the slice to (i,j,k) voxel indices in the
  // output.
  vtkMatrix4x4 *GetIndexMatrix();
  
  // Description:
  // Getting the IndexMatrix when discarding slices based on ECG signal
  vtkMatrix4x4 *GetIndexMatrix(int phase);

  // Description:
  // Insert the slice into the first volume, with optimization
  void OptimizedInsertSlice();

  // Description:
  // Insert the slice into the ith volume, with optimization
  void OptimizedInsertSlice(int i);

  // Description:
  // Actually clear the output volume(s), by calling InternalInternalClearOutput
  void InternalClearOutput();

  // Description:
  // Actually clear a single output volume
  void InternalInternalClearOutput(int i);

  // Description:
  // Clears the slice, slice axes and slice transform buffers for you
  // For use with discarding based on ECG signals
  void ClearSliceBuffers();

  // Description:
  // Calculate the mean heart rate and allowed maximum and minimum
  // heart rates
  int CalculateHeartRateParameters();

  // Description:
  // Similar to RequestInformation(), but applied to the output
  void InternalExecuteInformation();

  virtual int FillInputPortInformation(int port, vtkInformation* info);
  virtual int FillOutputPortInformation(int port, vtkInformation* info);
  virtual int ProcessRequest(vtkInformation*,
                             vtkInformationVector**,
                             vtkInformationVector*);
  virtual int RequestInformation(vtkInformation* request,
                                 vtkInformationVector** inputVector,
                                 vtkInformationVector* outputVector);
  virtual int RequestUpdateExtent(vtkInformation*,
                                 vtkInformationVector**,
                                 vtkInformationVector*);
  virtual int RequestData(vtkInformation *request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);
  virtual int ComputePipelineMTime(vtkInformation *request,
				   vtkInformationVector **inInfoVec,
				   vtkInformationVector *outInfoVec,
				   int requestFromOutputPort,
				   unsigned long* mtime);

private:
  vtkFreehandUltrasound2(const vtkFreehandUltrasound2&);
  void operator=(const vtkFreehandUltrasound2&);

  // Description:
  // Both StartReconstruction() and StartRealTimeReconstruction() need to run
  // this before reconstructing with triggering, to make sure that the signal
  // box is set up properly
  int TestBeforeReconstructingWithTriggering();

};

//----------------------------------------------------------------------------
inline char *vtkFreehandUltrasound2::GetInterpolationModeAsString()
{
  switch (this->InterpolationMode)
    {
    case VTK_FREEHAND_NEAREST:
      return "NearestNeighbor";
    case VTK_FREEHAND_LINEAR:
      return "Linear";
    default:
      return "";
    }
}

#endif





