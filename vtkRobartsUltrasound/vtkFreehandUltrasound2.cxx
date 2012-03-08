/*=========================================================================

Copyright (c) 2000,2002 David Gobbi.

=========================================================================*/

//#include <limits.h>
//#include <float.h>
//#include <math.h>
//#include <stdio.h>

// includes for mkdir
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include "vtkFreehandUltrasound2.h"
#include "vtkObjectFactory.h"

#include "fixed.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMath.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkMultiThreader.h"
#include "vtkCriticalSection.h"
#include "vtkTimerLog.h"
#include "vtkTrackerBuffer.h"
#include "vtkVideoSource2.h"
#include "vtkTrackerTool.h"
#include "vtkMutexLock.h"
#include "vtkImageThreshold.h"
#include "vtkImageClip.h"
#include "vtkImageFlip.h"
#include "vtkSignalBox.h"
#include "vtkVideoBuffer2.h"
#include "vtkVideoFrame2.h"

vtkCxxRevisionMacro(vtkFreehandUltrasound2, "$Revision: 1.20 $");
vtkStandardNewMacro(vtkFreehandUltrasound2);

// for keeping track of threading information
struct vtkFreehand2ThreadStruct
{
  vtkFreehandUltrasound2 *Filter;
  vtkImageData   *Input;
  vtkImageData   *Output;
  int Phase;
};

//----------------------------------------------------------------------------
// Constructor
// Just initialize objects and set initial values for attributes
//----------------------------------------------------------------------------
vtkFreehandUltrasound2::vtkFreehandUltrasound2()
{

  // set the video lag
  this->VideoLag = 0.0;

  // one PixelCount for each threadId, where 0 <= threadId < 4
  this->PixelCount[0] = 0;
  this->PixelCount[1] = 0;
  this->PixelCount[2] = 0;
  this->PixelCount[3] = 0;

  // set up the output parameters
  this->OutputSpacing[0] = 1.0;
  this->OutputSpacing[1] = 1.0;
  this->OutputSpacing[2] = 1.0;

  this->OutputOrigin[0] = -127.5;
  this->OutputOrigin[1] = -127.5;
  this->OutputOrigin[2] = -127.5;

  this->OutputExtent[0] = 0;
  this->OutputExtent[1] = 255;
  this->OutputExtent[2] = 0;
  this->OutputExtent[3] = 255;
  this->OutputExtent[4] = 0;
  this->OutputExtent[5] = 255;

  // TODO put back?
  //this->OldScalarType = VTK_UNSIGNED_CHAR;
  //this->OldNComponents = 1;
  
  //this->OldOutputSpacing[0] = 1.0;
  //this->OldOutputSpacing[1] = 1.0;
  //this->OldOutputSpacing[2] = 1.0;

  //this->OldOutputOrigin[0] = 0;
  //this->OldOutputOrigin[1] = 0;
  //this->OldOutputOrigin[2] = 0;

  //this->OldOutputExtent[0] = 0;
  //this->OldOutputExtent[1] = 0;
  //this->OldOutputExtent[2] = 0;
  //this->OldOutputExtent[3] = 0;
  //this->OldOutputExtent[4] = 0;
  //this->OldOutputExtent[5] = 0;
  
  // this will force ClearOutput() to run, which will allocate the output
  // and the accumulation buffer(s)
  this->NeedsClear = 1;

  this->InterpolationMode = VTK_FREEHAND_NEAREST;
  this->Compounding = 0;
  this->Optimization = 2;

  this->Slice = NULL;
  this->SliceAxes = NULL;
  this->SliceTransform = vtkTransform::New(); // initialized to identity
  this->IndexMatrix = NULL;
  this->LastIndexMatrix = NULL;

  this->ClipRectangle[0] = -1e8;
  this->ClipRectangle[1] = -1e8;
  this->ClipRectangle[2] = +1e8;
  this->ClipRectangle[3] = +1e8;

  this->FanAngles[0] = 0.0;
  this->FanAngles[1] = 0.0;
  this->FanOrigin[0] = 0.0;
  this->FanOrigin[1] = 0.0;
  this->FanDepth = +1e8;

  // one thread for each CPU is used for the reconstruction
  this->Threader = vtkMultiThreader::New();
  this->NumberOfThreads = 1; 
  
  // for running the reconstruction in the background
  this->VideoSource = NULL;
  this->TrackerTool = NULL;
  this->TrackerBuffer = vtkTrackerBuffer::New();
  this->ReconstructionRate = 0;
  this->ReconstructionThreadId = -1;
  this->RealTimeReconstruction = 0;
  this->ReconstructionFrameCount = 0;
  this->ActiveFlagLock = vtkCriticalSection::New();

  // parameters for rotating probes
  this->RotatingProbe = 0;
  this->FanRotation = 0;
  this->PreviousFanRotation = 0;
  this->RotationClipper = NULL;
  this->RotationThresholder = NULL;
  this->FlipHorizontalOnOutput = 0;
  this->FlipVerticalOnOutput = 0;

  // parameters for triggering (ECG-gating)
  this->Triggering = 0;
  this->SignalBox = NULL;
  this->CurrentPhase = -1;
  this->PreviousPhase = -1;

  // parameters for discarding outlier heart rates
  this->DiscardOutlierHeartRates = 0;
  this->ECGMonitoringTime = 10;
  this->NumECGMeasurementTrials = 5;
  this->PercentageIncreasedHeartRateAllowed = 20;
  this->PercentageDecreasedHeartRateAllowed = 20;
  this->MeanHeartRate = 0;
  this->SliceBuffer = NULL;
  this->SliceAxesBuffer = NULL;
  this->SliceTransformBuffer = NULL;

  // parameters for saving timestamps
  this->SaveInsertedTimestamps = 0;
  this->MaximumNumberOfInsertionsPerPhase = 250;
  this->InsertedTimestampsBuffer = NULL;
  this->InsertedTimestampsCounter = NULL;

  // pipeline setup
  this->SetNumberOfInputPorts(0);
  // sets the number of output volumes, creates the output ports on the VTK 5
  // pipeline, and creates the accumulation buffers
  this->NumberOfOutputVolumes = 0; // dummy so that the following does something

  this->SetNumberOfOutputVolumes(1);
}

//----------------------------------------------------------------------------
// Destructor
// Stop the reconstruction and delete stuff
//----------------------------------------------------------------------------
vtkFreehandUltrasound2::~vtkFreehandUltrasound2()
{
  this->StopRealTimeReconstruction();
  this->SetSlice(NULL);
  this->SetSliceTransform(NULL);
  this->SetSliceAxes(NULL);
  this->SetVideoSource(NULL);
  this->SetTrackerTool(NULL);

  if (this->IndexMatrix)
    {
    this->IndexMatrix->Delete();
    }
  if (this->LastIndexMatrix)
    {
    this->LastIndexMatrix->Delete();
    }
  for (int phase = 0; phase < this->NumberOfOutputVolumes; phase++)
    {
    this->AccumulationBuffers[phase]->Delete();
    }
  if (this->Threader)
    {
    this->Threader->Delete();
    }
  if (this->TrackerBuffer)
    {
    this->TrackerBuffer->Delete();
    }
	if (this->RotationClipper)
	  {
		this->RotationClipper->Delete();
	  }
	if (this->RotationThresholder)
	  {
		this->RotationThresholder->Delete();
	  }
  if (this->ActiveFlagLock)
    {
    this->ActiveFlagLock->Delete();
    }
  if (this->SliceBuffer)
    {
    for (int phase = 0; phase < this->NumberOfOutputVolumes; phase++)
      {
      if (this->SliceBuffer[phase])
        {
        this->SliceBuffer[phase]->Delete();
        }
      }
    }
  if (this->SliceAxesBuffer)
    {
    for (int phase = 0; phase < this->NumberOfOutputVolumes; phase++)
      {
      if (this->SliceAxesBuffer[phase])
        {
        this->SliceAxesBuffer[phase]->Delete();
        }
      }
    }
  if (this->SliceTransformBuffer)
    {
    for (int phase = 0; phase < this->NumberOfOutputVolumes; phase++)
      {
      if (this->SliceTransformBuffer[phase])
        {
        this->SliceTransformBuffer[phase]->Delete();
        }
      }
    }
  if (this->InsertedTimestampsBuffer)
    {
    for (int phase = 0; phase < this->NumberOfOutputVolumes; phase++)
      {
      delete [] this->InsertedTimestampsBuffer[phase];
      }
    delete [] this->InsertedTimestampsBuffer;
    }
  if (this->InsertedTimestampsCounter)
    {
    delete [] this->InsertedTimestampsCounter;
    }

}

//----------------------------------------------------------------------------
// PrintSelf
// Prints out attribute data
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Slice: " << this->Slice << "\n";
  if (this->Slice)
    {
    this->Slice->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "VideoSource: " << this->VideoSource << "\n";
  if (this->VideoSource)
    {
    this->VideoSource->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "TrackerTool: " << this->TrackerTool << "\n";
  if (this->TrackerTool)
    {
    this->TrackerTool->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "TrackerBuffer: " << this->TrackerBuffer << "\n";
  if (this->TrackerBuffer)
    {
    this->TrackerBuffer->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "VideoLag: " << this->VideoLag << "\n";
  os << indent << "AccumulationBuffers: " << this->AccumulationBuffers << "\n";
  if (this->AccumulationBuffers)
    {
    os << indent << "AccumulationBuffers[0]:\n";
    this->AccumulationBuffers[0]->PrintSelf(os,indent.GetNextIndent());
    }

  os << indent << "ClipRectangle: " << this->ClipRectangle[0] << " " <<
    this->ClipRectangle[1] << " " << this->ClipRectangle[2] << " " <<
    this->ClipRectangle[3] << "\n";
  os << indent << "FanAngles: " << this->FanAngles[0] << " " <<
    this->FanAngles[1] << "\n";
  os << indent << "FanOrigin: " << this->FanOrigin[0] << " " <<
    this->FanOrigin[1] << "\n";
  os << indent << "FanDepth: " << this->FanDepth << "\n";

  os << indent << "OutputSpacing: " << this->OutputSpacing[0] << " " <<
    this->OutputSpacing[1] << " " << this->OutputSpacing[2] << "\n";
  os << indent << "OutputOrigin: " << this->OutputOrigin[0] << " " <<
    this->OutputOrigin[1] << " " << this->OutputOrigin[2] << "\n";
  os << indent << "OutputExtent: " << this->OutputExtent[0] << " " <<
    this->OutputExtent[1] << " " << this->OutputExtent[2] << " " <<
    this->OutputExtent[3] << " " << this->OutputExtent[4] << " " <<
    this->OutputExtent[5] << "\n";

  /*os << indent << "OldOutputOrigin: " << this->OldOutputOrigin[0] << " " <<
    this->OldOutputOrigin[1] << " " << this->OldOutputOrigin[2] << "\n";
  os << indent << "OldOutputSpacing: " << this->OldOutputSpacing[0] << " " <<
    this->OldOutputSpacing[1] << " " << this->OldOutputSpacing[2] << "\n";
  os << indent << "OldOutputExtent: " << this->OldOutputExtent[0] << " " <<
    this->OldOutputExtent[1] << " " << this->OldOutputExtent[2] << " " <<
    this->OldOutputExtent[3] << " " << this->OldOutputExtent[4] << " " <<
    this->OldOutputExtent[5] << "\n";
  os << indent << "OldScalarType: " << this->OldScalarType << "\n";
  os << indent << "OldNComponents: " << this->OldNComponents << "\n";*/

  os << indent << "RotatingProbe: " << (this->RotatingProbe ? "On\n":"Off\n");
  os << indent << "FanRotation: " << this->FanRotation << "\n";
  os << indent << "PreviousFanRotation: " << this->PreviousFanRotation << "\n";
  os << indent << "FanRotationImageThreshold1: " << this->FanRotationImageThreshold1 << "\n";
  os << indent << "FanRotationImageThreshold2: " << this->FanRotationImageThreshold2 << "\n";
  os << indent << "FanRotationXShift: " << this->FanRotationXShift << "\n";
  os << indent << "FanRotationYShift: " << this->FanRotationYShift << "\n";
  os << indent << "FanDepthCm: " << this->FanDepthCm << "\n";
  os << indent << "RotationClipper: " << this->RotationClipper << "\n";
  if (this->RotationClipper)
    {
    this->RotationClipper->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "RotationThresholder: " << this->RotationThresholder << "\n";
  if (this->RotationThresholder)
    {
    this->RotationThresholder->PrintSelf(os,indent.GetNextIndent());
    }

  os << indent << "ImageIsFlipped: " << (this->ImageIsFlipped ? "On\n":"Off\n");
  os << indent << "FlipHorizontalOnOutput: " << (this->FlipHorizontalOnOutput ? "On\n":"Off\n");
  os << indent << "FlipVerticalOnOutput: " << (this->FlipVerticalOnOutput ? "On\n":"Off\n");
  os << indent << "NumberOfPixelsFromTipOfFanToBottomOfScreen: " << this->NumberOfPixelsFromTipOfFanToBottomOfScreen << "\n";

  os << indent << "SliceAxes: " << this->SliceAxes << "\n";
  if (this->SliceAxes)
    {
    this->SliceAxes->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "SliceTransform: " << this->SliceTransform << "\n";
  if (this->SliceTransform)
    {
    this->SliceTransform->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "InterpolationMode: " << this->GetInterpolationModeAsString() << "\n";
  os << indent << "Optimization: " << (this->Optimization ? "On\n":"Off\n");
  os << indent << "Compounding: " << (this->Compounding ? "On\n":"Off\n");
  os << indent << "IndexMatrix: " << this->IndexMatrix << "\n";
  if (this->IndexMatrix)
    {
    this->IndexMatrix->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "LastIndexMatrix: " << this->LastIndexMatrix << "\n";
  if (this->LastIndexMatrix)
    {
    this->LastIndexMatrix->PrintSelf(os,indent.GetNextIndent());
    }

  os << indent << "Triggering: " << (this->Triggering ? "On\n":"Off\n");
  os << indent << "SignalBox: " << this->SignalBox << "\n";
  if (this->SignalBox)
    {
    this->SignalBox->PrintSelf(os,indent.GetNextIndent());
    }
  os << indent << "NumberOfOutputVolumes: " << this->NumberOfOutputVolumes << "\n";

  os << indent << "DiscardOutlierHeartRates: " << (this->DiscardOutlierHeartRates ? "On\n":"Off\n");
  os << indent << "ECGMonitoringTime: " << this->ECGMonitoringTime << "\n";
  os << indent << "NumECGMeasurementTrials: " << this->NumECGMeasurementTrials << "\n";
  os << indent << "PercentageIncreasedHeartRateAllowed: " << this->PercentageIncreasedHeartRateAllowed << "\n";
  os << indent << "PercentageDecreasedHeartRateAllowed: " << this->PercentageDecreasedHeartRateAllowed << "\n";
  os << indent << "MeanHeartRate: " << this->MeanHeartRate << "\n";
  os << indent << "MaxAllowedHeartRate: " << this->MaxAllowedHeartRate << "\n";
  os << indent << "MinAllowedHeartRate: " << this->MinAllowedHeartRate << "\n";
  os << indent << "SliceBuffer: " << this->SliceBuffer << "\n";
  if (this->SliceBuffer)
    {
    if (this->SliceBuffer[0])
      {
      os << indent << "SliceBuffer[0]: \n";
      this->SliceBuffer[0]->PrintSelf(os,indent.GetNextIndent());
      }
    }
  os << indent << "SliceAxesBuffer: " << this->SliceAxesBuffer << "\n";
  if (this->SliceAxesBuffer)
    {
    if (this->SliceAxesBuffer[0])
      {
      os << indent << "SliceAxesBuffer[0]\n";
      this->SliceAxesBuffer[0]->PrintSelf(os,indent.GetNextIndent());
      }
    }
  os << indent << "SliceTransformBuffer: " << this->SliceTransformBuffer << "\n";
  if (this->SliceTransformBuffer)
    {
    if (this->SliceTransformBuffer[0])
      {
      os << indent << "SliceTransformBuffer[0]: \n";
      this->SliceTransformBuffer[0]->PrintSelf(os, indent.GetNextIndent());
      }
    }
  os << indent << "InsertedTimestampsCounter: " << this->InsertedTimestampsCounter << "\n";
  if (this->InsertedTimestampsCounter)
    {
    os << indent.GetNextIndent() << "InsertedTimestamps:\n";
    for (int i = 0; i < this->NumberOfOutputVolumes; i++)
      {
      os << indent.GetNextIndent() << this->InsertedTimestampsCounter[i] << "\n";
      }
    }

  os << indent << "SaveInsertedTimestamps: " << (this->SaveInsertedTimestamps ? "Yes\n":"No\n");
  os << indent << "MaximumNumberOfInsertionsPerPhase: " << this->MaximumNumberOfInsertionsPerPhase << "\n";
  os << indent << "InsertedTimestampsBuffer: " << this->InsertedTimestampsBuffer << "\n";

  os << indent << "NeedsClear: " << (this->Optimization ? "Yes\n":"No\n");
  os << indent << "NumberOfThreads: " << this->NumberOfThreads << "\n";
  os << indent << "ReconstructionThreadId: " << this->ReconstructionThreadId << "\n";

  os << indent << "Reconstruction Rate: " << this->ReconstructionRate << "\n";
  os << indent << "Realtime Reconstruction: " << (this->RealTimeReconstruction ? "On\n":"Off\n");
  os << indent << "Reconstruction Frame Count: " << this->ReconstructionFrameCount << "\n";
}


//****************************************************************************
// BASICS FOR 3D RECONSTRUCTION
//****************************************************************************

//----------------------------------------------------------------------------
// SetSlice
// Set the image slice to insert into the reconstruction volume
// If there is a video source, then set the slice to be a slice from the video
// source.  Otherwise, set the slice to the parameter.
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetSlice(vtkImageData *slice)
{  
	if(this->VideoSource)
	  {
	  this->Slice = this->VideoSource->GetOutput();
	  }
	else if(slice)
	  {
	  this->Slice = slice;
	  }
}

//----------------------------------------------------------------------------
// GetSlice
// If there is a video source, return the slice from the video source.
// Otherwise, return the slice attribute.
//----------------------------------------------------------------------------
vtkImageData* vtkFreehandUltrasound2::GetSlice()
{
	if(this->VideoSource)
	  {
	  return this->VideoSource->GetOutput(); 
	  }
	else
	  {
	  return this->Slice;
	  }
}

//----------------------------------------------------------------------------
// SetVideoSource
// Set the video source to input the slices from to the parameter
//----------------------------------------------------------------------------
vtkCxxSetObjectMacro(vtkFreehandUltrasound2,VideoSource,vtkVideoSource2);

//----------------------------------------------------------------------------
// SetTrackerTool
// Set the tracker tool to input transforms from to the parameter
//----------------------------------------------------------------------------
vtkCxxSetObjectMacro(vtkFreehandUltrasound2,TrackerTool,vtkTrackerTool);

//----------------------------------------------------------------------------
// GetOutput
// Get the algorithm output as a vtkImageData
// If triggering, returns the zeroth phase image
//----------------------------------------------------------------------------
vtkImageData *vtkFreehandUltrasound2::GetOutput()
{
  if (this->Triggering)
    {
    vtkErrorMacro(<< "Should not use GetOutput() when triggering - use GetOutput(phase) instead");
    return NULL;
    }

  if(this->GetOutputDataObject(0))
    {
    return vtkImageData::SafeDownCast(this->GetOutputDataObject(0));
    }
  else
    {
    return NULL;
    }
}

//----------------------------------------------------------------------------
// GetOutput(int)
// Get the algorithm output as a vtkImageData, for use with triggering
//----------------------------------------------------------------------------
vtkImageData *vtkFreehandUltrasound2::GetOutput(int port)
{
  if(this->GetOutputDataObject(port))
    {
    return vtkImageData::SafeDownCast(this->GetOutputDataObject(port));
    }
  else
    {
    return NULL;
    }
}

//----------------------------------------------------------------------------
// GetAccumulationBuffer
// Get the accumulation buffer
// If triggering, returns the zeroth accumulation buffer
//----------------------------------------------------------------------------
vtkImageData *vtkFreehandUltrasound2::GetAccumulationBuffer()
  {
  return this->GetAccumulationBuffer(0);
  }

//----------------------------------------------------------------------------
// GetAccumulationBuffer(int)
// Get the accumulation buffer, for use with triggering
//----------------------------------------------------------------------------
vtkImageData *vtkFreehandUltrasound2::GetAccumulationBuffer(int port)
  {
  if (port < this->GetNumberOfOutputVolumes())
    {
    return this->AccumulationBuffers[port];
    }
  else
    {
    return NULL;
    }
  }

//----------------------------------------------------------------------------
// ClearOutput
// Setup to clear the data volume, then call InternalClearOutput
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::ClearOutput()
{
  this->NeedsClear = 1;

  // if we are not currently reconstructing...
  if (this->ReconstructionThreadId == -1)
    {
    for (int phase = 0; phase < this->GetNumberOfOutputPorts(); phase++)
      {
		  this->GetOutput(phase)->UpdateInformation();
      this->InternalInternalClearOutput(phase);
      }
    // clear the buffers for discarding based on ECG and for keeping slice timestamps
    this->ClearSliceBuffers();
    }
  this->Modified();
  }

//----------------------------------------------------------------------------
// InternalClearOutput
// Actually clear the data volume, for  all of the ports
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::InternalClearOutput()
  {
  for (int phase = 0; phase < this->GetNumberOfOutputPorts(); phase++)
    {
    this->InternalInternalClearOutput(phase);
    }
  // clear the buffers for discarding based on ECG
  this->ClearSliceBuffers();
  }

//----------------------------------------------------------------------------
// InternalInternalClearOutput
// Actually clear the data volume, for one of the ports
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::InternalInternalClearOutput(int phase)
{
  this->NeedsClear = 0;

  // Set everything in the output (within the output extent) to 0
  int *outExtent = this->OutputExtent;
  this->GetOutput(phase)->SetExtent(outExtent);
	this->GetOutput(phase)->AllocateScalars();
  void *outPtr = this->GetOutput(phase)->GetScalarPointerForExtent(outExtent);
	memset(outPtr,0,((outExtent[1]-outExtent[0]+1)*
	   (outExtent[3]-outExtent[2]+1)*
	   (outExtent[5]-outExtent[4]+1)*
	   this->GetOutput(phase)->GetScalarSize()*this->GetOutput(phase)->GetNumberOfScalarComponents()));

  // clear the accumulation buffer too if we are compounding
	if (this->Compounding)
    {
      this->AccumulationBuffers[phase]->SetExtent(outExtent);
      this->AccumulationBuffers[phase]->AllocateScalars();
      void *accPtr = this->AccumulationBuffers[phase]->GetScalarPointerForExtent(outExtent);
      memset(accPtr,0,((outExtent[1]-outExtent[0]+1)*
	      (outExtent[3]-outExtent[2]+1)*
	      (outExtent[5]-outExtent[4]+1)*
	    this->AccumulationBuffers[phase]->GetScalarSize()*this->AccumulationBuffers[phase]->GetNumberOfScalarComponents()));
    }

  if (this->LastIndexMatrix)
    {
    this->LastIndexMatrix->Delete();
    this->LastIndexMatrix = NULL;
    }

  this->SetPixelCount(0,0);
  this->SetPixelCount(1,0);
  this->SetPixelCount(2,0);
  this->SetPixelCount(3,0);

}

//----------------------------------------------------------------------------
// ClearSliceBuffers
// Clears buffers for discarding based on ECG, and for keeping slice timestamps
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::ClearSliceBuffers()
  {
  // TODO assumes the video source will not be altered
  int *inExtent = this->GetSlice()->GetExtent();
  for (int phase = 0; phase < this->NumberOfOutputVolumes; phase++)
    {
    if (this->SliceBuffer)
      {
      if (this->SliceBuffer[phase])
        {
        this->SliceBuffer[phase]->SetExtent(inExtent);
        this->SliceBuffer[phase]->AllocateScalars();
        void *slicePtr = this->SliceBuffer[phase]->GetScalarPointerForExtent(inExtent);
        memset(slicePtr,0,((inExtent[1]-inExtent[0]+1)*
          (inExtent[3]-inExtent[2]+1)*
          (inExtent[5]-inExtent[4]+1)*
	      this->SliceBuffer[phase]->GetScalarSize()*this->SliceBuffer[phase]->GetNumberOfScalarComponents()));
        }
      }
    
    if (this->SliceAxesBuffer)
      {
      if (this->SliceAxesBuffer[phase])
        {
        this->SliceAxesBuffer[phase]->Identity();
        }
      }

    if (this->SliceTransformBuffer)
      {
      if (this->SliceTransformBuffer[phase])
        {
        vtkTransform::SafeDownCast(this->SliceTransformBuffer[phase])->Identity();
        }
      }

    if (this->InsertedTimestampsBuffer)
      {
      if (this->InsertedTimestampsBuffer[phase])
        {
        for (int i = 0; i < this->MaximumNumberOfInsertionsPerPhase; i++)
          {
          this->InsertedTimestampsBuffer[phase][i] = 0;
          }
        }
      }

    if (this->InsertedTimestampsCounter)
      {
      this->InsertedTimestampsCounter[phase] = 0;
      }
    }
  }



//****************************************************************************
// SET/GET IMAGING PARAMETERS
//****************************************************************************

//----------------------------------------------------------------------------
// GetClipExtent
// convert the ClipRectangle (which is in millimetre coordinates) into a
// clip extent that can be applied to the input data - number of pixels (+ or -)
// from the origin (the z component is copied from the inExt parameter)
// 
// clipExt = {x0, x1, y0, y1, z0, z1} <-- the "output" of this function is to
//                                        change this array
// inOrigin = {x, y, z} <-- the origin in mm
// inSpacing = {x, y, z} <-- the spacing in mm
// inExt = {x0, x1, y0, y1, z0, z1} <-- min/max possible extent, in pixels
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::GetClipExtent(int clipExt[6],
				     vtkFloatingPointType inOrigin[3],
				     vtkFloatingPointType inSpacing[3],
				     const int inExt[6])
{
  // Map the clip rectangle (millimetres) to pixels --> number of pixels (+ or -)
  // from the origin
  int x0 = (int)ceil((this->GetClipRectangle()[0]-inOrigin[0])/inSpacing[0]);
  int x1 = (int)floor((this->GetClipRectangle()[2]-inOrigin[0])/inSpacing[0]);
  int y0 = (int)ceil((this->GetClipRectangle()[1]-inOrigin[1])/inSpacing[1]);
  int y1 = (int)floor((this->GetClipRectangle()[3]-inOrigin[1])/inSpacing[1]);

  // Make sure that x0 <= x1 and y0 <= y1, otherwise swap 
  if (x0 > x1)
    {
    int tmp = x0; x0 = x1; x1 = tmp;
    }
  if (y0 > y1)
    {
    int tmp = y0; y0 = y1; y1 = tmp;
    }

  // make sure this lies within extent
  if (x0 < inExt[0])
    {
    x0 = inExt[0];
    }
  if (x1 > inExt[1])
    {
    x1 = inExt[1];
    }
  if (x0 > x1)
    {
    x0 = inExt[0];
    x1 = inExt[0]-1;
    }

  if (y0 < inExt[2])
    {
    y0 = inExt[2];
    }
  if (y1 > inExt[3])
    {
    y1 = inExt[3];
    }
  if (y0 > y1)
    {
    y0 = inExt[2];
    y1 = inExt[2]-1;
    }

  // Set the clip extent
  clipExt[0] = x0;
  clipExt[1] = x1;
  clipExt[2] = y0;
  clipExt[3] = y1;
  clipExt[4] = inExt[4];
  clipExt[5] = inExt[5];
}


//****************************************************************************
// SET/GET PARAMETERS FOR ROTATING PROBES
//****************************************************************************

//----------------------------------------------------------------------------
// SetRotatingProbe
// Set whether we want to use a rotating probe
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetRotatingProbe(int probe)
  {
  this->RotatingProbe = probe;
  
  if (probe)
    {
    // Setup the rotation objects
    this->RotationClipper = vtkImageClip::New();
		this->RotationClipper->ClipDataOn();
    this->RotationThresholder = vtkImageThreshold::New();
		this->RotationThresholder->ThresholdBetween(this->FanRotationImageThreshold1, this->FanRotationImageThreshold2);
		this->RotationThresholder->SetOutValue(1);
		this->RotationThresholder->SetInValue(0);
    }
  else
    {
    this->PreviousFanRotation = this->FanRotation;
    this->FanRotation = 0;
    }
  }

//----------------------------------------------------------------------------
// SetFanRotation
// Change the current fan rotation
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetFanRotation(int rot)
  {
  if (this->RotatingProbe)
    {
    this->FanRotation = rot;
    }
  }

//----------------------------------------------------------------------------
// SetFanRotationImageThreshold1
// Sets the lower bound for the thresholder used to calculate rotation
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetFanRotationImageThreshold1(int thresh)
  {
  this->FanRotationImageThreshold1 = thresh;
  if (this->RotationThresholder)
    {
    this->RotationThresholder->ThresholdBetween(this->FanRotationImageThreshold1, this->FanRotationImageThreshold2);
    this->RotationThresholder->SetOutValue(1);
    this->RotationThresholder->SetInValue(0);
    }
  }

//----------------------------------------------------------------------------
// SetFanRotationImageThreshold2
// Sets the upper bound for the thresholder used to calculate rotation
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetFanRotationImageThreshold2(int thresh)
  {
  this->FanRotationImageThreshold2 = thresh;
  if (this->RotationThresholder)
    {
    this->RotationThresholder->ThresholdBetween(this->FanRotationImageThreshold1, this->FanRotationImageThreshold2);
    this->RotationThresholder->SetOutValue(1);
    this->RotationThresholder->SetInValue(0);
    }
  }

//----------------------------------------------------------------------------
// SetPreviousFanRotation
// Change the previous fan rotation
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetPreviousFanRotation(int rot)
  {
  if (this->RotatingProbe)
    {
    this->PreviousFanRotation = rot;
    }
  }

//****************************************************************************
// SET/GET OPTIMIZATION AND COMPOUNDING 0PTIONS
//****************************************************************************

//----------------------------------------------------------------------------
// SetCompounding
// Sets whether or not we are compounding
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetCompounding(int comp)
{
  // TODO put back
	// we are switching from no compounding to compounding
	/*if (this->GetCompounding() == 0 &&  comp == 1)
	  {
		this->AccumulationBuffer->SetScalarType(VTK_UNSIGNED_SHORT);
		this->AccumulationBuffer->SetUpdateExtent(this->OutputExtent);
		this->AccumulationBuffer->SetWholeExtent(this->OutputExtent);
		this->AccumulationBuffer->SetExtent(this->OutputExtent);
		this->AccumulationBuffer->SetSpacing(this->OutputSpacing);
		this->AccumulationBuffer->SetOrigin(this->OutputOrigin);
		//this->AccumulationBuffer->AllocateScalars();
		//this->AccumulationBuffer->Update();
	  }
  */ 

	this->Compounding = comp;
}


//****************************************************************************
// VTK 5 PIPELINE
//****************************************************************************

//----------------------------------------------------------------------------
// GetMTime
// Account for the MTime of the transform and its matrix when determining
// the MTime of the filter - the MTime of the transform is the largest mTime of
// the superclass, the slice transform, and the slice transform's matrix
// TODO [David's note to self: this made sense in vtkImageReslice, but does it make
//  any sense here?]
//----------------------------------------------------------------------------
unsigned long int vtkFreehandUltrasound2::GetMTime()
{
  unsigned long mTime=this->Superclass::GetMTime();
  unsigned long time;

  if ( this->SliceTransform != NULL )
    {
    time = this->SliceTransform->GetMTime();
    mTime = ( time > mTime ? time : mTime );
    time = this->SliceTransform->GetMatrix()->GetMTime();
    mTime = ( time > mTime ? time : mTime );    
    }

  return mTime;
}

//----------------------------------------------------------------------------
// FillInputPortInformation
// Define the input port information - the input at port 0 needs vtkImageData
// as input
//----------------------------------------------------------------------------
int  vtkFreehandUltrasound2::FillInputPortInformation(
  int port, vtkInformation* info)
{
  if (port == 0)
    {
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    }
  return 0;
}

//----------------------------------------------------------------------------
// FillOutputPortInformation
// Define the output port information - all output ports produce vtkImageData
//---------------------------------------------------------------------------
int vtkFreehandUltrasound2::FillOutputPortInformation(
  int vtkNotUsed(port), vtkInformation* info)
{
  info->Set(vtkDataObject::DATA_TYPE_NAME(), "vtkImageData");
 
  return 1;
}

//----------------------------------------------------------------------------
// ProcessRequest
// The main method of an algorithm - called whenever there is a request
// We make sure that the REQUEST_DATA_NOT_GENERATED is set so that the data
// object is not initialized everytime an update is called
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::ProcessRequest(vtkInformation* request,
                              vtkInformationVector** inputVector,
                              vtkInformationVector* outputVector)
{
  if(request->Has(vtkDemandDrivenPipeline::REQUEST_DATA_NOT_GENERATED()))
    {
    // Mark all outputs as not generated so that the executive does
    // not try to handle initialization/finalization of the outputs.
    // We will do it here.
    int phase;
    for (phase=0; phase < outputVector->GetNumberOfInformationObjects(); ++phase)
      {
      vtkInformation* outInfo = outputVector->GetInformationObject(phase);
      outInfo->Set(vtkDemandDrivenPipeline::DATA_NOT_GENERATED(), 1);
      }
    }

  // Calls to RequestInformation, RequestUpdateExtent and RequestData are
  // handled here, in vtkImageAlgorithm's ProcessRequest
  return this->Superclass::ProcessRequest(request, inputVector, outputVector);
}

//----------------------------------------------------------------------------
// RequestInformation
// Asks the algorithm to provide as much information as it can about what the
// output data will look like once the algorithm has generated it.
// For both the first vtkInformation object and the data object associate with
// it, sets the whole extent, spacing and origin to match those of this
// object, and the scalar type and number of scalar components to match those
// of the slice.  Also updates the "old" attributes and NeedsClear if an
// parameter has changed.
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::RequestInformation(
	vtkInformation* vtkNotUsed(request),
	vtkInformationVector** vtkNotUsed(inInfo),
	vtkInformationVector* outInfoVector)
{
	// to avoid conflict between the main application thread and the
	// realtime reconstruction thread
	if (this->ReconstructionThreadId == -1)
	  {

    for (int phase = 0; phase < this->GetNumberOfOutputPorts(); phase++)
      {

		  vtkInformation *outInfo = outInfoVector->GetInformationObject(phase);
		  // would have been created by a call to REQUEST_DATA_OBJECT, presumably handled
		  // by vtkImageAlgorithm
		  vtkImageData *output = 
			  dynamic_cast<vtkImageData *>(outInfo->Get(vtkDataObject::DATA_OBJECT()));

		  // the whole extent, spacing, origin, PIPELINE scalar type (ex double; until REQUEST_DATA
		  // is called, the actual scalar type may be different) and number of scalar components of
		  // the object created by the REQUEST_DATA_OBJECT call
		  int oldwholeextent[6];
		  vtkFloatingPointType oldspacing[3];
		  vtkFloatingPointType oldorigin[3];
		  int oldtype = output->GetScalarType();
		  int oldncomponents = output->GetNumberOfScalarComponents();
		  output->GetWholeExtent(oldwholeextent);
		  output->GetSpacing(oldspacing);
		  output->GetOrigin(oldorigin);

		  // if we don't have a slice yet, then set the slice to be the output of the video source
		  if (this->GetVideoSource())
		    {
			  if (this->GetSlice() == 0)
			    {
				  this->SetSlice(this->GetVideoSource()->GetOutput());
			    } 
		    } 

		  // if we have a slice now...
		  if (this->GetSlice())
		    {
			  // get the newest slice information - updating origin and spacing and extent from pipeline
			  this->GetSlice()->UpdateInformation();

			  // for both the outInfo vtkInformation object and the data object associate with outInfo,
			  // set the whole extent, spacing and origin to match those of this object, and the scalar
			  // type and number of scalar components to match those of the slice
			  vtkDataObject::SetPointDataActiveScalarInfo(outInfo,
				  this->GetSlice()->GetScalarType(),
				  this->GetSlice()->GetNumberOfScalarComponents()+1);

			  output->SetScalarType(this->GetSlice()->GetScalarType());
			  output->SetNumberOfScalarComponents(this->GetSlice()->
				  GetNumberOfScalarComponents()+1);
			  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), this->OutputExtent, 6);
			  outInfo->Set(vtkDataObject::SPACING(), this->OutputSpacing, 3);
			  outInfo->Set(vtkDataObject::ORIGIN(), this->OutputOrigin, 3);
			  output->SetExtent(this->OutputExtent);
			  output->SetWholeExtent(this->OutputExtent);
			  output->SetSpacing(this->OutputSpacing);
			  output->SetOrigin(this->OutputOrigin);

			  // if the output has changed, then we need to clear
			  if (oldtype != output->GetScalarType() ||
				  oldncomponents != output->GetNumberOfScalarComponents() ||
				  oldwholeextent[0] != this->OutputExtent[0] ||
				  oldwholeextent[1] != this->OutputExtent[1] ||
				  oldwholeextent[2] != this->OutputExtent[2] ||
				  oldwholeextent[3] != this->OutputExtent[3] ||
				  oldwholeextent[4] != this->OutputExtent[4] ||
				  oldwholeextent[5] != this->OutputExtent[5] ||
				  oldspacing[0] != this->OutputSpacing[0] ||
				  oldspacing[1] != this->OutputSpacing[1] ||
				  oldspacing[2] != this->OutputSpacing[2] ||
				  oldorigin[0] != this->OutputOrigin[0] ||
				  oldorigin[1] != this->OutputOrigin[1] ||
				  oldorigin[2] != this->OutputOrigin[2])
			    {
				  this->NeedsClear = 1;
			    }

			  // if we are compounding, then adjust the accumulation buffer
			  if (this->Compounding)
			    {
			    int *extent = this->AccumulationBuffers[phase]->GetExtent();
          vtkImageData* accBuffer = this->AccumulationBuffers[phase];
			    accBuffer->SetWholeExtent(this->OutputExtent);
			    accBuffer->SetExtent(this->OutputExtent);
			    accBuffer->SetSpacing(this->OutputSpacing);
			    accBuffer->SetOrigin(this->OutputOrigin);
			    accBuffer->SetScalarType(this->GetOutput(phase)->GetScalarType());
			    accBuffer->SetUpdateExtent(this->OutputExtent);
			    accBuffer->Update();

			    // if the accumulation buffer has changed, we need to clear
			    if (extent[0] != this->OutputExtent[0] ||
				    extent[1] != this->OutputExtent[1] ||
				    extent[2] != this->OutputExtent[2] ||
				    extent[3] != this->OutputExtent[3] ||
				    extent[4] != this->OutputExtent[4] ||
				    extent[5] != this->OutputExtent[5])
			      {
				    this->NeedsClear = 1;
			      }
            
          }
		    }
	    }
    }
	return 1;
}

//----------------------------------------------------------------------------
// InternalExecuteInformation
// Gets the output ready to receive data, so we need to call it before the
// reconstruction starts.  Must update the information for the output and for
// the accumulation buffer.
// Looks similar to RequestInformation, but operates on this->GetOutput
// instead of on the output objects associated with information objects
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::InternalExecuteInformation() 
{
  for (int phase = 0; phase < this->GetNumberOfOutputPorts(); phase++)
    {

    vtkImageData *output = this->GetOutput(phase);
    vtkInformation *outInfo = output->GetPipelineInformation();
    int oldwholeextent[6];
    vtkFloatingPointType oldspacing[3];
    vtkFloatingPointType oldorigin[3];
    int oldtype = output->GetScalarType();
    int oldncomponents = output->GetNumberOfScalarComponents();
    output->GetWholeExtent(oldwholeextent);
    output->GetSpacing(oldspacing);
    output->GetOrigin(oldorigin);

    // if we don't have a slice yet, then set the slice to be the output of the video source
    if (this->GetVideoSource())
      {
      if (this->GetSlice() == 0)
        {
        this->SetSlice( this->GetVideoSource()->GetOutput());
        }
      }

    if (this->GetSlice())
      {
      this->GetSlice()->UpdateInformation();
      }    
   
    // set up the output dimensions and info here
    outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
	         this->OutputExtent, 6);
    outInfo->Set(vtkDataObject::SPACING(),
	         this->OutputSpacing, 3);
    outInfo->Set(vtkDataObject::ORIGIN(),
	         this->OutputOrigin, 3);
    output->SetScalarType(this->GetSlice()->GetScalarType());
	  output->SetNumberOfScalarComponents(this->GetSlice()->GetNumberOfScalarComponents()+1);
    output->SetExtent(this->OutputExtent);
    output->SetWholeExtent(this->OutputExtent);
    output->SetSpacing(this->OutputSpacing);
    output->SetOrigin(this->OutputOrigin);

    // check to see if output has changed
    if (oldtype != output->GetScalarType() ||
      oldncomponents != output->GetNumberOfScalarComponents() ||
      oldwholeextent[0] != this->OutputExtent[0] ||
      oldwholeextent[1] != this->OutputExtent[1] ||
      oldwholeextent[2] != this->OutputExtent[2] ||
      oldwholeextent[3] != this->OutputExtent[3] ||
      oldwholeextent[4] != this->OutputExtent[4] ||
      oldwholeextent[5] != this->OutputExtent[5] ||
      oldspacing[0] != this->OutputSpacing[0] ||
      oldspacing[1] != this->OutputSpacing[1] ||
      oldspacing[2] != this->OutputSpacing[2] ||
      oldorigin[0] != this->OutputOrigin[0] ||
      oldorigin[1] != this->OutputOrigin[1] ||
      oldorigin[2] != this->OutputOrigin[2])
      {
	    this->NeedsClear = 1;
      }

    // set up the accumulation buffer to be the same size as the
    // output
    if (this->Compounding)
      {
      int *extent = this->AccumulationBuffers[phase]->GetExtent();
      vtkImageData* accBuffer = this->AccumulationBuffers[phase];
      accBuffer->SetWholeExtent(this->OutputExtent);
      accBuffer->SetExtent(this->OutputExtent);
      accBuffer->SetSpacing(this->OutputSpacing);
      accBuffer->SetOrigin(this->OutputOrigin);
      accBuffer->SetScalarType(this->GetOutput(phase)->GetScalarType());
      accBuffer->SetUpdateExtent(this->OutputExtent);
      accBuffer->Update();

      if (extent[0] != this->OutputExtent[0] ||
        extent[1] != this->OutputExtent[1] ||
        extent[2] != this->OutputExtent[2] ||
        extent[3] != this->OutputExtent[3] ||
        extent[4] != this->OutputExtent[4] ||
        extent[5] != this->OutputExtent[5])
        {
        this->NeedsClear = 1;
        }
      }
    }
}

//----------------------------------------------------------------------------
// RequestUpdateExtent
// Asks for the input update extent necessary to produce a given output
// update extent.  Sets the update extent of the input information object
// to equal the whole extent of hte input information object - we need the
// entire whole extent of the input data object to generate the output
//----------------------------------------------------------------------------
int  vtkFreehandUltrasound2::RequestUpdateExtent(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *vtkNotUsed(outputVector))
{

  // TODO This dies for some reason, so take it out...
  // Set the update extent of the input information object to equal the
  // whole extent of the input information object
  /*int inExt[6];
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0); 
  inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), inExt); // get the whole extent of inInfo
  inInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), inExt, 6); // se the update extent of inInfo
  */

  return 1;
}

//----------------------------------------------------------------------------
// RequestData
// Asks for the output data object to be populated with the actual output data
// This doesn't really do much, because this is an unconventional image filter
// In most VTK classes this method is responsible for calling Execute, but since
// the output data has already been generated it just fools the pipeline into
// thinking that Execute has been called
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::RequestData(vtkInformation* request,
				       vtkInformationVector **vtkNotUsed(inInfo),
				       vtkInformationVector* outInfo)
{
  for (int phase = 0; phase < this->GetNumberOfOutputPorts(); phase++)
    {

    vtkDataObject *outObject = 
      outInfo->GetInformationObject(phase)->Get(vtkDataObject::DATA_OBJECT());

    // if we are not currently running a reconstruction and we need to clear, then
    // clear
    if (this->ReconstructionThreadId == -1 && this->NeedsClear == 1)
      {
      this->InternalClearOutput();
      }
    
    // This would have been done already in the call to ProcessRequest, so don't do it here
    outInfo->GetInformationObject(phase)->Set(vtkDemandDrivenPipeline::DATA_NOT_GENERATED(), 1);
    
    // Set the flag for the data object associated with port 0 that data has been generated -
    // sets the data released flag to 0 and sets a new update time
    ((vtkImageData *)outObject)->DataHasBeenGenerated();
    }

  return 1;
}

//----------------------------------------------------------------------------
// ComputePipelineMTime
// Compute the modified time for the pipeline - just returns the mtime of the
// input slice
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::ComputePipelineMTime(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **vtkNotUsed(inInfoVec),
  vtkInformationVector *vtkNotUsed(outInfoVec),
  int requestFromOutputPort,
  unsigned long* mtime)
{
  if (this->GetSlice())
    {
    *mtime = this->GetSlice()->GetPipelineMTime(); 
    }
  return 1;
}


//****************************************************************************
// ROUNDING CODE
//****************************************************************************

//----------------------------------------------------------------------------
// rounding functions, split and optimized for each type
// (because we don't want to round if the result is a float!)

// in the case of a tie between integers, the larger integer wins.

// The 'floor' function on x86 and mips is many times slower than these
// and is used a lot in this code, optimize for different CPU architectures
// static inline int vtkUltraFloor(double x)
// {
// #if defined mips || defined sparc
//   return (int)((unsigned int)(x + 2147483648.0) - 2147483648U);
// #elif defined i386
//   double tempval = (x - 0.25) + 3377699720527872.0; // (2**51)*1.5
//   return ((int*)&tempval)[0] >> 1;
// #else
//   return int(floor(x));
// #endif
// }

static inline int vtkUltraFloor(double x)
{
#if defined mips || defined sparc || defined __ppc__
  x += 2147483648.0;
  unsigned int i = (unsigned int)(x);
  return (int)(i - 2147483648U);
#elif defined i386 || defined _M_IX86
  union { double d; unsigned short s[4]; unsigned int i[2]; } dual;
  dual.d = x + 103079215104.0;  // (2**(52-16))*1.5
  return (int)((dual.i[1]<<16)|((dual.i[0])>>16));
#elif defined ia64 || defined __ia64__ || defined IA64
  x += 103079215104.0;
  long long i = (long long)(x);
  return (int)(i - 103079215104LL);
#else
  double y = floor(x);
  return (int)(y);
#endif
}

static inline int vtkUltraCeil(double x)
{
  return -vtkUltraFloor(-x - 1.0) - 1;
}

static inline int vtkUltraRound(double x)
{
  return vtkUltraFloor(x + 0.5);
}

static inline int vtkUltraFloor(float x)
{
  return vtkUltraFloor((double)x);
}

static inline int vtkUltraCeil(float x)
{
  return vtkUltraCeil((double)x);
}

static inline int vtkUltraRound(float x)
{
  return vtkUltraRound((double)x);
}

static inline int vtkUltraFloor(fixed x)
{
  return x.floor();
}

static inline int vtkUltraCeil(fixed x)
{
  return x.ceil();
}

static inline int vtkUltraRound(fixed x)
{
  return x.round();
}

// convert a float into an integer plus a fraction
template <class F>
static inline int vtkUltraFloor(F x, F &f)
{
  int ix = vtkUltraFloor(x);
  f = x - ix;
  return ix;
}

template <class F, class T>
static inline void vtkUltraRound(F val, T& rnd)
{
  rnd = vtkUltraRound(val);
}


//****************************************************************************
// SLEEP CODE
//***************************************************************************

//----------------------------------------------------------------------------
// vtkSleep
// platform-independent sleep function (duration in seconds)
//----------------------------------------------------------------------------
static inline void vtkSleep(double duration)
{
  duration = duration; // avoid warnings
  // sleep according to OS preference
#ifdef _WIN32
  Sleep(vtkUltraFloor(1000*duration));
#elif defined(__FreeBSD__) || defined(__linux__) || defined(sgi)
  struct timespec sleep_time, remaining_time;
  int seconds = vtkUltraFloor(duration);
  int nanoseconds = vtkUltraFloor(1000000000*(duration - seconds));
  sleep_time.tv_sec = seconds;
  sleep_time.tv_nsec = nanoseconds;
  nanosleep(&sleep_time, &remaining_time);
#endif
}

//----------------------------------------------------------------------------
// vtkThreadSleep
// Sleep until the specified absolute time has arrived.
// You must pass a handle to the current thread.  
// If '0' is returned, then the thread was aborted before or during the wait.
//----------------------------------------------------------------------------
static int vtkThreadSleep(struct ThreadInfoStruct *data, double time)
{
  for (;;)
    {
    // slice 10 millisecs off the time, since this is how long it will
    // take for this thread to start executing once it has been
    // re-scheduled
    // TODO vtkTimerLog::GetCurrentTime() is depreacated and replace with
    // GetUniversalTime in VTK 5:
    double remaining = time - vtkTimerLog::GetUniversalTime() - 0.01;

    // check to see if we have reached the specified time
    if (remaining <= 0)
      {
      return 1;
      }
    // check the ActiveFlag at least every 0.1 seconds
    if (remaining > 0.1)
      {
      remaining = 0.1;
      }

    // check to see if we are being told to quit 
    if (*(data->ActiveFlag) == 0)
      {
      return 0;
      }
    
    vtkSleep(remaining);
    }

  return 1;
}


//****************************************************************************
// INTERPOLATION CODE
//****************************************************************************

//----------------------------------------------------------------------------
// vtkGetUltraInterpFunc
// Sets interpolate (pointer to a function) to match the current interpolation
// mode - used for unoptimized versions only
//----------------------------------------------------------------------------
template <class F, class T>
static void vtkGetUltraInterpFunc(vtkFreehandUltrasound2 *self, 
                                    int (**interpolate)(F *point, 
                                                        T *inPtr, T *outPtr,
                                                        unsigned short *accPtr,
                                                        int numscalars, 
                                                        int outExt[6], 
                                                        int outInc[3]))
{
  switch (self->GetInterpolationMode())
    {
    case VTK_FREEHAND_NEAREST:
      *interpolate = &vtkNearestNeighborInterpolation;
      break;
    case VTK_FREEHAND_LINEAR:
      *interpolate = &vtkTrilinearInterpolation;
      break;
    }
}

///////////////// NEAREST NEIGHBOR INTERPOLATION ///////////////////////
// In the un-optimized version, each output voxel
// is converted into a set of look-up indices for the input data;
// then, the indices are checked to ensure they lie within the
// input data extent.

// In the optimized versions, the check is done in reverse:
// it is first determined which output voxels map to look-up indices
// within the input data extent.  Then, further calculations are
// done only for those voxels.  This means that 1) minimal work
// is done for voxels which map to regions outside fo the input
// extent (they are just set to the background color) and 2)
// the inner loops of the look-up and interpolation are
// tightened relative to the un-uptimized version.
////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// vtkNearestNeighborInterpolation - NOT OPTIMIZED
// Do nearest-neighbor interpolation of the input data 'inPtr' of extent 
// 'inExt' at the 'point'.  The result is placed at 'outPtr'.  
// If the lookup data is beyond the extent 'inExt', set 'outPtr' to
// the background color 'background'.  
// The number of scalar components in the data is 'numscalars'
//----------------------------------------------------------------------------
template <class F, class T>
static int vtkNearestNeighborInterpolation(F *point, T *inPtr, T *outPtr,
                                           unsigned short *accPtr, 
                                           int numscalars, 
                                           int outExt[6], int outInc[3])
{
  int i;
  // The nearest neighbor interpolation occurs here
  // The output point is the closest point to the input point - rounding
  // to get closest point
  int outIdX = vtkUltraRound(point[0])-outExt[0];
  int outIdY = vtkUltraRound(point[1])-outExt[2];
  int outIdZ = vtkUltraRound(point[2])-outExt[4];
  
  // fancy way of checking bounds
  if ((outIdX | (outExt[1]-outExt[0] - outIdX) |
       outIdY | (outExt[3]-outExt[2] - outIdY) |
       outIdZ | (outExt[5]-outExt[4] - outIdZ)) >= 0)
    {
    int inc = outIdX*outInc[0]+outIdY*outInc[1]+outIdZ*outInc[2];
    outPtr += inc;
    // accumulation buffer: do compounding
    if (accPtr)
      {
      accPtr += inc/outInc[0];
      int newa = *accPtr + 255;
      for (i = 0; i < numscalars; i++)
        {
        *outPtr = ((*inPtr++)*255 + (*outPtr)*(*accPtr))/newa;
        outPtr++;
        }
      *outPtr = 255;
      *accPtr = 65535;
      if (newa < 65535)
        {
        *accPtr = newa;
        }
      }
    // no accumulation buffer, replace what was there before
    else
      {
      for (i = 0; i < numscalars; i++)
        {
        *outPtr++ = *inPtr++;
        }
      *outPtr = 255;
      }
    return 1;
    }
  return 0;
} 

//----------------------------------------------------------------------------
// vtkFreehand2OptimizedNNHelper - OPTIMIZED, WITHOUT INTEGER MATHEMATICS
// Optimized nearest neighbor interpolation
//----------------------------------------------------------------------------
template<class T>
static inline void vtkFreehand2OptimizedNNHelper(int r1, int r2,
                                                double *outPoint,
                                                double *outPoint1,
												                        double *xAxis,
                                                T *&inPtr, T *outPtr,
                                                int *outExt, int *outInc,
                                                int numscalars, 
                                                unsigned short *accPtr)
{
  // with compounding
  if (accPtr)
    {

    for (int idX = r1; idX <= r2; idX++)
      {
      outPoint[0] = outPoint1[0] + idX*xAxis[0]; 
      outPoint[1] = outPoint1[1] + idX*xAxis[1];
      outPoint[2] = outPoint1[2] + idX*xAxis[2];

      int outIdX = vtkUltraRound(outPoint[0]) - outExt[0];
      int outIdY = vtkUltraRound(outPoint[1]) - outExt[2];
      int outIdZ = vtkUltraRound(outPoint[2]) - outExt[4];

      /* bounds checking turned off to improve performance
      if (outIdX < 0 || outIdX > outExt[1] - outExt[0] ||
          outIdY < 0 || outIdY > outExt[3] - outExt[2] ||
          outIdZ < 0 || outIdZ > outExt[5] - outExt[4])
        {
        cerr << "out of bounds!!!\n";
        inPtr += numscalars;
        return;
        }
      */

      int inc = outIdX*outInc[0] + outIdY*outInc[1] + outIdZ*outInc[2];
      T *outPtr1 = outPtr + inc;
      // divide by outInc[0] to accomodate for the difference
      // in the number of scalar pointers between the output
			// and the accumulation buffer
      unsigned short *accPtr1 = accPtr + ((unsigned short)(inc/outInc[0]));
	    unsigned short newa = *accPtr1 + ((unsigned short)(255)); 
	    int i = numscalars;
      do 
        {
        i--;
        *outPtr1 = ((*inPtr++)*255 + (*outPtr1)*(*accPtr1))/newa;
		    outPtr1++;
        }
      while (i);

      *outPtr1 = 255;
      *accPtr1 = 65535;
      if (newa < 65535)
        {
        *accPtr1 = newa;
        }
      }
    }

  // not compounding
  else
    {
    for (int idX = r1; idX <= r2; idX++)
      {
      outPoint[0] = outPoint1[0] + idX*xAxis[0]; 
      outPoint[1] = outPoint1[1] + idX*xAxis[1];
      outPoint[2] = outPoint1[2] + idX*xAxis[2];

      int outIdX = vtkUltraRound(outPoint[0]) - outExt[0];
      int outIdY = vtkUltraRound(outPoint[1]) - outExt[2];
      int outIdZ = vtkUltraRound(outPoint[2]) - outExt[4];

      /* bounds checking turned off to improve performance
      if (outIdX < 0 || outIdX > outExt[1] - outExt[0] ||
          outIdY < 0 || outIdY > outExt[3] - outExt[2] ||
          outIdZ < 0 || outIdZ > outExt[5] - outExt[4])
        {
        cerr << "out of bounds!!!\n";
        inPtr += numscalars;
        return;
        }
      */

      int inc = outIdX*outInc[0] + outIdY*outInc[1] + outIdZ*outInc[2];
      T *outPtr1 = outPtr + inc;
      int i = numscalars;
      do
        {
        i--;
		    // copy the input pointer value into the output pointer (this is where the intensities get copied)
		    *outPtr1++ = *inPtr++;
		    }
      while (i);
      *outPtr1 = 255;
      }
    } 
}

//----------------------------------------------------------------------------
// vtkFreehand2OptimizedNNHelper - OPTIMIZED, WITH INTEGER MATHEMATICS
// Optimized nearest neighbor interpolation, specifically optimized for fixed
// point (i.e. integer) mathematics
// Same as above, but with fixed type
//----------------------------------------------------------------------------
template <class T>
static inline void vtkFreehand2OptimizedNNHelper(int r1, int r2,
                                                fixed *outPoint,
                                                fixed *outPoint1, fixed *xAxis,
                                                T *&inPtr, T *outPtr,
                                                int *outExt, int *outInc,
                                                int numscalars, 
                                                unsigned short *accPtr)
{
  outPoint[0] = outPoint1[0] + r1*xAxis[0] - outExt[0];
  outPoint[1] = outPoint1[1] + r1*xAxis[1] - outExt[2];
  outPoint[2] = outPoint1[2] + r1*xAxis[2] - outExt[4];

  // Nearest-Neighbor, no extent checks, with accumulation
  if (accPtr)
    {
    for (int idX = r1; idX <= r2; idX++)
      {
      int outIdX = vtkUltraRound(outPoint[0]);
      int outIdY = vtkUltraRound(outPoint[1]);
      int outIdZ = vtkUltraRound(outPoint[2]);

      /* bounds checking turned off to improve performance
      if (outIdX < 0 || outIdX > outExt[1] - outExt[0] ||
          outIdY < 0 || outIdY > outExt[3] - outExt[2] ||
          outIdZ < 0 || outIdZ > outExt[5] - outExt[4])
        {
        cerr << "out of bounds!!!\n";
        inPtr += numscalars;
        return;
        }
      */

      int inc = outIdX*outInc[0] + outIdY*outInc[1] + outIdZ*outInc[2];
      T *outPtr1 = outPtr + inc;
      // divide by outInc[0] to accomodate for the difference
      // in the number of scalar pointers between the output
			// and the accumulation buffer
      unsigned short *accPtr1 = accPtr + ((unsigned short)(inc/outInc[0]));
      //TODO dies here
      unsigned short newa = *accPtr1 + ((unsigned short)(255));
      int i = numscalars;
      do 
        {
        i--;
        *outPtr1 = ((*inPtr++)*255 + (*outPtr1)*(*accPtr1))/newa;
        outPtr1++;
        }
      while (i);

      *outPtr1 = 255;
      *accPtr1 = 65535;
      if (newa < 65535)
        {
        *accPtr1 = newa;
        }

      outPoint[0] += xAxis[0];
      outPoint[1] += xAxis[1];
      outPoint[2] += xAxis[2];
      }
    }

  // Nearest-Neighbor, no extent checks, no accumulation
  else
    {
    for (int idX = r1; idX <= r2; idX++)
      {
      int outIdX = vtkUltraRound(outPoint[0]);
      int outIdY = vtkUltraRound(outPoint[1]);
      int outIdZ = vtkUltraRound(outPoint[2]);

      /* bounds checking turned off to improve performance
      if (outIdX < 0 || outIdX > outExt[1] - outExt[0] ||
          outIdY < 0 || outIdY > outExt[3] - outExt[2] ||
          outIdZ < 0 || outIdZ > outExt[5] - outExt[4])
        {
        cerr << "out of bounds!!!\n";
        inPtr += numscalars;
        return;
        }
      */

      int inc = outIdX*outInc[0] + outIdY*outInc[1] + outIdZ*outInc[2];
      T *outPtr1 = outPtr + inc;
      int i = numscalars;
      do
        {
        i--;
        *outPtr1++ = *inPtr++;
        }
      while (i);
      *outPtr1 = 255;

      outPoint[0] += xAxis[0];
      outPoint[1] += xAxis[1];
      outPoint[2] += xAxis[2];
      }
    } 
}

////////////////////// TRILINEAR INTERPOLATION //////////////////////////
// does reverse trilinear interpolation
// trilinear interpolation would use the pixel values to interpolate something
// in the middle we have the something in the middle and want to spread it to
// the discrete pixel values around it, in an interpolated way
/////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
// vtkTrilinearInterpolation
// Do trilinear interpolation of the input data 'inPtr' of extent 'inExt'
// at the 'point'.  The result is placed at 'outPtr'.  
// If the lookup data is beyond the extent 'inExt', set 'outPtr' to
// the background color 'background'.  
// The number of scalar components in the data is 'numscalars'
//----------------------------------------------------------------------------
template <class F, class T>
static int vtkTrilinearInterpolation(F *point, T *inPtr, T *outPtr,
                                     unsigned short *accPtr, int numscalars, 
                                     int outExt[6], int outInc[3])
{
  F fx, fy, fz;

  // convert point[0] into integer component and a fraction
  int outIdX0 = vtkUltraFloor(point[0], fx);
  // point[0] is unchanged, outIdX0 is the integer (floor), fx is the float
  int outIdY0 = vtkUltraFloor(point[1], fy);
  int outIdZ0 = vtkUltraFloor(point[2], fz);

  int outIdX1 = outIdX0 + (fx != 0); // ceiling
  int outIdY1 = outIdY0 + (fy != 0);
  int outIdZ1 = outIdZ0 + (fz != 0);

  // at this point in time we have the floor (outIdX0), the ceiling (outIdX1)
  // and the fractional component (fx) for x, y and z
  
  // bounds check
  if ((outIdX0 | (outExt[1]-outExt[0] - outIdX1) |
       outIdY0 | (outExt[3]-outExt[2] - outIdY1) |
       outIdZ0 | (outExt[5]-outExt[4] - outIdZ1)) >= 0)
    {
    // do reverse trilinear interpolation
    int factX0 = outIdX0*outInc[0];
    int factY0 = outIdY0*outInc[1];
    int factZ0 = outIdZ0*outInc[2];
    int factX1 = outIdX1*outInc[0];
    int factY1 = outIdY1*outInc[1];
    int factZ1 = outIdZ1*outInc[2];

    int factY0Z0 = factY0 + factZ0;
    int factY0Z1 = factY0 + factZ1;
    int factY1Z0 = factY1 + factZ0;
    int factY1Z1 = factY1 + factZ1;

    // increment between the output pointer and the 8 pixels to work on
    int idx[8];
    idx[0] = factX0 + factY0Z0;
    idx[1] = factX0 + factY0Z1;
    idx[2] = factX0 + factY1Z0;
    idx[3] = factX0 + factY1Z1;
    idx[4] = factX1 + factY0Z0;
    idx[5] = factX1 + factY0Z1;
    idx[6] = factX1 + factY1Z0;
    idx[7] = factX1 + factY1Z1;

    // remainders from the fractional components - difference between the fractional value and the ceiling
    F rx = 1 - fx;
    F ry = 1 - fy;
    F rz = 1 - fz;
      
    F ryrz = ry*rz;
    F ryfz = ry*fz;
    F fyrz = fy*rz;
    F fyfz = fy*fz;

    F fdx[8];
    fdx[0] = rx*ryrz;
    fdx[1] = rx*ryfz;
    fdx[2] = rx*fyrz;
    fdx[3] = rx*fyfz;
    fdx[4] = fx*ryrz;
    fdx[5] = fx*ryfz;
    fdx[6] = fx*fyrz;
    fdx[7] = fx*fyfz;
    
    F f, r, a;
    T *inPtrTmp, *outPtrTmp;
    
    // do compounding
    if (accPtr)
      {
      unsigned short *accPtrTmp;

      // loop over the eight voxels
      int j = 8;
      do 
        {
        j--;
        if (fdx[j] == 0)
          {
          continue;
          }
        inPtrTmp = inPtr;
        outPtrTmp = outPtr+idx[j];
        accPtrTmp = accPtr+ ((unsigned short)(idx[j]/outInc[0]));
        f = fdx[j];
		    r = F((*accPtrTmp)/255);
        a = f + r;

        int i = numscalars;
        do
          {
          i--;
          vtkUltraRound((f*(*inPtrTmp++) + r*(*outPtrTmp))/a, *outPtrTmp);
          outPtrTmp++;
          }
        while (i);

        *accPtrTmp = 65535;
        *outPtrTmp = 255;
        a *= 255;
        // don't allow accumulation buffer overflow
        if (a < F(65535))
          {
          vtkUltraRound(a, *accPtrTmp);
          }
		    }
      while (j);
	    }

    // no compounding
    else 
      {
      // loop over the eight voxels
      int j = 8;
      do
        {
        j--;
        if (fdx[j] == 0)
          {
          continue;
          }
        inPtrTmp = inPtr;
        outPtrTmp = outPtr+idx[j];
        // if alpha is nonzero then the pixel was hit before, so
        //  average with previous value
        if (outPtrTmp[numscalars])
          {
          f = fdx[j];
          F r = 1 - f;
          int i = numscalars;
          do
            {
            i--;
            vtkUltraRound(f*(*inPtrTmp++) + r*(*outPtrTmp), *outPtrTmp);
            outPtrTmp++;
            }
          while (i);
          }
        // alpha is zero, so just insert the new value
        else
          {
          int i = numscalars;
          do
            {
            i--;
            *outPtrTmp++ = *inPtrTmp++;
            }
          while (i);
          }          
        *outPtrTmp = 255;
        }
      while (j);
	  }
    return 1;
	}
	// if bounds check fails
  return 0;
}     


//****************************************************************************
// HELPER FUNCTIONS FOR THE RECONSTRUCTION
//****************************************************************************

//----------------------------------------------------------------------------
// vtkIsIdentityMatrix
// check a matrix to see whether it is the identity matrix
//----------------------------------------------------------------------------
static int vtkIsIdentityMatrix(vtkMatrix4x4 *matrix)
{
  static double identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  int i,j;

  for (i = 0; i < 4; i++)
    {
    for (j = 0; j < 4; j++)
      {
      if (matrix->GetElement(i,j) != identity[4*i+j])
        {
        return 0;
        }
      }
    }
  return 1;
}

//----------------------------------------------------------------------------
// intersectionHelper
// find approximate intersection of line with the plane x = x_min,
// y = y_min, or z = z_min (lower limit of data extent) 
//----------------------------------------------------------------------------
template<class F>
static inline
int intersectionHelper(F *point, F *axis, int *limit, int ai, int *inExt)
{
  F rd = limit[ai]*point[3]-point[ai]  + 0.5; 
    
  if (rd < inExt[0])
    { 
    return inExt[0];
    }
  else if (rd > inExt[1])
    {
    return inExt[1];
    }
  else
    {
    return int(rd);
    }
}

//----------------------------------------------------------------------------
// intersectionLow
// find the point just inside the extent
//----------------------------------------------------------------------------
template <class F>
static int intersectionLow(F *point, F *axis, int *sign,
                           int *limit, int ai, int *inExt)
{
  // approximate value of r
  int r = intersectionHelper(point,axis,limit,ai,inExt);

  // move back and forth to find the point just inside the extent
  for (;;)
    {
    F p = point[ai]+r*axis[ai];

    if ((sign[ai] < 0 && r > inExt[0] ||
         sign[ai] > 0 && r < inExt[1]) && 
        vtkUltraRound(p) < limit[ai])
      {
      r += sign[ai];
      }
    else
      {
      break;
      }
    }

  for (;;)
    {
    F p = point[ai]+(r-sign[ai])*axis[ai];

    if ((sign[ai] > 0 && r > inExt[0] ||
         sign[ai] < 0 && r < inExt[1]) && 
        vtkUltraRound(p) >= limit[ai])
      {
      r -= sign[ai];
      }
    else
      {
      break;
      }
    }

  return r;
}

//----------------------------------------------------------------------------
// intersectionHigh
// same as above, but for x = x_max
//----------------------------------------------------------------------------
template <class F>
static int intersectionHigh(F *point, F *axis, int *sign, 
                            int *limit, int ai, int *inExt)
{
  // approximate value of r
  int r = intersectionHelper(point,axis,limit,ai,inExt);
    
  // move back and forth to find the point just inside the extent
  for (;;)
    {
    F p = point[ai]+r*axis[ai];

    if ((sign[ai] > 0 && r > inExt[0] ||
         sign[ai] < 0 && r < inExt[1]) &&
        vtkUltraRound(p) > limit[ai])
      {
      r -= sign[ai];
      }
    else
      {
      break;
      }
    }

  for (;;)
    {
    F p = point[ai]+(r+sign[ai])*axis[ai];

    if ((sign[ai] < 0 && r > inExt[0] ||
         sign[ai] > 0 && r < inExt[1]) && 
        vtkUltraRound(p) <= limit[ai])
      {
      r += sign[ai];
      }
    else
      {
      break;
      }
    }

  return r;
}

//----------------------------------------------------------------------------
// isBounded
//----------------------------------------------------------------------------
template <class F>
static int isBounded(F *point, F *xAxis, int *inMin, 
                     int *inMax, int ai, int r)
{
  int bi = ai+1; 
  int ci = ai+2;
  if (bi > 2) 
    { 
    bi -= 3; // coordinate index must be 0, 1 or 2 
    } 
  if (ci > 2)
    { 
    ci -= 3;
    }

  F fbp = point[bi]+r*xAxis[bi];
  F fcp = point[ci]+r*xAxis[ci];

  int bp = vtkUltraRound(fbp);
  int cp = vtkUltraRound(fcp);
  
  return (bp >= inMin[bi] && bp <= inMax[bi] &&
          cp >= inMin[ci] && cp <= inMax[ci]);
}

//----------------------------------------------------------------------------
// vtkUltraFindExtentHelper
// This huge mess finds out where the current output raster
// line intersects the input volume
//----------------------------------------------------------------------------
static void vtkUltraFindExtentHelper(int &r1, int &r2, int sign, int *inExt)
{
  if (sign < 0)
    {
    int i = r1;
    r1 = r2;
    r2 = i;
    }
  
  // bound r1,r2 within reasonable limits
  if (r1 < inExt[0]) 
    {
    r1 = inExt[0];
    }
  if (r2 > inExt[1]) 
    {
    r2 = inExt[1];
    }
  if (r1 > r2) 
    {
    r1 = inExt[0];
    r2 = inExt[0]-1;
    }
}  

//----------------------------------------------------------------------------
// vtkUltraFindExtent
//----------------------------------------------------------------------------
template <class F>
static void vtkUltraFindExtent(int& r1, int& r2, F *point, F *xAxis, 
                                 int *inMin, int *inMax, int *inExt)
{
  int i, ix, iy, iz;
  int sign[3];
  int indx1[4],indx2[4];
  F p1,p2;

  // find signs of components of x axis 
  // (this is complicated due to the homogeneous coordinate)
  for (i = 0; i < 3; i++)
    {
    p1 = point[i];

    p2 = point[i]+xAxis[i];

    if (p1 <= p2)
      {
      sign[i] = 1;
      }
    else 
      {
      sign[i] = -1;
      }
    } 
  
  // order components of xAxis from largest to smallest
  ix = 0;
  for (i = 1; i < 3; i++)
    {
    if (((xAxis[i] < 0) ? (-xAxis[i]) : (xAxis[i])) >
        ((xAxis[ix] < 0) ? (-xAxis[ix]) : (xAxis[ix])))
      {
      ix = i;
      }
    }
  
  iy = ((ix > 1) ? ix-2 : ix+1);
  iz = ((ix > 0) ? ix-1 : ix+2);

  if (((xAxis[iy] < 0) ? (-xAxis[iy]) : (xAxis[iy])) >
      ((xAxis[iz] < 0) ? (-xAxis[iz]) : (xAxis[iz])))
    {
    i = iy;
    iy = iz;
    iz = i;
    }

  r1 = intersectionLow(point,xAxis,sign,inMin,ix,inExt);
  r2 = intersectionHigh(point,xAxis,sign,inMax,ix,inExt);
  
  // find points of intersections
  // first, find w-value for perspective (will usually be 1)
  for (i = 0; i < 3; i++)
    {
    p1 = point[i]+r1*xAxis[i];
    p2 = point[i]+r2*xAxis[i];

    indx1[i] = vtkUltraRound(p1);
    indx2[i] = vtkUltraRound(p2);
    }

  // passed through x face, check opposing face
  if (isBounded(point,xAxis,inMin,inMax,ix,r1))
    {
    if (isBounded(point,xAxis,inMin,inMax,ix,r2))
      {
      vtkUltraFindExtentHelper(r1,r2,sign[ix],inExt);
      return;
      }

    // check y face
    if (indx2[iy] < inMin[iy])
      {
      r2 = intersectionLow(point,xAxis,sign,inMin,iy,inExt);
      if (isBounded(point,xAxis,inMin,inMax,iy,r2))
        {
        vtkUltraFindExtentHelper(r1,r2,sign[ix],inExt);
        return;
        }
      }

    // check other y face
    else if (indx2[iy] > inMax[iy])
      {
      r2 = intersectionHigh(point,xAxis,sign,inMax,iy,inExt);
      if (isBounded(point,xAxis,inMin,inMax,iy,r2))
        {
        vtkUltraFindExtentHelper(r1,r2,sign[ix],inExt);
        return;
        }
      }
    
    // check z face
    if (indx2[iz] < inMin[iz])
      {
      r2 = intersectionLow(point,xAxis,sign,inMin,iz,inExt);
      if (isBounded(point,xAxis,inMin,inMax,iz,r2))
        {
        vtkUltraFindExtentHelper(r1,r2,sign[ix],inExt);
        return;
        }
      }

    // check other z face
    else if (indx2[iz] > inMax[iz])
      {
      r2 = intersectionHigh(point,xAxis,sign,inMax,iz,inExt);
      if (isBounded(point,xAxis,inMin,inMax,iz,r2))
        {
        vtkUltraFindExtentHelper(r1,r2,sign[ix],inExt);
        return;
        }
      }
    }
  
  // passed through the opposite x face
  if (isBounded(point,xAxis,inMin,inMax,ix,r2))
    {
    // check y face
    if (indx1[iy] < inMin[iy])
      {
      r1 = intersectionLow(point,xAxis,sign,inMin,iy,inExt);
      if (isBounded(point,xAxis,inMin,inMax,iy,r1))
        {
        vtkUltraFindExtentHelper(r1,r2,sign[ix],inExt);
        return;
        }
      }
    // check other y face
    else if (indx1[iy] > inMax[iy])
      {
      r1 = intersectionHigh(point,xAxis,sign,inMax,iy,inExt);
      if (isBounded(point,xAxis,inMin,inMax,iy,r1))
        {
        vtkUltraFindExtentHelper(r1,r2,sign[ix],inExt);
        return;
        }
      }
    
    // check other y face
    if (indx1[iz] < inMin[iz])
      {
      r1 = intersectionLow(point,xAxis,sign,inMin,iz,inExt);
      if (isBounded(point,xAxis,inMin,inMax,iz,r1))
        {
        vtkUltraFindExtentHelper(r1,r2,sign[ix],inExt);
        return;
        }
      }
    // check other z face
    else if (indx1[iz] > inMax[iz])
      {
      r1 = intersectionHigh(point,xAxis,sign,inMax,iz,inExt);
      if (isBounded(point,xAxis,inMin,inMax,iz,r1))
        {
        vtkUltraFindExtentHelper(r1,r2,sign[ix],inExt);
        return;
        }
      }
    }
  
  // line might pass through bottom face
  if ((indx1[iy] >= inMin[iy] && indx2[iy] < inMin[iy]) ||
      (indx1[iy] < inMin[iy] && indx2[iy] >= inMin[iy]))
    {
    r1 = intersectionLow(point,xAxis,sign,inMin,iy,inExt);
    if (isBounded(point,xAxis,inMin,inMax,iy,r1))
      {
      // line might pass through top face
      if ((indx1[iy] <= inMax[iy] && indx2[iy] > inMax[iy]) ||
          (indx1[iy] > inMax[iy] && indx2[iy] <= inMax[iy]))
        { 
        r2 = intersectionHigh(point,xAxis,sign,inMax,iy,inExt);
        if (isBounded(point,xAxis,inMin,inMax,iy,r2))
          {
          vtkUltraFindExtentHelper(r1,r2,sign[iy],inExt);
          return;
          }
        }
      
      // line might pass through in-to-screen face
      if (indx1[iz] < inMin[iz] && indx2[iy] < inMin[iy] ||
          indx2[iz] < inMin[iz] && indx1[iy] < inMin[iy])
        { 
        r2 = intersectionLow(point,xAxis,sign,inMin,iz,inExt);
        if (isBounded(point,xAxis,inMin,inMax,iz,r2))
          {
          vtkUltraFindExtentHelper(r1,r2,sign[iy],inExt);
          return;
          }
        }
      // line might pass through out-of-screen face
      else if (indx1[iz] > inMax[iz] && indx2[iy] < inMin[iy] ||
               indx2[iz] > inMax[iz] && indx1[iy] < inMin[iy])
        {
        r2 = intersectionHigh(point,xAxis,sign,inMax,iz,inExt);
        if (isBounded(point,xAxis,inMin,inMax,iz,r2))
          {
          vtkUltraFindExtentHelper(r1,r2,sign[iy],inExt);
          return;
          }
        } 
      }
    }
  
  // line might pass through top face
  if ((indx1[iy] <= inMax[iy] && indx2[iy] > inMax[iy]) ||
      (indx1[iy] > inMax[iy] && indx2[iy] <= inMax[iy]))
    {
    r2 = intersectionHigh(point,xAxis,sign,inMax,iy,inExt);
    if (isBounded(point,xAxis,inMin,inMax,iy,r2))
      {
      // line might pass through in-to-screen face
      if (indx1[iz] < inMin[iz] && indx2[iy] > inMax[iy] ||
          indx2[iz] < inMin[iz] && indx1[iy] > inMax[iy])
        {
        r1 = intersectionLow(point,xAxis,sign,inMin,iz,inExt);
        if (isBounded(point,xAxis,inMin,inMax,iz,r1))
          {
          vtkUltraFindExtentHelper(r1,r2,sign[iy],inExt);
          return;
          }
        }
      // line might pass through out-of-screen face
      else if (indx1[iz] > inMax[iz] && indx2[iy] > inMax[iy] || 
               indx2[iz] > inMax[iz] && indx1[iy] > inMax[iy])
        {
        r1 = intersectionHigh(point,xAxis,sign,inMax,iz,inExt);
        if (isBounded(point,xAxis,inMin,inMax,iz,r1))
          {
          vtkUltraFindExtentHelper(r1,r2,sign[iy],inExt);
          return;
          }
        }
      } 
    }
  
  // line might pass through in-to-screen face
  if ((indx1[iz] >= inMin[iz] && indx2[iz] < inMin[iz]) ||
      (indx1[iz] < inMin[iz] && indx2[iz] >= inMin[iz]))
    {
    r1 = intersectionLow(point,xAxis,sign,inMin,iz,inExt);
    if (isBounded(point,xAxis,inMin,inMax,iz,r1))
      {
      // line might pass through out-of-screen face
      if (indx1[iz] > inMax[iz] || indx2[iz] > inMax[iz])
        {
        r2 = intersectionHigh(point,xAxis,sign,inMax,iz,inExt);
        if (isBounded(point,xAxis,inMin,inMax,iz,r2))
          {
          vtkUltraFindExtentHelper(r1,r2,sign[iz],inExt);
          return;
          }
        }
      }
    }
  
  r1 = inExt[0];
  r2 = inExt[0] - 1;
}


//****************************************************************************
// REAL-TIME RECONSTRUCTION - OPTIMIZED
//****************************************************************************

//----------------------------------------------------------------------------
// vtkOptimizedInsertSlice
// Actually inserts the slice, with optimization.
//----------------------------------------------------------------------------
template <class F, class T>
static void vtkOptimizedInsertSlice(vtkFreehandUltrasound2 *self, // the freehand us
									vtkImageData *outData, // the output volume
									T *outPtr, // scalar pointer to the output volume over the output extent
									unsigned short *accPtr, // scalar pointer to the accumulation buffer over the output extent
									vtkImageData *inData, // input slice
									T *inPtr, // scalar pointer to the input volume over the input slice extent
									int inExt[6], // input slice extent (could have been split for threading)
									F matrix[4][4], // index matrix, output indices -> input indices
									int threadId) // current thread id
{
	int prevPixelCount = self->GetPixelCount();

	// local variables
	int id = 0;
	int i, numscalars; // numscalars = number of scalar components in the input image
	int idX, idY, idZ; // the x, y, and z pixel of the input image
	int inIncX, inIncY, inIncZ; // increments for the input extent
	int outExt[6]; // output extent
	int outMax[3], outMin[3]; // the max and min values of the output extents -
	// if outextent = (x0, x1, y0, y1, z0, z1), then
	// outMax = (x1, y1, z1) and outMin = (x0, y0, z0)
	int outInc[3]; // increments for the output extent
	int clipExt[6];
	unsigned long count = 0;
	unsigned long target;
	int r1,r2;
	// outPoint0, outPoint1, outPoint is a fancy way of incremetally multiplying the input point by
	// the index matrix to get the output point...  Outpoint is the result
	F outPoint0[3]; // temp, see above
	F outPoint1[3]; // temp, see above
	F outPoint[3]; // this is the final output point, created using Output0 and Output1
	F xAxis[3], yAxis[3], zAxis[3], origin[3]; // the index matrix (transform), broken up into axes and an origin
	vtkFloatingPointType inSpacing[3],inOrigin[3]; // input spacing and origin

	// input spacing and origin
	inData->GetSpacing(inSpacing);
	inData->GetOrigin(inOrigin);

	// number of pixels in the x and y directions b/w the fan origin and the slice origin
	double xf = (self->GetFanOrigin()[0]-inOrigin[0])/inSpacing[0];
	double yf = (self->GetFanOrigin()[1]-inOrigin[1])/inSpacing[1];

	if (self->GetFlipHorizontalOnOutput())
    {
    yf = (double)self->GetNumberOfPixelsFromTipOfFanToBottomOfScreen();
    }

	// fan depth squared
	double d2 = self->GetFanDepth()*self->GetFanDepth();
	// input spacing in the x and y directions
	double xs = inSpacing[0];
	double ys = inSpacing[1];
	// tan of the left and right fan angles
	double ml = tan(self->GetFanAngles()[0]*vtkMath::DoubleDegreesToRadians())/xs*ys;
	double mr = tan(self->GetFanAngles()[1]*vtkMath::DoubleDegreesToRadians())/xs*ys;
	// the tan of the right fan angle is always greater than the left one
	if (ml > mr)
	  {
		double tmp = ml; ml = mr; mr = tmp;
	  }

	// get the clip rectangle as an extent
	self->GetClipExtent(clipExt, inOrigin, inSpacing, inExt);

	// find maximum output range
	outData->GetExtent(outExt);

	for (i = 0; i < 3; i++)
	  {
		outMin[i] = outExt[2*i];
		outMax[i] = outExt[2*i+1];
	  }

	target = (unsigned long)
		((inExt[5]-inExt[4]+1)*(inExt[3]-inExt[2]+1)/50.0);
	target++;

	int wExtent[6]; // output whole extent
	outData->GetWholeExtent(wExtent);
	outData->GetIncrements(outInc);
	inData->GetContinuousIncrements(inExt, inIncX, inIncY, inIncZ);
	numscalars = inData->GetNumberOfScalarComponents();

	// break matrix into a set of axes plus an origin
	// (this allows us to calculate the transform Incrementally)
	for (i = 0; i < 3; i++)
	  {
		xAxis[i]  = matrix[i][0]; // remember that the matrix is the indexMatrix, and transforms
		yAxis[i]  = matrix[i][1];	// output pixels to input pixels
		zAxis[i]  = matrix[i][2];
		origin[i] = matrix[i][3];
	  }

  static int firstFrame = 1;

	// Loop through INPUT pixels - remember this is a 3D cube represented by the input extent
	for (idZ = inExt[4]; idZ <= inExt[5]; idZ++) // for each image...
	  {
		outPoint0[0] = origin[0]+idZ*zAxis[0]; // incremental transform
		outPoint0[1] = origin[1]+idZ*zAxis[1];
		outPoint0[2] = origin[2]+idZ*zAxis[2];

		for (idY = inExt[2]; idY <= inExt[3]; idY++) // for each horizontal line in the image...
		  {

			//TODO implement this for other options
			if (self->GetFlipHorizontalOnOutput())
			  {
        int dist = self->GetNumberOfPixelsFromTipOfFanToBottomOfScreen();
			  outPoint1[0] = outPoint0[0]+(dist-idY)*yAxis[0]; // incremental transform
			  outPoint1[1] = outPoint0[1]+(dist-idY)*yAxis[1];
			  outPoint1[2] = outPoint0[2]+(dist-idY)*yAxis[2];
			  }
			else
			  {
			  outPoint1[0] = outPoint0[0]+idY*yAxis[0]; // incremental transform
			  outPoint1[1] = outPoint0[1]+idY*yAxis[1];
			  outPoint1[2] = outPoint0[2]+idY*yAxis[2];
			  }

			if (!id)
			  {
				if (!(count%target)) 
				  {
					self->UpdateProgress(count/(50.0*target));  // progress between 0 and 1
				  }
				count++;
			  }

			// find intersections of x raster line with the output extent

      // this only changes r1 and r2
			vtkUltraFindExtent(r1,r2,outPoint1,xAxis,outMin,outMax,inExt);

			// next, handle the 'fan' shape of the input
			double y = (yf - idY);;
			if (ys < 0)
			  {
				y = -y;
			  }

      // first, check the angle range of the fan - choose r1 and r2 based
      // on the triangle that the fan makes from the fan origin to the bottom
      // line of the video image
			if (!(ml == 0 && mr == 0))
			  {
				// equivalent to: r1 < vtkUltraCeil(ml*y + xf + 1)
        // this is what the radius would be based on tan(fanAngle)
				if (r1 < -vtkUltraFloor(-(ml*y + xf + 1)))
				  {
					r1 = -vtkUltraFloor(-(ml*y + xf + 1));
				  }
				if (r2 > vtkUltraFloor(mr*y + xf - 1))
				  {
					r2 = vtkUltraFloor(mr*y + xf - 1);
				  }

				// next, check the radius of the fan - crop the triangle to the fan
        // depth
				double dx = (d2 - (y*y)*(ys*ys))/(xs*xs);

        // if we are outside the fan's radius, ex at the bottom lines
				if (dx < 0)
				  {
					r1 = inExt[0];
					r2 = inExt[0]-1;
				  }
        // if we are within the fan's radius, we have to adjust if we are in
        // the "ellipsoidal" (bottom) part of the fan instead of the top
        // "triangular" part
				else
				  {
					dx = sqrt(dx);
          // this is what r1 would be if we calculated it based on the
          // pythagorean theorem
					if (r1 < -vtkUltraFloor(-(xf - dx + 1)))
					  {
						r1 = -vtkUltraFloor(-(xf - dx + 1));
					  }
					if (r2 > vtkUltraFloor(xf + dx - 1))
					  {
						r2 = vtkUltraFloor(xf + dx - 1);
					  }
				  }
			  }

			// bound to the ultrasound clip rectangle
			if (r1 < clipExt[0])
			  {
				r1 = clipExt[0];
			  }
			if (r2 > clipExt[1])
			  {
				r2 = clipExt[1];
			  }

			if (r1 > r2)
			  {
				r1 = inExt[0];
				r2 = inExt[0]-1;
			  }

			// skip the portion of the slice to the left of the fan
			for (idX = inExt[0]; idX < r1; idX++)
			  {
				inPtr += numscalars;
			  }

      // multiplying the input point by the transform will give you fractional pixels,
      // so we need interpolation

			// interpolating linearly (code 1)
			if (self->GetInterpolationMode() == VTK_FREEHAND_LINEAR)
			  { 

				for (idX = r1; idX <= r2; idX++) // for all of the x pixels within the fan
				  {
					//TODO implement this for other options
					if (self->GetFlipVerticalOnOutput())
					  {
					  outPoint[0] = outPoint1[0] + (r1+r2-idX)*xAxis[0];
					  outPoint[1] = outPoint1[1] + (r1+r2-idX)*xAxis[1];
					  outPoint[2] = outPoint1[2] + (r1+r2-idX)*xAxis[2];
					  }
					else
					  {
					  outPoint[0] = outPoint1[0] + idX*xAxis[0];
					  outPoint[1] = outPoint1[1] + idX*xAxis[1];
					  outPoint[2] = outPoint1[2] + idX*xAxis[2];
					  }

					int hit = vtkTrilinearInterpolation(outPoint, inPtr, outPtr, accPtr, 
						numscalars, outExt, outInc); // hit is either 1 or 0

					inPtr += numscalars; // go to the next x pixel
					self->IncrementPixelCount(threadId, hit);
				  }
			  }

			// interpolating with nearest neighbor (code 0)
			else 
			  {
				vtkFreehand2OptimizedNNHelper(r1, r2, outPoint, outPoint1, xAxis, 
					inPtr, outPtr, outExt, outInc,
					numscalars, accPtr);
        // we added all the pixels between r1 and r2, so increment our count of the number of pixels added
				self->IncrementPixelCount(threadId, r2-r1+1); 
			  }

			// skip the portion of the slice to the right of the fan
			for (idX = r2+1; idX <= inExt[1]; idX++)
			  {
				inPtr += numscalars;
			  }

			inPtr += inIncY; // move to the next line
		  }
		inPtr += inIncZ; // move to the next image
	  }
}

//----------------------------------------------------------------------------
// ThreadedSliceExecute
// This method is passed a input and output region, and executes the filter
// algorithm to fill the output from the input.
// It just executes a switch statement to call the correct vtkOptimizedInsertSlice
// function for the regions data types.
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::ThreadedSliceExecute(vtkImageData *inData, // input data
						 vtkImageData *outData, // output data
						 int inExt[6], // input extent (could be split for this thread)
						 int threadId, // current thread id
             int phase) // phase for accumulation buffer
{

  // get scalar pointers for extents and output extent
  void *inPtr = inData->GetScalarPointerForExtent(inExt);
  int *outExt = this->OutputExtent;
  void *outPtr = outData->GetScalarPointerForExtent(outExt);
 
  // get the accumulation buffer and the scalar pointer for its extent, if we are compounding
  void *accPtr = NULL; 
  vtkImageData *accData = this->AccumulationBuffers[phase];

  if (this->Compounding)
    {
    accPtr = accData->GetScalarPointerForExtent(outExt);
    }
  else
    {
    accPtr = NULL;
    }

  // print out for debugging
  vtkDebugMacro(<< "OptimizedInsertSlice: inData = " << inData << ", outData = " << outData);
  
  // this filter expects that input is the same type as output.
  if (inData->GetScalarType() != outData->GetScalarType())
    {
    vtkErrorMacro(<< "OptimizedInsertSlice: input ScalarType, " 
		  << inData->GetScalarType()
		  << ", must match out ScalarType "<<outData->GetScalarType());
    return;
    }

  // use fixed-point math for optimization level 2
  if (this->GetOptimization() == 2)
    {
    // change transform matrix so that instead of taking 
    // input coords -> output coords it takes output indices -> input indices
    vtkMatrix4x4* matrix;
    if (!this->DiscardOutlierHeartRates)
      {
      matrix = this->GetIndexMatrix();
      }
    else
      {
      matrix = this->GetIndexMatrix(phase);
      }
    fixed newmatrix[4][4]; // fixed because optimization = 2
    for (int i = 0; i < 4; i++)
      {
      newmatrix[i][0] = matrix->GetElement(i,0);
      newmatrix[i][1] = matrix->GetElement(i,1);
      newmatrix[i][2] = matrix->GetElement(i,2);
      newmatrix[i][3] = matrix->GetElement(i,3);
      }

    switch (inData->GetScalarType())
      {
	    case VTK_SHORT:
		    vtkOptimizedInsertSlice(this, outData, (short *)(outPtr), 
                                (unsigned short *)(accPtr), 
                                inData, (short *)(inPtr), 
								                inExt, newmatrix, threadId);
        break;
	    case VTK_UNSIGNED_SHORT:
        vtkOptimizedInsertSlice(this,outData,(unsigned short *)(outPtr),
                                (unsigned short *)(accPtr), 
                                inData, (unsigned short *)(inPtr), 
								                inExt, newmatrix, threadId);
        break;
      case VTK_UNSIGNED_CHAR:
        vtkOptimizedInsertSlice(this, outData,(unsigned char *)(outPtr),
                                (unsigned short *)(accPtr), 
                                inData, (unsigned char *)(inPtr), 
								                inExt, newmatrix, threadId);
        break;
      default:
        vtkErrorMacro(<< "OptimizedInsertSlice: Unknown input ScalarType");
        return;
      }
    }

  // if we are not using fixed point math for optimization = 2, we are either doing no
  // optimization (0) or we are breaking into x, y, z components with no bounds checking for
  // nearest neighbor (1)
  else
    {
    // change transform matrix so that instead of taking 
    // input coords -> output coords it takes output indices -> input indices
    vtkMatrix4x4 *matrix;
    if (!this->DiscardOutlierHeartRates)
      {
      matrix = this->GetIndexMatrix();
      }
    else
      {
      matrix = this->GetIndexMatrix(phase);
      }
    double newmatrix[4][4];
    for (int i = 0; i < 4; i++)
      {
      newmatrix[i][0] = matrix->GetElement(i,0);
      newmatrix[i][1] = matrix->GetElement(i,1);
      newmatrix[i][2] = matrix->GetElement(i,2);
      newmatrix[i][3] = matrix->GetElement(i,3);
      }
  
    switch (inData->GetScalarType())
      {
      case VTK_SHORT:
        vtkOptimizedInsertSlice(this, outData, (short *)(outPtr), 
                                (unsigned short *)(accPtr), 
                                inData, (short *)(inPtr), 
                                inExt, newmatrix, threadId);
        break;
      case VTK_UNSIGNED_SHORT:
        vtkOptimizedInsertSlice(this,outData,(unsigned short *)(outPtr),
                                (unsigned short *)(accPtr), 
                                inData, (unsigned short *)(inPtr), 
                                inExt, newmatrix, threadId);
        break;
      case VTK_UNSIGNED_CHAR:
        vtkOptimizedInsertSlice(this, outData,(unsigned char *)(outPtr),
                                (unsigned short *)(accPtr), 
                                inData, (unsigned char *)(inPtr), 
                                inExt, newmatrix, threadId);
        break;
      default:
        vtkErrorMacro(<< "OptimizedInsertSlice: Unknown input ScalarType");
        return;
      }
    }
}

//----------------------------------------------------------------------------
// SplitSliceExtent
// For streaming and threads.  Splits the output update extent (startExt) into
// "total" pieces and calculates the split extent for the thread of thread id
// "num".  This method needs to be called "total" times
// with different values of num (thread id) to fill in the split extents for each
// therad.  This method returns the number of pieces resulting from a successful
// split - from 1 to "total".  If 1 is returned, the extent cannot be split.
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::SplitSliceExtent(int splitExt[6], // the extent of this split
					    int startExt[6],  //the original extent to be split up
					    int num, // current thread id
					    int total) // the maximum number of threads (pieces)
{
  int splitAxis; // the axis we should split along
  int min, max; // the min and max indices of the axis of interest

  // prints where we are, the starting extent, num and total for debugging
   vtkDebugMacro("SplitSliceExtent: ( " << startExt[0] << ", " << startExt[1]
		<< ", "
                << startExt[2] << ", " << startExt[3] << ", "
                << startExt[4] << ", " << startExt[5] << "), " 
                << num << " of " << total);
  
  // start with same extent - copy the extent from startExt to splitExt
  memcpy(splitExt, startExt, 6 * sizeof(int));

  // determine which axis we should split along - preference is z, then y, then x
  // as long as we can possibly split along that axis (i.e. as long as z0 != z1)
  splitAxis = 2; // at the end, shows whether we split along the z(2), y(1) or x(0) axis
  min = startExt[4]; // z0 of startExt
  max = startExt[5]; // z1 of startExt
  while (min == max)
    {
    --splitAxis;
    // we cannot split if the input extent is something like [50, 50, 100, 100, 0, 0]
    if (splitAxis < 0)
      { 
      vtkDebugMacro("  Cannot Split");
      return 1;
      }
    min = startExt[splitAxis*2];
    max = startExt[splitAxis*2+1];
    }

  // determine the actual number of pieces that will be generated (return value)
  int range = max - min + 1;
  // split the range over the maximum number of threads
  int valuesPerThread = (int)ceil(range/(double)total);
  // figure out the largest thread id used
  int maxThreadIdUsed = (int)ceil(range/(double)valuesPerThread) - 1;
  // if we are in a thread that will work on part of the extent, then figure
  // out the range that this thread should work on
  if (num < maxThreadIdUsed)
    {
    splitExt[splitAxis*2] = splitExt[splitAxis*2] + num*valuesPerThread;
    splitExt[splitAxis*2+1] = splitExt[splitAxis*2] + valuesPerThread - 1;
    }
  if (num == maxThreadIdUsed)
    {
    splitExt[splitAxis*2] = splitExt[splitAxis*2] + num*valuesPerThread;
    }

  // return the number of threads used
  return maxThreadIdUsed + 1;
}

//----------------------------------------------------------------------------
// vtkFreehand2ThreadedExecute
// This mess is really a simple function. All it does is call
// the ThreadedSliceExecute method after setting the correct
// extent for this thread. Its just a pain to calculate
// the correct extent.
//----------------------------------------------------------------------------
VTK_THREAD_RETURN_TYPE vtkFreehand2ThreadedExecute( void *arg )
{
  vtkFreehand2ThreadStruct *str; // contains the filter, input and output
  int ext[6], splitExt[6], total; // the input slice extent, the input slice extent
                                  // for this thread, and the total number of pieces
                                  // the extent can be split into (i.e. the number of
                                  // threads we should use)
  int threadId, threadCount; // thread id and number of threads, from the argument
  vtkImageData *input; // the slice input

  // get the thread id and number of threads
  threadId = ((ThreadInfoStruct *)(arg))->ThreadID;
  threadCount = ((ThreadInfoStruct *)(arg))->NumberOfThreads;

  // get the filter, input and output in the form of a vtkFreehand2ThreadStruct
  // and get the input extent
  str = (vtkFreehand2ThreadStruct *)(((ThreadInfoStruct *)(arg))->UserData);
  input = str->Input;
  input->GetUpdateExtent( ext );

  // execute the actual method with appropriate extent
  // first find out how many pieces the extent can be split into and calculate
  // the extent for this thread (the splitExt)
  total = str->Filter->SplitSliceExtent(splitExt, ext, threadId, threadCount);
  
  // if we can use this thread, then call ThreadedSliceExecute
  if (threadId < total)
    {
    str->Filter->ThreadedSliceExecute(str->Input, str->Output,
				      splitExt, threadId, str->Phase);
    }
  // else
  //   {
  //   otherwise don't use this thread. Sometimes the threads dont
  //   break up very well and it is just as efficient to leave a 
  //   few threads idle.
  //   }
  
  return VTK_THREAD_RETURN_VALUE;
}

//----------------------------------------------------------------------------
// MultiThread
// Setup the threader, and set the single method to be vtkFreehand2ThreadedExecute.
// Then execute vtkFreehand2ThreadedExecute.
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::MultiThread(vtkImageData *inData,
                                        vtkImageData *outData,
                                        int phase)
{
  // set up a vtkFreehand2ThreadStruct (defined above)
  vtkFreehand2ThreadStruct str;
  str.Filter = this;
  str.Input = inData;
  str.Output = outData;
  str.Phase = phase;

  // TODO this->NumberOfThreads is never updated past one when threads are
  // spawned or terminated
  this->Threader->SetNumberOfThreads(this->NumberOfThreads);
  // set the single method
  this->Threader->SetSingleMethod(vtkFreehand2ThreadedExecute, &str);
  // execute the single method using this->NumberOfThreads threads
  this->Threader->SingleMethodExecute();
}

//----------------------------------------------------------------------------
// OptimizedInsertSlice
// Given an input and output region, execute the filter algorithm to fill the
// output from the input - optimized by splitting into x,y,z components or
// with integer math, for the first volume
// It just calls MultiThread
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::OptimizedInsertSlice()
  {

  if (this->Triggering)
    {
    vtkErrorMacro(<< "Should not use OptimizedInsertSlice() when triggering - use OptimizedInsertSlice(phase) instead");
    return;
    }

  this->OptimizedInsertSlice(0);
  }

//----------------------------------------------------------------------------
// OptimizedInsertSlice
// Given an input and output region, execute the filter algorithm to fill the
// output from the input - optimized by splitting into x,y,z components or
// with integer math, for the ith volume
// It just calls MultiThread
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::OptimizedInsertSlice(int phase)
{
  if (this->ReconstructionThreadId == -1)
    {
		this->UpdateInformation();
    }
  if (this->NeedsClear)
    {
    this->InternalClearOutput();
    }

  vtkImageData *inData;
  if (!this->DiscardOutlierHeartRates)
    {
    inData = this->GetSlice();
    }
  else
    {
    if (!this->SliceBuffer)
      {
      vtkErrorMacro(<< "OptimizedInsertSlice(phase) doesn't have a slice buffer");
      return;
      }
    if (!this->SliceBuffer[phase])
      {
      vtkErrorMacro(<< "OptimizedInsertSlice(phase) doesn't have a slice buffer at this phase");
      return;
      }
    inData = this->SliceBuffer[phase];
    }
  vtkImageData *outData = this->GetOutput(phase);

  this->ActiveFlagLock->Lock();
  // if not in ReconstructionThread, update the slice here
  // (otherwise, the slice is updated in vtkReconstructionThread
  // to ensure synchronization with the tracking)
  if (this->ReconstructionThreadId == -1)
    {
    int clipExt[6];
    this->GetClipExtent(clipExt, inData->GetOrigin(), inData->GetSpacing(),
			inData->GetWholeExtent());
    inData->SetUpdateExtentToWholeExtent();
    inData->Update();
    }
  this->ActiveFlagLock->Unlock();
	this->MultiThread(inData, outData, phase);
  this->Modified();
}

//****************************************************************************
// REAL-TIME RECONSTRUCTION - NOT OPTIMIZED
//****************************************************************************

//----------------------------------------------------------------------------
// Actually inserts the slice - executes the filter for any type of data, for
// no optimization
// Given an input and output region, execute the filter algorithm to fill the
// output from the input - no optimization.
// (this one function is pretty much the be-all and end-all of the
// filter)
//----------------------------------------------------------------------------
template <class T>
static void vtkFreehandUltrasound2InsertSlice(vtkFreehandUltrasound2 *self,
                                             vtkImageData *outData,
					                                   T *outPtr,
                                             unsigned short *accPtr,
                                             vtkImageData *inData,
					                                   T *inPtr,
                                             int inExt[6],
                                             vtkMatrix4x4 *matrix)
{

  // local variables
  int numscalars;
  int idX, idY, idZ;
  int inIncX, inIncY, inIncZ;
  int outExt[6], outInc[3], clipExt[6];
  vtkFloatingPointType inSpacing[3], inOrigin[3];
  // the resulting point in the output volume (outPoint) from a point in the input slice
  // (inpoint)
  double outPoint[4], inPoint[4]; 

  // pointer to the nearest neighbor or trilinear interpolation function
  int (*interpolate)(double *point, T *inPtr, T *outPtr,
		     unsigned short *accPtr, int numscalars, int outExt[6], int outInc[3]);
  
  // slice spacing and origin
  inData->GetSpacing(inSpacing);
  inData->GetOrigin(inOrigin);
  // number of pixels in the x and y directions b/w the fan origin and the slice origin
  double xf = (self->GetFanOrigin()[0]-inOrigin[0])/inSpacing[0];
  double yf = (self->GetFanOrigin()[1]-inOrigin[1])/inSpacing[1]; 
  // fan depth squared 
  double d2 = self->GetFanDepth()*self->GetFanDepth();
  // absolute value of slice spacing
  double xs = fabs((double)(inSpacing[0]));
  double ys = fabs((double)(inSpacing[1]));
  // tan of the left and right fan angles
  double ml = tan(self->GetFanAngles()[0]*vtkMath::DoubleDegreesToRadians())/xs*ys;
  double mr = tan(self->GetFanAngles()[1]*vtkMath::DoubleDegreesToRadians())/xs*ys;
  // the tan of the right fan angle is always greater than the left one
  if (ml > mr)
    {
    double tmp = ml; ml = mr; mr = tmp;
    }
  // get the clip rectangle as an extent
  self->GetClipExtent(clipExt, inOrigin, inSpacing, inExt);

  // find maximum output range = output extent
  outData->GetExtent(outExt);

  // Get increments to march through data - ex move from the end of one x scanline of data to the
  // start of the next line
  outData->GetIncrements(outInc);
  inData->GetContinuousIncrements(inExt, inIncX, inIncY, inIncZ);
  numscalars = inData->GetNumberOfScalarComponents();
  
  // Set interpolation method - nearest neighbor or trilinear
  vtkGetUltraInterpFunc(self,&interpolate);

  // Loop through  slice pixels in the input extent and put them into the output volume
  for (idZ = inExt[4]; idZ <= inExt[5]; idZ++)
    {
      for (idY = inExt[2]; idY <= inExt[3]; idY++)
      {
	      for (idX = inExt[0]; idX <= inExt[1]; idX++)
        {

	      // if we are within the current clip extent
        if (idX >= clipExt[0] && idX <= clipExt[1] && 
	          idY >= clipExt[2] && idY <= clipExt[3])
          {
	        // current x/y index minus num pixels in the x/y direction b/w the fan origin and the slice origin
	        double x = (idX-xf);
          double y = (idY-yf);

	        // if we are within the fan
          if (((ml == 0 && mr == 0) || y > 0 &&
              ((x*x)*(xs*xs)+(y*y)*(ys*ys) < d2 && x/y >= ml && x/y <= mr)))
            {  
            inPoint[0] = idX;
            inPoint[1] = idY;
            inPoint[2] = idZ;
            inPoint[3] = 1;

	          //recall matrix = the index matrix --> transform voxels in the slice to indices in the output
            //formula: outPoint = matrix * inPoint
            matrix->MultiplyPoint(inPoint,outPoint);
            
	          // deal with w (homogeneous transform) if the transform was a perspective transform
            outPoint[0] /= outPoint[3]; 
            outPoint[1] /= outPoint[3]; 
            outPoint[2] /= outPoint[3];
            outPoint[3] = 1;
        
	          // interpolation functions return 1 if the interpolation was successful, 0 otherwise
            int hit = interpolate(outPoint, inPtr, outPtr, accPtr, numscalars, 
                        outExt, outInc);
	    
	          // increment the number of pixels inserted
	          self->IncrementPixelCount(0, hit);
            }
          }

        inPtr += numscalars; 
        }
      inPtr += inIncY;
      }
    inPtr += inIncZ;
    }
}

//----------------------------------------------------------------------------
// InsertSlice
// Given an input and output region, execute the filter algorithm to fill the
// output from the input, with no optimization, for the first volume
// It just executes a switch statement to call the
// vtkFreehandUltrasound2InsertSlice method
// TODO InsertSlice doesn't currently support rotations or triggering
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::InsertSlice()
  {

  if (this->Triggering)
    {
    vtkErrorMacro(<< "Should not use InsertSlice() when triggering - use InsertSlice(phase) instead");
    return;
    }

  this->InsertSlice(0);
  }

//----------------------------------------------------------------------------
// InsertSlice
// Given an input and output region, execute the filter algorithm to fill the
// output from the input, with no optimization, for the ith volume
// It just executes a switch statement to call the
// vtkFreehandUltrasound2InsertSlice method
// TODO InsertSlice doesn't currently support rotations or triggering or
// discarding based on ECG
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::InsertSlice(int phase)
{
  // if we are optimizing by either splitting into x, y, z components or with
  // integer math, then run the optimized insert slice function instead
  if (this->GetOptimization())
    {
    this->OptimizedInsertSlice(phase);
    return;
    }

  // if we are not reconstructing at the moment, then get the output ready to
  // receive data
  if (this->ReconstructionThreadId == -1)
    {
    this->InternalExecuteInformation();
    }

  // if we need to clear, then clear
  if (this->NeedsClear)
    {
    this->InternalClearOutput();
    }

  // get the slice, output data, accumulation buffer, whole extent == input extent (from the
  // slice) and output extent (from this object)
  vtkImageData *inData = this->GetSlice();
  vtkImageData *outData = this->GetOutput(phase);
  vtkImageData *accData = this->AccumulationBuffers[phase];
  int *inExt = inData->GetWholeExtent();
  int *outExt = this->OutputExtent;

  // TODO note that in optimizedInsertSlice(), this only happens if we are not reconstructing
  // in real time... otherwise done in vtkReconstructionThread
  inData->SetUpdateExtentToWholeExtent();
  inData->Update();
  void *inPtr = inData->GetScalarPointerForExtent(inExt);
  void *outPtr = outData->GetScalarPointerForExtent(outExt);
  void *accPtr = NULL;
  
  if (this->Compounding)
    {
    accPtr = accData->GetScalarPointerForExtent(outExt);
    }
  else
    {
    accPtr = NULL;
    }
  
  // this filter expects that input is the same type as output.
  if (inData->GetScalarType() != outData->GetScalarType())
    {
    vtkErrorMacro(<< "InsertSlice: input ScalarType, " 
                  << inData->GetScalarType()
                  << ", must match out ScalarType " 
                  << outData->GetScalarType());
    return;
    }

  vtkMatrix4x4 *matrix = this->GetIndexMatrix();

  // the LastIndexMatrix is a copy of the most recent index matrix used
  if (this->LastIndexMatrix == 0)
    {
    this->LastIndexMatrix = vtkMatrix4x4::New();
    }
  this->LastIndexMatrix->DeepCopy(matrix);
  
  // Call the vtkFreehandUltrasound2InsertSlice method to actually insert the slice
  switch (inData->GetScalarType())
    {
    case VTK_SHORT:
      vtkFreehandUltrasound2InsertSlice(this, outData, (short *)(outPtr), 
                             (unsigned short *)(accPtr), 
                             inData, (short *)(inPtr), 
                             inExt, matrix);
      break;
    case VTK_UNSIGNED_SHORT:
      vtkFreehandUltrasound2InsertSlice(this,outData,(unsigned short *)(outPtr),
                             (unsigned short *)(accPtr), 
                             inData, (unsigned short *)(inPtr), 
                             inExt, matrix);
      break;
    case VTK_UNSIGNED_CHAR:
      vtkFreehandUltrasound2InsertSlice(this, outData,(unsigned char *)(outPtr),
                             (unsigned short *)(accPtr), 
                             inData, (unsigned char *)(inPtr), 
                             inExt, matrix);
      break;
    default:
      vtkErrorMacro(<< "InsertSlice: Unknown input ScalarType");
      return;
    }
}


//****************************************************************************
// RECONSTRUCTION EXECUTION - BASICS
//****************************************************************************

//----------------------------------------------------------------------------
// SetSliceAxes
// Set the slice axes - axes of the slice to insert into the reconstruction
// volume, relative to the (x,y,z) axes of the reconstruction volume itself
//----------------------------------------------------------------------------
vtkCxxSetObjectMacro(vtkFreehandUltrasound2,SliceAxes,vtkMatrix4x4);

//----------------------------------------------------------------------------
// SetSliceTransform
// Set the slice transform - together with the slice axes, transforms from the
// local coordinate system of the slice to the coordinate system of the output
// reconstruction volume
//----------------------------------------------------------------------------
vtkCxxSetObjectMacro(vtkFreehandUltrasound2,SliceTransform,vtkLinearTransform);

//----------------------------------------------------------------------------
// GetIndexMatrix
// The transform matrix supplied by the user converts output coordinates
// to input coordinates.  
// To speed up the pixel lookup, the following function provides a
// matrix which converts output pixel indices to input pixel indices.
//----------------------------------------------------------------------------
vtkMatrix4x4* vtkFreehandUltrasound2::GetIndexMatrix()
  {
  if (this->DiscardOutlierHeartRates)
    {
    vtkErrorMacro(<< "Should not use GetIndexMatrix() when discarding based on ECG - use GetIndexMatrix(phase) instead");
    return NULL;
    }

  return this->GetIndexMatrix(0);
  }


//----------------------------------------------------------------------------
// GetIndexMatrix
// The transform matrix supplied by the user converts output coordinates
// to input coordinates.  
// To speed up the pixel lookup, the following function provides a
// matrix which converts output pixel indices to input pixel indices.
// For use with discarding slices based on the ECG signal
//----------------------------------------------------------------------------
vtkMatrix4x4 *vtkFreehandUltrasound2::GetIndexMatrix(int phase)
{
  // first verify that we have to update the matrix
  if (this->IndexMatrix == NULL)
    {
    this->IndexMatrix = vtkMatrix4x4::New();
    }

  vtkFloatingPointType inOrigin[3];
  vtkFloatingPointType inSpacing[3];
  vtkFloatingPointType outOrigin[3];
  vtkFloatingPointType outSpacing[3];

  this->GetSlice()->GetSpacing(inSpacing);
  this->GetSlice()->GetOrigin(inOrigin);
  this->GetOutput(0)->GetSpacing(outSpacing);
  this->GetOutput(0)->GetOrigin(outOrigin);  
  
  vtkTransform *transform = vtkTransform::New();
  vtkMatrix4x4 *inMatrix = vtkMatrix4x4::New();
  vtkMatrix4x4 *outMatrix = vtkMatrix4x4::New();

  if (!this->DiscardOutlierHeartRates)
    {
    if (this->SliceAxes)
      {
		  // cutting this out prevents the transform from being applied
      transform->SetMatrix(this->GetSliceAxes());
      }

    if (this->SliceTransform)
      {
      transform->PostMultiply();
      transform->Concatenate(this->SliceTransform->GetMatrix());
      }
    }
  else
    {
    if (!this->SliceAxesBuffer)
      {
      vtkErrorMacro(<< "GetIndexMatrix(phase) doesn't have a slice axes buffer");
      return NULL;
      }
    if (this->SliceAxesBuffer[phase])
      {
      // cutting this out prevents the transform from being applied
      transform->SetMatrix(this->SliceAxesBuffer[phase]);
      }
    else
      {
      vtkErrorMacro(<< "GetIndexMatrix(phase) doesn't have a slice axes buffer at this phase");
      return NULL;
      }

    if (!this->SliceTransformBuffer)
      {
      vtkErrorMacro(<< "GetIndexMatrix(phase) doesn't have a slice transform buffer");
      return NULL;
      }
    if (this->SliceTransformBuffer[phase])
      {
      transform->PostMultiply();
      transform->Concatenate(this->SliceTransformBuffer[phase]->GetMatrix());
      }
    else
      {
      vtkErrorMacro(<< "GetIndexMatrix(phase) doesn't have a slice transform buffer at this phase");
      return NULL;
      }
    }
  
  // check to see if we have an identity matrix
  int isIdentity = vtkIsIdentityMatrix(transform->GetMatrix());

  // the outMatrix takes OutputData indices to OutputData coordinates,
  // the inMatrix takes InputData coordinates to InputData indices
  for (int i = 0; i < 3; i++) 
    {
    if (inSpacing[i] != outSpacing[i] || inOrigin[i] != outOrigin[i])
      {
      isIdentity = 0;
      }
    inMatrix->Element[i][i] = inSpacing[i];
    inMatrix->Element[i][3] = inOrigin[i];
    outMatrix->Element[i][i] = 1.0f/outSpacing[i];
    outMatrix->Element[i][3] = -outOrigin[i]/outSpacing[i];
    }

	// outMatrix * (sliceTransform * sliceAxes) * inMatrix
  if (!isIdentity)
    {
    transform->PostMultiply();
    transform->Concatenate(outMatrix);
    transform->PreMultiply();
    transform->Concatenate(inMatrix);
    }

  // save the transform's matrix in this->IndexMatrix
  transform->GetMatrix(this->IndexMatrix);
  
  transform->Delete();
  inMatrix->Delete();
  outMatrix->Delete();

  return this->IndexMatrix;
}

//----------------------------------------------------------------------------
// SetPixelCount
// Set the number of pixels inserted by a particular threadId
//----------------------------------------------------------------------------
void  vtkFreehandUltrasound2::SetPixelCount(int threadId, int count)
{
  if( threadId < 4 && threadId >= 0)
    {
    this->PixelCount[threadId] = count;
    }
}

//----------------------------------------------------------------------------
// IncrementPixelCount
// Increment the number of pixels inserted by a particular threadId by a
// particular increment value
//----------------------------------------------------------------------------
void  vtkFreehandUltrasound2::IncrementPixelCount(int threadId, int increment)
{
  if( threadId < 4 && threadId >= 0)
    {
    this->PixelCount[threadId] += increment;
    }
}

//----------------------------------------------------------------------------
// GetPixelCount
// Get the total number of pixels inserted
//----------------------------------------------------------------------------
int  vtkFreehandUltrasound2::GetPixelCount()
{
  return ( this->PixelCount[0] + this->PixelCount[1] +
	   this->PixelCount[2] + this->PixelCount[3] );
}

//----------------------------------------------------------------------------
// vtkReconstructionThread
// This function is run in a background thread to perform the reconstruction.
// By running it in the background, it doesn't interfere with the display
// of the partially reconstructed volume.
//----------------------------------------------------------------------------
static void *vtkReconstructionThread(struct ThreadInfoStruct *data)
{
  vtkFreehandUltrasound2 *self = (vtkFreehandUltrasound2 *)(data->UserData);

  double prevtimes[10];
  double currtime = 0;  // most recent timestamp
  double lastcurrtime = 0;  // previous timestamp
  double timestamp = 0;  // video timestamp, corrected for lag
  double videolag = self->GetVideoLag();
  int numOutputVolumes = self->GetNumberOfOutputVolumes();
  int i;

  for (i = 0; i < 10; i++)
    {
    prevtimes[i] = 0.0;
    }

  // the tracker tool provides the position of each inserted slice
  if (!self->GetTrackerTool())
    {
	  cout << "Couldn't find tracker tool " << endl;
    return NULL;
    }
  else
    {
	  cout << "Found Tracker Tool"<<endl;
    }

  vtkMatrix4x4 *matrix = self->GetSliceAxes();
  vtkTrackerBuffer *buffer = self->GetTrackerTool()->GetBuffer();
  // if reconstructing previous data, use backup buffer
  if (!self->RealTimeReconstruction)
    { 
    buffer = self->TrackerBuffer;
    }

  vtkVideoSource2 *video = self->GetVideoSource();
  
  vtkImageData *inData = self->GetSlice();
  
  // wait for video to start (i.e. wait for timestamp to change)
  if (video && self->RealTimeReconstruction)
    {
    while (lastcurrtime == 0 || currtime == lastcurrtime)
      {
      int clipExt[6];
	    self->GetClipExtent(clipExt, inData->GetOrigin(), inData->GetSpacing(),
			  inData->GetWholeExtent());
      // TODO 3DPanoramicVolumeReconstructor has SetUpdateExtent(clipExt) instead of
      // SetUpdateExtentToWholeExtent()
      inData->SetUpdateExtentToWholeExtent();
      inData->Update();
	    if (self->GetCompounding())
	      {
        for (int j = 0; j < numOutputVolumes; j++)
          {
		      self->GetAccumulationBuffer(j)->SetUpdateExtentToWholeExtent();
		      self->GetAccumulationBuffer(j)->Update();
          }
	      }

      lastcurrtime = currtime;
      currtime = video->GetFrameTimeStamp();
      double timenow = vtkTimerLog::GetUniversalTime();
      double sleepuntil = currtime + 0.010;
      if (sleepuntil > timenow)
        {
		    vtkThreadSleep(data, sleepuntil);
	      }
      }
    }

  // THE RECONSTRUCTION LOOP
  // loop continuously until reconstruction thread is halted
  double starttime = 0;
  vtkTransform* tempTransform = vtkTransform::New();
  vtkMatrix4x4* sliceAxesInverseMatrix = vtkMatrix4x4::New();
  int rot; // current rotation
  // current phase, for triggering
  // may not be correct the first time, but there's a test to make sure the
  // first slice is not inserted
  int phase = 0;
  int prevPhase; // previous phase, for triggering
  double ecgRate; // current ecg rate
  double minHR = self->GetMinAllowedHeartRate(); // min allowed ecg rate
  double maxHR = self->GetMaxAllowedHeartRate(); // max allowed ecg rate
  // wait until the buffer is full if we are discarding
  int numInsertedIntoBuffer = 0;
  int rotating = self->GetRotatingProbe();
  int triggering = self->GetTriggering();
  int discard = self->GetDiscardOutlierHeartRates();
  int savingTimestamps = self->GetSaveInsertedTimestamps();
  vtkImageClip* clipper = self->GetRotationClipper();
  vtkImageThreshold* thresholder = self->GetRotationThresholder();
  vtkSignalBox* signalBox = self->GetSignalBox();

  for (i = 0;;)
    {
    lastcurrtime = currtime;

    // update the slice data - if reconstructing in real time, this is the only place
    // where this->Slice is actually updated - this is because even though we grab
    // from the video source in multiple places, we only do the Update() here!  Video
    // source only copies from buffer to output on the updates()

    // TODO 3DPanoramicVolumeReconstructor has SetUpdateExtent(clipExt) instead of
    // SetUpdateExtentToWholeExtent()

    int clipExt[6];

    if (!discard)
      {
      self->GetClipExtent(clipExt, inData->GetOrigin(), inData->GetSpacing(),	inData->GetWholeExtent());
      inData->SetUpdateExtentToWholeExtent();
      inData->Update();
      }
    else
      {
      self->GetClipExtent(clipExt, inData->GetOrigin(), inData->GetSpacing(),	inData->GetWholeExtent());
      self->GetSliceBuffer(phase)->SetUpdateExtent(inData->GetWholeExtent());
      inData->Update();
      }

    // get the timestamp for the video frame data
	  if (video)
      {
      currtime = video->GetFrameTimeStamp();
      timestamp = currtime - videolag;
	    }

	  if (starttime == 0)
      {
		  starttime = timestamp;
      }

    // Get the tracking matrix, using videolag if it's nonzero
    // recall that matrix = this->SliceAxes
    buffer->Lock();
    int flags = 0;
    if (video && (videolag > 0.0 || !self->RealTimeReconstruction))
      {
      flags = buffer->GetFlagsAndMatrixFromTime(matrix, timestamp);
      }
    else
      {
      buffer->GetMatrix(matrix, 0);
      flags = buffer->GetFlags(0);
	    if (!video)
        {
		    currtime = buffer->GetTimeStamp(0);
	      }
      }
    buffer->Unlock();

    if (buffer->GetNumberOfItems() > buffer->GetBufferSize())
      {
      printf("Overflowing tracker buffer!\n");
      }

	  // get the rotation
    if (rotating)
      {
      clipper->SetInput(inData);
      thresholder->SetInput(clipper->GetOutput());
      thresholder->Update();
      rot = self->CalculateFanRotationValue(thresholder);

      // ignore rotations of -1 or rotation differences greater than 20
      if (rot > 0 && abs(self->GetPreviousFanRotation() - rot) < 20)
        {
	      self->SetPreviousFanRotation(self->GetFanRotation());
	      self->SetFanRotation(rot);
        }
      else
        {
        self->SetPreviousFanRotation(self->GetFanRotation());
        }
      }

    // if we are not rotating, then fan rotation and previous fan rotation are zero

	  // now use the rotation to change the SliceTransform (vtkTransform)
    vtkMatrix4x4::Invert(matrix, sliceAxesInverseMatrix);
	  if (self->GetSliceTransform())
	    {
		    // TODO the code assumes the image is flipped
		    if (self->GetFanRotation() != self->GetPreviousFanRotation())
		    {
          tempTransform = (vtkTransform *) (self->GetSliceTransform());
          tempTransform->Identity();
          tempTransform->RotateY(self->GetFanRotation());
          // formula: sliceAxes * sliceTransform(rotation) * inv(sliceAxes)
          tempTransform->PostMultiply();
          tempTransform->Concatenate(matrix);
          tempTransform->PreMultiply();
          tempTransform->Concatenate(sliceAxesInverseMatrix);
		    }
	    }

    // sleep until the next video frame if we don't have an updated time
    if (currtime == lastcurrtime && self->RealTimeReconstruction)
      {
      double timenow = vtkTimerLog::GetUniversalTime();
      double sleepuntil = currtime + 0.033;
      if (sleepuntil > timenow)
        {
        // return if abort occurred during sleep
		    if (vtkThreadSleep(data, sleepuntil) == 0)
          {
			    return NULL;
		      }
	      }
      }
    // sleep until the next video frame if tool is not tracking properly
    else if (flags & (TR_MISSING | TR_OUT_OF_VIEW))
      {
      //printf("out of view\n");
      double timenow = vtkTimerLog::GetUniversalTime();
      double sleepuntil = currtime + 0.033;
      if (sleepuntil > timenow)
        {
        // return if abort occurred during sleep
		    if (vtkThreadSleep(data, sleepuntil) == 0)
          {
			    return NULL;
		      }
	      }
	    }
    // do the reconstruction
    else
      {

      // we are not triggering at all
      if (!triggering)
        {
        self->InsertSlice();
        if (savingTimestamps)
          {
          self->InsertSliceTimestamp(0,timestamp);
          self->IncrementSliceTimestampCounter(0);
          }
        }

      // decide whether to insert a slice if we are triggering
      // we know we have a signal box because we couldn't have gotten here without it
      // (checked in StartRealTimeReconstruction and StartReconstruction)

      // we are triggering but not discarding heart rates
      else if (!discard)
        {
        phase = signalBox->GetPhase();
        prevPhase = self->GetCurrentPhase();
        self->SetPreviousPhase(prevPhase);
        self->SetCurrentPhase(phase);
        ecgRate = signalBox->GetBPMRate();

        // don't want prevPhase = -1 - otherwise first slice will pass this test
        if (ecgRate > 0 && prevPhase != -1 && phase != prevPhase)
          {
          self->InsertSlice(phase);
          if (savingTimestamps)
            {
            self->InsertSliceTimestamp(phase, timestamp);
            self->IncrementSliceTimestampCounter(phase);
            }
          }
        }
      // we are triggering and discarding heart rates
      else
        {
        phase = signalBox->GetPhase();
        prevPhase = self->GetCurrentPhase();
        self->SetPreviousPhase(prevPhase);
        self->SetCurrentPhase(phase);
        ecgRate = signalBox->GetBPMRate();

        // if this is a valid slice to deal with...
        if (ecgRate > 0 && prevPhase != -1 && phase != prevPhase)
          {

          // if we are at the start of a new phase, then insert the slices
          // from the previous cycle into the output if the ECG rate is valid
          // and the buffer is full
          if (phase == 0 && ecgRate >= minHR && ecgRate <= maxHR && numInsertedIntoBuffer >= numOutputVolumes)
            {
            for (int count = 0; count < numOutputVolumes; count++)
              {
              self->InsertSlice(count);
              if (savingTimestamps)
                {
                self->IncrementSliceTimestampCounter(count);
                }
              }
            }

          // copy this slice and associated transforms into the buffers
          // TODO I don't like how these are doing DeepCopy()'s, but we don't know the phase
          // until after we have already dealt with the slice transforms, so it's tricky to
          // decide which index to put them in.  For some reason, triggering earlier doesn't
          // work as well.
          // Also, we really should be doing a
          // VideoSource2::Update() to update the frame time stamps, check extent changes, etc
          // in the video source... if we were to do an update and then grab the frame we actually
          // want, then that would involve two copies anyways and just copying the output into 
          // the buffer is much cleaner code
          self->SetSliceBuffer(phase, inData);
          self->SetSliceAxesBuffer(phase, self->GetSliceAxes());
          self->SetSliceTransformBuffer(phase, self->GetSliceTransform());
          if (savingTimestamps)
            {
            self->InsertSliceTimestamp(phase, timestamp);
            }
          numInsertedIntoBuffer++;
          }
        }

		  // get current reconstruction rate over last 10 updates
		  double tmptime = currtime;

      // calculate frame rate using computer clock, not timestamps
		  if (!self->RealTimeReconstruction)
        { 
			  tmptime = vtkTimerLog::GetUniversalTime();
		    }
		  double difftime = tmptime - prevtimes[i%10];
		  prevtimes[i%10] = tmptime;
		  if (i > 10 && difftime != 0)
        {
			  self->ReconstructionRate = (10.0/difftime);
		    }
		  i++;
	    }

    // check to see if we are being told to quit 
    int activeFlag = *(data->ActiveFlag);
    if (activeFlag == 0)
      {
      return NULL;
      }

    if (!self->RealTimeReconstruction)
      {
      // sleep for a millisecond, just to give the main application
      // thread some time
      vtkSleep(0.001);

      if (video)
        {
		    if (--self->ReconstructionFrameCount == 0)
          {
			    return NULL;
		      }
		    video->Seek(1);
	      }
      }
    }
}

//----------------------------------------------------------------------------
// StartRealTimeReconstruction
// Start doing real-time reconstruction from the video source.
// This will spawn a thread that does the reconstruction in the
// background.
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::StartRealTimeReconstruction()
{

  // if the reconstruction isn't running
  if (this->ReconstructionThreadId == -1)
    {
    this->RealTimeReconstruction = 1; // we are doing realtime reconstruction
    
    // Setup the slice axes matrix
    vtkMatrix4x4 *matrix = vtkMatrix4x4::New();
    this->SetSliceAxes(matrix);
    matrix->Delete();

    // Check that the signal box is setup properly if we are triggering
    if (this->Triggering)
      {
      if (!this->TestBeforeReconstructingWithTriggering())
        {
        return;
        }
      }

    // Calculate the heart rate parameters if we are excluding invalid slices
    if (this->DiscardOutlierHeartRates)
      {
      int success = this->CalculateHeartRateParameters();
      if (!success)
        {
        vtkWarningMacro( << "Could not calculate mean heart rate - the patient's heart rate is fluctuating too much");
        return;
        }
      }

    // so that we don't insert the slices from when calculating heart rate parameters, if we did that
    // TODO need to incorporate into non-real time
    this->CurrentPhase = -1;
    this->PreviousPhase = -1;
    this->ClearSliceBuffers();

    // TODO need to incorporate rotation too? perhaps cancel rotation out so that
    // it's ok the next time round?

    this->InternalExecuteInformation();

    //this->ActiveFlagLock->Lock(); // TODO needed?
    this->ReconstructionThreadId = \
      this->Threader->SpawnThread((vtkThreadFunctionType)\
				  &vtkReconstructionThread,
				  this);
    //this->ActiveFlagLock->Unlock(); // TODO needed?
    }
}

//----------------------------------------------------------------------------
// StopRealTimeReconstruction
// Stop the reconstruction started with StartRealTimeReconstruction
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::StopRealTimeReconstruction()
{	
  // if a reconstruction is currently running
  if (this->ReconstructionThreadId != -1)
    {
	  this->ActiveFlagLock->Lock();
	  cout << "Thread : " << this->ReconstructionThreadId <<" should terminate"<<endl;
	  int killingThread = this->ReconstructionThreadId;
	  Sleep(2000);
	  this->Threader->TerminateThread(killingThread);
	  cout << "Thread : " << this->ReconstructionThreadId <<" terminated"<<endl;
	  this->ReconstructionThreadId = -1;
	  this->ActiveFlagLock->Unlock();
	  if (this->TrackerTool)
	    {
	      // the vtkTrackerBuffer should be locked before changing or
	      // accessing the data in the buffer if the buffer is being used from
	      // multiple threads
	      this->TrackerTool->GetBuffer()->Lock();
	      this->TrackerBuffer->DeepCopy(this->TrackerTool->GetBuffer());
	      this->TrackerTool->GetBuffer()->Unlock();
	    }
    }
}

//----------------------------------------------------------------------------
// StartReconstruction, not real time
// Start doing a reconstruction from the video frames stored
// in the VideoSource buffer.  You should first use 'Seek'
// on the VideoSource to rewind first.  Then the reconstruction
// will advance through n frames one by one until the
// reconstruction is complete.  The reconstruction
// is performed in the background.
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::StartReconstruction(int frames)
{
  if (frames <= 0)
    {
    return;
    }

  // If the reconstruction isn't running (reconstructionThreadId == -1)
  if (this->ReconstructionThreadId == -1)
    {
    fprintf(stderr, "Reconstruction Start\n");
    this->RealTimeReconstruction = 0; // doing buffered reconstruction
    this->ReconstructionFrameCount = frames;

    // Setup the slice axes matrix
    vtkMatrix4x4 *matrix = vtkMatrix4x4::New();
    this->SetSliceAxes(matrix);
    matrix->Delete();

    // Check that the signal box is setup properly if we are triggering
    if (this->Triggering)
      {
      if (!this->TestBeforeReconstructingWithTriggering())
        {
        return;
        }
      }

    for (int phase = 0; phase < this->GetNumberOfOutputPorts(); phase++)
      {
      this->GetOutput(phase)->Update();
      }
	
    this->ReconstructionThreadId = \
      this->Threader->SpawnThread((vtkThreadFunctionType)\
				  &vtkReconstructionThread,
				  this);
    }
}

//----------------------------------------------------------------------------
// StopReconstruction, not real time
// Stop the reconstruction started wtih StartReconstruction() - returns the
// number of frames remaining to be reconstructed
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::StopReconstruction()
{
  // if a reconstruction is running
  if (this->ReconstructionThreadId != -1)
    {
    cout << "Thread : " << this->ReconstructionThreadId <<" should terminate"<<endl;
    this->Threader->TerminateThread(this->ReconstructionThreadId);
    cout << "Thread : " << this->ReconstructionThreadId <<" terminated"<<endl;
    this->ReconstructionThreadId = -1;
    return this->ReconstructionFrameCount;
    }
  return 0;
}


//****************************************************************************
// CODE FOR ROTATING PROBES
//****************************************************************************

//----------------------------------------------------------------------------
// GetFanRepresentation
// Returns an integer corresponding to the rotation value shown in the pixels
// of a single digit within the grabbed ultrasound frame.  Returns 0 if it
// could not interpret the pixels
// TODO hard coded for adult TEE
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::GetFanRepresentation (vtkImageThreshold* threshold, int array[12])
{
	int list[6];
	int B = 0;
	int W = 1;
	int result = -1;

	for (int i = 0; i < 6; i++)
	  {
		list[i] = threshold->GetOutput()->GetScalarComponentAsFloat(array[2*i], array[2*i+1],0,0);
	  }

  if ((list[0] == B) && (list[1] == W) && (list[2] == W) && (list[3] == B) && (list[4] == W) && (list[5] == W))
	  {
    result = 0;
	  }
	else if ((list[0] == W) && (list[1] == B) && (list[2] == W) && (list[3] == W) && (list[4] == B) && (list[5] == B))
	  {
    result = 1;  
	  }
	else if ((list[0] == B) && (list[1] == W) && (list[2] == B) && (list[3] == B) && (list[4] == B) && (list[5] == B))
	  {
    result = 2;
	  }
	else if ((list[0] == B) && (list[1] == B) && (list[2] == W) && (list[3] == B) && (list[4] == B) && (list[5] == W))
	  {
    result = 3;
	  }
	else if ((list[0] == W) && (list[1] == W) && (list[2] == W) && (list[3] == W) && (list[4] == W) && (list[5] == W))
	  {    
		result = 4;
	  }
	else if ((list[0] == B) && (list[1] == B) && (list[2] == B) && (list[3] == B) && (list[4] == B) && (list[5] == W))
	  {
    result = 5;
	  }
  else if ((list[0] == W) && (list[1] == B) && (list[2] == B) && (list[3] == B) && (list[4] == W) && (list[5] == W))
	  {
    result = 6;
	  }
  else if ((list[0] == B) && (list[1] == W) && (list[2] == W) && (list[3] == B) && (list[4] == W) && (list[5] == B))
	  {
    result = 7;
	  }
  else if ((list[0] == B) && (list[1] == B) && (list[2] == B) && (list[3] == B) && (list[4] == W) && (list[5] == W))
	  {
    result = 8;
	  }
  else if ((list[0] == B) && (list[1] == B) && (list[2] == B) && (list[3] == W) && (list[4] == B) && (list[5] == W))
	  {
    result = 9;
	  }
	else
	  {
		result = -1;
	  }

	delete[] &list;
	return result;
}

//----------------------------------------------------------------------------
// CalculateFanRotationValue
// Finds the current fan rotation based on the frame grabbed US image
// TODO hard coded for adult TEE
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::CalculateFanRotationValue(vtkImageThreshold* threshold)
{
  // rotation digits (ex for rotation 158, d1 = 1, d2 = 5, d3 = 8
	int d1, d2, d3;

	// not flipped
	if (this->GetImageIsFlipped() == 0)
	  {
		// first rotation digit
		int array3[12] = {72+this->GetFanRotationXShift(), 479-294+this->GetFanRotationYShift(), 72+this->GetFanRotationXShift(), 479-298+this->GetFanRotationYShift(),
											75+this->GetFanRotationXShift(), 479-294+this->GetFanRotationYShift(), 75+this->GetFanRotationXShift(), 479-299+this->GetFanRotationYShift(),
											71+this->GetFanRotationXShift(), 479-298+this->GetFanRotationYShift(), 77+this->GetFanRotationXShift(), 479-298+this->GetFanRotationYShift()};
		d3 = this->GetFanRepresentation(threshold, array3);
		// second rotation digit
		int array2[12] = {62+this->GetFanRotationXShift(), 479-294+this->GetFanRotationYShift(), 62+this->GetFanRotationXShift(), 479-298+this->GetFanRotationYShift(),
												65+this->GetFanRotationXShift(), 479-294+this->GetFanRotationYShift(), 65+this->GetFanRotationXShift(), 479-299+this->GetFanRotationYShift(),
												61+this->GetFanRotationXShift(), 479-298+this->GetFanRotationYShift(), 67+this->GetFanRotationXShift(), 479-298+this->GetFanRotationYShift()};
		d2 = this->GetFanRepresentation(threshold, array2);
		// third rotation digit
		int array1[12] = {52+this->GetFanRotationXShift(), 479-294+this->GetFanRotationYShift(), 52+this->GetFanRotationXShift(), 479-298+this->GetFanRotationYShift(),
												55+this->GetFanRotationXShift(), 479-294+this->GetFanRotationYShift(), 55+this->GetFanRotationXShift(), 479-299+this->GetFanRotationYShift(),
												51+this->GetFanRotationXShift(), 479-298+this->GetFanRotationYShift(), 57+this->GetFanRotationXShift(), 479-298+this->GetFanRotationYShift()};
		d1 = this->GetFanRepresentation(threshold, array1);
	  }

	// flipped
	else
	  {
		// first rotation digit
		int array3[12] = {502+this->GetFanRotationXShift(), 479-126+this->GetFanRotationYShift(), 502+this->GetFanRotationXShift(), 479-130+this->GetFanRotationYShift(),
														505+this->GetFanRotationXShift(), 479-126+this->GetFanRotationYShift(), 505+this->GetFanRotationXShift(), 479-131+this->GetFanRotationYShift(),
														501+this->GetFanRotationXShift(), 479-130+this->GetFanRotationYShift(), 507+this->GetFanRotationXShift(), 479-130+this->GetFanRotationYShift()};
		d3 = this->GetFanRepresentation(threshold, array3);

		// second rotation digit
		int array2[12] = {492+this->GetFanRotationXShift(), 479-126+this->GetFanRotationYShift(), 492+this->GetFanRotationXShift(), 479-130+this->GetFanRotationYShift(),
														495+this->GetFanRotationXShift(), 479-126+this->GetFanRotationYShift(), 495+this->GetFanRotationXShift(), 479-131+this->GetFanRotationYShift(),
														491+this->GetFanRotationXShift(), 479-130+this->GetFanRotationYShift(), 497+this->GetFanRotationXShift(), 479-130+this->GetFanRotationYShift()};
		d2 = this->GetFanRepresentation(threshold, array2);

		// third rotation digit
		int array1[12] = {482+this->GetFanRotationXShift(), 479-126+this->GetFanRotationYShift(), 482+this->GetFanRotationXShift(), 479-130+this->GetFanRotationYShift(),
														485+this->GetFanRotationXShift(), 479-126+this->GetFanRotationYShift(), 485+this->GetFanRotationXShift(), 479-131+this->GetFanRotationYShift(),
														481+this->GetFanRotationXShift(), 479-130+this->GetFanRotationYShift(), 487+this->GetFanRotationXShift(), 479-130+this->GetFanRotationYShift()};
		d1 = this->GetFanRepresentation(threshold, array1);
	  }

  // combine rotation digits to return rotation

  if (d3 >= 0)
	  {
    if (d2 >=0)
		  {
      if (d1 >=0)
			  {
        return d1*100+d2*10+d3;
			  }
      else
			  {
        return d2*10+d3;
			  }
		  }
    else
		  {
      return d3;
		  }
	  }
	else
	  {
		return -1;
	  }
}

//****************************************************************************
// FILLING HOLES
//****************************************************************************

//----------------------------------------------------------------------------
// vtkFreehandUltrasound2FillHolesInOutput
// Does the actual hole filling
//----------------------------------------------------------------------------
template <class T>
static void vtkFreehandUltrasound2FillHolesInOutput(vtkFreehandUltrasound2 *self,
						   vtkImageData *outData,
						   T *outPtr,
						   unsigned short *accPtr,
						   int outExt[6])
{
  int idX, idY, idZ;
  int incX, incY, incZ;
  int accIncX, accIncY, accIncZ;
  int startX, endX, numscalars;
  int c;

  // clip the extent by 1 voxel width relative to whole extent
  int *outWholeExt = outData->GetWholeExtent();
  int extent[6];
  for (int a = 0; a < 3; a++)
    {
    extent[2*a] = outExt[2*a];
    if (extent[2*a] == outWholeExt[2*a])
      {
      extent[2*a]++;
      }
    extent[2*a+1] = outExt[2*a+1];
    if (extent[2*a+1] == outWholeExt[2*a+1])
      {
      extent[2*a+1]--;
      }
    }

  // get increments for output and for accumulation buffer
  outData->GetIncrements(incX, incY, incZ);
  accIncX = 1;
  accIncY = incY/incX;
  accIncZ = incZ/incX;
  // number of components not including the alpha channel
  numscalars = outData->GetNumberOfScalarComponents() - 1;
   
  T *alphaPtr = outPtr + numscalars;
  T *outPtrZ, *outPtrY, *outPtrX;
  unsigned short *accPtrZ, *accPtrY, *accPtrX;

  // go through all voxels except the edge voxels
  for (idZ = extent[4]; idZ <= extent[5]; idZ++)
    {
    outPtrZ = outPtr + (idZ - outExt[4])*incZ;
    accPtrZ = accPtr + (idZ - outExt[4])*accIncZ;

    for (idY = extent[2]; idY <= extent[3]; idY++)
      {
      outPtrY = outPtrZ + (idY - outExt[2])*incY;
      accPtrY = accPtrZ + (idY - outExt[2])*accIncY;

      // find entry point
      alphaPtr = outPtrY + numscalars;
      for (startX = outExt[0]; startX <= outExt[1]; startX++)
	      {
	      // check the point on the row as well as the 4-connected voxels
        // break when alpha component is nonzero
	      if (*alphaPtr |
	          *(alphaPtr-incY) | *(alphaPtr+incY) |
	          *(alphaPtr-incZ) | *(alphaPtr+incZ))
	        {
	        break;
	        }
	      alphaPtr += incX;
	      }

      if (startX > outExt[1])
	      { // the whole row is empty, do nothing
	      continue;
	      }

      // find exit point
      alphaPtr = outPtrY + (outExt[1]-outExt[0])*incX + numscalars;
      for (endX = outExt[1]; endX >= outExt[0]; endX--)
	      {
	      // check the point on the row as well as the 4-connected voxels 
	      if (*alphaPtr |
	          *(alphaPtr-incY) | *(alphaPtr+incY) |
	          *(alphaPtr-incZ) | *(alphaPtr+incZ))
	        {
	        break;
	        }
	      alphaPtr -= incX;
	      }

      // go through the row, skip first and last voxel in row
      if (startX == outWholeExt[0])
	      {
	      startX++;
	      }
      if (endX == outWholeExt[1])
	      {
	      endX--;
	      }
      outPtrX = outPtrY + (startX - outExt[0])*incX;
      accPtrX = accPtrY + (startX - outExt[0])*accIncX;
      
      for (idX = startX; idX <= endX; idX++)
	      {
        // only do this for voxels that haven't been hit
	      if (outPtrX[numscalars] == 0)
	        { 
	        double sum[32];
	        for (c = 0; c < numscalars; c++) 
	          {
	          sum[c] = 0;
	          }
	        double asum = 0; 
	        int n = 0;
	        int nmin = 14; // half of the connected voxels plus one
	        T *blockPtr;
	        unsigned short *accBlockPtr;

          // for accumulation buffer
	        // sum the pixel values for the 3x3x3 block
          // START TURNED OFF FOR NOW
	        if (0) // (accPtr)
	          { // use accumulation buffer to do weighted average
	          for (int k = -accIncZ; k <= accIncZ; k += accIncZ)
	            {
	            for (int j = -accIncY; j <= accIncY; j += accIncY)
		            {
		            for (int i = -accIncX; i <= accIncX; i += accIncX)
		              {
		              int inc = j + k + i;
		              blockPtr = outPtrX + inc*incX;
		              accBlockPtr = accPtrX + inc;
		              if (blockPtr[numscalars] == 255)
		                {
		                n++;
		                for (c = 0; c < numscalars; c++)
		                  { // use accumulation buffer as weight
		                  sum[c] += blockPtr[c]*(*accBlockPtr);
		                  }
		                asum += *accBlockPtr;
		                }
		              }
		            }
	            }
	          
            // if less than half the neighbors have data, use larger block
	          if (n <= nmin && idX != startX && idX != endX &&
		            idX - outWholeExt[0] > 2 && outWholeExt[1] - idX > 2 &&
		            idY - outWholeExt[2] > 2 && outWholeExt[3] - idY > 2 &&
		            idZ - outWholeExt[4] > 2 && outWholeExt[5] - idZ > 2)
	            {
	            // weigh inner block by a factor of four (multiply three,
	            // plus we will be counting it again as part of the 5x5x5
	            // block)
	            asum *= 3;
	            for (c = 0; c < numscalars; c++) 
		            {
		            sum[c]*= 3;
		            }	      
	            nmin = 63;
	            n = 0;
	            for (int k = -accIncZ*2; k <= accIncZ*2; k += accIncZ)
		            {
		            for (int j = -accIncY*2; j <= accIncY*2; j += accIncY)
		              {
		              for (int i = -accIncX*2; i <= accIncX*2; i += accIncX)
		                {
		                int inc = j + k + i;
		                blockPtr = outPtrX + inc*incX;
		                accBlockPtr = accPtrX + inc;
                    // use accumulation buffer as weight
		                if (blockPtr[numscalars] == 255)
		                  { 
		                  n++;
		                  for (c = 0; c < numscalars; c++)
			                  {
			                  sum[c] += blockPtr[c]*(*accBlockPtr);
			                  }
		                  asum += *accBlockPtr; 
		                  }
		                }
		              }
		            }
	            }
	          }
          // END TURNED OFF FOR NOW

          // no accumulation buffer
	        else 
	          {
	          for (int k = -incZ; k <= incZ; k += incZ)
	            {
	            for (int j = -incY; j <= incY; j += incY)
		            {
		            for (int i = -incX; i <= incX; i += incX)
		              {
		              blockPtr = outPtrX + j + k + i;
		              if (blockPtr[numscalars] == 255)
		                {
		                n++;
		                for (int c = 0; c < numscalars; c++)
		                  {
		                  sum[c] += blockPtr[c];
		                  }
		                }
		              }
		            }
	            }
	          asum = n;
	    
            // if less than half the neighbors have data, use larger block,
	          // and count inner 3x3 block again to weight it by 2
	          if (n <= nmin && idX != startX && idX != endX &&
		            idX - outWholeExt[0] > 2 && outWholeExt[1] - idX > 2 &&
		            idY - outWholeExt[2] > 2 && outWholeExt[3] - idY > 2 &&
		            idZ - outWholeExt[4] > 2 && outWholeExt[5] - idZ > 2)
	            { 
	            // weigh inner block by a factor of four (multiply three,
	            // plus we will be counting it again as part of the 5x5x5
	            // block)
	            asum *= 3;
	            for (c = 0; c < numscalars; c++) 
		            {
		            sum[c]*= 3;
		            }
	            nmin = 63;
	            n = 0;
	            for (int k = -incZ*2; k <= incZ*2; k += incZ)
		            {
		            for (int j = -incY*2; j <= incY*2; j += incY)
		              {
		              for (int i = -incX*2; i <= incX*2; i += incX)
		                {
		                blockPtr = outPtrX + j + k + i;
		                if (blockPtr[numscalars] == 255)
		                  {
		                  n++;
		                  for (int c = 0; c < numscalars; c++)
			                  {
			                  sum[c] += blockPtr[c];
			                  }
		                  }
		                }
		              }
		            }
	            asum += n;
              }
	          }

	        // if more than half of neighboring voxels are occupied, then fill
	        if (n >= nmin)
	          {
	          for (int c = 0; c < numscalars; c++)
	            {
	            vtkUltraRound(sum[c]/asum, outPtrX[c]);
	            }
	          // set alpha to 1 now, change to 255 later
	          outPtrX[numscalars] = 1;
	          }
	        }
	        outPtrX += incX;
	      }
      }
    }

  // change alpha value '1' to value '255'
  alphaPtr = outPtr + numscalars;
  // go through all voxels this time
  for (idZ = outExt[4]; idZ <= outExt[5]; idZ++)
    {
    for (idY = outExt[2]; idY <= outExt[3]; idY++)
      {
      for (idX = outExt[0]; idX <= outExt[1]; idX++)
	      {
	      // convert '1' to 255
	      if (*alphaPtr == 1)
	        {
	        *alphaPtr = 255;
	        }
	      alphaPtr += incX;
	      }
      // add the continuous increment
      alphaPtr += (incY - (outExt[1]-outExt[0]+1)*incX);
      }
    // add the continuous increment
    alphaPtr += (incZ - (outExt[3]-outExt[2]+1)*incY);
    }
}

//----------------------------------------------------------------------------
// ThreadedFillExecute
// Calls vtkFreehandUltrasound2FillHolesInOutput, with templating for different
// types
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::ThreadedFillExecute(vtkImageData *outData,	
                                            	int outExt[6], int threadId, int phase)
{
  vtkImageData *accData = this->AccumulationBuffers[phase];
  void *outPtr = outData->GetScalarPointerForExtent(outExt);
  void *accPtr = NULL;
  
  if (this->Compounding)
    {
    accPtr = accData->GetScalarPointerForExtent(outExt);
    }

  switch (outData->GetScalarType())
    {
    case VTK_SHORT:
      vtkFreehandUltrasound2FillHolesInOutput(
                             this, outData, (short *)(outPtr), 
                             (unsigned short *)(accPtr), outExt);
      break;
    case VTK_UNSIGNED_SHORT:
      vtkFreehandUltrasound2FillHolesInOutput(
                             this, outData, (unsigned short *)(outPtr),
                             (unsigned short *)(accPtr), outExt);
      break;
    case VTK_UNSIGNED_CHAR:
      vtkFreehandUltrasound2FillHolesInOutput(
                             this, outData,(unsigned char *)(outPtr),
                             (unsigned short *)(accPtr), outExt); 
      break;
    default:
      vtkErrorMacro(<< "FillHolesInOutput: Unknown input ScalarType");
      return;
    }
}

//----------------------------------------------------------------------------
// vtkFreehand2ThreadedFillExecute
// This mess is really a simple function. All it does is call
// the ThreadedExecute method after setting the correct
// extent for this thread.
//----------------------------------------------------------------------------
VTK_THREAD_RETURN_TYPE vtkFreehand2ThreadedFillExecute( void *arg )
{
  vtkFreehand2ThreadStruct *str;
  int ext[6], splitExt[6], total;
  int threadId, threadCount;
  vtkImageData *output;

  threadId = ((ThreadInfoStruct *)(arg))->ThreadID;
  threadCount = ((ThreadInfoStruct *)(arg))->NumberOfThreads;

  str = (vtkFreehand2ThreadStruct *)(((ThreadInfoStruct *)(arg))->UserData);
  output = str->Output;
  output->GetExtent(ext);

  // execute the actual method with appropriate extent
  // first find out how many pieces extent can be split into.
  total = str->Filter->SplitSliceExtent(splitExt, ext, threadId, threadCount);
  
  if (threadId < total)
    {
    str->Filter->ThreadedFillExecute(str->Output, splitExt, threadId, str->Phase);
    }
  // else
  //   {
  //   otherwise don't use this thread. Sometimes the threads dont
  //   break up very well and it is just as efficient to leave a 
  //   few threads idle.
  //   }
  
  return VTK_THREAD_RETURN_VALUE;
}

//----------------------------------------------------------------------------
// MultiThreadFill
// Setup threading, and call vtkFreehand2ThreadedFillExecute
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::MultiThreadFill(vtkImageData *outData, int phase)
{
  vtkFreehand2ThreadStruct str;
  
  str.Filter = this;
  str.Input = 0;
  str.Output = outData;
  str.Phase = phase;
  
  this->Threader->SetNumberOfThreads(this->NumberOfThreads);
  
  // setup threading and the invoke threadedExecute
  this->Threader->SetSingleMethod(vtkFreehand2ThreadedFillExecute, &str);
  this->Threader->SingleMethodExecute();
}

//----------------------------------------------------------------------------
// FillHolesInOutput
// Fills holes in output by using the weighted average of the surrounding
// voxels (see Gobbi's thesis to find how it works)
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::FillHolesInOutput()
{
  this->UpdateInformation();
  if (this->NeedsClear)
    {
    this->InternalClearOutput();
    }

  for (int phase = 0; phase < this->GetNumberOfOutputPorts(); phase++)
    {
    vtkImageData *outData = this->GetOutput(phase);
    this->MultiThreadFill(outData, phase);
    }

  this->Modified(); 
}


//****************************************************************************
// Triggering (ECG-gating)
//****************************************************************************

//----------------------------------------------------------------------------
// SetSignalBox
// Sets the signal box for the triggering
// Automatically sets the number of output volumes to the number of phases as
// a side effect
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetSignalBox(vtkSignalBox *signalBox)
  {
  if (signalBox->GetNumberOfPhases() <= 1)
    {
    vtkErrorMacro(<< "The signal box must have at least 2 phases");
    return;
    }

  this->SignalBox = signalBox;
  this->SetNumberOfOutputVolumes(signalBox->GetNumberOfPhases());
  }

//----------------------------------------------------------------------------
// SetNumberOfOutputVolumes
// Sets the number of output volumes, for use with triggering
// Has the side effect of clearing the old accumulation buffers
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetNumberOfOutputVolumes(int num)
  {
  // sanity check
  if (num <= 0)
    {
    return;
    }

  if (this->SignalBox && num != this->SignalBox->GetNumberOfPhases())
    {
    vtkErrorMacro(<< "The number of output volumes must be equal to the number of phases in the signal box");
    return;
    }

  if (num == this->NumberOfOutputVolumes)
    {
    return;
    }

  // create the output objects via the VTK 5 pipeline
  this->SetNumberOfOutputPorts(num);

  // create the accumulation buffers
  vtkImageData **newAccumulationBuffers;
  int i;

  // if we are creating the accumulation buffers for the first time
  if (this->AccumulationBuffers == NULL)
    {
    this->AccumulationBuffers = new vtkImageData*[num];
    for (int i = 0; i < num; i++)
      {
      this->AccumulationBuffers[i] = vtkImageData::New();
      }
    }

  // if we already have accumulation buffers and are resizing
  // TODO check with example numbers to make sure that this makes sense!
  else
    {
    newAccumulationBuffers = new vtkImageData*[num];

    // copy over old accumulation buffers
    if (num > this->NumberOfOutputVolumes)
      {
      for (i = 0; i < this->NumberOfOutputVolumes; i++)
        {
        newAccumulationBuffers[i] = this->AccumulationBuffers[i];
        }
      }
    else
      {
      for (i = 0; i < num; i++)
        {
        newAccumulationBuffers[i] = this->AccumulationBuffers[i];
        }
      }

    // create new image buffers if necessary
    for (i = this->NumberOfOutputVolumes; i < num; i++)
      {
      newAccumulationBuffers[i] = vtkImageData::New();
      }

    // delete image buffers we no longer need
    for (i = num; i < this->NumberOfOutputVolumes; i++)
      {
      this->AccumulationBuffers[i]->Delete();
      }

    // delete the old array of accumulation buffers
    if (this->AccumulationBuffers)
      {
      delete [] this->AccumulationBuffers;
      }

    // and reset the array of accumulation buffers to the new one
    this->AccumulationBuffers = newAccumulationBuffers;
    }

  // if we are discarding invalid ECG signals, then create buffers
  if (this->DiscardOutlierHeartRates)
    {
    if (this->SliceBuffer)
      {
      for (i = 0; i < this->NumberOfOutputVolumes; i++)
        {
        if (this->SliceBuffer[i])
          {
          this->SliceBuffer[i]->Delete();
          }
        }
      delete [] this->SliceBuffer;
      this->SliceBuffer = NULL;
      }

    if (this->SliceAxesBuffer)
      {
      for (i = 0; i < this->NumberOfOutputVolumes; i++)
        {
        if (this->SliceAxesBuffer[i])
          {
          this->SliceAxesBuffer[i]->Delete();
          }
        }
      delete [] this->SliceAxesBuffer;
      this->SliceAxesBuffer = NULL;
      }

    if (this->SliceTransformBuffer)
      {
      for (i = 0; i < this->NumberOfOutputVolumes; i++)
        {
        if (this->SliceTransformBuffer[i])
          {
          this->SliceTransformBuffer[i]->Delete();
          }
        }
      delete [] this->SliceTransformBuffer;
      this->SliceTransformBuffer = NULL;
      }

    // create the new buffers
    this->SliceBuffer = new vtkImageData*[num];
    for (i = 0; i < num; i++)
      {
      this->SliceBuffer[i] = vtkImageData::New();
      }
    this->SliceAxesBuffer = new vtkMatrix4x4*[num];
    for (i = 0; i < num; i++)
      {
      this->SliceAxesBuffer[i] = vtkMatrix4x4::New();
      }
    this->SliceTransformBuffer = new vtkLinearTransform*[num];
    for (i = 0; i < num; i++)
      {
      this->SliceTransformBuffer[i] = vtkTransform::New();
      }
    }

  // if saving insertion times, then clear the old buffer and recreate
  if (this->SaveInsertedTimestamps)
    {
    // delete the old buffer
    if (this->InsertedTimestampsBuffer)
      {
      for (i = 0; i < this->NumberOfOutputVolumes; i++)
        {
        if (this->InsertedTimestampsBuffer[i])
          {
          delete [] this->InsertedTimestampsBuffer[i];
          }
        }
      delete [] this->InsertedTimestampsBuffer;
      this->InsertedTimestampsBuffer = NULL;
      }
    if (this->InsertedTimestampsCounter)
      {
      delete [] this->InsertedTimestampsCounter;
      }

    // create the new buffer
    int numPerPhase = this->MaximumNumberOfInsertionsPerPhase;
    this->InsertedTimestampsBuffer = new double*[num];
    for (i = 0; i < num; i++)
      {
      this->InsertedTimestampsBuffer[i] = new double[numPerPhase];
      }
    this->InsertedTimestampsCounter = new int[num];
    }

  // set the number of output volumes
  this->NumberOfOutputVolumes = num;

  }

//----------------------------------------------------------------------------
// SetSliceBuffer
// Inserts an image into the buffer, for when discarding invalid ECG's
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetSliceBuffer(int phase, vtkImageData* inData)
  {
  if (!this->DiscardOutlierHeartRates || phase < 0 || phase >= this->NumberOfOutputVolumes || !this->SliceBuffer)
    {
    return;
    }

  this->SliceBuffer[phase]->DeepCopy(inData);
  }

//----------------------------------------------------------------------------
// GetSliceBuffer
// Gets an image from the slice buffer, for when discarding invalid ECG's
//----------------------------------------------------------------------------
vtkImageData* vtkFreehandUltrasound2::GetSliceBuffer(int phase)
  {
  if (!this->DiscardOutlierHeartRates || phase < 0 || phase >= this->NumberOfOutputVolumes || !this->SliceBuffer)
    {
    return NULL;
    }

  return this->SliceBuffer[phase];
  }

//----------------------------------------------------------------------------
// SetSliceAxesBuffer
// Inserts a matrix into the slice axes buffer, for when discarding invalid
// ECG's
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetSliceAxesBuffer(int phase, vtkMatrix4x4* matrix)
  {
  if (!this->DiscardOutlierHeartRates || phase < 0 || phase >= this->NumberOfOutputVolumes || !this->SliceAxesBuffer)
    {
    return;
    }

  this->SliceAxesBuffer[phase]->DeepCopy(matrix);
  }

//----------------------------------------------------------------------------
// GetSliceAxesBuffer
// Gets a matrix from the buffer, for when discarding invalid ECG's
//----------------------------------------------------------------------------
vtkMatrix4x4* vtkFreehandUltrasound2::GetSliceAxesBuffer(int phase)
  {
  if (!this->DiscardOutlierHeartRates || phase < 0 || phase >= this->NumberOfOutputVolumes || !this->SliceAxesBuffer)
    {
    return NULL;
    }
  
  return this->SliceAxesBuffer[phase];
  }


//----------------------------------------------------------------------------
// SetSliceTransformBuffer
// Inserts a transform into the buffer, for when discarding invalid ECG's
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetSliceTransformBuffer(int phase, vtkLinearTransform* transform)
  {
  if (!this->DiscardOutlierHeartRates || phase < 0 || phase >= this->NumberOfOutputVolumes || !this->SliceTransformBuffer)
    {
    return;
    }

  this->SliceTransformBuffer[phase]->DeepCopy(transform);
  }

//----------------------------------------------------------------------------
// GetSliceTransformBuffer
// Gets a transform from the buffer, for when discarding invalid ECG's
//----------------------------------------------------------------------------
vtkLinearTransform* vtkFreehandUltrasound2::GetSliceTransformBuffer(int phase)
  {
  if (!this->DiscardOutlierHeartRates || phase < 0 || phase >= this->NumberOfOutputVolumes || !this->SliceTransformBuffer)
    {
    return NULL;
    }

  return this->SliceTransformBuffer[phase];
  }

//----------------------------------------------------------------------------
// TestBeforeReconstructingWithTriggering
// Makes sure that the signal box is running before starting the reconstruction
// (will start it for you if it's not already running)
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::TestBeforeReconstructingWithTriggering()
  {
  if (this->Triggering)
    {
    // Setup the signal box if necessary
    if (!this->SignalBox)
      {
      vtkErrorMacro(<< "you said to use triggering but did not supply a signal box");
      return 0;
      }
    else
      {
      if (!this->SignalBox->GetIsStarted())
        {
        vtkWarningMacro( << "tried to start the reconstruction before you started the signal box - starting signal box for you");
        this->SignalBox->Initialize();
        this->SignalBox->Start();
        }
      // if we still couldn't start...
      if (!this->SignalBox->GetIsStarted())
        {
        vtkErrorMacro(<< "I could not start the signal box for you");
        return 0;
        }
      }

    // makes sure that the number of output volumes is equal to the number
    // of phases captured by the signal box
    if (this->NumberOfOutputVolumes != this->SignalBox->GetNumberOfPhases())
      {
      vtkErrorMacro(<< "The number of output volumes must be equal to the number of phases in the signal box");
      return 0;
      }

    // makes sure we have at least two volumes (otherwise nothing gets
    // inserted because the phase never changes
    if (this->NumberOfOutputVolumes <= 1)
      {
      vtkErrorMacro(<< "The number of output volumes must be at least two");
      return 0;
      }
    }
  return 1;
  }

//****************************************************************************
// For discarding invalid heart rates with ECG information
//****************************************************************************

//----------------------------------------------------------------------------
// SetDiscardOutlierHeartRates
// Sets whether or not we are discarding outlier heart rates, and creates/
// deletes the buffers as necessary
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetDiscardOutlierHeartRates(int discard)
  {
  if ((discard && this->DiscardOutlierHeartRates) || (!discard && !this->DiscardOutlierHeartRates))
    {
    return;
    }

  int num = this->NumberOfOutputVolumes;
  int i;
  
  // create the buffers if we are discarding for the first time
  if (discard)
    {
    this->SliceBuffer = new vtkImageData*[num];
    for (i = 0; i < num; i++)
      {
      this->SliceBuffer[i] = vtkImageData::New();
      }
    this->SliceAxesBuffer = new vtkMatrix4x4*[num];
    for (i = 0; i < num; i++)
      {
      this->SliceAxesBuffer[i] = vtkMatrix4x4::New();
      }
    this->SliceTransformBuffer = new vtkLinearTransform*[num];
    for (i = 0; i < num; i++)
      {
      this->SliceTransformBuffer[i] = vtkTransform::New();
      }
    }

  // delete the buffers if we are no longer discarding
  if (!discard)
    {
    if (this->SliceBuffer)
       {
       for (i = 0; i < num; i++)
         {
         if (this->SliceBuffer[i])
           {
           this->SliceBuffer[i]->Delete();
           }
         }
       delete [] this->SliceBuffer;
       this->SliceBuffer = NULL;
       }

    if (this->SliceAxesBuffer)
      {
      for (i = 0; i < num; i++)
        {
        if (this->SliceAxesBuffer[i])
          {
          this->SliceAxesBuffer[i]->Delete();
          }
        }
      delete [] this->SliceAxesBuffer;
      this->SliceAxesBuffer = NULL;
      }

     if (this->SliceTransformBuffer)
       {
       for (i = 0; i < num; i++)
         {
         if (this->SliceTransformBuffer[i])
           {
           this->SliceTransformBuffer[i]->Delete();
           }
         }
       delete [] this->SliceTransformBuffer;
       this->SliceTransformBuffer = NULL;
       }
    }

  this->DiscardOutlierHeartRates = discard;

  }

//----------------------------------------------------------------------------
// Calculate the mean heart rate over the time specified by ECGMonitoringTime,
// and find the maximum and minimum heart rates allowed (as determined by
// PercentageIncreasedHeartRateAllowed and PercentageDecreasedHeartRateAllowed
// Returns whether or not it was successful
//----------------------------------------------------------------------------
int vtkFreehandUltrasound2::CalculateHeartRateParameters()
  {

  if (!this->DiscardOutlierHeartRates || !this->Triggering || !this->SignalBox)
    {
    return 0;
    }

  int haveConsistentHeartRateMeasurement = 0;
  double valsPerSecond = 10.0;
  int numVals = valsPerSecond * this->ECGMonitoringTime;
  double sleepTime = 1.0 / valsPerSecond;
  int i;
  double thisHR, meanHR, maxHR, minHR;
  double maxAllowedHR, minAllowedHR;
  int numTrials = 0;

  // Calculate the mean heart rate and make sure it's consistent enough before
  // accepting it
  while (!haveConsistentHeartRateMeasurement && numTrials < this->NumECGMeasurementTrials)
    {
    printf("starting trial %d\n", numTrials);
    meanHR = 0;
    minHR = 1000.0; // something unbelievably high sot hat the first test will pass
    maxHR = 0.0; // something unbelievably low so that the first test will pass
    for (i = 0; i < numVals; i++)
      {
      thisHR = this->SignalBox->GetBPMRate();
      if (thisHR != -1)
        {
        meanHR += thisHR;
        if (thisHR < minHR)
          {
          minHR = thisHR;
          }
        if (thisHR > maxHR)
          {
          maxHR = thisHR;
          }
        }
      vtkSleep(sleepTime);
      }
    meanHR = meanHR / numVals;
    maxAllowedHR = meanHR + (meanHR * this->PercentageIncreasedHeartRateAllowed / 100.0);
    minAllowedHR = meanHR - (meanHR * this->PercentageDecreasedHeartRateAllowed / 100.0);
    if (maxHR <= maxAllowedHR && minHR >= minAllowedHR)
      {
      haveConsistentHeartRateMeasurement = 1;
      }
    else
      {
      printf("trial %d is bad, trying again (mean %f, max %f, min %f)\n", numTrials, meanHR, maxHR, minHR);
      }
        
    numTrials++;
    }

  if (haveConsistentHeartRateMeasurement)
    {
    // set the heart rate parameters
    this->MeanHeartRate = meanHR;
    this->MaxAllowedHeartRate = maxAllowedHR;
    this->MinAllowedHeartRate = minAllowedHR;
    }
  
  printf("mean heart rate = %f\n", meanHR);
  printf("min heart rate allowed = %f\n", minAllowedHR);
  printf("max heart rate allowed = %f\n", maxAllowedHR);

  return haveConsistentHeartRateMeasurement;
  }


//****************************************************************************
// For saving inserted timestamps
//****************************************************************************

//----------------------------------------------------------------------------
// SetSaveInsertedTimestamps
// If turned on, keeps a record of the timestamps used to insert slices in a
// 2D double array
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetSaveInsertedTimestamps(int insert)
  {
  if ((insert && this->SaveInsertedTimestamps) || (!insert && !this->SaveInsertedTimestamps))
    {
    return;
    }

  int num = this->NumberOfOutputVolumes;
  int numPerPhase = this->MaximumNumberOfInsertionsPerPhase;

  // want to delete the old buffer regardless
  if (this->InsertedTimestampsBuffer)
    {
    for (int phase = 0; phase < num; phase++)
      {
      if (this->InsertedTimestampsBuffer[phase])
        {
        delete [] this->InsertedTimestampsBuffer[phase];
        }
      }
    delete [] this->InsertedTimestampsBuffer;
    }
  if (this->InsertedTimestampsCounter)
    {
    delete [] this->InsertedTimestampsCounter;
    }

  // create the new buffer if necessary
  if (insert)
    {
    this->InsertedTimestampsBuffer = new double*[num];
    for (int phase = 0; phase < num; phase++)
      {
      this->InsertedTimestampsBuffer[phase] = new double[numPerPhase];
      }
    this->InsertedTimestampsCounter = new int[num];
    }
    
  this->SaveInsertedTimestamps = insert;
  }

//----------------------------------------------------------------------------
// SetMaximumNumberOfInsertionsPerPhase
// Need to give the class an idea of how many insertions are expected per phase,
// so that we can allocate the 2D array that holds the timestamps
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SetMaximumNumberOfInsertionsPerPhase(int numPerPhase)
  {
  if (numPerPhase == this->MaximumNumberOfInsertionsPerPhase || numPerPhase < 1)
    {
    return;
    }

  this->MaximumNumberOfInsertionsPerPhase = numPerPhase;

  if (this->SaveInsertedTimestamps)
    {
    // delete the old buffer
    int num = this->NumberOfOutputVolumes;
    if (this->InsertedTimestampsBuffer)
      {
      for (int phase = 0; phase < num; phase++)
        {
        if (this->InsertedTimestampsBuffer[phase])
          {
          delete [] this->InsertedTimestampsBuffer[phase];
          }
        }
      delete [] this->InsertedTimestampsBuffer;
      }

    // create the new buffer
    this->InsertedTimestampsBuffer = new double*[num];
    for (int phase = 0; phase < num; phase++)
      {
      this->InsertedTimestampsBuffer[phase] = new double[numPerPhase];
      }
    }
  }

//----------------------------------------------------------------------------
// InsertSliceTimestamp
// Insert a timestamp into the buffer, if we are saving timestamps (note that
// the timestamp counter is handled in the vtkReconstructionThread, not here)
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::InsertSliceTimestamp(int phase, double timestamp)
  {
  if (this->SaveInsertedTimestamps)
    {
    int index = this->InsertedTimestampsCounter[phase];
    this->InsertedTimestampsBuffer[phase][index] = timestamp;
    }
  }

//----------------------------------------------------------------------------
// GetSliceTimestamp
// Get a timestamp from the buffer, if we are saving timestamps (note that
// the timestamp counter is handled int he vtkReconstructionThread, not here)
//----------------------------------------------------------------------------
double vtkFreehandUltrasound2::GetSliceTimestamp(int phase)
  {
  if (this->SaveInsertedTimestamps)
    {
    int index = this->InsertedTimestampsCounter[phase];
    return this->InsertedTimestampsBuffer[phase][index];
    }
  return -1;
  }

//----------------------------------------------------------------------------
// IncrementSliceTimestampCounter
// Simply increments the counter that keeps track of where we are in the
// timestamps array
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::IncrementSliceTimestampCounter(int phase)
  {
  if (this->InsertedTimestampsCounter[phase] == (this->MaximumNumberOfInsertionsPerPhase-1))
    {
    printf("Overflowing time buffer!\n");
    }

  if (this->SaveInsertedTimestamps)
    {
    this->InsertedTimestampsCounter[phase]++;
    }
  }

//****************************************************************************
// I/0
// TODO need IO for triggering
//****************************************************************************

//----------------------------------------------------------------------------
// vtkJoinPath2
// Combines a directory and a file to make a complete path
// directory = the directory, file = the filename, n = the number of characters
// in the array cp, result is stored in cp
//----------------------------------------------------------------------------
char *vtkJoinPath2(char *cp, int n, const char *directory, const char *file)
{
  int dlen = strlen(directory);
  int flen = strlen(file);

  if (n < (dlen + flen + 2))
    {
    return 0;
    }

  strncpy(cp,directory,n);
#ifdef _WIN32
  strncpy(cp+dlen,"\\",n-dlen);
#else
  strncpy(cp+dlen,"/",n-dlen);
#endif
  strncpy(cp+dlen+1,file,n-dlen);

  return cp;
}

//----------------------------------------------------------------------------
// vtkFreehandUltrasound2EatWhitespace
// Eats leading whitespace
//----------------------------------------------------------------------------
char *vtkFreehandUltrasound2EatWhitespace(char *text)
{
  int i = 0;

  for (i = 0; i < 128; i++)
    {
    switch (*text)
      {
      case ' ':
      case '\t':
      case '\r':
      case '\n':
        text++;
        break;
      default:
        return text;
        break;
      }
    }

  return 0;
}

//----------------------------------------------------------------------------
// vtkFreehandUltrasound2EatWhitespace
// Eats leading whitespace
//----------------------------------------------------------------------------
char *vtkFreehandUltrasound2EatWhitespaceWithEquals(char *text)
{
  int i = 0;

  for (i = 0; i < 128; i++)
    {
    switch (*text)
      {
      case ' ':
      case '\t':
      case '\r':
      case '\n':
      case '=':
        text++;
        break;
      default:
        return text;
        break;
      }
    }

  return 0;
}

//----------------------------------------------------------------------------
// vtkExtractArrayComponentsFromString
// T is either integer or double
//----------------------------------------------------------------------------
template <class T>
int vtkExtractArrayComponentsFromString(char *text, T *arrayToFill, int numIndices)
  {

  char delims[] = " ";
  char *result = NULL;
  T temp;

  text = vtkFreehandUltrasound2EatWhitespace(text);

  // loop through each component
  for (int i = 0; i < numIndices; i++)
    {

    // find the next token
    if (i == 0)
      {
      result = strtok(text, delims);
      }
    else
      {
      result = strtok(NULL, delims);
      }
    result = vtkFreehandUltrasound2EatWhitespace(result);

    if (result != NULL)
      {
      temp = (T) atof(result);
      arrayToFill[i] = temp;
      }
    else
      {
      return 0;
      }

    }

  return 1;

 }

//----------------------------------------------------------------------------
// SaveSummaryFile
// Save the summary data in the (relative!) directory specified.
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SaveSummaryFile(const char *directory)
  {

  if (this->ReconstructionThreadId != -1)
    {
    if (this->RealTimeReconstruction)
      {
      this->StopRealTimeReconstruction();
      }
    else
      {
      this->StopReconstruction();
      }
    }

    int res;
#ifdef _WIN32
  res = _mkdir(directory);
#else
  int mode = 0777;
  res = mkdir(directory, mode);
#endif

  char path[512];

  // write out the freehand information
  vtkJoinPath2(path,512,directory,"freehand_summary.txt");
  FILE *file = fopen(path,"w");

  fprintf(file, "# vtkFreehandUltrasound2 summary\n\n");

  if (this->InterpolationMode)
    {
    fprintf(file, "Interpolation = Trilinear;\n");
    }
  else
    {
    fprintf(file, "Interpolation = NearestNeighbor;\n");
    }
  if (this->Optimization == 0)
    {
    fprintf(file, "Optimization = None;\n");
    }
  else if (this->Optimization == 1)
    {
    fprintf(file, "Optimization = BreakAndBoundsCheck;\n");
    }
  else if (this->Optimization == 2)
    {
    fprintf(file, "Optimization = Fixed;\n");
    }
  if (this->Compounding)
    {
    fprintf(file, "Compounding = On;\n");
    }
  else
    {
    fprintf(file, "Compounding = Off;\n");
    }
  fprintf(file, "OutputSpacing = %7.5f %7.5f %7.5f;\n",
    this->OutputSpacing[0], this->OutputSpacing[1], this->OutputSpacing[2]);
  fprintf(file, "FanDepthCm = %d;\n", this->FanDepthCm);
  fprintf(file, "NumberOfOutputVolumes = %d;\n", this->NumberOfOutputVolumes);
  if (this->DiscardOutlierHeartRates)
    {
    fprintf(file, "MeanHeartRate = %f;\n", this->MeanHeartRate);
    fprintf(file, "PercentageIncreasedHeartRateAllowed = %f;\n", this->PercentageIncreasedHeartRateAllowed);
    fprintf(file, "PercentageDecreasedHeartRateAllowed = %f;\n", this->PercentageDecreasedHeartRateAllowed);
    }
  fprintf(file, "VideoLag = %f;\n", this->VideoLag);

  if (this->SaveInsertedTimestamps)
    {
    int count;
    for (int phase = 0; phase < this->NumberOfOutputVolumes; phase++)
      {
      fprintf(file, "\nTimestamps for phase %d\n", phase);
      count = this->InsertedTimestampsCounter[phase];
      fprintf(file, "num timestamps = %d\n", count);
      for (int i = 0; i < count; i++)
        {
        fprintf(file, "%f\n", this->InsertedTimestampsBuffer[phase][i]);
        }
      }
    }

  fclose(file);
  }

//----------------------------------------------------------------------------
// SaveRawData
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
// TODO should save rotation information too
// TODO in dynamic code, should save phase information too
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::SaveRawData(const char *directory, int frames)
{
  if (this->ReconstructionThreadId != -1)
    {
    if (this->RealTimeReconstruction)
      {
      this->StopRealTimeReconstruction();
      }
    else
      {
      this->StopReconstruction();
      }
    }

  int res;
#ifdef _WIN32
  res = _mkdir(directory);
#else
  int mode = 0777;
  res = mkdir(directory, mode);
#endif

  // mkdir will return -1 if the directory was already created, so take this out
  /*if (res < 0)
    {
    vtkErrorMacro(<< "couldn't create directory " << directory);
    return;
    }*/

  char path[512];

  // write out the freehand information
  vtkJoinPath2(path,512,directory,"freehand.txt");
  FILE *file = fopen(path,"w");

  // get an image from the video source, for spacingings and such
  vtkImageData *image = this->VideoSource->GetOutput();
  image->UpdateInformation();

  fprintf(file, "# vtkFreehandUltrasound2 output\n\n");
  fprintf(file, "VideoSpacing = %7.5f %7.5f %7.5f;\n",
	  image->GetSpacing()[0], image->GetSpacing()[1], image->GetSpacing()[2]);
  fprintf(file, "VideoOrigin = %7.3f %7.3f %7.3f;\n",
	  image->GetOrigin()[0], image->GetOrigin()[1], image->GetOrigin()[2]);
  fprintf(file, "VideoFrameSize = %d %d %d;\n",
    this->VideoSource->GetFrameSize()[0], this->VideoSource->GetFrameSize()[1],
    this->VideoSource->GetFrameSize()[2]);
  fprintf(file, "VideoOutputFormat = %d;\n", this->VideoSource->GetOutputFormat());
  fprintf(file, "VideoFrameBufferSize = %d;\n", this->VideoSource->GetFrameBufferSize());
  fprintf(file, "VideoOutputExtent = %d %d %d %d %d %d;\n",
    this->VideoSource->GetOutputWholeExtent()[0], this->VideoSource->GetOutputWholeExtent()[1],
    this->VideoSource->GetOutputWholeExtent()[2], this->VideoSource->GetOutputWholeExtent()[3],
    this->VideoSource->GetOutputWholeExtent()[4], this->VideoSource->GetOutputWholeExtent()[5]);
  fprintf(file, "OutputSpacing = %7.5f %7.5f %7.5f;\n",
    this->OutputSpacing[0], this->OutputSpacing[1], this->OutputSpacing[2]);
  fprintf(file, "OutputOrigin = %7.3f %7.3f %7.3f;\n",
	  this->OutputOrigin[0], this->OutputOrigin[1], this->OutputOrigin[2]);
  fprintf(file, "OutputExtent = %d %d %d %d %d %d;\n",
	  this->OutputExtent[0], this->OutputExtent[1],
	  this->OutputExtent[2], this->OutputExtent[3],
	  this->OutputExtent[4], this->OutputExtent[5]);
  fprintf(file, "ClipRectangle = %7.3f %7.3f %7.3f %7.3f;\n",
	  this->ClipRectangle[0], this->ClipRectangle[1],
	  this->ClipRectangle[2], this->ClipRectangle[3]);
  fprintf(file, "FanAngles = %7.2f %7.2f;\n",
	  this->FanAngles[0], this->FanAngles[1]);
  fprintf(file, "FanOrigin = %7.3f %7.3f;\n",
	  this->FanOrigin[0], this->FanOrigin[1]);
  fprintf(file, "FanDepth = %7.3f;\n", this->FanDepth);
  fprintf(file, "VideoLag = %5.3f;\n", this->VideoLag);
  fprintf(file, "RotatingProbe = %d;\n", this->RotatingProbe);
  fprintf(file, "FanRotationImageThreshold1 = %d;\n", this->FanRotationImageThreshold1);
  fprintf(file, "FanRotationImageThreshold2 = %d;\n", this->FanRotationImageThreshold2);
  fprintf(file, "FanRotationXShift = %d;\n", this->FanRotationXShift);
  fprintf(file, "FanRotationYShift = %d;\n", this->FanRotationYShift);
  fprintf(file, "FanDepthCm = %d;\n", this->FanDepthCm);
  fprintf(file, "ImageIsFlipped = %d;\n", this->ImageIsFlipped);
  fprintf(file, "FlipHorizontalOnOutput = %d;\n", this->FlipHorizontalOnOutput);
  fprintf(file, "FlipVerticalOnOutput = %d;\n", this->FlipVerticalOnOutput);
  fprintf(file, "NumberOfPixelsFromTipOfFanToBottomOfScreen = %d;\n", this->NumberOfPixelsFromTipOfFanToBottomOfScreen);
  fprintf(file, "InterpolationMode = %d;\n", this->InterpolationMode);
  fprintf(file, "Optimization = %d;\n", this->Optimization);
  fprintf(file, "Compounding = %d;\n", this->Compounding);
  fclose(file);

  // write out the tracking information
  vtkJoinPath2(path,512,directory,"track.txt");
  this->TrackerBuffer->WriteToFile(path);

  // write out video information
  vtkJoinPath2(path,512,directory,"video.txt");
  char filePath[512];

  vtkJoinPath2(filePath,512,directory,"z");
  this->VideoSource->WriteFramesAsPNG(path, filePath, frames);
}

//----------------------------------------------------------------------------
// ReadRawData
// Read the raw data from the specified directory and use it for the
// following reconstructions.
// TODO should read rotation data too
//----------------------------------------------------------------------------
void vtkFreehandUltrasound2::ReadRawData(const char *directory)
{
  
if (this->ReconstructionThreadId != -1)
    {
    if (this->RealTimeReconstruction)
      {
      this->StopRealTimeReconstruction();
      }
    else
      {
      this->StopReconstruction();
      }
    }

  char path[512];
  char text[128];
  char *cp;
  char *tag;
  char *vals;
  int flag;

  // placeholders for the values we get in
  double double1;
  int int1;
  double doubleArray2[2];
  double doubleArray3[3];
  int intArray3[3];
  double doubleArray4[4];
  double doubleArray6[6];
  int intArray6[6];

  // read in the freehand information
  vtkJoinPath2(path,512,directory,"freehand.txt");
  FILE *file = fopen(path,"r");

  if (file == 0)
    {
    vtkErrorMacro(<< "can't open file " << path);
    return;
    }

  // will break out on EOF
  while (fgets(text, 128, file))
    {
    
    // eat leading whitespace
    cp = vtkFreehandUltrasound2EatWhitespace(text);

    // skip over empty lines or comments
    if (cp == 0 || *cp == '\0' || *cp == '#')
      {
      continue;
      }

    // find the '=' sign
    vals = strchr(cp,'=');
    
    // eat whitespace and equals
    vals = vtkFreehandUltrasound2EatWhitespaceWithEquals(vals);

    // do the matching
    flag = 0;
    if (strstr(cp,"VideoSpacing"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, doubleArray3, 3);
      if (flag) { this->VideoSource->SetDataSpacing(doubleArray3); }
      }
    else if (strstr(cp,"VideoOrigin"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, doubleArray3, 3);
      if (flag) { this->VideoSource->SetDataOrigin(doubleArray3); }
      }
    else if (strstr(cp,"VideoFrameSize"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, intArray3, 3);
      if (flag) { this->VideoSource->SetFrameSize(intArray3); }
      }
    else if (strstr(cp,"VideoOutputFormat"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) { this->VideoSource->SetOutputFormat(int1); }
      }
    else if (strstr(cp,"VideoFrameBufferSize"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) { this->VideoSource->SetFrameBufferSize(int1); }
      }
    else if (strstr(cp,"VideoOutputExtent"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, intArray6, 6);
      if (flag) { this->VideoSource->SetOutputWholeExtent(intArray6); }
      }
    else if (strstr(cp,"OutputSpacing"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, doubleArray3, 3);
      if (flag) {this->SetOutputSpacing(doubleArray3); }
      }
    else if (strstr(cp,"OutputOrigin"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, doubleArray3, 3);
      if (flag) {this->SetOutputOrigin(doubleArray3); }
      }
    else if (strstr(cp,"OutputExtent"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, intArray6, 6);
      if (flag) {this->SetOutputExtent(intArray6); }
      }
    else if (strstr(cp,"ClipRectangle"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, doubleArray4, 4);
      if (flag) {this->SetClipRectangle(doubleArray4); }
      }
    else if (strstr(cp,"FanAngles"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, doubleArray2, 2);
      if (flag) {this->SetFanAngles(doubleArray2); }
      }
    else if (strstr(cp,"FanOrigin"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, doubleArray2, 2);
      if (flag) {this->SetFanOrigin(doubleArray2); }
      }
    else if (strstr(cp,"FanDepth"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &double1, 1);
      if (flag) {this->SetFanDepth(double1); }
      }
    else if (strstr(cp,"VideoLag"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &double1, 1);
      if (flag) {this->SetVideoLag(double1); }
      }
    else if (strstr(cp,"RotatingProbe"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetRotatingProbe(int1); }
      }
    else if (strstr(cp,"FanRotationImageThreshold1"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetFanRotationImageThreshold1(int1); }
      }
    else if (strstr(cp,"FanRotationImageThreshold2"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetFanRotationImageThreshold1(int1); }
      }
    else if (strstr(cp,"FanRotationXShift"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetFanRotationXShift(int1); }
      }
    else if (strstr(cp,"FanRotationYShift"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetFanRotationYShift(int1); }
      }
    else if (strstr(cp,"FanDepthCm"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetFanDepthCm(int1); }
      }
    else if (strstr(cp,"ImageIsFlipped"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetImageIsFlipped(int1); }
      }
    else if (strstr(cp,"FlipHorizontalOnOutput"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetFlipHorizontalOnOutput(int1); }
      }
    else if (strstr(cp,"FlipVerticalOnOutput"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetFlipVerticalOnOutput(int1); }
      }
    else if (strstr(cp,"NumberOfPixelsFromTipOfFanToBottomOfScreen"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetNumberOfPixelsFromTipOfFanToBottomOfScreen(int1); }
      }
    else if (strstr(cp,"InterpolationMode"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetInterpolationMode(int1); }
      }
    else if (strstr(cp,"Optimization"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetOptimization(int1); }
      }
    else if (strstr(cp,"Compounding"))
      {
      flag = vtkExtractArrayComponentsFromString(vals, &int1, 1);
      if (flag) {this->SetCompounding(int1); }
      }
    else
      {
      vtkErrorMacro(<< "error reading file (invalid tag) " << path);
      return;
      }

    if (!flag)
      {
      vtkErrorMacro(<< "error reading file (invalid values) " << path);
      return;
      }

    }

  fclose(file);

  // TODO put back!
  // read in the tracking information
  //vtkJoinPath2(path,512,directory,"track.txt");
  //this->TrackerBuffer->ReadFromFile(path);

    // TODO read in videoinformation too
}