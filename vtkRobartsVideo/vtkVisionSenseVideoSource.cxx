/*=========================================================================

  File: vtkVisionSenseVideoSource.cxx
  Author: Kyle Charbonneau <kcharbon@imaging.robarts.ca>
  Language: C++
  Description: 
    This class is used in conjunction wtih vtkVisionSenseNetworkSource to
    get images from the VisionSense stereo endocam and play them as a
    vtkVideoSource for use in the AtamaiViewer or Slicer. This class
    represents the actual video source object that can then be played
    in a render pane.
    
    Before initializing this class a vtkVisionSenseNetworkSource must be
    connected to the camera to download images off of it. For the 
    vtkVisionSenseNetworkSource to update there must be a video source 
    set to both the left camera and a video source set to the right camera, 
    as the network source will not update its buffer until both video
    sources have updated theirs. To link this class to a network source use
    SetSource().
     
=========================================================================*/

#include "vtkVisionSenseVideoSource.h"
#include "vtkTimerLog.h"
#include "vtkObjectFactory.h"
#include "vtkCriticalSection.h"
#include "vtkUnsignedCharArray.h"
#include "vtkMutexLock.h"
#include "vtkVisionSenseNetworkSource.h"

#include <string> 

vtkVisionSenseVideoSource* vtkVisionSenseVideoSource::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkVisionSenseVideoSource");
  if(ret)
    {
    return (vtkVisionSenseVideoSource*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkVisionSenseVideoSource;
}

//----------------------------------------------------------------------------
vtkVisionSenseVideoSource::vtkVisionSenseVideoSource()
{
  this->Initialized = 0;
  this->Source = NULL;
  this->Camera = 0;
}

//----------------------------------------------------------------------------
vtkVisionSenseVideoSource::~vtkVisionSenseVideoSource()
{
  this->vtkVisionSenseVideoSource::ReleaseSystemResources();
}  

//----------------------------------------------------------------------------
void vtkVisionSenseVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Image Size: " << this->ImageSize << "\n";
  
}

//----------------------------------------------------------------------------
void vtkVisionSenseVideoSource::Initialize()
{
  if (this->Initialized) 
  {
    return;
  }

  if (this->Source == NULL) 
  {
    vtkErrorMacro(<<"Initialize: Must set source first!");
  }
   
  // Calculate image requirements and allocate buffer
  this->ImageSize = this->Source->GetWidth() * this->Source->GetHeight() * 3;
  
  // Setup some needed values
  vtkVideoSource::SetOutputFormat(VTK_RGB);
  vtkVideoSource::SetFrameSize(this->Source->GetWidth(), this->Source->GetHeight(), 1);

  // Initialization worked
  this->Initialized = 1;

  // Update frame buffer  to reflect any changes
  this->UpdateFrameBuffer();
}  

//----------------------------------------------------------------------------
void vtkVisionSenseVideoSource::ReleaseSystemResources()
{
  this->Initialized = 0;
}

void vtkVisionSenseVideoSource::InternalGrab()
{

  // get a thread lock on the frame buffer
  this->FrameBufferMutex->Lock();

  if (this->AutoAdvance)
    {
    this->AdvanceFrameBuffer(1);
    if (this->FrameIndex + 1 < this->FrameBufferSize)
      {
      this->FrameIndex++;
      }
    }

  int index = this->FrameBufferIndex % this->FrameBufferSize;
  while (index < 0)
    {
    index += this->FrameBufferSize;
    }

  // Get pointer to data from the network source
  char *buffer;
  if (this->Camera)
  {
    buffer = this->Source->GetRightBuffer();
  }
  else
  {
    buffer = this->Source->GetLeftBuffer();
  }
  
  // Get a pointer to the location of the frame buffer
  char *ptr = (char *) reinterpret_cast<vtkUnsignedCharArray *>(this->FrameBuffer[index])->GetPointer(0);

  // Copy image into frame buffer
  memcpy(ptr, buffer, this->ImageSize);

  this->FrameBufferTimeStamps[index] = vtkTimerLog::GetUniversalTime();

  if (this->FrameCount++ == 0)
    {
    this->StartTimeStamp = this->FrameBufferTimeStamps[index];
    }

  this->Modified();

  this->FrameBufferMutex->Unlock();
}

void vtkVisionSenseVideoSource::UpdateFrameBuffer()
{
  int i, oldExt;
  int ext[3];
  vtkDataArray *buffer;

  // clip the ClipRegion with the FrameSize
  for (i = 0; i < 3; i++)
    {
    oldExt = this->FrameBufferExtent[2*i+1] - this->FrameBufferExtent[2*i] + 1;
    this->FrameBufferExtent[2*i] = ((this->ClipRegion[2*i] > 0) 
                             ? this->ClipRegion[2*i] : 0);  
    this->FrameBufferExtent[2*i+1] = ((this->ClipRegion[2*i+1] < 
                                       this->FrameSize[i]-1) 
                             ? this->ClipRegion[2*i+1] : this->FrameSize[i]-1);

    ext[i] = this->FrameBufferExtent[2*i+1] - this->FrameBufferExtent[2*i] + 1;
    if (ext[i] < 0)
      {
      this->FrameBufferExtent[2*i] = 0;
      this->FrameBufferExtent[2*i+1] = -1;
      ext[i] = 0;
      }

    if (oldExt > ext[i])
      { // dimensions of framebuffer changed
      this->OutputNeedsInitialization = 1;
      }
    }

  // total number of bytes required for the framebuffer
  int bytesPerRow = ext[0]*(this->FrameBufferBitsPerPixel/8);
  bytesPerRow = ((bytesPerRow + this->FrameBufferRowAlignment - 1) /
                 this->FrameBufferRowAlignment)*this->FrameBufferRowAlignment;
  int totalSize = bytesPerRow * ext[1] * ext[2];
  i = this->FrameBufferSize;

  while (--i >= 0)
    {
    buffer = reinterpret_cast<vtkDataArray *>(this->FrameBuffer[i]);
    if (buffer->GetDataType() != VTK_UNSIGNED_CHAR ||
        buffer->GetNumberOfComponents() != 1 ||
        buffer->GetNumberOfTuples() != totalSize)
      {
      buffer->Delete();
      buffer = vtkUnsignedCharArray::New();
      this->FrameBuffer[i] = buffer;
      buffer->SetNumberOfComponents(1);
      buffer->SetNumberOfTuples(totalSize);
      }
    }
}

void vtkVisionSenseVideoSource::SetLeftCamera()
{
  this->Camera = 0;
}
void vtkVisionSenseVideoSource::SetRightCamera()
{
  this->Camera = 1;
}
void vtkVisionSenseVideoSource::SetSource(vtkVisionSenseNetworkSource *source) 
{
  this->Source = source;
}
void vtkVisionSenseVideoSource::SetFrameSize(int x, int y, int z) 
{
  vtkErrorMacro(<<"SetFrameSize: Frame size is set automatically");
}
void vtkVisionSenseVideoSource::SetOutputFormat(int format) 
{
  vtkErrorMacro(<<"SetOutputFormat: Output format is fixed to RGB");
}
