/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkOpenCVVideoSource.cxx

  Copyright (c) John Baxter, Robarts Research Institute
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkOpenCVVideoSource.h"

#include "vtkObjectFactory.h"
#include "vtkDataArray.h"
#include "vtkUnsignedCharArray.h"
#include "vtkCriticalSection.h"
#include "vtkTimerLog.h"

#include <iostream>
#include <cv.h>
#include <highgui.h>

class vtkOpenCVVideoSourceInternal
{
public:
	vtkOpenCVVideoSourceInternal(vtkOpenCVVideoSource* s) {
		parent = s;
		capture = 0;
		cvRawImage = 0;
		cvProcImage = 0;
	}
	~vtkOpenCVVideoSourceInternal() {
		if( this->capture ) cvReleaseCapture( &(this->capture) );
		if( this->cvRawImage ) cvReleaseImage( &(this->cvRawImage) );
		if( this->cvProcImage ) cvReleaseImage( &(this->cvProcImage) );
	}

	vtkOpenCVVideoSource*	parent;
	double					lastTime;

	CvCapture*				capture;
	IplImage*				cvRawImage;
	IplImage*				cvProcImage;

};

vtkStandardNewMacro(vtkOpenCVVideoSource);

//----------------------------------------------------------------------------
vtkOpenCVVideoSource::vtkOpenCVVideoSource()
{
	
	//set up the internal class (handles pretty much everything)
	this->Internal = new vtkOpenCVVideoSourceInternal(this);
	this->OpenCVFirstBufferLock = vtkMutexLock::New();
	this->OpenCVSecondBufferLock = vtkMutexLock::New();

	//set some reasonable default values
	this->vtkVideoSource::SetFrameRate( 5.0f );
	this->videoSourceNumber = 0;
	this->FrameBufferBitsPerPixel = 24;
	this->vtkVideoSource::SetOutputFormat(VTK_RGB);
	
	this->EnumerateSources();
	this->Initialized = false;
}

//----------------------------------------------------------------------------
vtkOpenCVVideoSource::~vtkOpenCVVideoSource()
{
  this->vtkOpenCVVideoSource::ReleaseSystemResources();
  this->OpenCVFirstBufferLock->Delete();
  this->OpenCVSecondBufferLock->Delete();
  delete this->Internal;
}  

//----------------------------------------------------------------------------
void vtkOpenCVVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkOpenCVVideoSource::EnumerateSources(){
	this->maxSource = 0;
	bool stillLooking = true;
	while( stillLooking ){
		if( !cvCreateCameraCapture( this->maxSource ) ){
			stillLooking = false;
		}
		this->maxSource++;
	}
}

//----------------------------------------------------------------------------
void vtkOpenCVVideoSource::SetVideoSourceNumber(unsigned int n){
	
	//determine is source is available
	if( n < 0 || n > maxSource ){
		vtkErrorMacro(<<"Cannot use that video source number. Must be an available source");
		return;
	}

	//if we haven't changed number, don't do anything
	if( this->videoSourceNumber == n ) return;

	//create a new capture device for the specified identifier, and re-initialize
	this->videoSourceNumber = n;
	if( this->Initialized ){
		this->ReleaseSystemResources();
		this->Initialize();
	}

}

unsigned int vtkOpenCVVideoSource::GetVideoSourceNumber(){
	return this->videoSourceNumber;
}

//----------------------------------------------------------------------------
void vtkOpenCVVideoSource::SetFrameSize(int x, int y, int z) {
	vtkErrorMacro(<<"Frame size cannot be set, but is automatically determined by OpenCV.");
}

void vtkOpenCVVideoSource::SetOutputFormat(int format) {
	
	//vtkErrorMacro(<<"Use the VideoFormatDialog() method to set the output format.");
}

//----------------------------------------------------------------------------
void vtkOpenCVVideoSource::Initialize()
{
	//INITIALIZE OPENCV
	if( this->Initialized ) return;

	//Grab the appropriate capture device
	cvReleaseCapture( &(this->Internal->capture) );
	this->Internal->capture = cvCreateCameraCapture( this->videoSourceNumber );
	if( !this->Internal->capture ){
		vtkErrorMacro(<<"Could not create OpenCV capture device.");
		ReleaseSystemResources();
		return;
	}

	//collect information regarding image size, frame dimensions, etc...
	this->Internal->cvRawImage = cvQueryFrame( this->Internal->capture );
	if( !this->Internal->cvRawImage ){
		vtkErrorMacro(<<"Could not create OpenCV capture device.");
		ReleaseSystemResources();
		return;
	}
	this->ImageSize = this->Internal->cvRawImage->imageSize;
	this->vtkVideoSource::SetFrameSize( this->Internal->cvRawImage->width, this->Internal->cvRawImage->height, 1 );
	this->Internal->cvProcImage = (IplImage*) cvClone( this->Internal->cvRawImage );

	// Initialization worked
	this->Initialized = 1;

	// Update frame buffer  to reflect any changes
	this->UpdateFrameBuffer();
}  

//----------------------------------------------------------------------------
void vtkOpenCVVideoSource::ReleaseSystemResources()
{

	//UNINITIALIZE OPENCV
	delete this->Internal;
	this->Internal = new vtkOpenCVVideoSourceInternal( this );

	this->Initialized = 0;
}

//----------------------------------------------------------------------------
void vtkOpenCVVideoSource::UpdateFrameBuffer()
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
        buffer->GetNumberOfComponents() != 3 ||
        buffer->GetNumberOfTuples() != totalSize)
      {
      buffer->Delete();
      buffer = vtkUnsignedCharArray::New();
      this->FrameBuffer[i] = buffer;
      buffer->SetNumberOfComponents(3);
      buffer->SetNumberOfTuples(ext[0] * ext[1] * ext[2]);

      }
    }
}


void vtkOpenCVVideoSource::InternalGrab()
{

  //get a lock on the OpenCV buffer and fetch the images
  this->OpenCVFirstBufferLock->Lock();
  this->Internal->cvRawImage = cvQueryFrame( this->Internal->capture );
  this->OpenCVSecondBufferLock->Lock();
  if( this->Internal->cvRawImage ) cvConvertImage( this->Internal->cvRawImage, this->Internal->cvProcImage, CV_CVTIMG_FLIP + CV_CVTIMG_SWAP_RB );
  this->OpenCVFirstBufferLock->Unlock();
  //unlock 2nd buffer lock after final memcpy to framebuffer

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
  while (index < 0) index += this->FrameBufferSize;

  // Get a pointer to the location of the frame buffer
  char *ptr = (char *) reinterpret_cast<vtkUnsignedCharArray *>(this->FrameBuffer[index])->GetPointer(0);
  
  //attempt the find the processed buffer
  char* buffer = 0;
  if( this->Internal->cvProcImage ) buffer = this->Internal->cvProcImage->imageData;
  
  //if we can't find the buffer, error and return
  if( !buffer || !ptr ){
	  vtkErrorMacro(<<"Could not access both buffers.");
  }else{
	  memcpy(ptr, buffer, this->ImageSize);
  }

  // Copy image into frame buffer and release the old image
  this->OpenCVSecondBufferLock->Unlock();

  this->FrameBufferTimeStamps[index] = vtkTimerLog::GetUniversalTime();

  if (this->FrameCount++ == 0)
    {
    this->StartTimeStamp = this->FrameBufferTimeStamps[index];
    }

  this->Modified();

  this->FrameBufferMutex->Unlock();
}