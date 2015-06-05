
/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkVideoECGBuffer2.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkVideoECGBuffer2.h"
#include "vtkObjectFactory.h"
#include "vtkVideoFrame2.h"
#include "vtkDoubleArray.h"
#include "vtkCriticalSection.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

#if (VTK_MAJOR_VERSION <= 5)
vtkCxxRevisionMacro(vtkVideoECGBuffer2, "$Revision: 1.1 $");
#endif

vtkStandardNewMacro(vtkVideoECGBuffer2);

//----------------------------------------------------------------------------
vtkVideoECGBuffer2::vtkVideoECGBuffer2()
{
  this->FrameArray = 0;
  this->TimeStampArray = vtkDoubleArray::New();
  this->FrameFormat = vtkVideoFrame2::New();
  this->Mutex = vtkCriticalSection::New();
  this->BufferSize = 0;
  this->NumberOfItems = 0;
  this->CurrentIndex = 0;
  this->CurrentTimeStamp = 0.0;

  // serves to instantiate the frame array and the time stamp array
  this->SetBufferSize(30);
}

//----------------------------------------------------------------------------
vtkVideoECGBuffer2::~vtkVideoECGBuffer2()
{
  this->SetBufferSize(0);
  this->NumberOfItems = 0;

  if (this->FrameArray)
    {
    delete [] this->FrameArray;
    }

  if (this->TimeStampArray)
    {
    this->TimeStampArray->Delete();
    }

  this->FrameFormat->Delete();
  this->Mutex->Delete();
}

//----------------------------------------------------------------------------
void vtkVideoECGBuffer2::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
// Sets the buffer size, and copies the maximum number of the most current old
// frames and timestamps
void vtkVideoECGBuffer2::SetBufferSize(int bufsize)
{

  if (bufsize < 0)
  {
    vtkErrorMacro("SetBufferSize: invalid buffer size");
    return;
  }

  if (bufsize == this->BufferSize && bufsize != 0)
  { return; }

  int i;
  vtkVideoFrame2 **framearray;
  vtkDoubleArray* timestamps;
  vtkIntArray* phase;

  // if we don't have a frame array, we'll make one
  if (this->FrameArray == 0)
  {
    if (bufsize > 0)
    {
    this->ECGPhaseArray->SetNumberOfValues(this->BufferSize);
      for (i = 0; i < bufsize; i++) {
      this->ECGPhaseArray->SetValue(i, -1.0);
      }
      this->Modified();
    }
  }
  // if we already have a frame array and are changing its buffer size
  else
  {
    if (bufsize > 0)
    {
    phase = vtkIntArray::New();
      phase->SetNumberOfValues(bufsize);
    }
    else
    {
      phase = NULL;
    }

    int index = this->CurrentIndex;

    // if the new buffer is smaller than the old buffer
    if (this->BufferSize > bufsize)
      {
      // copy the most recent frames and timestamps
      for (i = 0; i < bufsize; i++)
        {
        phase->SetValue(i, this->ECGPhaseArray->GetValue(index));
        index = (index + 1) % this->BufferSize;
        if (index < 0)
          {
          // because '%' can give negative results on some platforms
          index = index + this->BufferSize;
          }
        }
      // delete the older frames and timestamps that won't fit in the new buffer
      for (i = 0; i < this->BufferSize - bufsize; i++)
        {
         this->ECGPhaseArray->SetValue(index, -1.0);
         index = (index + 1) % this->BufferSize;
         if (index < 0)
            {
          // because '%' can give negative results on some platforms
          index = index + this->BufferSize;
            }
        }
      }
    // if the new buffer is bigger than the old buffer
    else if (bufsize > this->BufferSize)
      {
      // create new frames and timestamps
      for (i = 0; i < bufsize - this->BufferSize; i++)
        {
        phase->SetValue(i, -1.0);
        }
      // copy the old frames and timestamps
      for (i = bufsize-this->BufferSize; i < bufsize; i++)
        {
        phase->SetValue(i, this->ECGPhaseArray->GetValue(index));
        index = (index + 1) % this->BufferSize;
        if (index < 0)
          {
          // because '%' can give negative results on some platforms
          index = index + this->BufferSize;
          }
        }
  }

    if (this->ECGPhaseArray)
      {
        this->ECGPhaseArray->Delete();
      }
    this->ECGPhaseArray = phase;

  vtkVideoBuffer2::SetBufferSize(bufsize);

    this->Modified();
    }
}

//----------------------------------------------------------------------------
void vtkVideoECGBuffer2::AddItem(vtkVideoFrame2* frame, double time, int phase)
{
  if (time <= this->CurrentTimeStamp)
    {
    return;
    }

  // don't add a frame if it doesn't match the buffer frame format
  int frameSize[3];
  int frameExtent[6];
  int frameFormatSize[3];
  int frameFormatExtent[6];
  frame->GetFrameSize(frameSize);
  frame->GetFrameExtent(frameExtent);
  this->FrameFormat->GetFrameSize(frameFormatSize);
  this->FrameFormat->GetFrameExtent(frameFormatExtent);
  if (frameSize[0] != frameFormatSize[0] ||
    frameSize[1] != frameFormatSize[1] ||
    frameSize[2] != frameFormatSize[2] ||
    frameExtent[0] != frameFormatExtent[0] ||
    frameExtent[1] != frameFormatExtent[1] ||
    frameExtent[2] != frameFormatExtent[2] ||
    frameExtent[3] != frameFormatExtent[3] ||
    frameExtent[4] != frameFormatExtent[4] ||
    frameExtent[5] != frameFormatExtent[5] ||
    frame->GetPixelFormat() != this->FrameFormat->GetPixelFormat() ||
    frame->GetBitsPerPixel() != this->FrameFormat->GetBitsPerPixel() ||
    frame->GetRowAlignment() != this->FrameFormat->GetRowAlignment() ||
    frame->GetTopDown() != this->FrameFormat->GetTopDown() ||
    frame->GetOpacity() != this->FrameFormat->GetOpacity() ||
  frame->GetCompression() != this->FrameFormat->GetCompression() ||
  frame->GetFrameGrabberType() != this->FrameFormat->GetFrameGrabberType() )
    {
    return;
    }

  // add the frame and timestamp, and update buffer
  this->CurrentTimeStamp = time;
  this->FrameArray[this->CurrentIndex] = frame;
  this->TimeStampArray->SetValue(this->CurrentIndex, time);
  this->ECGPhaseArray->SetValue(this->CurrentIndex, phase);
  this->NumberOfItems++;
  if (this->NumberOfItems > this->BufferSize)
    {
    this->NumberOfItems = this->BufferSize;
    }

  this->Modified();
}
