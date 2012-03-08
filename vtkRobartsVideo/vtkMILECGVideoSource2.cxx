/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMILECGVideoSource2.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkMILECGVideoSource2.h"
#include "vtkTimerLog.h"
#include "vtkObjectFactory.h"
#include "vtkCriticalSection.h"
#include "vtkDataArray.h"
#include "vtkVideoECGBuffer2.h"
#include "vtkVideoFrame2.h"

vtkCxxRevisionMacro(vtkMILECGVideoSource2, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkMILECGVideoSource2);

//----------------------------------------------------------------------------
vtkMILECGVideoSource2::vtkMILECGVideoSource2()
{
		this->CurrentPhase = -1;
}

//----------------------------------------------------------------------------
vtkMILECGVideoSource2::~vtkMILECGVideoSource2()
{
  this->vtkMILECGVideoSource2::ReleaseSystemResources();

  if (this->MILDigitizerDCF != NULL)
    {
    delete [] this->MILDigitizerDCF;
    this->MILDigitizerDCF = NULL;
    }

  this->SetMILSystemType(0);
}  

//----------------------------------------------------------------------------
void vtkMILECGVideoSource2::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkMILECGVideoSource2::InternalGrab()
{
  this->Buffer->Lock();

  if (this->AutoAdvance)
    {
    this->AdvanceFrameBuffer(1);
    if (this->FrameIndex + 1 < this->Buffer->GetBufferSize())
      {
      this->FrameIndex++;
      }
    }

  double indexTime = this->CreateTimeStampForFrame(this->LastFrameCount + 1);

  //void *ptr = reinterpret_cast<vtkDataArray *>(this->Buffer->GetFrame(0)->GetVoidPointer(0));
  unsigned char *ptr = reinterpret_cast<unsigned char *>(this->Buffer->GetFrame(0)->GetVoidPointer(0));
  int depth = this->Buffer->GetFrameFormat()->GetBitsPerPixel()/8;

  int frameExtent[6];
  this->Buffer->GetFrameFormat()->GetFrameExtent(frameExtent);

  int offsetX = frameExtent[0];
  int offsetY = frameExtent[2];

  int sizeX = frameExtent[1] - frameExtent[0] + 1;
  int sizeY = frameExtent[3] - frameExtent[2] + 1;

  if (sizeX > 0 && sizeY > 0)
    {
    if (depth == 1)
      {
      MbufGet2d(this->MILBufID,offsetX,offsetY,sizeX,sizeY,ptr);
      }
    else if (depth == 3)
      {
      MbufGetColor2d(this->MILBufID,M_RGB24+M_PACKED,M_ALL_BAND,
                     offsetX,offsetY,sizeX,sizeY,ptr);
      }
    else if (depth == 4) 
      {
      MbufGetColor2d(this->MILBufID,M_RGB32+M_PACKED,M_ALL_BAND,
                     offsetX,offsetY,sizeX,sizeY,ptr);
      }
    }

  // add the new frame and the current time to the buffer
  this->Buffer->AddItem(this->Buffer->GetFrame(0), indexTime, this->CurrentPhase);

  if (this->FrameCount++ == 0)
    {
    this->StartTimeStamp = this->Buffer->GetTimeStamp(0);
    }

  this->Modified();

  this->Buffer->Unlock();
}

void vtkMILECGVideoSource2::SetECGPhase(int newPhase) {
  //this->Buffer->Lock();
  this->CurrentPhase = newPhase;
  //this->Buffer->Unlock();
}

int vtkMILECGVideoSource2::GetECGPhase() {
  return this->CurrentPhase;
}