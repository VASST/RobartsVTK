/*=========================================================================

  Program:   Shapetape for VTK
  Module:    $RCSfile: vtkTrackerToolSensor.cxx,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2007/04/19 12:48:53 $
  Version:   $Revision: 1.1 $

==========================================================================

Copyright (c)

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

#include "vtkTrackerToolSensor.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkDoubleArray.h"
#include "vtkAmoebaMinimizer.h"
#include "vtkTrackerBuffer.h"
#include "vtkObjectFactory.h"

//----------------------------------------------------------------------------
vtkTrackerToolSensor* vtkTrackerToolSensor::New()
{

  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkTrackerToolSensor");
  if(ret)
    {
    return (vtkTrackerToolSensor*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkTrackerToolSensor;
}

//----------------------------------------------------------------------------
vtkTrackerToolSensor::vtkTrackerToolSensor()
{
  this->TempMatrix = vtkMatrix4x4::New();

  this->CalibrationMatrix = vtkMatrix4x4::New();
  for (int i = 0; i < VTK_MAX_SENSORS; i++) {
    this->Transform[i] = vtkTransform::New();
	this->Buffer[i] = vtkTrackerBuffer::New();
	this->Buffer[i]->SetToolCalibrationMatrix(this->CalibrationMatrix);
  }
  this->TapeLength = 0;
  this->InterpolationInterval = 0;

  //this->ErrorValue = 0;
  this->NumberOfSensors = 0;
}

//----------------------------------------------------------------------------
vtkTrackerToolSensor::~vtkTrackerToolSensor()
{
  for (int i = 0; i < VTK_MAX_SENSORS; i++) {
    this->Transform[i]->Delete();
	this->Buffer[i]->Delete();
  }
}


vtkTransform * vtkTrackerToolSensor::GetTransform(int num){
	return this->Transform[num];
}

vtkTransform * vtkTrackerToolSensor::GetTransform(){
	return this->Transform[0];
}
//----------------------------------------------------------------------------
void vtkTrackerToolSensor::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkTrackerTool::PrintSelf(os,indent);

  os << indent << "Tracker: " << this->Tracker << "\n";
  os << indent << "CalibrationMatrix: " << this->CalibrationMatrix << "\n";
  this->CalibrationMatrix->PrintSelf(os,indent.GetNextIndent());
  os << indent << "NumberOfSensors: " << this->NumberOfSensors << "\n";
/*
  for (int i = 0; i < 10; i++) {
    os << indent << "Transform " << i << " :" <<this->Transform[i] << "\n";
    this->Transform[i]->PrintSelf(os,indent.GetNextIndent());
    os << indent << "Buffer " << i << " :" << this->Buffer[i] << "\n";
    this->Buffer[i]->PrintSelf(os,indent.GetNextIndent());
  }//*/
  
}

//----------------------------------------------------------------------------
// the update copies the latest matrix from the buffer
void vtkTrackerToolSensor::Update()
{
  for (int i = 0; i < this->NumberOfSensors; i++) {
	this->Buffer[i]->Lock();
	this->Flags = this->Buffer[i]->GetFlags(0);

	if ((this->Flags & (TR_MISSING | TR_OUT_OF_VIEW))  == 0) {
		this->Buffer[i]->GetMatrix(this->TempMatrix, 0);
		this->Transform[i]->SetMatrix(this->TempMatrix);
    } 
	this->TimeStamp = this->Buffer[i]->GetTimeStamp(0);
	this->Buffer[i]->Unlock();
  }
}

//----------------------------------------------------------------------------
void vtkTrackerToolSensor::SetCalibrationMatrix(vtkMatrix4x4 *vmat)
{
  int i, j;

  for (i = 0; i < 4; i++) 
    {
    for (j = 0; j < 4; j++)
      {
		if (vtkTrackerToolSensor::CalibrationMatrix->GetElement(i,j) != vmat->GetElement(i,j))
	{
	break;
	}
      }
    if (j < 4)
      { 
      break;
      }
    }

  if (i < 4 || j < 4) // the matrix is different
    {
    this->CalibrationMatrix->DeepCopy(vmat);
    this->Modified();
    }
}

//----------------------------------------------------------------------------
void vtkTrackerToolSensor::SetTracker(vtkTrackerSensor *tracker)
{
  if (tracker == this->Tracker)
    {
    return;
    }




  if (this->Tracker) {
	for (int i = 0; i < VTK_MAX_SENSORS; i++) {
		this->Buffer[i]->SetWorldCalibrationMatrix(NULL);
	}
    this->Tracker->Delete();
  }

  if (tracker) {
    tracker->Register(this);
    this->Tracker = tracker;
    for (int i = 0; i < VTK_MAX_SENSORS; i++) {
		this->Buffer[i]->SetWorldCalibrationMatrix(tracker->GetWorldCalibrationMatrix());
	}
  }
  else {
    this->Tracker = NULL;
  }

  this->Modified();
}

//----------------------------------------------------------------------------
void vtkTrackerToolSensor::SensorUpdate(int sensor, vtkMatrix4x4 *matrix, long flags, double timestamp) 
{
  if (sensor < 0 && sensor >= this->NumberOfSensors) {
	  return;
  }
  vtkTrackerBuffer *buffer = this->Buffer[sensor];
  
  buffer->Lock();
  buffer->AddItem(matrix, flags, timestamp);
  buffer->Unlock();
//  buffer
}



//----------------------------------------------------------------------------
void vtkTrackerToolSensor::SetNumberOfSensors(int number) 
{
  this->NumberOfSensors = number;
}

//----------------------------------------------------------------------------
vtkTrackerBuffer * vtkTrackerToolSensor::GetBuffer(int sensor) 
{
  if (sensor < 0 && sensor >= this->NumberOfSensors) {
	  return NULL;
  }

  return this->Buffer[sensor];

}