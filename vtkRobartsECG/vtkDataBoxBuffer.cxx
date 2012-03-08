/*=========================================================================

  Program:   Data Acquisition box for the USB 9800 series for VTK
  Module:    $RCSfile: vtkDataBoxBuffer.cxx,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2007/04/19 12:48:52 $
  Version:   $Revision: 1.1 $

==========================================================================

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
#include "vtkDataBoxBuffer.h"
#include "vtkMath.h"
#include "vtkCriticalSection.h"
#include "vtkObjectFactory.h"

#include <stdio.h>
#include <stdlib.h>

//----------------------------------------------------------------------------
vtkDataBoxBuffer* vtkDataBoxBuffer::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkDataBoxBuffer");
  if(ret)
    {
    return (vtkDataBoxBuffer*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkDataBoxBuffer;
}

//----------------------------------------------------------------------------
vtkDataBoxBuffer::vtkDataBoxBuffer()
{

  this->BufferSize = 100000;
  this->TimeStampArray = vtkDoubleArray::New();
  this->TimeStampArray->SetNumberOfValues(this->BufferSize);

  this->NumberOfItems = 0;
  this->CurrentIndex = 0;
  this->CurrentTimeStamp = 0.0;
  this->numberOfChannels=0;
//  this->OutFile="";

  this->Mutex = vtkCriticalSection::New();
 
}

void vtkDataBoxBuffer::SetNumberOfChannels(int value){
	while (this->SignalArrayVector.size() != 0) {
		this->SignalArrayVector.pop_back();
	}
	for (int i=0; i < value; i++) {
		vtkDoubleArray * newArray = vtkDoubleArray::New();
		newArray->SetNumberOfValues(this->BufferSize);
		this->SignalArrayVector.push_back(newArray);
	}
	this->numberOfChannels = value;
}

int vtkDataBoxBuffer::GetNumberOfChannels(){
	return this->numberOfChannels;
}

//----------------------------------------------------------------------------
void vtkDataBoxBuffer::DeepCopy(vtkDataBoxBuffer *buffer)
{
  this->numberOfChannels = buffer->GetNumberOfChannels();
  this->SetNumberOfChannels(this->numberOfChannels);
	
  this->SetBufferSize(buffer->GetBufferSize());

  for (int i = 0; i < this->BufferSize; i++)
    {
	  	for (int j=0; j < SignalArrayVector.size(); j++) {
			this->SignalArrayVector[j]->SetValue(i, buffer->SignalArrayVector[j]->GetValue(i));
		}
    this->TimeStampArray->SetValue(i, buffer->TimeStampArray->GetValue(i));
    }

  this->CurrentIndex = buffer->CurrentIndex;
  this->NumberOfItems = buffer->NumberOfItems;
  this->CurrentTimeStamp = buffer->CurrentTimeStamp;

}

//----------------------------------------------------------------------------
vtkDataBoxBuffer::~vtkDataBoxBuffer()
{  
  this->SignalArray->Delete();
  this->TimeStampArray->Delete();

  this->Mutex->Delete();
}

//----------------------------------------------------------------------------
void vtkDataBoxBuffer::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkObject::PrintSelf(os,indent);
  
  os << indent << "BufferSize: " << this->BufferSize << "\n";
  os << indent << "NumberOfItems: " << this->NumberOfItems << "\n";
}

//----------------------------------------------------------------------------
void vtkDataBoxBuffer::SetBufferSize(int n)
{
  if (n == this->BufferSize)
    {
    return;
    }
  // right now, there is no effort made to save the previous contents
  this->NumberOfItems = 0;
  this->CurrentIndex = 0;
  this->CurrentTimeStamp = 0.0;
 
  this->BufferSize = n;
  this->TimeStampArray->SetNumberOfValues(this->BufferSize);
  for (int i=0; i < this->SignalArrayVector.size(); i++) {
	  vtkDoubleArray * newArray = vtkDoubleArray::New();
	  newArray->SetNumberOfValues(this->BufferSize);
	  this->SignalArrayVector.push_back(newArray);
  }

  this->Modified();
}  

//----------------------------------------------------------------------------
void vtkDataBoxBuffer::AddItem(double signal, int channel, double time)
{
  if (time < this->CurrentTimeStamp) { return; }
  else if (time == this->CurrentTimeStamp) {
	if (++this->CurrentIndex >= this->BufferSize) {
		this->CurrentIndex = 0;
		this->NumberOfItems = this->BufferSize;
	}
  }

  this->CurrentTimeStamp = time;

  if (this->CurrentIndex > this->NumberOfItems)
    {
    this->NumberOfItems = this->CurrentIndex;
    }
  
  this->SignalArrayVector[channel]->SetValue(this->CurrentIndex, signal);
  this->TimeStampArray->SetValue(this->CurrentIndex, time);

  this->Modified();
}


//----------------------------------------------------------------------------
int vtkDataBoxBuffer::GetSignal(int i, int channel)
{
  i = ((this->CurrentIndex - i) % this->BufferSize);

  if (i < 0)
    {
    i += this->BufferSize;
    }

  return this->SignalArrayVector[channel]->GetValue(i);
}

//----------------------------------------------------------------------------
double vtkDataBoxBuffer::GetTimeStamp(int i)
{
  i = ((this->CurrentIndex - i) % this->BufferSize);

  if (i < 0)
    {
    i += this->BufferSize;
    }

  return this->TimeStampArray->GetValue(i);
}

//----------------------------------------------------------------------------
// do a simple divide-and-conquer search for the transform
// that best matches the given timestamp
int vtkDataBoxBuffer::GetIndexFromTime(double time)
{
  int lo = this->NumberOfItems-1;
  int hi = 0;

  double tlo = this->GetTimeStamp(lo);
  double thi = this->GetTimeStamp(hi);

  if (time <= tlo)
    {
    return lo;
    }
  else if (time >= thi)
    {
    return hi;
    }

  for (;;)
    {
    if (lo-hi == 1)
      {
      if (time - tlo > thi - time)
	{
	return hi;
	}
      else
	{
	return lo;
	}
      }

    int mid = (lo+hi)/2;
    double tmid = this->GetTimeStamp(mid);
  
    if (time < tmid)
      {
      hi = mid;
      thi = tmid;
      }
    else
      {
      lo = mid;
      tlo = tmid;
      }
    }
}

//----------------------------------------------------------------------------
char *vtkDataBoxBufferEatWhitespace(char *text)
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
// Write the tracking information to a file
void vtkDataBoxBuffer::WriteToFile(const char *filename)
{
  int n;
  double timestamp;//, timestamp0=0;
  FILE *file;

  file = fopen(filename,"w");
  if (file == 0) {
    vtkErrorMacro( << "can't open file " << filename);
    return;
  }

  n = this->GetNumberOfItems();
  
  while (--n >= 0){
    timestamp = this->GetTimeStamp(n);
    fprintf(file,"%14.5f\t",timestamp);
//	fprintf(file,"%7.6f ",timestamp-timestamp0);

	for (int i = 0; i < this->numberOfChannels; i++)  {
		fprintf(file,"%10.6f\t", this->SignalArrayVector[i]->GetValue(n) );
	}
	fprintf(file,"\n");

	//timestamp0 = timestamp;
  }

  fclose(file);
}
