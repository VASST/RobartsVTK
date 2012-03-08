/*=========================================================================
  
  Program:   Heart Signal Box for VTK
  Module:    $RCSfile: vtkVideoLinkedUSBECGBox.cxx,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2009/02/26 17:49:52 $
  Version:   $Revision: 1.3 $

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

#include <limits.h>
#include <float.h>
#include <math.h>
#include "vtkVideoLinkedUSBECGBox.h"
#include "vtkMultiThreader.h"
#include "vtkMutexLock.h"
#include "vtkCriticalSection.h"
#include "vtkObjectFactory.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <io.h>

vtkVideoLinkedUSBECGBox* vtkVideoLinkedUSBECGBox::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkVideoLinkedUSBECGBox");
  if(ret)
    {
    return (vtkVideoLinkedUSBECGBox*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkVideoLinkedUSBECGBox;
}

vtkVideoLinkedUSBECGBox::vtkVideoLinkedUSBECGBox()
{
  this->videoSource = NULL; 
}

vtkVideoLinkedUSBECGBox::~vtkVideoLinkedUSBECGBox()
{
  if (this->ThreadId != -1)
    {
    this->Threader->TerminateThread(this->ThreadId);
    this->ThreadId = -1;
    }

  this->Threader->Delete();
  this->UpdateMutex->Delete();

}
  
void vtkVideoLinkedUSBECGBox::PrintSelf(ostream& os, vtkIndent indent)
{
   this->Superclass::PrintSelf(os,indent);
}
  
static void *vtkVideoLinkedUSBECGBoxThread(vtkMultiThreader::ThreadInfo *data)
{
	vtkVideoLinkedUSBECGBox *self = (vtkVideoLinkedUSBECGBox *)(data->UserData);


	for (int i = 0;; i++) {

		#ifdef _WIN32
			Sleep(1);
		#else
		#ifdef unix
		#ifdef linux
			usleep(5000);
		#endif
		#endif
		#endif

		// query the hardware tracker
		self->UpdateMutex->Lock();
		self->Update();
		self->UpdateTime.Modified();
		self->UpdateMutex->Unlock();
    
		// check to see if we are being told to quit 
		data->ActiveFlagLock->Lock();
		int activeFlag = *(data->ActiveFlag);
		data->ActiveFlagLock->Unlock();

	  if (activeFlag == 0) {
			return NULL;
		}
	}
}

int vtkVideoLinkedUSBECGBox::GetECG(void) {

	int signal = vtkUSBECGBox::GetECG();
	if (this->videoSource) {
		this->videoSource->SetECGPhase(ECGPhase);
	}
	return signal;
}

void vtkUSBECGBox::CalculateECGRate() {
	if (this->Timestamp <= this->ExpectedSignalTimeStamp) {
		//this->ECGRateBPM = 0;
		//this->ECGPhase = -1;
		return;
	}

	ECGRateBPMArray.push_back(float((1/(this->Timestamp-this->StartSignalTimeStamp))*60));
	if (ECGRateBPMArray.size() > this->Average) {
		int BPM_Sum = 0;
		for (int i=1; i <= this->Average; i++) {
			BPM_Sum += ECGRateBPMArray[ECGRateBPMArray.size()-i];
		}
		this->ECGRateBPM = int(BPM_Sum/this->Average);
		if ((ECGRateBPMArray.size()+10) > this->Average) {
			ECGRateBPMArray.erase( ECGRateBPMArray.begin() );
		}
		double increase = ((this->Timestamp-this->StartSignalTimeStamp)*0.4) > 1 ? 1 : ((this->Timestamp-this->StartSignalTimeStamp)*0.4) < 0.3 ? 0.3 : ((this->Timestamp-this->StartSignalTimeStamp)*0.4);
		this->ExpectedSignalTimeStamp = this->Timestamp+increase;
		this->StartSignalTimeStamp = this->Timestamp;
	}

}


void vtkVideoLinkedUSBECGBox::Start()
{
	int Count=sampleSize;
	if (this->IsStarted)
		return;

	/* Collect the values with cbAInScan() in BACKGROUND mode
        Parameters:
             BoardNum    :the number used by CB.CFG to describe this board
             LowChan     :low channel of the scan
             HighChan    :high channel of the scan
             Count       :the total number of A/D samples to collect
             Rate        :sample rate in samples per second
             Gain        :the gain for the board
             ADData[]    :the array for the collected data values
             Options     :data collection options 
	*/

    this->ULStat = cbAInScan (this->BoardNum, this->channel[0], this->channel[1], Count, &this->SampleRate, this->Gain, ADData, this->OPTIONS);
	if (this->ULStat > 0) {
		return;
	}
	this->UpdateMutex->Lock();

	if (this->ThreadId == -1){
		this->ThreadId = this->Threader->SpawnThread((vtkThreadFunctionType)&vtkVideoLinkedUSBECGBoxThread,this);
	}
	this->UpdateMutex->Unlock();
	this->IsStarted=1;
}