/*=========================================================================

  Program:   Heart Signal Box for VTK
  Module:    $RCSfile: vtkTestECGSignalBox.h,v $
  Creator:   Chris Wedlake <cwedlake@imaging.robarts.ca>
  Language:  C++
  Author:    $Author: cwedlake $
  Date:      $Date: 2009/09/23 17:31:37 $
  Version:   $Revision: 1.0 $


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
#include "vtkTestECGSignalBox.h"
#include "vtkObjectFactory.h"
#include "vtkMultiThreader.h"
#include "vtkMutexLock.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/timeb.h>
#include <io.h>

//-------------------------------------------------------------------------
vtkTestECGSignalBox* vtkTestECGSignalBox::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkTestECGSignalBox");
  if(ret)
    {
    return (vtkTestECGSignalBox*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkTestECGSignalBox;
}

//-------------------------------------------------------------------------
vtkTestECGSignalBox::vtkTestECGSignalBox()
{
  this->videoSource = NULL; 
}

//-------------------------------------------------------------------------
// everything is done in the constructor
vtkTestECGSignalBox::~vtkTestECGSignalBox()
{
}

//-------------------------------------------------------------------------
void vtkTestECGSignalBox::PrintSelf(ostream& os, vtkIndent indent)
{
 // vtkObject::PrintSelf(os,indent);
}

//-------------------------------------------------------------------------
int vtkTestECGSignalBox::GetECG(void) {
  int signal = vtkSignalBox::GetECG();
  if (this->videoSource) {
    this->videoSource->SetECGPhase(ECGPhase);
  }
  return signal;
}

//-------------------------------------------------------------------------
static void *vtkTestECGSignalBoxThread(vtkMultiThreader::ThreadInfo *data)
{

  vtkTestECGSignalBox *self = (vtkTestECGSignalBox *)(data->UserData);

  for (int i = 0;; i++) {

    #ifdef _WIN32
      Sleep(self->GetSleepInterval());
    #else
    #ifdef unix
    #ifdef linux
      usleep(self->GetSleepInterval() * 1000);
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

void vtkTestECGSignalBox::SetVideoSource(vtkTestECGVideoSource * video) {
  this->videoSource = video;
}

//-------------------------------------------------------------------------
void vtkTestECGSignalBox::Start()
{

  if (this->IsStarted)
    {
    return;
    }

  if (!this->IsInitialized)
    {
    this->Initialize();
    }

  // if initialization fails
  if (!this->IsInitialized)
    {
    return;
    }
    
  this->UpdateMutex->Lock();

  if (this->ThreadId == -1)
    {
    this->ThreadId = this->Threader->SpawnThread((vtkThreadFunctionType)&vtkTestECGSignalBoxThread,this);
    }
  this->UpdateMutex->Unlock();

  this->IsStarted=1;
}