/*=========================================================================

  File: vtkReplayImageVideoSource.cxx
  Author: Chris Wedlake <cwedlake@robarts.ca>

  Language: C++
  Description:

  =========================================================================

  Copyright (c) Chris Wedlake, cwedlake@robarts.ca

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

#include "vtkReplayImageVideoSource.h"
#include "vtkTimerLog.h"
#include "vtkObjectFactory.h"
#include "vtkCriticalSection.h"
#include "vtkUnsignedCharArray.h"
#include "vtkMutexLock.h"
#include "vtkSmartPointer.h"

#include "vtkJPEGReader.h"
#include "vtkJPEGWriter.h"
#include "vtkPNGReader.h"
#include "vtkBMPReader.h"
#include "vtkTIFFReader.h"
#include "vtkImageData.h"
#include "vtkPointData.h"

#include "vtkImageFlip.h"

#include <string>
#include <algorithm>

// #include <windows.h>
// #include <tchar.h>
#include <stdio.h>
//#include <strsafe.h>


#include <vtkDirectory.h>
#include <vtkSortFileNames.h>
#include <vtkStringArray.h>
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

//#pragma comment(lib, "User32.lib")

vtkReplayImageVideoSource* vtkReplayImageVideoSource::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkReplayImageVideoSource");
  if(ret)
    {
      return (vtkReplayImageVideoSource*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkReplayImageVideoSource;
}

//----------------------------------------------------------------------------
vtkReplayImageVideoSource::vtkReplayImageVideoSource()
{

  this->Initialized = 0;
  this->pauseFeed = 0;
  this->currentLength = 0;

  this->vtkVideoSource::SetOutputFormat(VTK_RGB);
  this->vtkVideoSource::SetFrameBufferSize( 54 );
  this->vtkVideoSource::SetFrameRate( 15.0f );
  this->SetFrameSize(1680,1048,1);
  this->SetFrameSizeAutomatically = true;
  this->imageIndex=-1;
}

//----------------------------------------------------------------------------
vtkReplayImageVideoSource::~vtkReplayImageVideoSource()
{
  this->vtkReplayImageVideoSource::ReleaseSystemResources();
  for (unsigned int i = 0; i < this->loadedData.size(); i++) {
    this->loadedData[i]->Delete();
  }
  this->loadedData.clear();
}

//----------------------------------------------------------------------------
void vtkReplayImageVideoSource::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}

//----------------------------------------------------------------------------
void vtkReplayImageVideoSource::Initialize()
{
  if (this->Initialized)
    {
      return;
    }


  // Initialization worked
  this->Initialized = 1;

  // Update frame buffer  to reflect any changes
  this->UpdateFrameBuffer();
}

//----------------------------------------------------------------------------
void vtkReplayImageVideoSource::ReleaseSystemResources()
{
  this->Initialized = 0;
}

void vtkReplayImageVideoSource::InternalGrab()
{

  if (this->loadedData.size() == 0)
    {
      return;
    }

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

  this->imageIndex = ++this->imageIndex % this->loadedData.size();

  void *buffer = this->loadedData[this->imageIndex]->GetScalarPointer();

  unsigned char *ptr = reinterpret_cast<vtkUnsignedCharArray *>(this->FrameBuffer[index])->GetPointer(0);

  //int ImageSize = (this->FrameBufferExtent[1]-this->FrameBufferExtent[0])*(this->FrameBufferExtent[3]-this->FrameBufferExtent[2]);

  memcpy(ptr, buffer, this->NumberOfScalarComponents*(this->FrameSize[0]-1)*(this->FrameSize[1]-1));

  this->FrameBufferTimeStamps[index] = vtkTimerLog::GetUniversalTime();

  if (this->FrameCount++ == 0)
    {
      this->StartTimeStamp = this->FrameBufferTimeStamps[index];
    }

  this->Modified();

  this->FrameBufferMutex->Unlock();
}

//----------------------------------------------------------------------------
// platform-independent sleep function
static inline void vtkSleep(double duration)
{
  duration = duration; // avoid warnings
  // sleep according to OS preference
#ifdef _WIN32
  Sleep((int)(1000*duration));
#elif defined(__FreeBSD__) || defined(__linux__) || defined(sgi)
  struct timespec sleep_time, dummy;
  sleep_time.tv_sec = (int)duration;
  sleep_time.tv_nsec = (int)(1000000000*(duration-sleep_time.tv_sec));
  nanosleep(&sleep_time,&dummy);
#endif
}

//----------------------------------------------------------------------------
// Sleep until the specified absolute time has arrived.
// You must pass a handle to the current thread.
// If '0' is returned, then the thread was aborted before or during the wait.
static int vtkThreadSleep(vtkMultiThreader::ThreadInfo *data, double time)
{
  // loop either until the time has arrived or until the thread is ended
  for (int i = 0;; i++)
    {
      double remaining = time - vtkTimerLog::GetUniversalTime();

      // check to see if we have reached the specified time
      if (remaining <= 0)
  {
    if (i == 0)
      {
        vtkGenericWarningMacro("Dropped a video frame.");
      }
    return 1;
  }
      // check the ActiveFlag at least every 0.1 seconds
      if (remaining > 0.1)
  {
    remaining = 0.1;
  }

      // check to see if we are being told to quit
      data->ActiveFlagLock->Lock();
      int activeFlag = *(data->ActiveFlag);
      data->ActiveFlagLock->Unlock();

      if (activeFlag == 0)
  {
    break;
  }

      vtkSleep(remaining);
    }

  return 0;
}

//----------------------------------------------------------------------------
// this function runs in an alternate thread to asyncronously grab frames
static void *vtkReplayImageVideoSourceRecordThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkReplayImageVideoSource *self = (vtkReplayImageVideoSource *)(data->UserData);

  double startTime = vtkTimerLog::GetUniversalTime();
  double rate = self->GetFrameRate();
  int frame = 0;

  do
    {
      self->InternalGrab();
      frame++;
    }
  while (vtkThreadSleep(data, startTime + frame/rate));

  return NULL;
}

//----------------------------------------------------------------------------
// Set the source to grab frames continuously.
// You should override this as appropriate for your device.
void vtkReplayImageVideoSource::Record()
{
  // We don't actually record data.
  return;
}

//----------------------------------------------------------------------------
// this function runs in an alternate thread to 'play the tape' at the
// specified frame rate.
static void *vtkReplayImageVideoSourcePlayThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkVideoSource *self = (vtkVideoSource *)(data->UserData);

  double startTime = vtkTimerLog::GetUniversalTime();
  double rate = self->GetFrameRate();
  int frame = 0;

  do
    {
      self->Seek(1);
      frame++;
    }
  while (vtkThreadSleep(data, startTime + frame/rate));

  return NULL;
}

//----------------------------------------------------------------------------
// Set the source to play back recorded frames.
// You should override this as appropriate for your device.
void vtkReplayImageVideoSource::Play()
{
  if (this->Recording)
    {
      this->Stop();
    }

  if (!this->Playing)
    {
      this->Initialize();

      this->Playing = 1;
      this->Modified();
      this->PlayerThreadId =
  this->PlayerThreader->SpawnThread((vtkThreadFunctionType)\
            &vtkReplayImageVideoSourcePlayThread,this);
    }
}

//----------------------------------------------------------------------------
// Stop continuous grabbing or playback.  You will have to override this
// if your class overrides Play() and Record()
void vtkReplayImageVideoSource::Stop()
{
  if (this->Playing || this->Recording)
    {
      this->PlayerThreader->TerminateThread(this->PlayerThreadId);
      this->PlayerThreadId = -1;
      this->Playing = 0;
      this->Recording = 0;
      this->Modified();
    }
}

void vtkReplayImageVideoSource::Pause() {
  this->pauseFeed = 1;
}

void vtkReplayImageVideoSource::UnPause() {
  this->pauseFeed = 0;
}

void vtkReplayImageVideoSource::Restart() {
  this->imageIndex = -1;
}

void vtkReplayImageVideoSource::LoadFile(char * filename)
{

  bool applyFlip = false;

  std::string str(filename);
  std::string ext = "";

  for(unsigned int i=0; i<str.length(); i++)
  {
    if(str[i] == '.')
    {
      for(unsigned int j = i; j<str.length(); j++)
      {
        ext += str[j];
      }
      break;
    }
  }

  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  vtkImageData * data = vtkImageData::New();

  vtkSmartPointer<vtkImageReader2> reader;

  if (ext == ".jpg")
  {
    reader = vtkSmartPointer<vtkJPEGReader>::New();
  }
  else if (ext == ".png")
  {
    reader = vtkSmartPointer<vtkPNGReader>::New();
  }
  else if (ext == ".bmp")
  {
    reader = vtkSmartPointer<vtkBMPReader>::New();

  }
  else if (ext == ".tif")
  {
    reader = vtkSmartPointer<vtkTIFFReader>::New();
    applyFlip = true;
  }
  else
  {
    return;
  }

  if (reader->CanReadFile(filename))
  {
    reader->SetFileName(filename);
    reader->Update();
    reader->Modified();
#if (VTK_MAJOR_VERSION <= 5)
    reader->GetOutput()->Update();
#endif
  }
  else
  {
    cerr << "Unable To Read File:" << filename << endl;
    return;
  }

  int extents[6];
  reader->GetOutput()->GetExtent(extents);
  if (extents[1]-extents[0]+1 != this->FrameSize[0] ||
     extents[3]-extents[2]+1 != this->FrameSize[1] ||
     extents[5]-extents[4]+1 != this->FrameSize[2] )
  {
    if (this->SetFrameSizeAutomatically)
    {
      this->SetFrameSize(extents[1]-extents[0]+1, extents[3]-extents[2]+1,extents[5]-extents[4]+1);
      this->SetFrameSizeAutomatically = false;
    }
    else
    {
       vtkErrorMacro("Unable to open file as size doesn't match video source");
      return;
    }
  }

  if (applyFlip == true)
  {
    vtkSmartPointer<vtkImageFlip> flip = vtkSmartPointer<vtkImageFlip>::New();
    flip->SetInputConnection(reader->GetOutputPort());
    flip->SetFilteredAxis(1);
    flip->Modified();
    flip->Update();

    data->DeepCopy(flip->GetOutput());

  }
  else
  {
    data->DeepCopy(reader->GetOutput());
  }

  this->loadedData.push_back(data);

}


int vtkReplayImageVideoSource::LoadFolder(char * folder, char * filetype)
{

  char* fullPath = new char[1024];
  vtkSmartPointer<vtkDirectory> dir = vtkSmartPointer<vtkDirectory>::New();

  fullPath = strncpy( fullPath, folder,1024);
  fullPath = strncat( fullPath, "/",1024);

  int hFind = dir->Open(fullPath);

  if(hFind != 1){
    return -1;
  }

  vtkSmartPointer<vtkSortFileNames> sort = vtkSmartPointer<vtkSortFileNames>::New();
  sort->SetInputFileNames(dir->GetFiles());
  sort->SkipDirectoriesOn();
  sort->NumericSortOn();

  for(int i = 0; i < sort->GetFileNames()->GetNumberOfValues(); i++){

    char *file = new char[1024];
    file = strncpy(file, fullPath,1024);
    file = strncat(file, sort->GetFileNames()->GetValue(i),1024);
    //std::cout << file << std::endl;

    this->LoadFile(file);
  }

  return 0;
}

void vtkReplayImageVideoSource::Clear()
{

}

void vtkReplayImageVideoSource::SetClipRegion(int x0, int x1, int y0, int y1,
                int z0, int z1)
{
  vtkVideoSource::SetClipRegion(x0,x1,y0,y1,z0,z1);
  //return;
}
