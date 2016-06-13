/*=========================================================================

Program:   Camera Calibration
Module:    $RCSfile: QCaptureThread.cpp,v $
Creator:   Adam Rankin <arankin@robarts.ca>
Language:  C++
Author:    $Author: Adam Rankin $

==========================================================================

Copyright (c) Adam Rankin, arankin@robarts.ca

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

#include "QCaptureThread.h"

// PLUS includes
#include <PlusCommon.h>

// Qt includes
#include <QImage>

// Open CV includes
#include <opencv2/imgproc.hpp>

//----------------------------------------------------------------------------
QCaptureThread::QCaptureThread(QObject *parent /*= 0*/)
  : QThread(parent)
  , CameraCapture(NULL)
  , CameraIndex(-1)
  , abort(false)
{

}

//----------------------------------------------------------------------------
QCaptureThread::~QCaptureThread()
{
  QMutexLocker locker(&LocalMutex);
  abort = true;

  wait();
}

//----------------------------------------------------------------------------
bool QCaptureThread::StartCapture(int cameraIndex)
{
  if (!isRunning())
  {
    if( CommonMutex == NULL )
    {
      LOG_ERROR("Common mutex not set. Cannot co-operate with other capture threads.");
      return false;
    }

    CommonMutex->lock();
    if( !CameraCapture->InitializeCamera(cameraIndex) )
    {
      LOG_ERROR("Unable to initialize camera with camera index: " << cameraIndex);
      CommonMutex->unlock();
      return false;
    }
    CommonMutex->unlock();

    QMutexLocker locker(&LocalMutex);
    CameraIndex = cameraIndex;

    start(LowPriority);
  }

  return true;
}

//----------------------------------------------------------------------------
void QCaptureThread::StopCapture(bool shouldWait /* = true */)
{
  if (isRunning())
  {
    QMutexLocker locker(&LocalMutex);
    abort = true;

    if( shouldWait )
    {
      wait();
    }

    CommonMutex->lock();
    CameraCapture->ReleaseCamera(CameraIndex);
    CommonMutex->unlock();
  }
}

//----------------------------------------------------------------------------
void QCaptureThread::SetCommonMutex(QMutex* mutex)
{
  if( isRunning() )
  {
    LOG_ERROR("Critical failure. Common mutex changed while thread is running.");
    return;
  }

  QMutexLocker locker(&LocalMutex);
  if( mutex != NULL )
  {
    this->CommonMutex = mutex;
  }
}

//----------------------------------------------------------------------------
void QCaptureThread::SetOpenCVInternals(OpenCVInternals& cameraCapture)
{
  QMutexLocker locker(&LocalMutex);
  this->CameraCapture = &cameraCapture;
}

//----------------------------------------------------------------------------
void QCaptureThread::run()
{
  while(true)
  {
    QMutexLocker locker(&LocalMutex);
    if( this->CameraCapture != NULL )
    {
      if( CameraCapture->QueryFrame(CameraIndex, CapturedImage) )
      {
        emit capturedImage(CapturedImage, CameraIndex);
      }
    }

    if( abort )
    {
      return;
    }
  }
}