#include "OpenCVCameraCapture.h"
#include "PlusCommon.h"

//----------------------------------------------------------------------------
OpenCVCameraCapture::OpenCVCameraCapture()
{
}

//----------------------------------------------------------------------------
OpenCVCameraCapture::~OpenCVCameraCapture()
{
  for ( std::map<int, cv::VideoCapture*>::iterator it = cameraFeeds.begin(); it != cameraFeeds.end(); ++it )
  {
    it->second->release();
    delete it->second;
  }
}

//----------------------------------------------------------------------------
// return true if a particular feed is available
bool OpenCVCameraCapture::IsFeedAvailable( int i ) const
{
  return cameraFeeds.find(i) != cameraFeeds.end();
}

//----------------------------------------------------------------------------
bool OpenCVCameraCapture::InitializeCamera(int cameraIndex)
{
  if( cameraFeeds.find(cameraIndex) != cameraFeeds.end() )
  {
    LOG_INFO("Camera with index " << cameraIndex << " already initialized.");
    return true;
  }
  cv::VideoCapture* vidCap = new cv::VideoCapture(cameraIndex + cv::CAP_MSMF);
  if( !vidCap->isOpened() )
  {
    delete vidCap;
    return false;
  }
  cameraFeeds[cameraIndex] = vidCap;

  return true;
}

//----------------------------------------------------------------------------
bool OpenCVCameraCapture::ReleaseCamera(int cameraIndex)
{
  if( cameraFeeds.find(cameraIndex) == cameraFeeds.end() )
  {
    return true;
  }

  cameraFeeds[cameraIndex]->release();
  delete cameraFeeds[cameraIndex];
  cameraFeeds.erase(cameraFeeds.find(cameraIndex));

  return true;
}

//----------------------------------------------------------------------------
bool OpenCVCameraCapture::QueryFrame(int cameraIndex, cv::Mat& outFrame, int flags /* = 0 */)
{
  IndexFeedMapIterator iter = cameraFeeds.find(cameraIndex);
  if( iter == cameraFeeds.end() )
  {
    LOG_ERROR("Unable to locate camera with index: " << cameraIndex);
    return false;
  }
  else
  {
    *cameraFeeds[cameraIndex] >> outFrame;
  }

  return true;
}

//----------------------------------------------------------------------------
int OpenCVCameraCapture::CameraCount() const
{
  return cameraFeeds.size();
}