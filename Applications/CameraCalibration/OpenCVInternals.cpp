#include "OpenCVInternals.h"
#include "PlusCommon.h"

//----------------------------------------------------------------------------
OpenCVInternals::OpenCVInternals()
{
}

//----------------------------------------------------------------------------
OpenCVInternals::~OpenCVInternals()
{
  for ( std::map<int, cv::VideoCapture*>::iterator it = cameraFeeds.begin(); it != cameraFeeds.end(); ++it )
  {
    it->second->release();
    delete it->second;
  }
}

//----------------------------------------------------------------------------
// return true if a particular feed is available
bool OpenCVInternals::IsFeedAvailable( int i ) const
{
  return cameraFeeds.find(i) != cameraFeeds.end();
}

//----------------------------------------------------------------------------
bool OpenCVInternals::InitializeCamera(int cameraIndex)
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

  //create structures for the intrinsic parameters
  cv::Mat empty;
  intrinsic_matrix[cameraIndex] = empty;
  distortion_coeffs[cameraIndex] = empty;

  return true;
}

//----------------------------------------------------------------------------
bool OpenCVInternals::ReleaseCamera(int cameraIndex)
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
bool OpenCVInternals::QueryFrame(int cameraIndex, cv::Mat& outFrame, int flags /* = 0 */)
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
int OpenCVInternals::CameraCount() const
{
  return cameraFeeds.size();
}

//----------------------------------------------------------------------------
void OpenCVInternals::SetIntrinsicMatrix(int cameraIndex, cv::Mat& matrix)
{
  IndexFeedMapIterator iter = cameraFeeds.find(cameraIndex);
  if( iter == cameraFeeds.end() )
  {
    LOG_ERROR("Unable to locate camera with index: " << cameraIndex);
    return;
  }
  else
  {
    intrinsic_matrix[cameraIndex] = matrix;
  }
}

//----------------------------------------------------------------------------
cv::Mat* OpenCVInternals::GetInstrinsicMatrix(int cameraIndex)
{
  IndexFeedMapIterator iter = cameraFeeds.find(cameraIndex);
  if( iter == cameraFeeds.end() )
  {
    LOG_ERROR("Unable to locate camera with index: " << cameraIndex);
    return NULL;
  }
  else
  {
    return &intrinsic_matrix[cameraIndex];
  }
}

//----------------------------------------------------------------------------
void OpenCVInternals::SetDistortionCoeffs(int cameraIndex, cv::Mat& matrix)
{
  IndexFeedMapIterator iter = cameraFeeds.find(cameraIndex);
  if( iter == cameraFeeds.end() )
  {
    LOG_ERROR("Unable to locate camera with index: " << cameraIndex);
    return;
  }
  else
  {
    distortion_coeffs[cameraIndex] = matrix;
  }
}

//----------------------------------------------------------------------------
cv::Mat* OpenCVInternals::GetDistortionCoeffs(int cameraIndex)
{
  IndexFeedMapIterator iter = cameraFeeds.find(cameraIndex);
  if( iter == cameraFeeds.end() )
  {
    LOG_ERROR("Unable to locate camera with index: " << cameraIndex);
    return NULL;
  }
  else
  {
    return &distortion_coeffs[cameraIndex];
  }
}
