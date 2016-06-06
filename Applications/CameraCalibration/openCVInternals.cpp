#include "openCVInternals.h"

OpenCVInternals::OpenCVInternals()
{
  nCameras = NUM_CAMERAS;

  for ( int i = 0; i < nCameras; i++ )
  {
    feedAvailable[i] = false;
    cameraFeeds[i] = new cv::VideoCapture(i + cv::CAP_MSMF);

    //create structures for the intrinsic parameters
    intrinsic_matrix[i] = 0;
    distortion_coeffs[i]= 0;
    if ( cameraFeeds[i] )
    {
      feedAvailable[i] = true;
    }
  }
}

OpenCVInternals::~OpenCVInternals()
{
  for ( int i = 0; i < nCameras; i++ )
  {
    delete cameraFeeds[i];
  }
}

// return true if a particular feed is available
bool OpenCVInternals::isFeedAvailable( int i )
{
  return feedAvailable[i];
}

int OpenCVInternals::numOfCamera()
{
  return nCameras;
}