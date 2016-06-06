#ifndef __OPENCVINTERNALS_H__
#define __OPENCVINTERNALS_H__

#include <opencv2/videoio.hpp>

class OpenCVInternals 
{
public:
  static const int NUM_CAMERAS = 2;

	OpenCVInternals();
	~OpenCVInternals();

	// return true if a particular feed is available
	bool isFeedAvailable( int i );

	int numOfCamera(); 

	cv::VideoCapture *cameraFeeds[NUM_CAMERAS];
	cv::Mat intrinsic_matrix[NUM_CAMERAS];
	cv::Mat distortion_coeffs[NUM_CAMERAS];

private:
	bool feedAvailable[2];
	int nCameras;
};

#endif // of __OPENCVINTERNALS_H__