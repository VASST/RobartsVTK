#ifndef __OPENCVINTERNALS_H__
#define __OPENCVINTERNALS_H__

#include <map>
#include <opencv2/videoio.hpp>

class OpenCVCameraCapture 
{
  typedef std::map<int, cv::VideoCapture*> IndexFeedMap;
  typedef IndexFeedMap::iterator IndexFeedMapIterator;
  typedef IndexFeedMap::const_iterator IndexFeedMapConstIterator;

public:
	OpenCVCameraCapture();
	~OpenCVCameraCapture();

	// return true if a particular feed is available
	bool IsFeedAvailable( int i ) const;

  bool InitializeCamera(int cameraIndex);
  bool ReleaseCamera(int cameraIndex);
  bool QueryFrame(int cameraIndex, cv::Mat& outFrame, int flags = 0);

	int CameraCount() const;

protected:
	std::map<int, cv::VideoCapture*> cameraFeeds;
};

#endif // of __OPENCVINTERNALS_H__