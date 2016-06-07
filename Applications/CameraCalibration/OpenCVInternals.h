#ifndef __OPENCVINTERNALS_H__
#define __OPENCVINTERNALS_H__

#include <map>
#include <opencv2/videoio.hpp>

class OpenCVInternals 
{
  typedef std::map<int, cv::VideoCapture*> IndexFeedMap;
  typedef IndexFeedMap::iterator IndexFeedMapIterator;
  typedef IndexFeedMap::const_iterator IndexFeedMapConstIterator;

  typedef std::map<int, cv::Mat> IndexMatrixMap;
  typedef IndexMatrixMap::iterator IndexMatrixMapIterator;
  typedef IndexMatrixMap::const_iterator IndexMatrixMapConstIterator;

public:
	OpenCVInternals();
	~OpenCVInternals();

	// return true if a particular feed is available
	bool IsFeedAvailable( int i ) const;

  bool InitializeCamera(int cameraIndex);
  bool ReleaseCamera(int cameraIndex);
  bool QueryFrame(int cameraIndex, cv::Mat& outFrame, int flags = 0);

	int CameraCount() const;

  void SetIntrinsicMatrix(int cameraIndex, cv::Mat& matrix);
  cv::Mat* GetInstrinsicMatrix(int cameraIndex);

  void SetDistortionCoeffs(int cameraIndex, cv::Mat& matrix);
  cv::Mat* GetDistortionCoeffs(int cameraIndex);

protected:
	std::map<int, cv::VideoCapture*> cameraFeeds;
	std::map<int, cv::Mat> intrinsic_matrix;
	std::map<int, cv::Mat> distortion_coeffs;
};

#endif // of __OPENCVINTERNALS_H__