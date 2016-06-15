/*=========================================================================

Program:   Camera Calibration
Module:    $RCSfile: QComputeThread.cpp,v $
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

#include "QComputeThread.h"

// PLUS includes
#include <PlusCommon.h>

// Qt includes
#include <QImage>

// Open CV includes
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

//----------------------------------------------------------------------------
QComputeThread::QComputeThread(int computeIndex, QObject *parent /*= 0*/)
  : QThread(parent)
  , ComputeIndex(computeIndex)
  , Computation(COMPUTATION_NONE)
{

}

//----------------------------------------------------------------------------
QComputeThread::~QComputeThread()
{
  wait();
}

//----------------------------------------------------------------------------
void QComputeThread::run()
{
  QMutexLocker locker(&Mutex);

  if( Computation == COMPUTATION_MONO_CALIBRATE )
  {
    LeftCameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    LeftDistCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    double reprojectionError = cv::calibrateCamera(PatternPoints, LeftImagePoints, ImageSize,
                               LeftCameraMatrix, LeftDistCoeffs, RotationsVector, TranslationsVector, Flags);

    double totalAvgErr = ComputeReprojectionErrors(PatternPoints, LeftImagePoints, RotationsVector, TranslationsVector, LeftCameraMatrix,
                         LeftDistCoeffs, PerViewErrors, false);

    emit monoCalibrationComplete(ComputeIndex, CameraIndex, LeftCameraMatrix, LeftDistCoeffs, ImageSize, reprojectionError, totalAvgErr, RotationsVector, TranslationsVector, PerViewErrors);
  }
  else if( Computation == COMPUTATION_STEREO_CALIBRATE )
  {
    if( IntrinsicsAvailable )
    {
      double reprojError = cv::stereoCalibrate(PatternPoints,
                           LeftImagePoints, RightImagePoints,
                           LeftCameraMatrix, LeftDistCoeffs,
                           RightCameraMatrix, RightDistCoeffs,
                           ImageSize,
                           RotationMatrix,
                           TranslationMatrix,
                           EssentialMatrix,
                           FundamentalMatrix,
                           Flags,
                           TerminationCriteria);
      emit stereoCalibrationComplete(ComputeIndex, reprojError, RotationMatrix, TranslationMatrix, EssentialMatrix, FundamentalMatrix);
    }
    else
    {
      LeftCameraMatrix = cv::Mat::eye(3, 3, CV_64F);
      RightCameraMatrix = cv::Mat::eye(3, 3, CV_64F);
      LeftDistCoeffs = cv::Mat::zeros(8, 1, CV_64F);
      RightDistCoeffs = cv::Mat::zeros(8, 1, CV_64F);

      double reprojError = cv::stereoCalibrate(PatternPoints,
                           LeftImagePoints, RightImagePoints,
                           LeftCameraMatrix, LeftDistCoeffs,
                           RightCameraMatrix, RightDistCoeffs,
                           ImageSize,
                           RotationMatrix,
                           TranslationMatrix,
                           EssentialMatrix,
                           FundamentalMatrix,
                           Flags,
                           TerminationCriteria);
      emit stereoCalibrationComplete(ComputeIndex, reprojError, LeftCameraMatrix, LeftDistCoeffs, RightCameraMatrix, RightDistCoeffs, RotationMatrix, TranslationMatrix, EssentialMatrix, FundamentalMatrix);
    }
  }

  return;
}

//----------------------------------------------------------------------------
bool QComputeThread::CalibrateCamera(int cameraIndex, const std::vector<std::vector<cv::Point3f> >& patternPoints, const std::vector<std::vector<cv::Point2f> >& imagePoints,
                                     const cv::Size& imageSize, int flags)
{
  if (!isRunning())
  {
    {
      QMutexLocker locker(&Mutex);
      CameraIndex = cameraIndex;
      Computation = COMPUTATION_MONO_CALIBRATE;
      PatternPoints = patternPoints;
      LeftImagePoints = imagePoints;
      ImageSize = imageSize;
      Flags = flags;
    }

    start(LowPriority);
  }

  return false;
}

//----------------------------------------------------------------------------
bool QComputeThread::StereoCalibrate(const std::vector<std::vector<cv::Point3f> >& patternPoints, const std::vector< std::vector< cv::Point2f > >& leftImagePoints, const std::vector< std::vector< cv::Point2f > >& rightImagePoints, const cv::Mat& leftCameraMatrix, const cv::Mat& leftDistCoeffs, const cv::Mat& rightCameraMatrix, const cv::Mat& rightDistCoeffs, const cv::Size& imageSize, int flags, const cv::TermCriteria& termCriteria)
{
  if (!isRunning())
  {
    {
      QMutexLocker locker(&Mutex);
      Computation = COMPUTATION_STEREO_CALIBRATE;
      Flags = flags | CV_CALIB_FIX_INTRINSIC;
      IntrinsicsAvailable = true;
      PatternPoints = patternPoints;
      LeftImagePoints = leftImagePoints;
      RightImagePoints = rightImagePoints;
      LeftCameraMatrix = leftCameraMatrix;
      RightCameraMatrix = rightCameraMatrix;
      LeftDistCoeffs = leftDistCoeffs;
      RightDistCoeffs = rightDistCoeffs;
      ImageSize = imageSize;
      TerminationCriteria = termCriteria;
    }

    start(LowPriority);
  }

  return false;
  
}

//----------------------------------------------------------------------------
bool QComputeThread::StereoCalibrate(const std::vector<std::vector<cv::Point3f> >& patternPoints, const std::vector< std::vector< cv::Point2f > >& leftImagePoints, const std::vector< std::vector< cv::Point2f > >& rightImagePoints, const cv::Size& imageSize, int flags, const cv::TermCriteria& termCriteria)
{
  if (!isRunning())
  {
    {
      QMutexLocker locker(&Mutex);
      Computation = COMPUTATION_STEREO_CALIBRATE;
      Flags = flags;
      IntrinsicsAvailable = false;
      PatternPoints = patternPoints;
      LeftImagePoints = leftImagePoints;
      RightImagePoints = rightImagePoints;
      ImageSize = imageSize;
      TerminationCriteria = termCriteria;
    }

    start(LowPriority);
  }

  return false;
}

//----------------------------------------------------------------------------
double QComputeThread::ComputeReprojectionErrors(const std::vector<std::vector<cv::Point3f> >& objectPoints, const std::vector<std::vector<cv::Point2f> >& imagePoints, const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs, const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs, std::vector<float>& perViewErrors, bool fisheye)
{
  std::vector<cv::Point2f> imagePoints2;
  size_t totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());

  for(size_t i = 0; i < objectPoints.size(); ++i )
  {
    if (fisheye)
    {
      cv::fisheye::projectPoints(objectPoints[i], imagePoints2, rvecs[i], tvecs[i], cameraMatrix, distCoeffs);
    }
    else
    {
      cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
    }
    err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);

    size_t n = objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err*err/n);
    totalErr        += err*err;
    totalPoints     += n;
  }

  return std::sqrt(totalErr/totalPoints);
}
