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
#include <QMutexLocker>

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
    PatternPoints = std::vector<std::vector<cv::Point3f> >(1);
    CalcBoardCornerPositions(Height, Width, QuadSize, PatternPoints[0], Pattern);
    PatternPoints.resize(LeftImagePoints.size(), PatternPoints[0]);

    LeftCameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    LeftDistCoeffs = cv::Mat::zeros(8, 1, CV_64F);

    double reprojectionError = cv::calibrateCamera(PatternPoints, LeftImagePoints, ImageSize,
                               LeftCameraMatrix, LeftDistCoeffs, RotationsVector, TranslationsVector, Flags);

    double totalAvgErr = ComputeReprojectionErrors(PatternPoints, LeftImagePoints, RotationsVector, TranslationsVector, LeftCameraMatrix,
                         LeftDistCoeffs, PerViewErrors);

    emit monoCalibrationComplete(ComputeIndex, CameraIndex, LeftCameraMatrix, LeftDistCoeffs, ImageSize, reprojectionError, totalAvgErr, RotationsVector, TranslationsVector, PerViewErrors);
  }
  else if( Computation == COMPUTATION_STEREO_CALIBRATE )
  {
    PatternPoints = std::vector<std::vector<cv::Point3f> >(1);
    CalcBoardCornerPositions(Height, Width, QuadSize, PatternPoints[0], Pattern);
    PatternPoints.resize(LeftImagePoints.size(), PatternPoints[0]);

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
  else if( Computation == COMPUTATION_PROCESS_IMAGE )
  {
    ImagePoints.clear();

    // make a copy of the current feed
    cv::Mat ResultImage = Image.clone();

    cv::Size boardSize(Width, Height);

    int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;

    bool found(false);
    switch( Pattern ) // Find feature points on the input format
    {
    case QComputeThread::CHESSBOARD:
      found = cv::findChessboardCorners( ResultImage, boardSize, ImagePoints, chessBoardFlags );
      break;
    case QComputeThread::CIRCLES_GRID:
      found = cv::findCirclesGrid( ResultImage, boardSize, ImagePoints );
      break;
    case QComputeThread::ASYMMETRIC_CIRCLES_GRID:
      found = cv::findCirclesGrid( ResultImage, boardSize, ImagePoints, cv::CALIB_CB_ASYMMETRIC_GRID );
      break;
    default:
      found = false;
      break;
    }

    if ( found )
    {
      // improve the found corners' coordinate accuracy for chessboard
      if( Pattern == QComputeThread::CHESSBOARD)
      {
        cv::Mat viewGray;
        cv::cvtColor(ResultImage, viewGray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(viewGray, ImagePoints, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1 ));
      }

      // Draw the corners.
      cv::drawChessboardCorners(ResultImage, boardSize, cv::Mat(ImagePoints), found);
    }

    emit patternProcessingComplete(ComputeIndex, CameraIndex, ImagePoints, ResultImage);
  }

  return;
}

//----------------------------------------------------------------------------
bool QComputeThread::CalibrateCamera(int cameraIndex,
                                     int width,
                                     int height,
                                     double size,
                                     CalibrationPattern pattern,
                                     const std::vector<std::vector<cv::Point2f> >& imagePoints,
                                     const cv::Size& imageSize, int flags)
{
  if (!isRunning())
  {
    {
      QMutexLocker locker(&Mutex);
      CameraIndex = cameraIndex;
      Computation = COMPUTATION_MONO_CALIBRATE;
      Pattern = pattern;
      LeftImagePoints = imagePoints;
      ImageSize = imageSize;
      Width = width;
      Height = height;
      QuadSize = QuadSize;
      Flags = flags;
    }

    start(LowPriority);
  }

  return false;
}

//----------------------------------------------------------------------------
bool QComputeThread::StereoCalibrate(int width,
                                     int height,
                                     double size,
                                     CalibrationPattern pattern,
                                     const std::vector< std::vector< cv::Point2f > >& leftImagePoints,
                                     const std::vector< std::vector< cv::Point2f > >& rightImagePoints,
                                     const cv::Mat& leftCameraMatrix,
                                     const cv::Mat& leftDistCoeffs,
                                     const cv::Mat& rightCameraMatrix,
                                     const cv::Mat& rightDistCoeffs,
                                     const cv::Size& imageSize,
                                     int flags,
                                     const cv::TermCriteria& termCriteria)
{
  if (!isRunning())
  {
    {
      QMutexLocker locker(&Mutex);
      Computation = COMPUTATION_STEREO_CALIBRATE;
      Flags = flags | CV_CALIB_FIX_INTRINSIC;
      IntrinsicsAvailable = true;
      Pattern = pattern;
      LeftImagePoints = leftImagePoints;
      RightImagePoints = rightImagePoints;
      LeftCameraMatrix = leftCameraMatrix;
      RightCameraMatrix = rightCameraMatrix;
      LeftDistCoeffs = leftDistCoeffs;
      RightDistCoeffs = rightDistCoeffs;
      ImageSize = imageSize;
      Width = width;
      Height = height;
      QuadSize = QuadSize;
      TerminationCriteria = termCriteria;
    }

    start(LowPriority);
  }

  return false;

}

//----------------------------------------------------------------------------
bool QComputeThread::StereoCalibrate(int width,
                                     int height,
                                     double size,
                                     CalibrationPattern pattern,
                                     const std::vector< std::vector< cv::Point2f > >& leftImagePoints,
                                     const std::vector< std::vector< cv::Point2f > >& rightImagePoints,
                                     const cv::Size& imageSize,
                                     int flags,
                                     const cv::TermCriteria& termCriteria)
{
  if (!isRunning())
  {
    {
      QMutexLocker locker(&Mutex);
      Computation = COMPUTATION_STEREO_CALIBRATE;
      Flags = flags;
      IntrinsicsAvailable = false;
      Pattern = pattern;
      LeftImagePoints = leftImagePoints;
      RightImagePoints = rightImagePoints;
      ImageSize = imageSize;
      Width = width;
      Height = height;
      QuadSize = QuadSize;
      TerminationCriteria = termCriteria;
    }

    start(LowPriority);
  }

  return false;
}

//----------------------------------------------------------------------------
bool QComputeThread::LocatePatternInImage(int cameraIndex, 
                                          int width,
                                          int height,
                                          CalibrationPattern pattern, 
                                          const cv::Mat& image)
{
  if (!isRunning())
  {
    {
      QMutexLocker locker(&Mutex);
      Computation = COMPUTATION_PROCESS_IMAGE;
      CameraIndex = cameraIndex;
      Pattern = pattern;
      Image = image;
      Width = width;
      Height = height;
    }

    start(LowPriority);
  }

  return false;
}

//----------------------------------------------------------------------------
double QComputeThread::ComputeReprojectionErrors(const std::vector<std::vector<cv::Point3f> >& objectPoints, const std::vector<std::vector<cv::Point2f> >& imagePoints, const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs, const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs, std::vector<float>& perViewErrors)
{
  std::vector<cv::Point2f> imagePoints2;
  size_t totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());

  for(size_t i = 0; i < objectPoints.size(); ++i )
  {
    cv::projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);

    err = norm(imagePoints[i], imagePoints2, cv::NORM_L2);

    size_t n = objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err*err/n);
    totalErr        += err*err;
    totalPoints     += n;
  }

  return std::sqrt(totalErr/totalPoints);
}

//----------------------------------------------------------------------------
void QComputeThread::CalcBoardCornerPositions(int height, int width, double quadSize,
                                                           std::vector<cv::Point3f>& outCorners,
                                                           QComputeThread::CalibrationPattern patternType)
{
  outCorners.clear();

  switch(patternType)
  {
  case QComputeThread::CHESSBOARD:
  case QComputeThread::CIRCLES_GRID:
    for( int i = 0; i < height; ++i )
      for( int j = 0; j < width; ++j )
      {
        outCorners.push_back(cv::Point3f(j*quadSize, i*quadSize, 0));
      }
      break;

  case QComputeThread::ASYMMETRIC_CIRCLES_GRID:
    for( int i = 0; i < height; i++ )
      for( int j = 0; j < width; j++ )
      {
        outCorners.push_back(cv::Point3f((2*j + i % 2)*quadSize, i*quadSize, 0));
      }
      break;
  default:
    break;
  }
}