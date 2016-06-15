/*=========================================================================

Program:   Camera Calibration
Module:    $RCSfile: QComputeThread.h,v $
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

#ifndef __QCOMPUTETHREAD_H__
#define __QCOMPUTETHREAD_H__

#include <QThread>
#include <opencv2/core.hpp>

class QComputeThread : public QThread
{
  enum ComputationType
  {
    COMPUTATION_NONE,
    COMPUTATION_MONO_CALIBRATE,
    COMPUTATION_STEREO_CALIBRATE
  };

  Q_OBJECT

public:
  QComputeThread(int computeIndex, QObject *parent = 0);
  ~QComputeThread();

  /// Calibrate a single camera
  bool CalibrateCamera(int cameraIndex,
                       const std::vector<std::vector<cv::Point3f> >& patternPoints,
                       const std::vector<std::vector<cv::Point2f> >& imagePoints,
                       const cv::Size& imageSize,
                       int flags = 0);

  /// This prototype is called if the intrinsics for both cameras has already been determined
  bool StereoCalibrate(const std::vector<std::vector<cv::Point3f> >& patternPoints,
                       const std::vector< std::vector< cv::Point2f > >& leftImagePoints,
                       const std::vector< std::vector< cv::Point2f > >& rightImagePoints,
                       const cv::Mat& leftCameraMatrix,
                       const cv::Mat& leftDistCoeffs,
                       const cv::Mat& rightCameraMatrix,
                       const cv::Mat& rightDistCoeffs,
                       const cv::Size& imageSize,
                       int flags,
                       const cv::TermCriteria& termCriteria);

  /// This prototype is called if the intrinsics are being determined at the same time
  bool StereoCalibrate(const std::vector<std::vector<cv::Point3f> >& patternPoints,
                       const std::vector< std::vector< cv::Point2f > >& leftImagePoints,
                       const std::vector< std::vector< cv::Point2f > >& rightImagePoints,
                       const cv::Size& imageSize,
                       int flags,
                       const cv::TermCriteria& termCriteria);

protected:
  double ComputeReprojectionErrors(const std::vector<std::vector<cv::Point3f> >& objectPoints,
                                   const std::vector<std::vector<cv::Point2f> >& imagePoints,
                                   const std::vector<cv::Mat>& rvecs,
                                   const std::vector<cv::Mat>& tvecs,
                                   const cv::Mat& cameraMatrix,
                                   const cv::Mat& distCoeffs,
                                   std::vector<float>& perViewErrors,
                                   bool fisheye);

signals:
  void monoCalibrationComplete(int computeIndex,
                               int cameraIndex,
                               const cv::Mat& cameraMatrix,
                               const cv::Mat& distCoeffs,
                               const cv::Size& imageSize,
                               double reprojError,
                               double totalAvgErr,
                               const std::vector<cv::Mat>& rvecs,
                               const std::vector<cv::Mat>& tvecs,
                               const std::vector<float>& perViewErrors);

  /// This prototype is fired if the intrinsics have already been determined
  void stereoCalibrationComplete(int computeIndex,
                                 double reprojError,
                                 const cv::Mat& rotationMatrix,
                                 const cv::Mat& translationMatrix,
                                 const cv::Mat& essentialMatrix,
                                 const cv::Mat& fundamentalMatrix);

  /// This prototype is fired if the intrinsics are being determined as well
  void stereoCalibrationComplete(int computeIndex,
                                 double reprojError,
                                 const cv::Mat& leftCameraMatrix,
                                 const cv::Mat& leftDistCoeffs,
                                 const cv::Mat& rightCameraMatrix,
                                 const cv::Mat& rightDistCoeffs,
                                 const cv::Mat& rotationMatrix,
                                 const cv::Mat& translationMatrix,
                                 const cv::Mat& essentialMatrix,
                                 const cv::Mat& fundamentalMatrix);

protected:
  void run() Q_DECL_OVERRIDE;

  // Variables related to both
  int ComputeIndex;
  ComputationType Computation;
  QMutex Mutex;
  std::vector<std::vector<cv::Point3f> > PatternPoints;
  std::vector<std::vector<cv::Point2f> > LeftImagePoints;
  cv::Size ImageSize;
  int Flags;
  cv::Mat LeftCameraMatrix;
  cv::Mat LeftDistCoeffs;

  // Mono related variables
  int CameraIndex;
  std::vector<float> PerViewErrors;
  std::vector<cv::Mat> RotationsVector;
  std::vector<cv::Mat> TranslationsVector;

  // Stereo related variables
  bool IntrinsicsAvailable;
  std::vector<std::vector<cv::Point2f> > RightImagePoints;
  cv::Mat RotationMatrix;
  cv::Mat TranslationMatrix;
  cv::Mat RightCameraMatrix;
  cv::Mat RightDistCoeffs;
  cv::Mat EssentialMatrix;
  cv::Mat FundamentalMatrix;
  cv::TermCriteria TerminationCriteria;
};

#endif
