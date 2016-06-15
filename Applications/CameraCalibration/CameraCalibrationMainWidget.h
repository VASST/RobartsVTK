/*=========================================================================

Program:   tracking with GUI
Module:    $RCSfile: CameraCalibrationMainWidget.h,v $
Creator:   Elvis C. S. Chen <chene@robarts.ca>
Language:  C++
Author:    $Author: Elvis Chen $
Date:      $Date: 2011/07/04 15:28:30 $
Version:   $Revision: 0.99 $

==========================================================================

Copyright (c) Elvis C. S. Chen, elvis.chen@gmail.com

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

#ifndef __CameraCalibrationMainWidget_H__
#define __CameraCalibrationMainWidget_H__

// C++ includes
#include <vector>
#include <map>

// QT include
#include <QWidget>
#include <QSettings>

// VTK includes
#include <vtkSmartPointer.h>

// local includes
#include "OpenCVCameraCapture.h"
#include "QCaptureThread.h"
#include "QComputeThread.h"

#include <fstream>
#include <cv.h>

#include "ui_CameraCalibrationMainWidget.h"

// QT forward declaration
class QButtonGroup;
class QDoubleSpinBox;
class QDoubleSpinBox;
class QGroupBox;
class QPushButton;
class QSpinBox;
class QStatusBar;
class QTimer;
class QVTKWidget;

// PLUS forward declaration
class vtkPlusDataCollector;
class vtkPlusTransformRepository;
class vtkPlusChannel;
class vtkPlusDataSource;
class PlusDeviceSetSelectorWidget;
class PlusToolStateDisplayWidget;
class PlusStatusIcon;

// VTK forward declaration
class vtkActor;
class vtkLandmarkTransform;
class vtkPoints;
class vtkRenderer;
class vtkTransform;

class CameraCalibrationMainWidget : public QWidget
{
  Q_OBJECT

public:
  void SetPLUSTrackingChannel(const std::string& trackingChannel);

  void LoadLeftCameraParameters(const std::string& fileName);
  void LoadRightCameraParameters(const std::string& fileName);

protected:
  /// Initialize all UI elements
  void InitUI();

  /// Thread-safe retrieval of latest captured image
  bool RetrieveLatestImage(int cameraIndex, cv::Mat& outImage);
  bool RetrieveLatestLeftImage(cv::Mat& outImage);
  bool RetrieveLatestRightImage(cv::Mat& outImage);

  /// Process a single checkerboard
  bool ProcessCheckerBoard( const cv::Mat& image, int width, int height, double size, std::vector<cv::Point2f>& outCorners );

  /// Start collecting data from the chosen device set
  bool StartDataCollection();

  /// Calculate the corner positions of the board given the parametric description of a checkerboard
  /// \param height number of vertical squares
  /// \param width number of horizontal squares
  /// \param quadSize size of the board in unit of choice
  /// \param outCorners vector of 3d points to export the points too
  void CalcBoardCornerPositions(int height, int width, double quadSize, std::vector<cv::Point3f>& outCorners);

  bool ComputeIntrinsicsAndDistortion( int cameraIndex );

  int GetBoardWidthCalib() const;
  int GetBoardHeightCalib() const;
  double GetBoardQuadSizeCalib() const;

  void SetIntrinsicMatrix(int cameraIndex, const cv::Mat& matrix);
  cv::Mat& GetInstrinsicMatrix(int cameraIndex);

  void SetDistortionCoeffs(int cameraIndex, const cv::Mat& matrix);
  cv::Mat& GetDistortionCoeffs(int cameraIndex);

  void ShowStatusMessage(const char* message);

  double ComputeReprojectionErrors( const std::vector<std::vector<cv::Point3f> >& objectPoints,
                                    const std::vector<std::vector<cv::Point2f> >& imagePoints,
                                    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                                    const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
                                    std::vector<float>& perViewErrors, bool fisheye);

  void SaveMonoCameraParameters( const std::string& filename, const cv::Size& imageSize, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                                 const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                                 const std::vector<float>& reprojErrs, const std::vector<std::vector<cv::Point2f> >& imagePoints,
                                 double totalAvgErr );

  void SaveStereoCameraParameters( const std::string& filename, const cv::Size& imageSize,
                                   const cv::Mat& leftCameraMatrix, const cv::Mat& leftDistCoeffs,
                                   const cv::Mat& rightCameraMatrix, const cv::Mat& rightDistCoeffs,
                                   const cv::Mat& rotationMatrix, const cv::Mat& translationMatrix,
                                   const std::vector<std::vector<cv::Point2f> >& leftImagePoints, const std::vector<std::vector<cv::Point2f> >& rightImagePoints,
                                   double reprojError, const cv::Mat& essentialMatrix, const cv::Mat& fundamentalMatrix );

  /// Load mono camera parameters from file
  void LoadMonoCameraParameters(int cameraIndex, const std::string& fileName);

  /// Reset the capture counts when any parameter affecting capturing changes
  void ResetCaptureCount();

protected slots:
  /*!
  Connect to devices described in the argument configuration file in response by clicking on the Connect button
  \param aConfigFile DeviceSet configuration file path and name
  */
  void ConnectToDevicesByConfigFile(std::string);

  /// Refresh contents (e.g. GUI elements) of toolbox according to the state in the toolbox controller
  virtual void RefreshGUI();

  /// When the image capture thread fires a signal with a new image, store it and show it
  void OnImageCaptured(const cv::Mat& image, int cameraIndex);

  /// When a mono camera calibration is finished, this slot is called
  void OnMonoComputationFinished(int computeIndex, int cameraIndex, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const cv::Size& imageSize,
                                 double reprojError, double totalAvgErr, const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs, const std::vector<float>& perViewErrors);

  /// When a stereo calibration is finished, this slot is called
  void OnStereoComputationFinished(int computeIndex, double reprojError, const cv::Mat& rotationMatrix, const cv::Mat& translationMatrix, const cv::Mat& essentialMatrix, const cv::Mat& fundamentalMatrix);
  void OnStereoComputationFinished(int computeIndex, double reprojError, const cv::Mat& leftIntrinsicMatrix, const cv::Mat& leftDistCoeffs, const cv::Mat& rightIntrinsicMatrix, const cv::Mat& rightDistCoeffs,
                                   const cv::Mat& rotationMatrix, const cv::Mat& translationMatrix, const cv::Mat& essentialMatrix, const cv::Mat& fundamentalMatrix);

  // start the video feeds
  void OnLeftCameraIndexChanged( int index );
  void OnRightCameraIndexChanged( int index );

  /// Capture and process images to find checkerboard corners
  void CaptureAndProcessStereoImages();
  void CaptureAndProcessLeftImage();
  void CaptureAndProcessRightImage();
  void CaptureAndProcessImage(int cameraIndex);

  void CalibBoardWidthValueChanged(int i);
  void CalibBoardHeightValueChanged(int i);
  void CalibBoardQuadSizeValueChanged(double i);

  // compute the intrinsics
  void ComputeLeftIntrinsic();
  void ComputeRightIntrinsic();
  void ComputeStereoCalibration();

  // optical flow tracking
  void OpticalFlowTracking();

protected:
  QSettings AppSettings;
  OpenCVCameraCapture* CVInternals;
  std::map<int, cv::Mat> CameraImages;
  std::map<int, int> CaptureCount;
  std::map<int, cv::Mat> IntrinsicMatrix;
  std::map<int, cv::Mat> DistortionCoefficients;
  int StereoCaptureCount;
  std::map<int, std::vector<std::vector<cv::Point2f> > > ChessboardCornerPoints;
  std::map<int, std::vector<std::vector<cv::Point2f> > > ChessboardCornerPointsCount;
  int MinBoardNeeded;
  int LeftCameraIndex;
  int RightCameraIndex;
  QCaptureThread*                               LeftCameraCaptureThread;
  QCaptureThread*                               RightCameraCaptureThread;
  std::map<int, QComputeThread*>                ComputeThreads;
  int                                           LatestComputeIndex;
  QMutex                                        OpenCVInternalsMutex;
  QMutex                                        LeftImageStorageMutex;
  QMutex                                        RightImageStorageMutex;
  QTimer*                                       OpticalFlowTrackingTimer;
  QTimer*                                       GUITimer;
  QStatusBar*                                   StatusBar;
  bool LeftIntrinsicAvailable;
  bool RightIntrinsicAvailable;
  bool LeftDistortionAvailable;
  bool RightDistortionAvailable;
  std::map<int, std::vector< std::vector< cv::Point2f > > > StereoImagePoints;
  std::map<int, std::vector< std::vector< cv::Point2f > > > ReprojectionPoints;
  std::map<int, std::vector< std::vector< cv::Point2f > > > HomographyPoints;

  /// PLUS related variables
  std::string TrackingDataChannelName;
  vtkSmartPointer< vtkPlusDataCollector >       DataCollector;
  vtkPlusChannel*                               TrackingDataChannel;
  vtkSmartPointer< vtkPlusTransformRepository > TransformRepository;
  PlusDeviceSetSelectorWidget*                  DeviceSetSelectorWidget;
  PlusToolStateDisplayWidget*                   ToolStateDisplayWidget;
  PlusStatusIcon*                               StatusIcon;

public:
  CameraCalibrationMainWidget(QWidget* parent = 0);
  ~CameraCalibrationMainWidget();

private:
  Ui::CameraCalibrationMainWidget ui;
};

#endif