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
#include "OpenCVInternals.h"
#include "mathUtil.h"

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
  enum processingMode
  {
    forCalibration, // save the points into external structures
    forLandmark, // save the points to vtkPoints
    forEvaluation, // save the points to file
    forEvaluation2,
    forFundamental // for computing the fundamental matrix
  };

  void SetPLUSTrackingChannel(const std::string& trackingChannel);

protected:
  /// centralized place to create all vtk objects
  void CreateVTKObjects();

  /// centralized place to setup all vtk pipelines
  void SetupVTKPipeline();

  // process a single checkerboard
  int ProcessCheckerBoard( int cameraIndex, int width, int height, double size, processingMode mode,
                           vtkPoints* source, vtkPoints* target, std::string videoTitle );

  bool StartDataCollection();

  void CalcBoardCornerPositions(int height, int width, double quadSize, std::vector<cv::Point3d>& corners);

  void ComputeIntrinsicsAndDistortion( int cameraIndex, double& totalAvgErr, std::vector<float>& perViewErrors );
  void ValidateStylusVideo( int cameraIndex, std::string videoTitle, double* pos );

  void InitUI();

  int GetBoardWidthCalib() const;
  int GetBoardHeightCalib() const;
  double GetBoardQuadSizeCalib() const;

  void ShowStatusMessage(const char* message);

  double ComputeReprojectionErrors( const std::vector<std::vector<cv::Point3d> >& objectPoints,
                                    const std::vector<std::vector<cv::Point2f> >& imagePoints,
                                    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                                    const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
                                    std::vector<float>& perViewErrors, bool fisheye);

protected slots:
  /*!
  Connect to devices described in the argument configuration file in response by clicking on the Connect button
  \param aConfigFile DeviceSet configuration file path and name
  */
  void ConnectToDevicesByConfigFile(std::string);

  /// Refresh contents (e.g. GUI elements) of toolbox according to the state in the toolbox controller
  virtual void RefreshContent();

  // OpenCV
  void ResetCalibrationCheckerboards();

  // start the video feeds
  void OnLeftCameraIndexChanged( int index );
  void OnRightCameraIndexChanged( int index );

  // start opencv video
  void UpdateLeftVideo();
  void UpdateRightVideo();

  // process the recorded image
  void CaptureAndProcessLeftImage();
  void CaptureAndProcessRightImage();

  void CalibBoardWidthValueChanged(int i);
  void CalibBoardHeightValueChanged(int i);
  void CalibBoardQuadSizeValueChanged(double i);

  void ValidationBoardWidthValueChanged(int i);
  void ValidationBoardHeightValueChanged(int i);
  void ValidationBoardQuadSizeValueChanged(double i);

  // compute the intrinsics
  void ComputeLeftIntrinsic();
  void ComputeRightIntrinsic();

  // for finding the landmark registration
  // between the optical axis and the attached reference tool
  void DrawLeftChessBoardCorners();
  void DrawRightChessBoardCorners();
  void ComputeHMDRegistration();

  // collect a single point using the sharp tool
  void CollectStylusPoint();
  // perform landmark registration of the calibration checkerboard to reference tool
  void PerformBoardRegistration();

  // collect a single point using the sharp tool
  void CollectRegPoint();
  // perform landmark registration of the validation checkerboard to reference tool
  void PerformRegBoardRegistration();

  // optical flow tracking
  void OpticalFlowTracking();

  // validation
  void ValidateStylus();
  void ValidateStylusStartTimer( bool v );
  void ValidateChess();
  void ValidateValidChess();
  void ValidateChessVisual();
  void ValidateChessVisualTimer( bool v);
  void ValidateChessStartTimer( bool v );

  // file operation
  void LoadLeftIntrinsic();
  void LoadRightIntrinsic();
  void LoadLeftDistortion();
  void LoadRightDistortion();
  void LoadLeftLandmark();
  void LoadRightLandmark();
  void LoadChessRegistration();
  void LoadValidChessRegistration();

  void WriteIntrinsicsToFile( int cameraIndex, const std::string& filename );
  void WriteDistortionToFile( int cameraIndex, const std::string& filename );

  /// acquire images from both camera, find the corners, and store them
  void StereoAcquire();
  void ComputeFundamentalMatrix();

protected:
  QSettings appSettings;
  OpenCVInternals* CVInternals;
  std::map<int, cv::Mat> CameraImages;
  std::map<int, int> CaptureCount;
  std::map<int, std::vector<std::vector<cv::Point2f> > > image_points;
  std::map<int, std::vector<std::vector<cv::Point2f> > > point_counts;
  int MinBoardNeeded;
  int LeftCameraIndex;
  int RightCameraIndex;
  ofstream* fileOutput[4];
  QTimer*                                       TrackingDataTimer;
  QTimer*                                       LeftCameraTimer;
  QTimer*                                       RightCameraTimer;
  QTimer*                                       ValidateStylusTimer;
  QTimer*                                       ValidateChessTimer;
  QTimer*                                       ValidateTrackingTimer;
  QTimer*                                       GUITimer;
  QStatusBar*                                   StatusBar;
  vtkSmartPointer< vtkPoints >                  BoardSource;
  vtkSmartPointer< vtkPoints >                  BoardTarget;
  vtkSmartPointer< vtkLandmarkTransform >       BoardRegTransform;
  vtkSmartPointer< vtkPoints >                  ValidBoardSource;
  vtkSmartPointer< vtkPoints >                  ValidBoardTarget;
  vtkSmartPointer< vtkLandmarkTransform >       ValidBoardRegTransform;
  vtkSmartPointer< vtkPoints >                  BoardCornerLeftSource;
  vtkSmartPointer< vtkPoints >                  BoardCornerLeftTarget;
  vtkSmartPointer< vtkPoints >                  BoardCornerRightSource;
  vtkSmartPointer< vtkPoints >                  BoardCornerRightTarget;
  vtkSmartPointer< vtkLandmarkTransform >       LeftLandmarkTransform;
  vtkSmartPointer< vtkLandmarkTransform >       RightLandmarkTransform;
  bool LeftLandmarkAvailable, RightLandmarkAvailable;
  bool LeftIntrinsicAvailable, RightIntrinsicAvailable;
  bool LeftDistortionAvailable, RightDistortionAvailable;
  bool BoardRegAvailable, ValidBoardAvailable;
  std::vector< p3 > ImagePointsLeft, ImagePointsRight;
  std::vector< p3 > ReprojectionPointsLeft, ReprojectionPointsRight;
  std::vector< p3 > HomographyPointsLeft, HomographyPointsRight;
  std::vector< p3 > TrackedPointsLeft, TrackedPointsRight;
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

#endif // of __CameraCalibrationMainWidget_H__