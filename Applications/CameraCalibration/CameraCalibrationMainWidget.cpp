/*=========================================================================

Program:   tracking with GUI
Module:    $RCSfile: CameraCalibrationMainWidget.cpp,v $
Creator:   Elvis C. S. Chen <chene@robarts.ca>
Modifications: Adam Rankin <arankin@robarts.ca>
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

#include "CameraCalibrationMainWidget.h"

// C++ includes
#include <cstdio>
#include <cstdlib>
#include <vector>

// QT includes
#include <QButtonGroup>
#include <QDesktopWidget>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QMainWindow>
#include <QPixmap>
#include <QPushButton>
#include <QSpinBox>
#include <QStatusBar>
#include <QString>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>

// local includes
#include "CameraCalibrationMainWidget.h"

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/cvconfig.h>

// PLUS includes
#ifdef _WIN32
#include <MediaFoundationCaptureLibrary.h>
#include <MediaFoundationVideoDevice.h>
#include <MediaFoundationVideoDevices.h>
#endif
#include <PlusDeviceSetSelectorWidget.h>
#include <PlusStatusIcon.h>
#include <PlusToolStateDisplayWidget.h>
#include <vtkPlusAccurateTimer.h>
#include <vtkPlusChannel.h>
#include <vtkPlusDataCollector.h>
#include <vtkPlusDataSource.h>
#include <vtkPlusTransformRepository.h>

//----------------------------------------------------------------------------

namespace
{

QImage cvMatToQImage(const cv::Mat& src)
{
  cv::Mat temp;

  switch(src.type())
  {
  case CV_8UC1:
    cvtColor(src, temp, CV_GRAY2RGB); // cvtColor makes a copy
    break;
  case CV_8UC3:
    cvtColor(src, temp, CV_BGR2RGB); // cvtColor makes a copy
    break;
  }

  QImage dest;
  if( temp.isContinuous() )
  {
    dest = QImage((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
  }

  dest.bits();
  return dest;
}

const static double VIDEO_WIDGET_SCREEN_WIDTH_RATIO = 0.25;
const static int INVALID_CAMERA_INDEX = -1;
}

CameraCalibrationMainWidget::CameraCalibrationMainWidget(QWidget* parent)
  : QWidget(parent)
  , AppSettings("cameraCalibration.ini", QSettings::IniFormat)
  , TrackingDataChannel(NULL)
  , LeftCameraIndex(INVALID_CAMERA_INDEX)
  , RightCameraIndex(INVALID_CAMERA_INDEX)
  , MinBoardNeeded(6)
  , TrackingDataChannelName("")
  , StereoCaptureCount(0)
  , DeviceSetSelectorWidget(NULL)
  , ToolStateDisplayWidget(NULL)
  , StatusIcon(NULL)
  , GUITimer(NULL)
  , OpticalFlowTrackingTimer(NULL)
  , LeftIntrinsicAvailable(false)
  , RightIntrinsicAvailable(false)
  , LeftDistortionAvailable(false)
  , RightDistortionAvailable(false)
  , DataCollector(vtkSmartPointer< vtkPlusDataCollector >::New())
{
  // Set up UI
  ui.setupUi(this);

  DeviceSetSelectorWidget = new PlusDeviceSetSelectorWidget(ui.groupBox_DataCollection);
  DeviceSetSelectorWidget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
  DeviceSetSelectorWidget->SetDeviceSetComboBoxMaximumSizeRatio(0.2);
  connect( DeviceSetSelectorWidget, SIGNAL( ConnectToDevicesByConfigFileInvoked(std::string) ), this, SLOT( ConnectToDevicesByConfigFile(std::string) ) );
  ToolStateDisplayWidget = new PlusToolStateDisplayWidget(ui.groupBox_DataCollection);
  ToolStateDisplayWidget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);

  QVBoxLayout* dataCollectionLayout = qobject_cast<QVBoxLayout*>(ui.groupBox_DataCollection->layout());
  if( dataCollectionLayout != NULL )
  {
    dataCollectionLayout->addWidget(DeviceSetSelectorWidget);
    dataCollectionLayout->addWidget(ToolStateDisplayWidget);
  }

  if( parent != NULL )
  {
    QMainWindow* mainWindow = qobject_cast<QMainWindow*>(parent);
    if( mainWindow != NULL )
    {
      StatusBar = mainWindow->statusBar();
      StatusIcon = new PlusStatusIcon(mainWindow);
      StatusBar->addPermanentWidget(StatusIcon);
    }
  }

  CVInternals = new OpenCVInternals();

  LeftCameraCaptureThread = new QCaptureThread();
  RightCameraCaptureThread = new QCaptureThread();
  LeftCameraCaptureThread->SetOpenCVInternals(*CVInternals);
  RightCameraCaptureThread->SetOpenCVInternals(*CVInternals);
  LeftCameraCaptureThread->SetCommonMutex(&OpenCVInternalsMutex);
  RightCameraCaptureThread->SetCommonMutex(&OpenCVInternalsMutex);

  InitUI();

  fileOutput[0] = new ofstream( "Left_Results.csv" );
  fileOutput[1] = new ofstream( "Right_Results.csv" );
  fileOutput[2] = new ofstream( "triangulation.csv" );
  fileOutput[3] = new ofstream( "Results.csv" );

  for(int i = 0; i < 2; i++)
  {
    *(fileOutput[i]) << "IndexNumber" << ", " <<"ImageSegX" << ", " << "ImageSegY" << ", "
                     << "HomographyX" << ", " << "HomographyY" << ", " << "HomographyZ" << ", "
                     << "ReprojectionX" << ", " << "ReprojectionY" << ", "
                     << "ActualPosX" << ", " << "ActualPosY" << ", " << "ActualPosZ" << std::endl;
  }

  *(fileOutput[2]) << "triangulate_x" << ", " << "triangulate_y" << ", " << "triangulate_z" << std::endl;

  *(fileOutput[3]) << "IndexNumber" << ", " <<"ImageSegX" << ", " << "ImageSegY" << ", "
                   << "HomographyX" << ", " << "HomographyY" << ", " << "HomographyZ" << ", "
                   << "ReprojectionX" << ", " << "ReprojectionY" << ", "
                   << "ActualPosX" << ", " << "ActualPosY" << ", " << "ActualPosZ" << ", , ,"
                   << "IndexNumber" << ", " << "ImageSegX" << ", " << "ImageSegY" << ", "
                   << "HomographyX" << ", " << "HomographyY" << ", " << "HomographyZ" << ", "
                   << "ReprojectionX" << ", " << "ReprojectionY" << ", "
                   << "ActualPosX" << ", " << "ActualPosY" << ", " << "ActualPosZ" << ", , ,"
                   << "triangulate_x" << ", " << "triangulate_y" << ", " << "triangulate_z" << std::endl;
}

//---------------------------------------------------------
CameraCalibrationMainWidget::~CameraCalibrationMainWidget()
{
  AppSettings.sync();
  DataCollector->Stop();

  LeftCameraCaptureThread->StopCapture(true);
  delete LeftCameraCaptureThread;
  RightCameraCaptureThread->StopCapture(true);
  delete RightCameraCaptureThread;
  delete CVInternals;

  for(int i = 0; i < 4; i++)
  {
    if( fileOutput[i] )
    {
      fileOutput[i]->close();
      delete fileOutput[i];
    }
  }

  delete GUITimer;
  delete OpticalFlowTrackingTimer;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::InitUI()
{
  // create timers;
  OpticalFlowTrackingTimer = new QTimer( this );
  GUITimer = new QTimer( this );

  GUITimer->start(33);

  qRegisterMetaType< cv::Mat >("cv::Mat");
  connect(LeftCameraCaptureThread, SIGNAL( capturedImage(const cv::Mat&, int) ), this, SLOT(OnImageCaptured(const cv::Mat&, int)));
  connect(RightCameraCaptureThread, SIGNAL( capturedImage(const cv::Mat&, int) ), this, SLOT(OnImageCaptured(const cv::Mat&, int)));

  ui.spinBox_BoardWidthCalib->setValue(AppSettings.value("calib/boardWidth", 7).toInt());
  ui.spinBox_BoardHeightCalib->setValue(AppSettings.value("calib/boardHeight", 5).toInt());
  ui.doubleSpinBox_QuadSizeCalib->setValue(AppSettings.value("calib/quadSize", 18.125).toDouble());

  connect( GUITimer, SIGNAL( timeout() ), this, SLOT( RefreshGUI() ) );

  // Disable acquisition until cameras are selected
  ui.pushButton_CaptureLeft->setEnabled(false);
  ui.pushButton_CaptureRight->setEnabled(false);
  ui.pushButton_ComputeLeftIntrinsic->setEnabled(false);
  ui.pushButton_ComputeRightIntrinsic->setEnabled(false);
  ui.pushButton_StereoAcquire->setEnabled(false);
  ui.pushButton_StereoCompute->setEnabled(false);
  ui.pushButton_VisualTracking->setEnabled(false);

  // Configuration
  connect( ui.spinBox_BoardWidthCalib, SIGNAL( valueChanged( int ) ),
           this, SLOT( CalibBoardWidthValueChanged( int ) ) );
  connect( ui.spinBox_BoardHeightCalib, SIGNAL( valueChanged( int ) ),
           this, SLOT( CalibBoardHeightValueChanged( int ) ) );
  connect( ui.doubleSpinBox_QuadSizeCalib, SIGNAL( valueChanged( double ) ),
           this, SLOT( CalibBoardQuadSizeValueChanged( double ) ) );

  // Calibration
  connect( ui.pushButton_CaptureLeft, SIGNAL( clicked() ),
           this, SLOT( CaptureAndProcessLeftImage() ) );
  connect( ui.pushButton_CaptureRight, SIGNAL( clicked() ),
           this, SLOT( CaptureAndProcessRightImage() ) );
  connect( ui.pushButton_ComputeLeftIntrinsic, SIGNAL( clicked() ),
           this, SLOT( ComputeLeftIntrinsic() ) );
  connect( ui.pushButton_ComputeRightIntrinsic, SIGNAL( clicked() ),
           this, SLOT( ComputeRightIntrinsic() ) );
  connect( ui.pushButton_StereoAcquire, SIGNAL( clicked() ),
           this, SLOT( StereoAcquire() ) );
  connect( ui.pushButton_StereoCompute, SIGNAL( clicked() ),
           this, SLOT( ComputeFundamentalMatrix() ) );

  // Validation
  connect( OpticalFlowTrackingTimer, SIGNAL( timeout() ),
           this, SLOT( OpticalFlowTracking() ) );

  ui.comboBox_LeftCamera->addItem("None", QVariant(INVALID_CAMERA_INDEX));
  ui.comboBox_RightCamera->addItem("None", QVariant(INVALID_CAMERA_INDEX));
#ifdef _WIN32
  MfVideoCapture::MediaFoundationCaptureLibrary::GetInstance().BuildListOfDevices();
  for( int i = 0; i < MfVideoCapture::MediaFoundationVideoDevices::GetInstance().GetCount(); ++i )
  {
    MfVideoCapture::MediaFoundationVideoDevice* device = MfVideoCapture::MediaFoundationVideoDevices::GetInstance().GetDevice(i);
    QString name = QString::fromWCharArray(device->GetName());
    ui.comboBox_LeftCamera->addItem(name, QVariant(i));
    ui.comboBox_RightCamera->addItem(name, QVariant(i));
  }
#else
  // TODO : other camera libraries to detect cameras
#endif

  connect( ui.comboBox_LeftCamera, SIGNAL( currentIndexChanged( int ) ),
           this, SLOT(OnLeftCameraIndexChanged( int ) ) );
  connect( ui.comboBox_RightCamera, SIGNAL( currentIndexChanged( int ) ),
           this, SLOT(OnRightCameraIndexChanged( int ) ) );
}

//----------------------------------------------------------------------------
bool CameraCalibrationMainWidget::RetrieveLatestImage(int cameraIndex, cv::Mat& outImage)
{
  if( cameraIndex == LeftCameraIndex )
  {
    // Copy image to cv::Mat in storage
    QMutexLocker locker(&LeftImageStorageMutex);
    CameraImages[LeftCameraIndex].copyTo(outImage);

    return true;
  }
  else
  {
    QMutexLocker locker(&RightImageStorageMutex);
    CameraImages[RightCameraIndex].copyTo(outImage);

    return true;
  }

  return false;
}

//----------------------------------------------------------------------------
bool CameraCalibrationMainWidget::RetrieveLatestLeftImage(cv::Mat& outImage)
{
  if( this->LeftCameraIndex == INVALID_CAMERA_INDEX )
  {
    return false;
  }

  return this->RetrieveLatestImage(LeftCameraIndex, outImage);
}

//----------------------------------------------------------------------------
bool CameraCalibrationMainWidget::RetrieveLatestRightImage(cv::Mat& outImage)
{
  if( this->RightCameraIndex == INVALID_CAMERA_INDEX )
  {
    return false;
  }

  return this->RetrieveLatestImage(RightCameraIndex, outImage);
}

//---------------------------------------------------------
int CameraCalibrationMainWidget::GetBoardWidthCalib() const
{
  return ui.spinBox_BoardWidthCalib->value();
}

//---------------------------------------------------------
int CameraCalibrationMainWidget::GetBoardHeightCalib() const
{
  return ui.spinBox_BoardHeightCalib->value();
}

//---------------------------------------------------------
double CameraCalibrationMainWidget::GetBoardQuadSizeCalib() const
{
  return ui.doubleSpinBox_QuadSizeCalib->value();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::ShowStatusMessage(const char* message)
{
  if( StatusBar != NULL )
  {
    StatusBar->showMessage(message);
  }
}

//----------------------------------------------------------------------------
double CameraCalibrationMainWidget::ComputeReprojectionErrors(const std::vector<std::vector<cv::Point3f> >& objectPoints,
    const std::vector<std::vector<cv::Point2f> >& imagePoints,
    const std::vector<cv::Mat>& rvecs,
    const std::vector<cv::Mat>& tvecs,
    const cv::Mat& cameraMatrix ,
    const cv::Mat& distCoeffs,
    std::vector<float>& perViewErrors,
    bool fisheye)
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

//---------------------------------------------------------
void CameraCalibrationMainWidget::SetPLUSTrackingChannel(const std::string& trackingChannel)
{
  this->TrackingDataChannelName = trackingChannel;
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::LoadLeftCameraParameters(const std::string& fileName)
{
  if( LeftCameraIndex == INVALID_CAMERA_INDEX )
  {
    LOG_ERROR("Unable to load parameters for left camera. Camera isn't selected.");
    ShowStatusMessage("Unable to load parameters for left camera. Camera isn't selected.");
    return;
  }

  this->LoadCameraParameters(LeftCameraIndex, fileName);

  LeftIntrinsicAvailable = true;
  LeftDistortionAvailable = true;
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::LoadRightCameraParameters(const std::string& fileName)
{
  if( RightCameraIndex == INVALID_CAMERA_INDEX )
  {
    LOG_ERROR("Unable to load parameters for right camera. Camera isn't selected.");
    ShowStatusMessage("Unable to load parameters for right camera. Camera isn't selected.");
    return;
  }

  this->LoadCameraParameters(RightCameraIndex, fileName);

  RightIntrinsicAvailable = true;
  RightDistortionAvailable = true;
}

//---------------------------------------------------------
// initialize the data collection system
bool CameraCalibrationMainWidget::StartDataCollection()
{
  if( DataCollector->ReadConfiguration(vtkPlusConfig::GetInstance()->GetDeviceSetConfigurationData()) != PLUS_SUCCESS )
  {
    LOG_ERROR("Unable to parse configuration file.");
    return false;
  }

  if ( DataCollector->Start() != PLUS_SUCCESS )
  {
    return false;
  }

  if( DataCollector->GetChannel(TrackingDataChannel, TrackingDataChannelName) != PLUS_SUCCESS )
  {
    LOG_WARNING("Channel: " << TrackingDataChannelName << " is not found. Falling back to any available channel.");
    if( DataCollector->GetFirstChannel(TrackingDataChannel) != PLUS_SUCCESS )
    {
      LOG_ERROR("No channels to fall back too. Aborting.");
      return false;
    }
  }

  return true;
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CalcBoardCornerPositions(int height, int width, double quadSize, std::vector<cv::Point3f>& outCorners)
{
  outCorners.clear();

  for( int i = 0; i < height; ++i )
  {
    for( int j = 0; j < width; ++j )
    {
      outCorners.push_back(cv::Point3d(j*quadSize, i*quadSize, 0));
    }
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ConnectToDevicesByConfigFile(std::string aConfigFile)
{
  QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));

  // Disconnect
  // Empty parameter string means disconnect from device
  if ( aConfigFile.empty() )
  {
    LOG_INFO("Disconnect request successful");
    DeviceSetSelectorWidget->ClearDescriptionSuffix();
    DeviceSetSelectorWidget->SetConnectionSuccessful(false);
    DeviceSetSelectorWidget->ShowResetTrackerButton(false);
    ToolStateDisplayWidget->InitializeTools(NULL, false);

    DataCollector = vtkSmartPointer< vtkPlusDataCollector >::New();

    QApplication::restoreOverrideCursor();
    return;
  }

  LOG_INFO("Connect using configuration file: " << aConfigFile);

  // Read configuration
  vtkSmartPointer<vtkXMLDataElement> configRootElement = vtkSmartPointer<vtkXMLDataElement>::Take(vtkXMLUtilities::ReadElementFromFile(aConfigFile.c_str()));
  if (configRootElement == NULL)
  {
    LOG_ERROR("Unable to read configuration from file " << aConfigFile);

    DeviceSetSelectorWidget->SetConnectionSuccessful(false);
    ToolStateDisplayWidget->InitializeTools(NULL, false);

    QApplication::restoreOverrideCursor();
    return;
  }

  LOG_INFO("Device set configuration is read from file: " << aConfigFile);
  vtkPlusConfig::GetInstance()->SetDeviceSetConfigurationData(configRootElement);

  // If connection has been successfully created then start data collection
  if ( !DeviceSetSelectorWidget->GetConnectionSuccessful() )
  {
    LOG_INFO("Connect to devices");

    // Disable main window
    this->setEnabled(false);

    // Create dialog
    QDialog* connectDialog = new QDialog(this, Qt::Dialog);
    connectDialog->setMinimumSize(QSize(360,80));
    connectDialog->setWindowTitle(tr("Camera Calibration"));
    connectDialog->setStyleSheet("QDialog { background-color: rgb(224, 224, 224); }");

    QLabel* connectLabel = new QLabel(QString("Connecting to devices, please wait..."), connectDialog);
    connectLabel->setFont(QFont("SansSerif", 16));

    QHBoxLayout* layout = new QHBoxLayout();
    layout->addWidget(connectLabel);

    connectDialog->setLayout(layout);
    connectDialog->show();

    QApplication::processEvents();

    // Connect to devices
    if ( !this->StartDataCollection() )
    {
      LOG_ERROR("Unable to start collecting data!");
      DeviceSetSelectorWidget->SetConnectionSuccessful(false);
      ToolStateDisplayWidget->InitializeTools(NULL, false);
    }
    else
    {
      DeviceCollection aCollection;
      if( DataCollector->GetDevices(aCollection) != PLUS_SUCCESS )
      {
        LOG_ERROR("Unable to load the list of devices.");
        QApplication::restoreOverrideCursor();
        return;
      }

      // Successful connection
      DeviceSetSelectorWidget->SetConnectionSuccessful(true);
      ToolStateDisplayWidget->InitializeTools(TrackingDataChannel, true);

      vtkPlusConfig::GetInstance()->SaveApplicationConfigurationToFile();
    }

    // Close dialog
    connectDialog->done(0);
    connectDialog->hide();
    delete connectDialog;
  }

  // Re-enable main window
  this->setEnabled(true);
  QApplication::restoreOverrideCursor();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::RefreshGUI()
{
  if( ToolStateDisplayWidget != NULL && ToolStateDisplayWidget->IsInitialized() )
  {
    ToolStateDisplayWidget->Update();
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::OnLeftCameraIndexChanged(int index)
{
  int cameraIndex = ui.comboBox_LeftCamera->currentData().toInt();

  if( LeftCameraIndex == cameraIndex )
  {
    return;
  }

  if( LeftCameraIndex != INVALID_CAMERA_INDEX )
  {
    LeftCameraCaptureThread->StopCapture(true);

    // Remove camera entry from maps
    CameraImages.erase(CameraImages.find(LeftCameraIndex));
    CaptureCount.erase(CaptureCount.find(LeftCameraIndex));
    ChessboardCornerPoints.erase(ChessboardCornerPoints.find(LeftCameraIndex));
    ChessboardCornerPointsCount.erase(ChessboardCornerPointsCount.find(LeftCameraIndex));
  }

  if( cameraIndex == INVALID_CAMERA_INDEX )
  {
    ui.pushButton_CaptureLeft->setEnabled(false);
    ui.pushButton_StereoAcquire->setEnabled(false);

    LeftCameraIndex = cameraIndex;
    return;
  }

  ui.pushButton_ComputeLeftIntrinsic->setEnabled(false);

  if( !LeftCameraCaptureThread->StartCapture(cameraIndex) )
  {
    LOG_ERROR("Unable to start capturing thread for camera index: " << cameraIndex);
    LeftCameraIndex = INVALID_CAMERA_INDEX;
    return;
  }

  LeftCameraIndex = cameraIndex;
  CameraImages[LeftCameraIndex];
  CaptureCount[LeftCameraIndex] = 0;
  StereoCaptureCount = 0;
  ChessboardCornerPoints[LeftCameraIndex];
  ChessboardCornerPointsCount[LeftCameraIndex];

  ui.pushButton_CaptureLeft->setEnabled(true);

  if( RightCameraIndex != INVALID_CAMERA_INDEX )
  {
    ui.pushButton_StereoAcquire->setEnabled(true);
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::OnRightCameraIndexChanged(int index)
{
  int cameraIndex = ui.comboBox_RightCamera->currentData().toInt();

  if( RightCameraIndex == cameraIndex )
  {
    return;
  }

  if( RightCameraIndex != INVALID_CAMERA_INDEX )
  {
    RightCameraCaptureThread->StopCapture(true);

    // Remove camera entry from maps
    CameraImages.erase(CameraImages.find(RightCameraIndex));
    CaptureCount.erase(CaptureCount.find(RightCameraIndex));
    ChessboardCornerPoints.erase(ChessboardCornerPoints.find(RightCameraIndex));
    ChessboardCornerPointsCount.erase(ChessboardCornerPointsCount.find(RightCameraIndex));
  }

  if( cameraIndex == INVALID_CAMERA_INDEX )
  {
    ui.pushButton_CaptureRight->setEnabled(false);
    ui.pushButton_StereoAcquire->setEnabled(false);

    RightCameraIndex = cameraIndex;
    return;
  }

  ui.pushButton_ComputeRightIntrinsic->setEnabled(false);

  if( !RightCameraCaptureThread->StartCapture(cameraIndex) )
  {
    LOG_ERROR("Unable to start capturing thread for camera index: " << cameraIndex);
    RightCameraIndex = INVALID_CAMERA_INDEX;
    return;
  }

  RightCameraIndex = cameraIndex;
  CameraImages[RightCameraIndex];
  CaptureCount[RightCameraIndex] = 0;
  StereoCaptureCount = 0;
  ChessboardCornerPoints[RightCameraIndex];
  ChessboardCornerPointsCount[RightCameraIndex];

  ui.pushButton_CaptureRight->setEnabled(true);

  if( LeftCameraIndex != INVALID_CAMERA_INDEX )
  {
    ui.pushButton_StereoAcquire->setEnabled(true);
  }

  // Acquire a frame, determine aspect ratio and change scaled view size
  // TODO : figure out why this widget can't be resized
  /*
  cv::Mat image;
  if( CVInternals->QueryFrame(RightCameraIndex, image) )
  {
    double aspectRatio = (double)image.cols / image.rows;

    QDesktopWidget widget;
    QRect screenSize = widget.availableGeometry(widget.screenNumber(this));

    // Take up VIDEO_WIDGET_SCREEN_WIDTH_RATIO of the screen width, scale height to aspect ratio
    ui.scaledView_RightVideo->setPixmap(QPixmap::fromImage(cvMatToQImage(image)));
    ui.scaledView_RightVideo->setFixedSize(screenSize.width() * VIDEO_WIDGET_SCREEN_WIDTH_RATIO, screenSize.width() * VIDEO_WIDGET_SCREEN_WIDTH_RATIO * aspectRatio);
  }
  */
}

//---------------------------------------------------------
// process a single checkerboard
bool CameraCalibrationMainWidget::ProcessCheckerBoard( const cv::Mat& image, int width, int height, double size, std::vector<cv::Point2f>& outCorners )
{
  LOG_INFO("Checkerboard dimension: " << width << "x" << height << "x" << size);

  // make a copy of the current feed
  // need to do so as the current feed may be self-updating
  cv::Mat copy(image);

  cv::Size board_sz(width, height);
  int board_n = width * height;

  cv::Point text_origin;
  int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;

  //if(!s.useFisheye)
  {
    // fast check erroneously fails with high distortions like fisheye
    chessBoardFlags |= cv::CALIB_CB_FAST_CHECK;
  }
  bool found = cv::findChessboardCorners( copy, board_sz, outCorners, chessBoardFlags );

  //if the checkerboard is not entirely visible, chuck this image and return
  if ( !found )
  {
    return false;
  }

  cv::Mat viewGray;
  cv::cvtColor(copy, viewGray, cv::COLOR_BGR2GRAY);
  cv::cornerSubPix( viewGray, outCorners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1 ));

  // sometimes OpenCV returned a flipped image
  bool flip = false;

  if ( outCorners[board_n-1].x < outCorners[0].x )
  {
    flip = true;
    LOG_INFO("Image is flipped.");
  }

  cv::drawChessboardCorners( copy, board_sz, cv::Mat(outCorners), found );

  cv::imshow( "Chessboard Corners", copy );
  cv::waitKey(25);

  return true;
}

//---------------------------------------------------------
// process the recorded image
void CameraCalibrationMainWidget::CaptureAndProcessLeftImage()
{
  CaptureAndProcessImage(LeftCameraIndex);
  if( CaptureCount[LeftCameraIndex] >= MinBoardNeeded )
  {
    ui.pushButton_ComputeLeftIntrinsic->setEnabled(true);
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::CaptureAndProcessRightImage()
{
  CaptureAndProcessImage(RightCameraIndex);
  if( CaptureCount[RightCameraIndex] >= MinBoardNeeded )
  {
    ui.pushButton_ComputeRightIntrinsic->setEnabled(true);
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CaptureAndProcessImage(int cameraIndex)
{
  cv::Mat image;
  if( this->RetrieveLatestImage(cameraIndex, image) )
  {
    std::vector<cv::Point2f> corners;
    if( ProcessCheckerBoard( image, GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(), corners) )
    {
      // Store located corners in local database of corners
      ChessboardCornerPoints[cameraIndex].push_back(corners);
      CaptureCount[cameraIndex]++;
      std::stringstream ss;
      ss << "Success! Captured " << CaptureCount[cameraIndex] << " calibration image thus far.";
      LOG_INFO(ss.str());
      ShowStatusMessage(ss.str().c_str());
    }
    else
    {
      LOG_WARNING("Unable to locate checkerboard in camera image. Try again.");
      ShowStatusMessage("Unable to locate checkerboard in camera image. Try again.");
    }
  }
  else
  {
    LOG_ERROR("Unable to retrieve latest iamge.");
    ShowStatusMessage("Unable to retrieve latest image.");
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CalibBoardWidthValueChanged(int i)
{
  AppSettings.setValue("calib/boardWidth", i);
  ResetCaptureCount();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CalibBoardHeightValueChanged(int i)
{
  AppSettings.setValue("calib/boardHeight", i);
  ResetCaptureCount();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CalibBoardQuadSizeValueChanged(double i)
{
  AppSettings.setValue("calib/quadSize", i);
  ResetCaptureCount();
}

//---------------------------------------------------------
// compute the intrinsics
void CameraCalibrationMainWidget::ComputeLeftIntrinsic()
{
  double totalAvgErr;
  std::vector<float> perViewErrors;
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  if( !this->ComputeIntrinsicsAndDistortion( LeftCameraIndex, totalAvgErr, rvecs, tvecs, perViewErrors ) )
  {
    return;
  }

  QString saveFileName = QFileDialog::getSaveFileName(this, "Save intrinsics and distortion", vtkPlusConfig::GetInstance()->GetOutputDirectory().c_str(), "*.xml");

  if( saveFileName.indexOf(".xml") == -1 )
  {
    saveFileName.append(".xml");
  }

  cv::Mat image;
  if( this->RetrieveLatestLeftImage(image) )
  {
    this->SaveCameraParameters( saveFileName.toStdString(), cv::Size(image.size[0], image.size[1]), *CVInternals->GetInstrinsicMatrix(LeftCameraIndex),
                                       *CVInternals->GetDistortionCoeffs(LeftCameraIndex), rvecs, tvecs, perViewErrors, ChessboardCornerPoints[LeftCameraIndex], totalAvgErr );
  }

  LOG_INFO("Camera parameters written computed and saved to: " << saveFileName.toStdString());
  ShowStatusMessage("Success. Camera parameters written to file.");
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ComputeRightIntrinsic()
{
  double totalAvgErr;
  std::vector<float> perViewErrors;
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  if( !this->ComputeIntrinsicsAndDistortion( RightCameraIndex, totalAvgErr, rvecs, tvecs, perViewErrors ) )
  {
    return;
  }

  QString saveFileName = QFileDialog::getSaveFileName(this, "Save intrinsics and distortion", vtkPlusConfig::GetInstance()->GetOutputDirectory().c_str(), "*.xml");

  if( saveFileName.indexOf(".xml") == -1 )
  {
    saveFileName.append(".xml");
  }

  cv::Mat image;
  if( this->RetrieveLatestRightImage(image) )
  {
    this->SaveCameraParameters( saveFileName.toStdString(), cv::Size(image.size[0], image.size[1]), *CVInternals->GetInstrinsicMatrix(RightCameraIndex),
                                       *CVInternals->GetDistortionCoeffs(RightCameraIndex), rvecs, tvecs, perViewErrors, ChessboardCornerPoints[RightCameraIndex], totalAvgErr );
  }

  LOG_INFO("Camera parameters written computed and saved to: " << saveFileName.toStdString());
  ShowStatusMessage("Success. Camera parameters written to file.");
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::SaveCameraParameters( const std::string& filename, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
    const std::vector<float>& reprojErrs, const std::vector<std::vector<cv::Point2f> >& imagePoints,
    double totalAvgErr )
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  fs << "calibration_time_utc" << vtkPlusAccurateTimer::GetUniversalTime();

  fs << "camera_matrix" << cameraMatrix;

  fs << "distortion_coefficients" << distCoeffs;

  fs << "image_width" << imageSize.width;
  fs << "image_height" << imageSize.height;
  fs << "board_width" << ui.spinBox_BoardWidthCalib->value();
  fs << "board_height" << ui.spinBox_BoardHeightCalib->value();
  fs << "square_size" << ui.doubleSpinBox_QuadSizeCalib->value();

  fs << "avg_reprojection_error" << totalAvgErr;
  if (!reprojErrs.empty())
  {
    fs << "per_view_reprojection_errors" << cv::Mat(reprojErrs);
  }

  if(!rvecs.empty() && !tvecs.empty() )
  {
    CV_Assert(rvecs[0].type() == tvecs[0].type());
    cv::Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
    for( size_t i = 0; i < rvecs.size(); i++ )
    {
      cv::Mat r = bigmat(cv::Range(int(i), int(i+1)), cv::Range(0,3));
      cv::Mat t = bigmat(cv::Range(int(i), int(i+1)), cv::Range(3,6));

      CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
      CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
      //*.t() is MatExpr (not Mat) so we can use assignment operator
      r = rvecs[i].t();
      t = tvecs[i].t();
    }

    fs << "extrinsic_parameters" << bigmat;
  }

  if(!imagePoints.empty() )
  {
    cv::Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
    for( size_t i = 0; i < imagePoints.size(); i++ )
    {
      cv::Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
      cv::Mat imgpti(imagePoints[i]);
      imgpti.copyTo(r);
    }
    fs << "image_points" << imagePtMat;
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::LoadCameraParameters(int cameraIndex, const std::string& fileName)
{
  cv::FileStorage fs(fileName, cv::FileStorage::READ);

  cv::Mat cameraMatrix;
  fs["camera_matrix"] >> cameraMatrix;
  cv::Mat distCoeffs;
  fs["distortion_coefficients"] >> distCoeffs;

  QMutexLocker locker(&OpenCVInternalsMutex);
  CVInternals->SetIntrinsicMatrix(cameraIndex, cameraMatrix);
  CVInternals->SetDistortionCoeffs(cameraIndex, distCoeffs);
}

//---------------------------------------------------------
bool CameraCalibrationMainWidget::ComputeIntrinsicsAndDistortion( int cameraIndex, double& totalAvgErr, std::vector<cv::Mat>& rotationsVector,
    std::vector<cv::Mat>& translationsVector, std::vector<float>& perViewErrors )
{
  if ( CaptureCount[cameraIndex] < MinBoardNeeded )
  {
    std::stringstream ss;
    ss << "Not enough board images recorded: " << CaptureCount[cameraIndex] << "/" << MinBoardNeeded;
    LOG_ERROR(ss.str());
    ShowStatusMessage(ss.str().c_str());
    return false;
  }

  std::vector<std::vector<cv::Point3f> > objectPoints(1);
  CalcBoardCornerPositions(GetBoardHeightCalib(), GetBoardWidthCalib(), GetBoardQuadSizeCalib(), objectPoints[0]);
  objectPoints.resize(ChessboardCornerPoints[cameraIndex].size(), objectPoints[0]);

  cv::Mat cameraMatrix(cv::Mat::eye(3, 3, CV_64F));
  cv::Mat distCoeffs(cv::Mat::zeros(8, 1, CV_64F));

  double reprojectionError = cv::calibrateCamera(objectPoints, ChessboardCornerPoints[cameraIndex], cv::Size(CameraImages[cameraIndex].size[0], CameraImages[cameraIndex].size[1]),
                             cameraMatrix, distCoeffs, rotationsVector, translationsVector, CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);

  {
    QMutexLocker locker(&OpenCVInternalsMutex);
    CVInternals->SetIntrinsicMatrix(cameraIndex, cameraMatrix);
    CVInternals->SetDistortionCoeffs(cameraIndex, distCoeffs);
  }

  totalAvgErr = ComputeReprojectionErrors(objectPoints, ChessboardCornerPoints[cameraIndex], rotationsVector, translationsVector, cameraMatrix,
                                          distCoeffs, perViewErrors, false);

  std::stringstream ss;
  ss << "Camera calibrated with reprojection error: " << reprojectionError;
  LOG_INFO(ss.str());
  ShowStatusMessage(ss.str().c_str());

  if ( cameraIndex == LeftCameraIndex )
  {
    LeftIntrinsicAvailable = LeftDistortionAvailable = true;
  }
  else if ( cameraIndex == RightCameraIndex )
  {
    RightIntrinsicAvailable = RightDistortionAvailable = true;
  }

  return true;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::OpticalFlowTracking()
{
  // only do something when we have stereo
  if ( this->CVInternals->CameraCount() < 2 )
  {
    return;
  }

  // get both images
  cv::Mat leftImage;
  this->RetrieveLatestLeftImage(leftImage);
  cv::Mat rightImage;
  this->RetrieveLatestRightImage(rightImage);

  // size of the images
  cv::Size img_sz(leftImage.size[0], leftImage.size[1]);

  // detect a red ball
  cv::Scalar hsv_min(150, 50, 100, 0);
  cv::Scalar hsv_max(180, 255, 255, 0);

  // create a threshold image
  cv::Mat leftThresholded( img_sz, IPL_DEPTH_8U, 1 );
  cv::Mat rightThresholded( img_sz, IPL_DEPTH_8U, 1 );
  cv::Mat leftHSV( img_sz, IPL_DEPTH_8U, 3 );
  cv::Mat rightHSV( img_sz, IPL_DEPTH_8U, 3 );

  // consult page 161 of the OpenCV book

  // convert color space to HSV as so we can segment the image
  // based on the color
  cv::cvtColor(leftImage, leftHSV, CV_BGR2HSV);
  cv::cvtColor(rightImage, rightHSV, CV_BGR2HSV);

  // threshold the HSV image
  cv::inRange(leftHSV, hsv_min, hsv_max, leftThresholded);
  cv::inRange(rightHSV, hsv_min, hsv_max, rightThresholded);

  // apply a gaussian filter to smooth the binary image
  cv::boxFilter(leftThresholded, leftThresholded, -1, cv::Size(5,5));
  cv::boxFilter(rightThresholded, rightThresholded, -1, cv::Size(5,5));

  // use Hough detector to find the sphere/circle
  std::vector<cv::Vec3f> leftCircles;
  cv::HoughCircles(leftThresholded, leftCircles, cv::HOUGH_GRADIENT, 2, leftThresholded.rows / 4);

  std::vector<cv::Vec3f> rightCircles;
  cv::HoughCircles(rightThresholded, rightCircles, cv::HOUGH_GRADIENT, 2, leftThresholded.rows / 4);

  for ( unsigned int i = 0; i < leftCircles.size(); i++ )
  {
    cv::circle(leftImage, cv::Point(leftCircles[i][0], leftCircles[i][1]), 3, cv::Scalar(0, 255, 0), -1, 8, 0);
    cv::circle(leftImage, cv::Point(leftCircles[i][0], leftCircles[i][1]), leftCircles[i][2], cv::Scalar(255, 0, 0), 3, 8, 0);
  }

  for ( unsigned int i = 0; i < rightCircles.size(); i++ )
  {
    cv::circle(rightImage, cv::Point(rightCircles[i][0], rightCircles[i][1]), 3, cv::Scalar(0, 255, 0), -1, 8, 0);
    cv::circle(rightImage, cv::Point(rightCircles[i][0], rightCircles[i][1]), rightCircles[i][2], cv::Scalar(255, 0, 0), 3, 8, 0);
  }

  cv::imshow("Left Result", leftImage);
  cv::imshow("Right Result", rightImage);
  cv::waitKey();
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ResetCaptureCount()
{
  CaptureCount[LeftCameraIndex] = 0;
  CaptureCount[RightCameraIndex] = 0;
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::OnImageCaptured(const cv::Mat& image, int cameraIndex)
{
  if( cameraIndex == LeftCameraIndex )
  {
    // Copy image to cv::Mat in storage
    QMutexLocker locker(&LeftImageStorageMutex);
    image.copyTo(CameraImages[LeftCameraIndex]);

    QImage img = cvMatToQImage(CameraImages[LeftCameraIndex]);
    ui.scaledView_LeftVideo->setPixmap(QPixmap::fromImage(img));
    ui.scaledView_LeftVideo->update();
  }
  else
  {
    QMutexLocker locker(&RightImageStorageMutex);
    image.copyTo(CameraImages[RightCameraIndex]);

    QImage img = cvMatToQImage(CameraImages[RightCameraIndex]);
    ui.scaledView_RightVideo->setPixmap(QPixmap::fromImage(img));
    ui.scaledView_RightVideo->update();
  }
}

//---------------------------------------------------------
// acquire images from both camera, find the corners, and store them
void CameraCalibrationMainWidget::StereoAcquire()
{
  // only do something when we have stereo
  if ( this->CVInternals->CameraCount() < 2 )
  {
    return;
  }

  // get both images
  cv::Mat leftImage;
  this->RetrieveLatestLeftImage(leftImage);
  cv::Mat rightImage;
  this->RetrieveLatestRightImage(rightImage);

  // make a copy
  cv::Mat leftCopy = leftImage.clone();
  cv::Mat rightCopy = rightImage.clone();

  // BW image
  cv::Mat leftGray(cv::Size(leftImage.size[0], leftImage.size[1]), CV_8UC1);
  cv::Mat rightGray(cv::Size(leftImage.size[0], leftImage.size[1]), CV_8UC1);

  cv::Size board_sz = cv::Size( GetBoardWidthCalib(), GetBoardHeightCalib() );
  int board_n = GetBoardWidthCalib() * GetBoardHeightCalib();
  std::vector<cv::Point2f> leftCorners;
  std::vector<cv::Point2f> rightCorners;

  bool leftFound = cv::findChessboardCorners(leftCopy, board_sz, leftCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
  bool rightFound = cv::findChessboardCorners(rightCopy, board_sz, rightCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

  // make sure we found equal number of corners from both images
  if ( !leftFound || !rightFound )
  {
    return;
  }

  // get subpixel accuracy
  cv::cvtColor( leftCopy, leftGray, CV_BGR2GRAY );
  cv::cvtColor( rightCopy, rightGray, CV_BGR2GRAY );

  cv::cornerSubPix(leftGray, leftCorners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ) );
  cv::cornerSubPix(rightGray, rightCorners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ) );

  cv::drawChessboardCorners(leftCopy, board_sz, leftCorners, leftFound);
  cv::drawChessboardCorners(rightCopy, board_sz, rightCorners, leftFound);

  cv::imshow("LeftResult", leftCopy);
  cv::imshow("RightResult", rightCopy);

  bool leftFlip = false;
  bool rightFlip = false;

  // sometimes OpenCV returns the corner in the wrong order
  if ( leftCorners[board_n-1].x < leftCorners[0].x )
  {
    leftFlip = true;
  }

  if ( rightCorners[board_n-1].x < rightCorners[0].x )
  {
    rightFlip = true;
  }

  // now, enter everything into the external storage
  cv::Vec3d point;
  point[2] = 0.0;
  for ( int i = 0, idx=(board_n-1); i < leftCorners.size(); i++, idx-- )
  {
    if ( !leftFlip )
    {
      point[0] = leftCorners[i].x;
      point[1] = leftCorners[i].y;
    }
    else
    {
      point[0] = leftCorners[idx].x;
      point[1] = leftCorners[idx].y;
    }
    StereoImagePoints[LeftCameraIndex].push_back( point );
  }

  LOG_INFO("Collected: " << StereoImagePoints[LeftCameraIndex].size() << " left points so far.");

  point[2] = 0.0;
  for ( int i = 0, idx=(board_n-1); i < rightCorners.size(); i++, idx-- )
  {
    if ( !rightFlip )
    {
      point[0] = rightCorners[i].x;
      point[1] = rightCorners[i].y;
    }
    else
    {
      point[0] = rightCorners[idx].x;
      point[1] = rightCorners[idx].y;
    }
    StereoImagePoints[RightCameraIndex].push_back( point );
  }

  LOG_INFO("Collected: " << StereoImagePoints[RightCameraIndex].size() << " right points so far.");

  if( StereoCaptureCount >= MinBoardNeeded )
  {
    ui.pushButton_StereoCompute->setEnabled(true);
  }

  StereoCaptureCount++;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ComputeFundamentalMatrix()
{
  int nPoints = StereoImagePoints[LeftCameraIndex].size();

  // make sure we have more than 8 points and same for both cameras
  if ( nPoints != StereoImagePoints[RightCameraIndex].size() || nPoints < 8 )
  {
    return;
  }

  cv::Mat image_pointsL(nPoints, 2, CV_32FC1);
  cv::Mat image_pointsR( nPoints, 2, CV_32FC1 );
  cv::Mat object_points( nPoints, 3, CV_32FC1 );

  // copy over
  int board_n = GetBoardWidthCalib() * GetBoardHeightCalib();

  for ( int i = 0; i < nPoints; i++ )
  {
    image_pointsL.at<float>(i,0) = StereoImagePoints[LeftCameraIndex][i].x;
    image_pointsL.at<float>(i,1) = StereoImagePoints[LeftCameraIndex][i].y;
    image_pointsR.at<float>(i,0) = StereoImagePoints[LeftCameraIndex][i].x;
    image_pointsR.at<float>(i,1) = StereoImagePoints[LeftCameraIndex][i].y;

    object_points.at<float>(i,0) = (float)((i%board_n)/GetBoardWidthCalib()) * 17.0;
    object_points.at<float>(i,1) = (float)((i%board_n)/GetBoardWidthCalib()) * 17.0;
    object_points.at<float>(i,2) = 0.0;
  }

  cv::Mat intrinsic_matrixL( 3, 3, CV_64FC1 );
  cv::Mat intrinsic_matrixR( 3, 3, CV_64FC1 );
  cv::Mat rmat( 3, 3, CV_64FC1 );
  cv::Mat tmat( 3, 1, CV_64FC1 );
  cv::Mat distortion_coeffsL( 4, 1,CV_64FC1 );
  cv::Mat distortion_coeffsR( 4, 1,CV_64FC1 );
  cv::Mat Essential_matrix( 3, 3, CV_64FC1 );
  cv::Mat Fundamental_matrix( 3, 3, CV_64FC1 );
  cv::Mat point_counts( nPoints / board_n, 1, CV_32SC1 );
  cv::Mat fundamental_matrix( 3, 3, CV_32FC1 );
  cv::Mat status( 1, nPoints, CV_8UC1 );

  for ( int i = 0; i < nPoints/board_n; i++ )
  {
    point_counts.at<float>(i,0) = board_n;
  }

  if ( LeftIntrinsicAvailable && RightIntrinsicAvailable )
  {
    // if intrinsics are available, use them
    cv::stereoCalibrate(object_points, image_pointsL, image_pointsR, *CVInternals->GetInstrinsicMatrix(LeftCameraIndex), *CVInternals->GetDistortionCoeffs(LeftCameraIndex), *CVInternals->GetInstrinsicMatrix(RightCameraIndex), *CVInternals->GetDistortionCoeffs(RightCameraIndex),
                        cv::Size(CameraImages[0].size[0], CameraImages[0].size[1]), rmat, tmat, Essential_matrix, Fundamental_matrix, CV_CALIB_FIX_INTRINSIC, cv::TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 1e-6));
  }
  else
  {
    cv::stereoCalibrate(object_points, image_pointsL, image_pointsR, intrinsic_matrixL, distortion_coeffsL, intrinsic_matrixR, distortion_coeffsR, cv::Size(CameraImages[0].size[0], CameraImages[0].size[1]),
                        rmat, tmat, Essential_matrix, Fundamental_matrix, 0, cv::TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 1e-6));
  }

  fundamental_matrix = cv::findFundamentalMat(image_pointsL, image_pointsR, CV_FM_RANSAC, 1.0, 0.99);
}