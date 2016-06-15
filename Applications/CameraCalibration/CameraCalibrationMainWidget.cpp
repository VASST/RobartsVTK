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
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

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
const static std::string LEFT_RESULT_IMAGE_WINDOW_NAME = "Left Result";
const static std::string RIGHT_RESULT_IMAGE_WINDOW_NAME = "Right Result";
const static std::string PATTERN_RESULT_IMAGE_WINDOW_NAME = "Pattern Result";
}

//----------------------------------------------------------------------------
CameraCalibrationMainWidget::CameraCalibrationMainWidget(QWidget* parent)
  : QWidget(parent)
  , AppSettings("cameraCalibration.ini", QSettings::IniFormat)
  , TrackingDataChannel(NULL)
  , LatestComputeIndex(0)
  , Pattern(QComputeThread::CHESSBOARD)
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

  CVInternals = new OpenCVCameraCapture();

  LeftCameraCaptureThread = new QCaptureThread();
  RightCameraCaptureThread = new QCaptureThread();
  LeftCameraCaptureThread->SetOpenCVInternals(*CVInternals);
  RightCameraCaptureThread->SetOpenCVInternals(*CVInternals);
  LeftCameraCaptureThread->SetCommonMutex(&OpenCVInternalsMutex);
  RightCameraCaptureThread->SetCommonMutex(&OpenCVInternalsMutex);

  InitUI();
}

//----------------------------------------------------------------------------
CameraCalibrationMainWidget::~CameraCalibrationMainWidget()
{
  AppSettings.sync();
  DataCollector->Stop();

  LeftCameraCaptureThread->StopCapture(true);
  delete LeftCameraCaptureThread;
  RightCameraCaptureThread->StopCapture(true);
  delete RightCameraCaptureThread;
  delete CVInternals;

  delete GUITimer;
  delete OpticalFlowTrackingTimer;
}

//----------------------------------------------------------------------------
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
  connect( ui.spinBox_BoardWidthCalib, SIGNAL( valueChanged( int ) ), this, SLOT( CalibBoardWidthValueChanged( int ) ) );
  connect( ui.spinBox_BoardHeightCalib, SIGNAL( valueChanged( int ) ), this, SLOT( CalibBoardHeightValueChanged( int ) ) );
  connect( ui.doubleSpinBox_QuadSizeCalib, SIGNAL( valueChanged( double ) ), this, SLOT( CalibBoardQuadSizeValueChanged( double ) ) );

  // Calibration
  connect( ui.pushButton_CaptureLeft, SIGNAL( clicked() ), this, SLOT( CaptureAndProcessLeftImage() ) );
  connect( ui.pushButton_CaptureRight, SIGNAL( clicked() ), this, SLOT( CaptureAndProcessRightImage() ) );
  connect( ui.pushButton_ComputeLeftIntrinsic, SIGNAL( clicked() ), this, SLOT( ComputeLeftIntrinsic() ) );
  connect( ui.pushButton_ComputeRightIntrinsic, SIGNAL( clicked() ), this, SLOT( ComputeRightIntrinsic() ) );
  connect( ui.pushButton_StereoAcquire, SIGNAL( clicked() ), this, SLOT( CaptureAndProcessStereoImagesAsync() ) );
  connect( ui.pushButton_StereoCompute, SIGNAL( clicked() ), this, SLOT( ComputeStereoCalibrationAsync() ) );

  // Validation
  connect( OpticalFlowTrackingTimer, SIGNAL( timeout() ), this, SLOT( OpticalFlowTracking() ) );

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

  connect( ui.comboBox_LeftCamera, SIGNAL( currentIndexChanged( int ) ), this, SLOT(OnLeftCameraIndexChanged( int ) ) );
  connect( ui.comboBox_RightCamera, SIGNAL( currentIndexChanged( int ) ), this, SLOT(OnRightCameraIndexChanged( int ) ) );
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

//----------------------------------------------------------------------------
int CameraCalibrationMainWidget::GetBoardWidthCalib() const
{
  return ui.spinBox_BoardWidthCalib->value();
}

//----------------------------------------------------------------------------
int CameraCalibrationMainWidget::GetBoardHeightCalib() const
{
  return ui.spinBox_BoardHeightCalib->value();
}

//----------------------------------------------------------------------------
double CameraCalibrationMainWidget::GetBoardQuadSizeCalib() const
{
  return ui.doubleSpinBox_QuadSizeCalib->value();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::SetIntrinsicMatrix(int cameraIndex, const cv::Mat& matrix)
{
  if( cameraIndex != INVALID_CAMERA_INDEX )
  {
    matrix.copyTo(IntrinsicMatrix[cameraIndex]);
  }
}

//----------------------------------------------------------------------------
cv::Mat& CameraCalibrationMainWidget::GetInstrinsicMatrix(int cameraIndex)
{
  return IntrinsicMatrix[cameraIndex];
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::SetDistortionCoeffs(int cameraIndex, const cv::Mat& matrix)
{
  if( cameraIndex != INVALID_CAMERA_INDEX )
  {
    matrix.copyTo(DistortionCoefficients[cameraIndex]);
  }
}

//----------------------------------------------------------------------------
cv::Mat& CameraCalibrationMainWidget::GetDistortionCoeffs(int cameraIndex)
{
  return DistortionCoefficients[cameraIndex];
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::ShowStatusMessage(const std::string& message)
{
  if( StatusBar != NULL )
  {
    StatusBar->showMessage(QString::fromStdString(message));
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::SetPLUSTrackingChannel(const std::string& trackingChannel)
{
  this->TrackingDataChannelName = trackingChannel;
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::SetCalibrationPattern(QComputeThread::CalibrationPattern pattern)
{
  if( this->Pattern != pattern )
  {
    this->Pattern = pattern;

    ResetCaptureCount();
    if( LeftCameraIndex != INVALID_CAMERA_INDEX )
    {
      EraseCameraEntries(LeftCameraIndex);
      InitializeCameraEntries(LeftCameraIndex);
    }
    if( RightCameraIndex != INVALID_CAMERA_INDEX )
    {
      EraseCameraEntries(RightCameraIndex);
      InitializeCameraEntries(RightCameraIndex);
    }
  }
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

  this->LoadMonoCameraParameters(LeftCameraIndex, fileName);

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

  this->LoadMonoCameraParameters(RightCameraIndex, fileName);

  RightIntrinsicAvailable = true;
  RightDistortionAvailable = true;
}

//----------------------------------------------------------------------------
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

//----------------------------------------------------------------------------
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

    EraseCameraEntries(LeftCameraIndex);
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
  ResetCaptureCount();
  InitializeCameraEntries(LeftCameraIndex);

  ui.pushButton_CaptureLeft->setEnabled(true);

  if( RightCameraIndex != INVALID_CAMERA_INDEX )
  {
    ui.pushButton_StereoAcquire->setEnabled(true);
  }
}

//----------------------------------------------------------------------------
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

    EraseCameraEntries(RightCameraIndex);
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
  ResetCaptureCount();
  InitializeCameraEntries(RightCameraIndex);

  ui.pushButton_CaptureRight->setEnabled(true);

  if( LeftCameraIndex != INVALID_CAMERA_INDEX )
  {
    ui.pushButton_StereoAcquire->setEnabled(true);
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::InitializeCameraEntries(int cameraIndex)
{
  CameraImages[cameraIndex];
  IntrinsicMatrix[cameraIndex];
  DistortionCoefficients[cameraIndex];
  ChessboardCornerPoints[cameraIndex];
  ChessboardCornerPointsCount[cameraIndex];
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CaptureAndProcessLeftImage()
{
  CaptureAndProcessImageAsync(LeftCameraIndex);
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CaptureAndProcessRightImage()
{
  CaptureAndProcessImageAsync(RightCameraIndex);
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CaptureAndProcessImageAsync(int cameraIndex)
{
  cv::Mat image;
  if( this->RetrieveLatestImage(cameraIndex, image) )
  {
    LOG_INFO("Board dimensions: " << GetBoardWidthCalib() << "x" << GetBoardHeightCalib() << "x" << GetBoardQuadSizeCalib());

    ComputeThreads[LatestComputeIndex] = new QComputeThread(LatestComputeIndex);
    qRegisterMetaType< cv::Mat >("cv::Mat");
    qRegisterMetaType< std::vector<cv::Point2f> >("std::vector<cv::Point2f>");
    connect( ComputeThreads[LatestComputeIndex],
             SIGNAL(patternProcessingComplete(int, int, const std::vector<cv::Point2f>&, const cv::Mat&)),
             this,
             SLOT(OnBoardPatternProcessingFinished(int, int, const std::vector<cv::Point2f>&, const cv::Mat&)) );
    ComputeThreads[LatestComputeIndex]->LocatePatternInImage(cameraIndex, GetBoardWidthCalib(), GetBoardHeightCalib(), Pattern, image);
    LatestComputeIndex++;
  }
  else
  {
    LOG_ERROR("Unable to retrieve latest image. Cannot process for board pattern.");
    ShowStatusMessage("Unable to retrieve latest image. Cannot process for board pattern.");
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

//----------------------------------------------------------------------------
// compute the intrinsics
void CameraCalibrationMainWidget::ComputeLeftIntrinsic()
{
  ui.pushButton_CaptureLeft->setEnabled(false);
  ui.pushButton_ComputeLeftIntrinsic->setEnabled(false);
  ui.pushButton_ComputeLeftIntrinsic->setText("Computing...");
  if( !this->ComputeIntrinsicsAndDistortionAsync( LeftCameraIndex ) )
  {
    return;
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::ComputeRightIntrinsic()
{
  ui.pushButton_CaptureRight->setEnabled(false);
  ui.pushButton_ComputeRightIntrinsic->setEnabled(false);
  ui.pushButton_ComputeRightIntrinsic->setText("Computing...");
  if( !this->ComputeIntrinsicsAndDistortionAsync( RightCameraIndex ) )
  {
    return;
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::ComputeStereoCalibrationAsync()
{
  cv::Mat image;
  if( this->RetrieveLatestLeftImage(image) || this->RetrieveLatestRightImage(image) )
  {
    // make sure we have same for both cameras
    if ( StereoImagePoints[LeftCameraIndex].size() != StereoImagePoints[RightCameraIndex].size() )
    {
      return;
    }

    ComputeThreads[LatestComputeIndex] = new QComputeThread(LatestComputeIndex);

    qRegisterMetaType< cv::Mat >("cv::Mat");
    if ( LeftIntrinsicAvailable && RightIntrinsicAvailable )
    {
      connect( ComputeThreads[LatestComputeIndex],
               SIGNAL(stereoCalibrationComplete(int, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)),
               this,
               SLOT(OnStereoComputationFinished(int, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)) );
      ComputeThreads[LatestComputeIndex]->StereoCalibrate(GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(), Pattern,
          StereoImagePoints[LeftCameraIndex], StereoImagePoints[RightCameraIndex],
          GetInstrinsicMatrix(LeftCameraIndex), GetDistortionCoeffs(LeftCameraIndex),
          GetInstrinsicMatrix(RightCameraIndex), GetDistortionCoeffs(RightCameraIndex),
          cv::Size(image.size[0], image.size[1]),
          CV_CALIB_FIX_INTRINSIC,
          cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-6));
    }
    else
    {
      connect( ComputeThreads[LatestComputeIndex],
               SIGNAL(stereoCalibrationComplete(int, double, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)),
               this,
               SLOT(OnStereoComputationFinished(int, double, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)) );
      ComputeThreads[LatestComputeIndex]->StereoCalibrate(GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(), Pattern,
          StereoImagePoints[LeftCameraIndex], StereoImagePoints[RightCameraIndex],
          cv::Size(image.size[0], image.size[1]),
          0,
          cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-6));
    }
    LatestComputeIndex++;

    ui.pushButton_StereoAcquire->setEnabled(false);
    ui.pushButton_StereoCompute->setEnabled(false);
    ui.pushButton_StereoCompute->setText("Computing...");
  }
  else
  {
    LOG_ERROR("Unable to grab image to determine image size.");
    ShowStatusMessage("Unable to grab image to determine image size.");
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::SaveMonoCameraParameters( const std::string& filename, const cv::Size& imageSize, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
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
void CameraCalibrationMainWidget::SaveStereoCameraParameters(const std::string& filename, const cv::Size& imageSize,
    const cv::Mat& leftCameraMatrix, const cv::Mat& leftDistCoeffs,
    const cv::Mat& rightCameraMatrix, const cv::Mat& rightDistCoeffs,
    const cv::Mat& rotationMatrix, const cv::Mat& translationMatrix,
    const std::vector<std::vector<cv::Point2f> >& leftImagePoints, const std::vector<std::vector<cv::Point2f> >& rightImagePoints,
    double reprojError, const cv::Mat& essentialMatrix, const cv::Mat& fundamentalMatrix)
{
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  fs << "calibration_time_utc" << vtkPlusAccurateTimer::GetUniversalTime();

  fs << "left_camera_matrix" << leftCameraMatrix;
  fs << "left_distortion_coefficients" << leftDistCoeffs;
  fs << "right_camera_matrix" << rightCameraMatrix;
  fs << "right_distortion_coefficients" << rightDistCoeffs;

  fs << "image_width" << imageSize.width;
  fs << "image_height" << imageSize.height;
  fs << "board_width" << ui.spinBox_BoardWidthCalib->value();
  fs << "board_height" << ui.spinBox_BoardHeightCalib->value();
  fs << "square_size" << ui.doubleSpinBox_QuadSizeCalib->value();

  fs << "avg_reprojection_error" << reprojError;

  fs << "rotation_matrix" << rotationMatrix;
  fs << "translation_matrix" << translationMatrix;

  if(!leftImagePoints.empty() )
  {
    cv::Mat imagePtMat((int)leftImagePoints.size(), (int)leftImagePoints[0].size(), CV_32FC2);
    for( size_t i = 0; i < leftImagePoints.size(); i++ )
    {
      cv::Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
      cv::Mat imgpti(leftImagePoints[i]);
      imgpti.copyTo(r);
    }
    fs << "left_image_points" << imagePtMat;
  }

  if(!rightImagePoints.empty() )
  {
    cv::Mat imagePtMat((int)rightImagePoints.size(), (int)rightImagePoints[0].size(), CV_32FC2);
    for( size_t i = 0; i < rightImagePoints.size(); i++ )
    {
      cv::Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
      cv::Mat imgpti(rightImagePoints[i]);
      imgpti.copyTo(r);
    }
    fs << "right_image_points" << imagePtMat;
  }

  fs << "fundamental_matrix" << fundamentalMatrix;
  fs << "essential_matrix" << essentialMatrix;
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::LoadMonoCameraParameters(int cameraIndex, const std::string& fileName)
{
  cv::FileStorage fs(fileName, cv::FileStorage::READ);

  cv::Mat cameraMatrix;
  fs["camera_matrix"] >> cameraMatrix;
  cv::Mat distCoeffs;
  fs["distortion_coefficients"] >> distCoeffs;

  SetIntrinsicMatrix(cameraIndex, cameraMatrix);
  SetDistortionCoeffs(cameraIndex, distCoeffs);
}

//----------------------------------------------------------------------------
bool CameraCalibrationMainWidget::ComputeIntrinsicsAndDistortionAsync( int cameraIndex )
{
  if ( CaptureCount[cameraIndex] < MinBoardNeeded )
  {
    std::stringstream ss;
    ss << "Not enough board images recorded: " << CaptureCount[cameraIndex] << "/" << MinBoardNeeded;
    LOG_ERROR(ss.str());
    ShowStatusMessage(ss.str());
    return false;
  }

  ComputeThreads[LatestComputeIndex] = new QComputeThread(LatestComputeIndex);
  qRegisterMetaType< cv::Mat >("cv::Mat");
  qRegisterMetaType< cv::Size >("cv::Size");
  qRegisterMetaType< std::vector<cv::Mat> >("std::vector<cv::Mat>");
  qRegisterMetaType< std::vector<float> >("std::vector<float>");
  connect( ComputeThreads[LatestComputeIndex],
           SIGNAL(monoCalibrationComplete(int, int, const cv::Mat&, const cv::Mat&, const cv::Size&, double, double, const std::vector<cv::Mat>&, const std::vector<cv::Mat>&, const std::vector<float>&)),
           this,
           SLOT(OnMonoComputationFinished(int, int, const cv::Mat&, const cv::Mat&, const cv::Size&, double, double, const std::vector<cv::Mat>&, const std::vector<cv::Mat>&, const std::vector<float>&)) );
  ComputeThreads[LatestComputeIndex]->CalibrateCamera(cameraIndex, GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(), Pattern, ChessboardCornerPoints[cameraIndex], cv::Size(CameraImages[cameraIndex].size[0], CameraImages[cameraIndex].size[1]), CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
  LatestComputeIndex++;
  return true;
}

//----------------------------------------------------------------------------
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

  cv::imshow(LEFT_RESULT_IMAGE_WINDOW_NAME, leftImage);
  cv::imshow(RIGHT_RESULT_IMAGE_WINDOW_NAME, rightImage);
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::ResetCaptureCount()
{
  CaptureCount[LeftCameraIndex] = 0;
  CaptureCount[RightCameraIndex] = 0;
  StereoCaptureCount = 0;
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::EraseCameraEntries(int cameraIndex)
{
  // Remove camera entry from maps
  CameraImages.erase(CameraImages.find(cameraIndex));
  CaptureCount.erase(CaptureCount.find(cameraIndex));
  ChessboardCornerPoints.erase(ChessboardCornerPoints.find(cameraIndex));
  ChessboardCornerPointsCount.erase(ChessboardCornerPointsCount.find(cameraIndex));
  IntrinsicMatrix.erase(IntrinsicMatrix.find(cameraIndex));
  DistortionCoefficients.erase(DistortionCoefficients.find(cameraIndex));
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
  else if( cameraIndex == RightCameraIndex )
  {
    QMutexLocker locker(&RightImageStorageMutex);
    image.copyTo(CameraImages[RightCameraIndex]);

    QImage img = cvMatToQImage(CameraImages[RightCameraIndex]);
    ui.scaledView_RightVideo->setPixmap(QPixmap::fromImage(img));
    ui.scaledView_RightVideo->update();
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::OnMonoComputationFinished(int computeIndex, int cameraIndex, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const cv::Size& imageSize, double reprojError, double totalAvgErr, const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs, const std::vector<float>& perViewErrors)
{
  SetIntrinsicMatrix(cameraIndex, cameraMatrix);
  SetDistortionCoeffs(cameraIndex, distCoeffs);

  std::stringstream ss;
  ss << "Camera calibrated with re-projection error: " << reprojError;
  LOG_INFO(ss.str());
  ShowStatusMessage(ss.str());

  if ( cameraIndex == LeftCameraIndex )
  {
    LeftIntrinsicAvailable = LeftDistortionAvailable = true;
  }
  else if ( cameraIndex == RightCameraIndex )
  {
    RightIntrinsicAvailable = RightDistortionAvailable = true;
  }

  QString saveFileName = QFileDialog::getSaveFileName(this, "Save intrinsics and distortion", vtkPlusConfig::GetInstance()->GetOutputDirectory().c_str(), "*.xml");

  if( saveFileName.indexOf(".xml") == -1 )
  {
    saveFileName.append(".xml");
  }

  this->SaveMonoCameraParameters( saveFileName.toStdString(), imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, perViewErrors, ChessboardCornerPoints[cameraIndex], totalAvgErr );

  LOG_INFO("Camera parameters written computed and saved to: " << saveFileName.toStdString());
  ShowStatusMessage("Success. Camera parameters written to file.");

  disconnect( ComputeThreads[computeIndex],
              SIGNAL(monoCalibrationComplete(int, int, const cv::Mat&, const cv::Mat&, const cv::Size&, double, double, const std::vector<cv::Mat>&, const std::vector<cv::Mat>&, const std::vector<float>&)),
              this,
              SLOT(OnMonoComputationFinished(int, int, const cv::Mat&, const cv::Mat&, const cv::Size&, double, double, const std::vector<cv::Mat>&, const std::vector<cv::Mat>&, const std::vector<float>&)) );
  ComputeThreads[computeIndex]->wait();
  delete ComputeThreads[computeIndex];
  ComputeThreads.erase(ComputeThreads.find(computeIndex));

  if( cameraIndex == LeftCameraIndex )
  {
    ui.pushButton_CaptureLeft->setEnabled(true);
    ui.pushButton_ComputeLeftIntrinsic->setEnabled(true);
    ui.pushButton_ComputeLeftIntrinsic->setText("Compute && Save");
  }
  else
  {
    ui.pushButton_CaptureRight->setEnabled(true);
    ui.pushButton_ComputeRightIntrinsic->setEnabled(true);
    ui.pushButton_ComputeRightIntrinsic->setText("Compute && Save");
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::OnStereoComputationFinished(int computeIndex, double reprojError, const cv::Mat& rotationMatrix, const cv::Mat& translationMatrix,
    const cv::Mat& essentialMatrix, const cv::Mat& fundamentalMatrix)
{
  QString saveFileName = QFileDialog::getSaveFileName(this, "Save stereo parameters", vtkPlusConfig::GetInstance()->GetOutputDirectory().c_str(), "*.xml");
  cv::FileStorage fs(saveFileName.toStdString(), cv::FileStorage::WRITE);
  fs << "reproj_error" << reprojError;
  fs << "rotation_matrix" << rotationMatrix;
  fs << "translation_matrix" << translationMatrix;
  fs << "fundamental_matrix" << fundamentalMatrix;
  fs << "essential_matrix" << essentialMatrix;

  LOG_INFO("Stereo calibration performed with average re-projection error of " << reprojError);
  ShowStatusMessage("Stereo calibration performed.");

  disconnect( ComputeThreads[LatestComputeIndex],
              SIGNAL(stereoCalibrationComplete(int, double, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)),
              this,
              SLOT(OnStereoComputationFinished(int, double, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)) );
  ComputeThreads[computeIndex]->wait();
  delete ComputeThreads[computeIndex];
  ComputeThreads.erase(ComputeThreads.find(computeIndex));

  ui.pushButton_StereoAcquire->setEnabled(true);
  ui.pushButton_StereoCompute->setEnabled(true);
  ui.pushButton_StereoCompute->setText("Compute && Save");
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::OnStereoComputationFinished(int computeIndex, double reprojError, const cv::Mat& leftIntrinsicMatrix, const cv::Mat& leftDistCoeffs,
    const cv::Mat& rightIntrinsicMatrix, const cv::Mat& rightDistCoeffs, const cv::Mat& rotationMatrix, const cv::Mat& translationMatrix,
    const cv::Mat& essentialMatrix, const cv::Mat& fundamentalMatrix)
{
  SetIntrinsicMatrix(LeftCameraIndex, leftIntrinsicMatrix);
  SetIntrinsicMatrix(RightCameraIndex, rightIntrinsicMatrix);
  SetDistortionCoeffs(LeftCameraIndex, leftDistCoeffs);
  SetDistortionCoeffs(RightCameraIndex, rightDistCoeffs);

  LOG_INFO("Stereo and dual mono calibrations performed with average reprojection error of " << reprojError);
  ShowStatusMessage("Stereo and dual mono calibrations performed.");

  QString saveFileName = QFileDialog::getSaveFileName(this, "Save camera parameters", vtkPlusConfig::GetInstance()->GetOutputDirectory().c_str(), "*.xml");

  if( saveFileName.indexOf(".xml") == -1 )
  {
    saveFileName.append(".xml");
  }

  cv::Mat image;
  if( this->RetrieveLatestLeftImage(image) || this->RetrieveLatestRightImage(image) )
  {
    this->SaveStereoCameraParameters( saveFileName.toStdString(), cv::Size(image.size[0], image.size[1]), leftIntrinsicMatrix,
                                      leftDistCoeffs, rightIntrinsicMatrix, rightDistCoeffs, rotationMatrix, translationMatrix,
                                      StereoImagePoints[LeftCameraIndex], StereoImagePoints[RightCameraIndex], reprojError, essentialMatrix, fundamentalMatrix );
  }

  disconnect( ComputeThreads[LatestComputeIndex],
              SIGNAL(stereoCalibrationComplete(int, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)),
              this,
              SLOT(OnStereoComputationFinished(int, const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&)) );
  ComputeThreads[computeIndex]->wait();
  delete ComputeThreads[computeIndex];
  ComputeThreads.erase(ComputeThreads.find(computeIndex));

  ui.pushButton_StereoAcquire->setEnabled(true);
  ui.pushButton_StereoCompute->setEnabled(true);
  ui.pushButton_StereoCompute->setText("Compute && Save");
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::OnBoardPatternProcessingFinished(int computeIndex, int cameraIndex, const std::vector<cv::Point2f>& outImagePoints, const cv::Mat& resultImage)
{
  if( outImagePoints.empty() )
  {
    LOG_WARNING("Unable to locate checkerboard in camera image. Try again.");
    ShowStatusMessage("Unable to locate checkerboard in camera image. Try again.");
  }
  else
  {
    // Store located corners in local database of corners
    ChessboardCornerPoints[cameraIndex].push_back(outImagePoints);
    CaptureCount[cameraIndex]++;
    std::stringstream ss;
    ss << "Success! Captured " << CaptureCount[cameraIndex] << " calibration image thus far.";
    LOG_INFO(ss.str());
    ShowStatusMessage(ss.str());

    cv::imshow( PATTERN_RESULT_IMAGE_WINDOW_NAME, resultImage );

    if( CaptureCount[cameraIndex] >= MinBoardNeeded )
    {
      if( cameraIndex == LeftCameraIndex )
      {
        ui.pushButton_ComputeLeftIntrinsic->setEnabled(true);
      }
      else if( cameraIndex == RightCameraIndex )
      {
        ui.pushButton_ComputeRightIntrinsic->setEnabled(true);
      }
    }
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::OnLeftBoardPatternProcessingFinished(int computeIndex, int cameraIndex, const std::vector<cv::Point2f>& outImagePoints, const cv::Mat& resultImage)
{
  if( outImagePoints.empty() )
  {
    LOG_WARNING("Unable to locate checkerboard in left camera image. Try again.");
    ShowStatusMessage("Unable to locate checkerboard in left camera image. Try again.");
  }
  else
  {
    cv::imshow(LEFT_RESULT_IMAGE_WINDOW_NAME, resultImage);

    {
      QMutexLocker locker(&StereoResultMutex);
      // TODO : what if one succeeds and the other fails
      StereoImagePoints[LeftCameraIndex].push_back(outImagePoints);
      StereoCaptureCount++;
    }

    if( StereoCaptureCount >= 2*MinBoardNeeded ) // StereoCaptureCount is for each individual camera
    {
      ui.pushButton_StereoCompute->setEnabled(true);
    }

    std::stringstream ss;
    ss << "Captured " << StereoCaptureCount << " images so far. " << MinBoardNeeded << " needed.";
    LOG_INFO(ss.str());
    ShowStatusMessage(ss.str());
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::OnRightBoardPatternProcessingFinished(int computeIndex, int cameraIndex, const std::vector<cv::Point2f>& outImagePoints, const cv::Mat& resultImage)
{
  if( outImagePoints.empty() )
  {
    LOG_WARNING("Unable to locate checkerboard in right camera image. Try again.");
    ShowStatusMessage("Unable to locate checkerboard in right camera image. Try again.");
  }
  else
  {
    cv::imshow(RIGHT_RESULT_IMAGE_WINDOW_NAME, resultImage);
    StereoImagePoints[RightCameraIndex].push_back(outImagePoints);
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CaptureAndProcessStereoImagesAsync()
{
  cv::Mat leftImage;
  cv::Mat rightImage;
  // get both images
  if( this->RetrieveLatestLeftImage(leftImage) && this->RetrieveLatestRightImage(rightImage) )
  {
    LOG_INFO("Board dimensions: " << GetBoardWidthCalib() << "x" << GetBoardHeightCalib() << "x" << GetBoardQuadSizeCalib());

    ComputeThreads[LatestComputeIndex] = new QComputeThread(LatestComputeIndex);
    qRegisterMetaType< cv::Mat >("cv::Mat");
    qRegisterMetaType< std::vector<cv::Point2f> >("std::vector<cv::Point2f>");
    connect( ComputeThreads[LatestComputeIndex],
             SIGNAL(patternProcessingComplete(int, int, const std::vector<cv::Point2f>&, const cv::Mat&)),
             this,
             SLOT(OnLeftBoardPatternProcessingFinished(int, int, const std::vector<cv::Point2f>&, const cv::Mat&)) );
    ComputeThreads[LatestComputeIndex]->LocatePatternInImage(LeftCameraIndex, GetBoardWidthCalib(), GetBoardHeightCalib(), Pattern, leftImage);
    LatestComputeIndex++;

    ComputeThreads[LatestComputeIndex] = new QComputeThread(LatestComputeIndex);
    connect( ComputeThreads[LatestComputeIndex],
             SIGNAL(patternProcessingComplete(int, int, const std::vector<cv::Point2f>&, const cv::Mat&)),
             this,
             SLOT(OnRightBoardPatternProcessingFinished(int, int, const std::vector<cv::Point2f>&, const cv::Mat&)) );
    ComputeThreads[LatestComputeIndex]->LocatePatternInImage(RightCameraIndex, GetBoardWidthCalib(), GetBoardHeightCalib(), Pattern, rightImage);
    LatestComputeIndex++;
  }
  else
  {
    LOG_ERROR("Unable to retrieve latest images. Cannot process for board pattern.");
    ShowStatusMessage("Unable to retrieve latest images. Cannot process for board pattern.");
  }
}