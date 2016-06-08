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
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMainWindow>
#include <QPushButton>
#include <QSpinBox>
#include <QStatusBar>
#include <QString>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>

// local includes
#include "CameraCalibrationMainWidget.h"
#include "mathUtil.h"

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
#include <vtkPlusChannel.h>
#include <vtkPlusDataCollector.h>
#include <vtkPlusDataSource.h>
#include <vtkPlusTransformRepository.h>

// VTK Includes
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkLandmarkTransform.h>
#include <vtkMatrix4x4.h>
#include <vtkPoints.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataReader.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTransform.h>
#include <vtkXMLDataElement.h>

CameraCalibrationMainWidget::CameraCalibrationMainWidget(QWidget* parent)
  : QWidget(parent)
  , appSettings("cameraCalibration.ini", QSettings::IniFormat)
  , TrackingDataChannel(NULL)
  , LeftCameraIndex(-1)
  , RightCameraIndex(-1)
  , MinBoardNeeded(6)
  , TrackingDataChannelName("")
  , DeviceSetSelectorWidget(NULL)
  , ToolStateDisplayWidget(NULL)
  , StatusIcon(NULL)
  , GUITimer(NULL)
  , TrackingDataTimer(NULL)
  , LeftCameraTimer(NULL)
  , RightCameraTimer(NULL)
  , ValidateStylusTimer(NULL)
  , ValidateChessTimer(NULL)
  , ValidateTrackingTimer(NULL)
{
  // Set up UI
  ui.setupUi(this);

  DeviceSetSelectorWidget = new PlusDeviceSetSelectorWidget(ui.groupBox_DataCollection);
  DeviceSetSelectorWidget->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
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

  InitUI();

  // VTK related objects
  CreateVTKObjects();
  SetupVTKPipeline();

  fileOutput[0] = new ofstream( "Left_Results.csv" );
  fileOutput[1] = new ofstream( "Right_Results.csv" );
  fileOutput[2] = new ofstream( "triangulation.csv" );
  fileOutput[3] = new ofstream( "Results.csv" );

  for(int i = 0; i < 2; i++)
    *(fileOutput[i]) << "IndexNumber" << ", " <<"ImageSegX" << ", " << "ImageSegY" << ", "
                     << "HomographyX" << ", "
                     << "HomographyY" << ", "
                     << "HomographyZ" << ", "
                     << "ReprojectionX" << ", "
                     << "ReprojectionY" << ", "
                     << "ActualPosX" << ", " << "ActualPosY" << ", " << "ActualPosZ" << std::endl;

  *(fileOutput[2]) << "triangulate_x" << ", "
                   << "triangulate_y" << ", "
                   << "triangulate_z" << std::endl;

  *(fileOutput[3]) << "IndexNumber" << ", " <<"ImageSegX" << ", " << "ImageSegY" << ", "
                   << "HomographyX" << ", "
                   << "HomographyY" << ", "
                   << "HomographyZ" << ", "
                   << "ReprojectionX" << ", "
                   << "ReprojectionY" << ", "
                   << "ActualPosX" << ", " << "ActualPosY" << ", " << "ActualPosZ" << ", , ,"
                   << "IndexNumber" << ", " <<"ImageSegX" << ", " << "ImageSegY" << ", "
                   << "HomographyX" << ", "
                   << "HomographyY" << ", "
                   << "HomographyZ" << ", "
                   << "ReprojectionX" << ", "
                   << "ReprojectionY" << ", "
                   << "ActualPosX" << ", " << "ActualPosY" << ", " << "ActualPosZ" << ", , ,"
                   << "triangulate_x" << ", "
                   << "triangulate_y" << ", "
                   << "triangulate_z" << std::endl;
}

//---------------------------------------------------------
CameraCalibrationMainWidget::~CameraCalibrationMainWidget()
{
  appSettings.sync();
  DataCollector->Stop();

  delete CVInternals;

  for(int i = 0; i < 4; i++)
  {
    if( fileOutput[i] )
    {
      fileOutput[i]->close();
      delete fileOutput[i];
    }
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::InitUI()
{
  // create timers;
  LeftCameraTimer = new QTimer( this );
  RightCameraTimer = new QTimer( this );
  ValidateStylusTimer = new QTimer( this );
  ValidateChessTimer = new QTimer( this );
  ValidateTrackingTimer = new QTimer( this );
  GUITimer = new QTimer( this );

  // Load settings
  ui.spinBox_BoardWidthCalib->setValue(appSettings.value("calib/boardWidth", 7).toInt());
  ui.spinBox_BoardHeightCalib->setValue(appSettings.value("calib/boardHeight", 5).toInt());
  ui.doubleSpinBox_QuadSizeCalib->setValue(appSettings.value("calib/quadSize", 18.125).toDouble());

  ui.spinBox_BoardWidthValid->setValue(appSettings.value("validation/boardWidth", 4).toInt());
  ui.spinBox_BoardHeightValid->setValue(appSettings.value("validation/boardHeight", 3).toInt());
  ui.doubleSpinBox_QuadSizeValid->setValue(appSettings.value("validation/quadSize", 20.0).toDouble());

  connect( GUITimer, SIGNAL( timeout() ), this, SLOT( RefreshContent() ) );

  // Disable acquisition until cameras are selected
  ui.groupBox_IntrinsicsLeftAcquisition->setEnabled(false);
  ui.groupBox_IntrinsicsRightAcquisition->setEnabled(false);
  ui.pushButton_RecordLeftHMD->setEnabled(false);
  ui.pushButton_RecordRightHMD->setEnabled(false);
  ui.groupBox_FundamentalMatrix->setEnabled(false);
  ui.widget_LeftContainerFiles->setEnabled(false);
  ui.widget_RightContainerFiles->setEnabled(false);

  // set up all the signal/slots here
  connect( LeftCameraTimer, SIGNAL( timeout() ),
           this, SLOT( UpdateLeftVideo() ) );
  connect( RightCameraTimer, SIGNAL( timeout() ),
           this, SLOT( UpdateRightVideo() ) );
  connect( ui.pushButton_LoadLeftIntrinsic, SIGNAL( clicked() ),
           this, SLOT( LoadLeftIntrinsic() ) );
  connect( ui.pushButton_LoadRightIntrinsic, SIGNAL( clicked() ),
           this, SLOT( LoadRightIntrinsic() ) );
  connect( ui.pushButton_LoadLeftLandmark, SIGNAL( clicked() ),
           this, SLOT( LoadLeftLandmark() ) );
  connect( ui.pushButton_LoadRightLandmark, SIGNAL( clicked() ),
           this, SLOT( LoadRightLandmark() ) );
  connect( ui.pushButton_LoadLeftDistortion, SIGNAL( clicked() ),
           this, SLOT( LoadLeftDistortion() ) );
  connect( ui.pushButton_LoadRightDistortion, SIGNAL( clicked() ),
           this, SLOT( LoadRightDistortion() ) );
  connect( ui.pushButton_LoadCalibChessReg, SIGNAL( clicked() ),
           this, SLOT( LoadChessRegistration() ) );
  connect( ui.pushButton_LoadValidChessReg, SIGNAL( clicked() ),
           this, SLOT( LoadValidChessRegistration() ) );

  // Configuration
  connect( ui.spinBox_BoardWidthCalib, SIGNAL( valueChanged( int ) ),
           this, SLOT( CalibBoardWidthValueChanged( int ) ) );
  connect( ui.spinBox_BoardHeightCalib, SIGNAL( valueChanged( int ) ),
           this, SLOT( CalibBoardHeightValueChanged( int ) ) );
  connect( ui.doubleSpinBox_QuadSizeCalib, SIGNAL( valueChanged( double ) ),
           this, SLOT( CalibBoardQuadSizeValueChanged( double ) ) );

  connect( ui.spinBox_BoardWidthValid, SIGNAL( valueChanged( int ) ),
           this, SLOT( ValidationBoardWidthValueChanged( int ) ) );
  connect( ui.spinBox_BoardHeightValid, SIGNAL( valueChanged( int ) ),
           this, SLOT( ValidationBoardHeightValueChanged( int ) ) );
  connect( ui.doubleSpinBox_QuadSizeValid, SIGNAL( valueChanged( double ) ),
           this, SLOT( ValidationBoardQuadSizeValueChanged( double ) ) );

  // Calibration
  connect( ui.pushButton_CaptureLeft, SIGNAL( clicked() ),
           this, SLOT( CaptureAndProcessLeftImage() ) );
  connect( ui.pushButton_CaptureRight, SIGNAL( clicked() ),
           this, SLOT( CaptureAndProcessRightImage() ) );
  connect( ui.pushButton_ComputeLeftIntrinsic, SIGNAL( clicked() ),
           this, SLOT( ComputeLeftIntrinsic() ) );
  connect( ui.pushButton_ComputeRightIntrinsic, SIGNAL( clicked() ),
           this, SLOT( ComputeRightIntrinsic() ) );
  connect( ui.pushButton_CollectCalibChess, SIGNAL( clicked() ),
           this, SLOT( CollectStylusPoint() ) );
  connect( ui.pushButton_RegCalibChess, SIGNAL( clicked() ),
           this, SLOT( PerformBoardRegistration() ) );
  connect( ui.pushButton_CollectValidChess, SIGNAL( clicked() ),
           this, SLOT( CollectRegPoint() ) );
  connect( ui.pushButton_RegValidChess, SIGNAL( clicked() ),
           this, SLOT( PerformRegBoardRegistration() ) );
  connect( ui.pushButton_StereoAcquire, SIGNAL( clicked() ),
           this, SLOT( StereoAcquire() ) );
  connect( ui.pushButton_StereoCompute, SIGNAL( clicked() ),
           this, SLOT( ComputeFundamentalMatrix() ) );
  connect( ui.pushButton_RecordLeftHMD, SIGNAL( clicked() ),
           this, SLOT( DrawLeftChessBoardCorners() ) );
  connect( ui.pushButton_RecordRightHMD, SIGNAL( clicked() ),
           this, SLOT( DrawRightChessBoardCorners() ) );
  connect( ui.pushButton_ComputeHMDLandmarks, SIGNAL( clicked() ),
           this, SLOT( ComputeHMDRegistration() ) );

  // Validation
  connect( ui.pushButton_ValidateStylus, SIGNAL( toggled( bool ) ),
           this, SLOT( ValidateStylusStartTimer( bool ) ) );
  connect( ValidateStylusTimer, SIGNAL( timeout() ),
           this, SLOT( ValidateStylus() ) );
  connect( ui.pushButton_ValidCalibChess, SIGNAL( clicked() ),
           this, SLOT( ValidateChess() ) );
  connect( ui.pushButton_ValidateVisual, SIGNAL( toggled( bool ) ),
           this, SLOT( ValidateChessStartTimer( bool ) ) );
  connect( ValidateChessTimer, SIGNAL( timeout() ),
           this, SLOT( ValidateChessVisual() ) );
  connect( ui.pushButton_ValidateValidChess, SIGNAL( clicked() ),
           this, SLOT( ValidateValidChess() ) );
  connect( ui.pushButton_VisualTracking, SIGNAL( toggled( bool ) ),
           this, SLOT( ValidateChessVisualTimer( bool) ) );
  connect( ValidateTrackingTimer, SIGNAL( timeout() ),
           this, SLOT( OpticalFlowTracking() ) );

  ui.comboBox_LeftCamera->addItem("None", QVariant(-1));
  ui.comboBox_RightCamera->addItem("None", QVariant(-1));
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
double CameraCalibrationMainWidget::ComputeReprojectionErrors(const std::vector<std::vector<cv::Point3d> >& objectPoints,
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
      projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
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

//---------------------------------------------------------
// centralized place to create all vtk objects
void CameraCalibrationMainWidget::CreateVTKObjects()
{
  DataCollector = vtkSmartPointer< vtkPlusDataCollector >::New();
  BoardRegTransform = vtkSmartPointer< vtkLandmarkTransform >::New();
  LeftLandmarkTransform = vtkSmartPointer< vtkLandmarkTransform >::New();
  RightLandmarkTransform = vtkSmartPointer< vtkLandmarkTransform >::New();
  BoardSource = vtkSmartPointer< vtkPoints >::New();
  BoardTarget = vtkSmartPointer< vtkPoints >::New();
  BoardCornerLeftSource = vtkSmartPointer< vtkPoints >::New();
  BoardCornerLeftTarget = vtkSmartPointer< vtkPoints >::New();
  BoardCornerRightSource = vtkSmartPointer< vtkPoints >::New();
  BoardCornerRightTarget = vtkSmartPointer< vtkPoints >::New();
  ValidBoardSource = vtkSmartPointer< vtkPoints >::New();
  ValidBoardTarget = vtkSmartPointer< vtkPoints >::New();
  ValidBoardRegTransform = vtkSmartPointer< vtkLandmarkTransform >::New();
}

//---------------------------------------------------------
// centralized place to setup all vtk pipelines
void CameraCalibrationMainWidget::SetupVTKPipeline()
{
  LeftLandmarkAvailable = RightLandmarkAvailable =
                            LeftIntrinsicAvailable = RightIntrinsicAvailable =
                                  LeftDistortionAvailable = RightDistortionAvailable =
                                        BoardRegAvailable = ValidBoardAvailable = false ;
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
void CameraCalibrationMainWidget::CalcBoardCornerPositions(int height, int width, double quadSize, std::vector<cv::Point3d>& corners)
{
  corners.clear();

  for( int i = 0; i < height; ++i )
  {
    for( int j = 0; j < width; ++j )
    {
      corners.push_back(cv::Point3d(j*quadSize, i*quadSize, 0));
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
    GUITimer->stop();

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
      GUITimer->start(33);

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
void CameraCalibrationMainWidget::RefreshContent()
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

  if( cameraIndex == -1 )
  {
    LeftCameraTimer->stop();
    ui.groupBox_IntrinsicsLeftAcquisition->setEnabled(false);
    ui.pushButton_RecordLeftHMD->setEnabled(false);
    ui.groupBox_FundamentalMatrix->setEnabled(false);
    ui.widget_LeftContainerFiles->setEnabled(false);

    if( LeftCameraIndex != -1 )
    {
      CVInternals->ReleaseCamera(LeftCameraIndex);
      // Remove camera entry from maps
      CameraImages.erase(CameraImages.find(LeftCameraIndex));
      CaptureCount.erase(CaptureCount.find(LeftCameraIndex));
      image_points.erase(image_points.find(LeftCameraIndex));
      point_counts.erase(point_counts.find(LeftCameraIndex));
    }

    LeftCameraIndex = cameraIndex;
    return;
  }

  if( !CVInternals->InitializeCamera(cameraIndex) )
  {
    LOG_ERROR("Unable to initialize camera with camera index: " << cameraIndex);
    return;
  }

  LeftCameraIndex = cameraIndex;
  CameraImages[LeftCameraIndex];
  CaptureCount[LeftCameraIndex];
  image_points[LeftCameraIndex];
  point_counts[LeftCameraIndex];

  ui.groupBox_IntrinsicsLeftAcquisition->setEnabled(true);
  ui.pushButton_RecordLeftHMD->setEnabled(true);
  ui.widget_LeftContainerFiles->setEnabled(true);

  if( ui.groupBox_IntrinsicsRightAcquisition->isEnabled() )
  {
    ui.groupBox_FundamentalMatrix->setEnabled(true);
  }

  LeftCameraTimer->start(33);
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::OnRightCameraIndexChanged(int index)
{
  int cameraIndex = ui.comboBox_RightCamera->currentData().toInt();
  if( cameraIndex == -1 )
  {
    RightCameraTimer->stop();
    ui.groupBox_IntrinsicsRightAcquisition->setEnabled(false);
    ui.widget_RightContainerFiles->setEnabled(false);
    ui.pushButton_RecordRightHMD->setEnabled(false);
    ui.groupBox_FundamentalMatrix->setEnabled(false);

    if( RightCameraIndex != -1 )
    {
      CVInternals->ReleaseCamera(RightCameraIndex);
      // Remove camera entry from maps
      CameraImages.erase(CameraImages.find(RightCameraIndex));
      CaptureCount.erase(CaptureCount.find(RightCameraIndex));
      image_points.erase(image_points.find(RightCameraIndex));
      point_counts.erase(point_counts.find(RightCameraIndex));
    }

    RightCameraIndex = cameraIndex;
    return;
  }

  if( !CVInternals->InitializeCamera(cameraIndex) )
  {
    LOG_ERROR("Unable to initialize camera with camera index: " << cameraIndex);
    return;
  }

  RightCameraIndex = cameraIndex;
  CameraImages[RightCameraIndex];
  CaptureCount[RightCameraIndex];
  image_points[RightCameraIndex];
  point_counts[RightCameraIndex];

  ui.groupBox_IntrinsicsRightAcquisition->setEnabled(true);
  ui.pushButton_RecordRightHMD->setEnabled(true);
  ui.widget_RightContainerFiles->setEnabled(true);

  if( ui.groupBox_IntrinsicsLeftAcquisition->isEnabled() )
  {
    ui.groupBox_FundamentalMatrix->setEnabled(true);
  }

  RightCameraTimer->start(33);
}

//---------------------------------------------------------
// start opencv video
void CameraCalibrationMainWidget::UpdateLeftVideo()
{
  if( !CVInternals->QueryFrame(LeftCameraIndex, CameraImages[LeftCameraIndex]) )
  {
    return;
  }

  //QImage imgIn(CameraImages[LeftCameraIndex].data, CameraImages[LeftCameraIndex].cols, CameraImages[LeftCameraIndex].rows, CameraImages[LeftCameraIndex].step, QImage::Format_RGB888);

  //ui.openGLWidget_LeftVideo->setPixmap(QPixmap::fromImage(imgIn));
  //ui.openGLWidget_LeftVideo->update();
  cv::imshow("Left Result", CameraImages[LeftCameraIndex]);
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::UpdateRightVideo()
{
  if( !CVInternals->QueryFrame(RightCameraIndex, CameraImages[RightCameraIndex]) )
  {
    return;
  }

  //QImage imgIn(CameraImages[RightCameraIndex].data, CameraImages[RightCameraIndex].cols, CameraImages[RightCameraIndex].rows, CameraImages[RightCameraIndex].step, QImage::Format_RGB888);

  //ui.openGLWidget_RightVideo->setPixmap(QPixmap::fromImage(imgIn));
  //ui.openGLWidget_RightVideo->update();
  cv::imshow("Right Result", CameraImages[RightCameraIndex]);
}

//---------------------------------------------------------
// file operation
void CameraCalibrationMainWidget::LoadLeftIntrinsic()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open Left Intrinsic" ),
                     QDir::currentPath(),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  CVInternals->SetIntrinsicMatrix(LeftCameraIndex, cv::imread( FileName.toStdString() ));

  LeftIntrinsicAvailable = true;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadRightIntrinsic()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open Right Intrinsic" ),
                     QDir::currentPath(),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  CVInternals->SetIntrinsicMatrix(RightCameraIndex, cv::imread( FileName.toStdString() ) );

  RightIntrinsicAvailable = true;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadLeftDistortion()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open left Distortion" ),
                     QDir::currentPath(),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  CVInternals->SetDistortionCoeffs(LeftCameraIndex, cv::imread( FileName.toStdString() ) );

  LeftDistortionAvailable = true;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadRightDistortion()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open right Distortion" ),
                     QDir::currentPath(),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  CVInternals->SetDistortionCoeffs(RightCameraIndex, cv::imread( FileName.toStdString() ) );

  RightDistortionAvailable = true;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadLeftLandmark()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open Left Landmark Transform" ),
                     QDir::currentPath(),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  cv::Mat landmarkMat = cv::imread( FileName.toStdString() );

  LeftLandmarkTransform->GetMatrix()->DeepCopy( landmarkMat.ptr<double>(0) );

  LeftLandmarkAvailable = true;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadRightLandmark()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open Right Landmark Transform" ),
                     QDir::currentPath(),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  cv::Mat landmarkMat = cv::imread( FileName.toStdString() );

  RightLandmarkTransform->GetMatrix()->DeepCopy( landmarkMat.ptr<double>(0) );
  RightLandmarkAvailable = true;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadValidChessRegistration()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open Validation Checkerboard Registration" ),
                     QDir::currentPath(),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  cv::Mat matrix = cv::imread( FileName.toStdString() );

  ValidBoardRegTransform->GetMatrix()->DeepCopy( matrix.ptr<double>(0) );

  ValidBoardAvailable = true;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadChessRegistration()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open Calibration Checkerboard Registration" ),
                     QDir::currentPath(),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  cv::Mat matrix = cv::imread( FileName.toStdString() );

  BoardRegTransform->GetMatrix()->DeepCopy( matrix.ptr<double>(0) );

  BoardRegAvailable = true;
}

//---------------------------------------------------------
// process a single checkerboard
int CameraCalibrationMainWidget::ProcessCheckerBoard( int cameraIndex, int width, int height, double size, processingMode mode,
    vtkPoints *source, vtkPoints *target, std::string videoTitle )
{
  // double check if the video feed is available
  if ( cameraIndex >= CVInternals->CameraCount() && !CVInternals->IsFeedAvailable( cameraIndex ) )
  {
    return CaptureCount[cameraIndex];
  }

  LOG_INFO("Checkerboard dimension: " << width << "x" << height << "x" << size);

  CVInternals->QueryFrame(cameraIndex, CameraImages[cameraIndex]);

  // make a copy of the current feed
  // need to do so as the current feed may be self-updating
  cv::Mat copy(CameraImages[cameraIndex]);

  cv::Size board_sz(width, height);
  int board_n = width * height;
  std::vector<cv::Point2f> corners;

  cv::Point text_origin;
  int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;

  //if(!s.useFisheye)
  {
    // fast check erroneously fails with high distortions like fisheye
    chessBoardFlags |= cv::CALIB_CB_FAST_CHECK;
  }
  bool found = cv::findChessboardCorners( copy, board_sz, corners, chessBoardFlags );

  //if the checkerboard is not entirely visible, chuck this image and return
  if ( !found )
  {
    return CaptureCount[cameraIndex];
  }

  cv::Mat viewGray;
  cv::cvtColor(copy, viewGray, cv::COLOR_BGR2GRAY);
  cv::cornerSubPix( viewGray, corners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1 ));

  // sometimes OpenCV returned a flipped image
  bool flip = false;

  if ( corners[board_n-1].x < corners[0].x )
  {
    flip = true;
    LOG_INFO("Image is flipped.");
  }

  if ( mode == forCalibration )
  {
    image_points[cameraIndex].push_back(corners);
    CaptureCount[cameraIndex]++;
    cv::drawChessboardCorners( copy, board_sz, cv::Mat(corners), found );

    cv::imshow( videoTitle, copy );
    cv::waitKey(25);
  }
  else if ( ( mode == forLandmark || mode == forEvaluation || mode == forEvaluation2 ) && DataCollector->IsStarted() )
  {
    PlusTrackedFrame frame;
    TrackingDataChannel->GetTrackedFrame(frame);
    if( TransformRepository->SetTransforms(frame) != PLUS_SUCCESS)
    {
      LOG_ERROR("Unable to load transforms into repository. Aborting.");
      return CaptureCount[cameraIndex];
    }
    bool isValid(false);
    PlusTransformName checkboardToReferenceName("Checkerboard", "Reference");
    vtkSmartPointer<vtkMatrix4x4> checkboardToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(checkboardToReferenceName, checkboardToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate checkerboard to reference transform.");
      return CaptureCount[cameraIndex];
    }

    PlusTransformName HMDToReferenceName("HMD", "Reference");
    vtkSmartPointer<vtkMatrix4x4> HMDToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(HMDToReferenceName, HMDToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate HMD to reference transform.");
      return CaptureCount[cameraIndex];
    }
    HMDToReferenceTransform->Invert();

    vtkSmartPointer<vtkTransform> tempTransform = vtkSmartPointer<vtkTransform>::New();
    tempTransform->PostMultiply();
    tempTransform->Identity();

    if ( mode == forEvaluation || mode == forLandmark )
    {
      tempTransform->Concatenate( BoardRegTransform );
      tempTransform->Concatenate( checkboardToReferenceTransform );
      tempTransform->Concatenate( HMDToReferenceTransform );
    }
    else if ( mode == forEvaluation2 )
    {
      tempTransform->Concatenate( ValidBoardRegTransform );
      tempTransform->Concatenate( checkboardToReferenceTransform );
      tempTransform->Concatenate( HMDToReferenceTransform );
    }

    if ( mode == forEvaluation || mode == forEvaluation2 )
    {
      if ( cameraIndex == 0 )
      {
        tempTransform->Concatenate( LeftLandmarkTransform->GetMatrix() );
      }
      else if ( cameraIndex == 1 )
      {
        tempTransform->Concatenate( RightLandmarkTransform->GetMatrix() );
      }
    }

    tempTransform->Update();

    int point_count = board_n;
    cv::Mat objectPoints(point_count, 3, CV_64FC1);
    cv::Mat imagePoints(point_count, 2, CV_64FC1);
    cv::Mat rotationMat(3, 3, CV_64FC1 );
    cv::Mat pointofinterest(3, 1, CV_64FC1);
    cv::Mat fproject(3, 1, CV_64FC1);

    cv::Mat MP( 1, 3, CV_64FC1 ); // the 3D point in CvMat
    cv::Mat Mres( 1, 2, CV_64FC1 ); // the projected points in pixel
    cv::Mat MR( 3, 1, CV_64FC1 ); // the rotation, set to 0
    cv::Mat Mt( 3, 1, CV_64FC1 ); // the translation, set to 0

    // set the rotation/translation to none
    MR.at<double>(0,0) = 0;
    MR.at<double>(1,0) = 0;
    MR.at<double>(2,0) = 0;
    Mt.at<double>(0,0) = 0;
    Mt.at<double>(1,0) = 0;
    Mt.at<double>(2,0) = 0;

    double Phm[4];
    for ( int i = 0, j = 0, idx=(board_n-1); j < height; j++ )
    {
      for ( int k = 0; k < width; k++, i++, idx-- )
      {
        // objectPoints are the "virtual" corners of the checkerboard
        // in its local coordinate system.  Note the z==0
        objectPoints.at<double>(i, 0) = k*size;
        objectPoints.at<double>(i, 1) = j*size;
        objectPoints.at<double>(i, 2) = 0;

        // imagePoints are the actual corners detected by the openCV
        // sometimes OpenCV does not return the detected corners from
        // the top-left to the bottom-right order, so we need to
        // reverse the order ourselves
        if ( flip )
        {
          imagePoints.at<double>(i, 0) = corners[idx].x;
          imagePoints.at<double>(i, 1) = corners[idx].y;
        }
        else
        {
          imagePoints.at<double>(i, 0) = corners[i].x;
          imagePoints.at<double>(i, 1) = corners[i].y;;
        }
      }
    }

    // finds the pose of the checkerboard
    // BUG:!!  NOTE THAT OPENCV MAY FIND A FLIPPED ROTATION!
    std::vector<cv::Mat> rotationsVector;
    std::vector<cv::Mat> translationsVector;
    cv::calibrateCamera(objectPoints, imagePoints, board_sz, *CVInternals->GetInstrinsicMatrix(cameraIndex), *CVInternals->GetDistortionCoeffs(cameraIndex), rotationsVector, translationsVector);
    cv::Rodrigues(rotationsVector, rotationMat);

    // export the corners to an external data structure
    if ( cameraIndex == 0 )
    {
      ImagePointsLeft.resize( point_count );
      HomographyPointsLeft.resize( point_count );
      TrackedPointsLeft.resize( point_count );
      ReprojectionPointsLeft.resize( point_count );
    }
    else if ( cameraIndex == 1 )
    {
      ImagePointsRight.resize( point_count );
      HomographyPointsRight.resize( point_count );
      TrackedPointsRight.resize( point_count );
      ReprojectionPointsRight.resize( point_count );
    }
    p3 tempp3;

    for ( int index = 0; index < point_count; index ++ )
    {
      // pointofinterest are the "virtual" corners of the checkerboard
      // in its local coordinate system.  Note the z==0
      pointofinterest.at<double>(0,0) = objectPoints.at<double>(index, 0);
      pointofinterest.at<double>(1,0) = objectPoints.at<double>(index, 1);
      pointofinterest.at<double>(2,0) = objectPoints.at<double>(index, 2);

      // fproject is the re-projected 3D points of the corner using
      // the homography found by opencv
      /** Matrix transform: dst = A*B + C*/
      cv::gemm(rotationMat, pointofinterest, 1., translationsVector, 1., fproject);

      // for vtk
      //
      // Phm are the "virtual" corners of the checkerboard
      // in its local coordinate system.  Note the z==0
      //
      Phm[0] = objectPoints.at<double>(index, 0);
      Phm[1] = objectPoints.at<double>(index, 1);
      Phm[2] = objectPoints.at<double>(index, 2);
      Phm[3] = 1.0;

      tempTransform->MultiplyPoint( Phm, Phm );

      if ( mode == forLandmark )
      {
        source->InsertNextPoint( Phm );
        target->InsertNextPoint( fproject.ptr<double>(0) );
      }
      else if ( mode == forEvaluation || mode == forEvaluation2 )
      {
        // Phm are points in 3D since tempTransform incorporates the camera's landmark transformation
        MP.at<double>(0,0) = Phm[0];
        MP.at<double>(0,1) = Phm[1];
        MP.at<double>(0,2) = Phm[2];

        // projection
        cv::projectPoints(MP, MR, Mt, *CVInternals->GetInstrinsicMatrix(cameraIndex), *CVInternals->GetDistortionCoeffs(cameraIndex), Mres);

        // Mres is the projected 2D pixel of Phm
        text_origin.x = Mres.at<double>(0,0);
        text_origin.y = Mres.at<double>(0,1);
        cv::circle(copy, text_origin, 3, cv::Scalar(255, 0, 0), 2);

        // detected corners
        tempp3.x = imagePoints.at<double>(index, 0);
        tempp3.y = imagePoints.at<double>(index, 1);
        tempp3.z = 0.0;

        if ( cameraIndex == 0 )
        {
          ImagePointsLeft[index] = tempp3;
        }
        else if ( cameraIndex == 1 )
        {
          ImagePointsRight[index] = tempp3;
        }

        // projected homography 3D points
        tempp3.x = *fproject.ptr<double>(0);
        tempp3.y = *fproject.ptr<double>(1);
        tempp3.z = *fproject.ptr<double>(2);

        if ( cameraIndex == 0 )
        {
          HomographyPointsLeft[index] = tempp3;
        }
        else if ( cameraIndex == 1 )
        {
          HomographyPointsRight[index] = tempp3;
        }

        // re-projected 2D points
        tempp3.x = Mres.at<double>(0,0);
        tempp3.y = Mres.at<double>(0,1);
        tempp3.z = 0.0;

        if ( cameraIndex == 0 )
        {
          ReprojectionPointsLeft[index] = tempp3;
        }
        else if ( cameraIndex == 1 )
        {
          ReprojectionPointsRight[index] = tempp3;
        }

        // tracked 3D points
        tempp3.x = Phm[0];
        tempp3.y = Phm[1];
        tempp3.z = Phm[2];

        if ( cameraIndex == 0 )
        {
          TrackedPointsLeft[index] = tempp3;
        }
        else if ( cameraIndex == 1 )
        {
          TrackedPointsRight[index] = tempp3;
        }

        *(fileOutput[cameraIndex]) << index << ", " << imagePoints.at<double>(index, 0) << ", " << imagePoints.at<double>(index, 1) << ", "
                                   << *fproject.ptr<double>(0) << ", "
                                   << *fproject.ptr<double>(1) << ", "
                                   << *fproject.ptr<double>(2) << ", "
                                   << Mres.at<double>(0, 0) << ", "
                                   << Mres.at<double>(0, 1) << ", "
                                   << Phm[0] << ", " << Phm[1] << ", " << Phm[2] << std::endl;

        LOG_INFO(imagePoints.at<double>(index, 0) << " " << imagePoints.at<double>(index, 1) << " "
                 << *fproject.ptr<double>(0) << " " << *fproject.ptr<double>(1) << " " << *fproject.ptr<double>(2) << "  "
                 << Mres.at<double>(0, 0) << " "
                 << Mres.at<double>(0, 1) << " "
                 << Phm[0] << " " << Phm[1] << " " << Phm[2]);
      }

      text_origin.x = imagePoints.at<double>(index, 0);
      text_origin.y = imagePoints.at<double>(index, 1);
      cv::circle(copy, text_origin, 3, cv::Scalar(0, 255, 0), 2);
    }
  }

  return CaptureCount[cameraIndex];
}

//---------------------------------------------------------
// process the recorded image
void CameraCalibrationMainWidget::CaptureAndProcessLeftImage()
{
  int beforeAttempt = CaptureCount[LeftCameraIndex];
  int n = ProcessCheckerBoard( LeftCameraIndex,
                               GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
                               forCalibration, 0, 0, std::string( "Results" ) );

  if( CaptureCount[LeftCameraIndex] - beforeAttempt == 0 )
  {
    LOG_WARNING("Unable to locate checkerboard in camera image. Try again.");
  }
  else
  {
    LOG_INFO("Success! Captured " << n << " left calibration image thus far.");
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::CaptureAndProcessRightImage()
{
  int beforeAttempt = CaptureCount[RightCameraIndex];
  int n = ProcessCheckerBoard( RightCameraIndex,
                               GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
                               forCalibration, 0, 0, std::string( "Results" ) );

  if( CaptureCount[RightCameraIndex] - beforeAttempt == 0 )
  {
    LOG_WARNING("Unable to locate checkerboard in camera image. Try again.");
  }
  else
  {
    LOG_INFO("Success! Captured " << n << " right calibration image thus far.");
  }
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CalibBoardWidthValueChanged(int i)
{
  appSettings.setValue("calib/boardWidth", i);
  ResetCalibrationCheckerboards();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CalibBoardHeightValueChanged(int i)
{
  appSettings.setValue("calib/boardHeight", i);
  ResetCalibrationCheckerboards();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::CalibBoardQuadSizeValueChanged(double i)
{
  appSettings.setValue("calib/quadSize", i);
  ResetCalibrationCheckerboards();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::ValidationBoardWidthValueChanged(int i)
{
  appSettings.setValue("validation/boardWidth", i);
  ResetCalibrationCheckerboards();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::ValidationBoardHeightValueChanged(int i)
{
  appSettings.setValue("validation/boardHeight", i);
  ResetCalibrationCheckerboards();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWidget::ValidationBoardQuadSizeValueChanged(double i)
{
  appSettings.setValue("validation/quadSize", i);
  ResetCalibrationCheckerboards();
}

//---------------------------------------------------------
// compute the intrinsics
void CameraCalibrationMainWidget::ComputeLeftIntrinsic()
{
  double totalAvgErr;
  std::vector<float> perViewErrors;
  this->ComputeIntrinsicsAndDistortion( LeftCameraIndex, totalAvgErr, perViewErrors );

  // TODO : file dialogs
  this->WriteIntrinsicsToFile( LeftCameraIndex, "Left_Intrinsics.xml" );
  this->WriteDistortionToFile( LeftCameraIndex, "Left_Distortion.xml" );

  LOG_INFO("Left intrinsics/distortions computed and saved to: " << "Left_Intrinsics.xml");
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ComputeRightIntrinsic()
{
  double totalAvgErr;
  std::vector<float> perViewErrors;
  this->ComputeIntrinsicsAndDistortion( RightCameraIndex, totalAvgErr, perViewErrors );

  // TODO : file dialogs
  this->WriteIntrinsicsToFile( RightCameraIndex, "Right_Intrinsics.xml" );
  this->WriteDistortionToFile( RightCameraIndex, "Right_Distortion.xml" );

  LOG_INFO("Right intrinsics/distortions computed and saved to: " << "Right_Intrinsics.xml");
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::WriteIntrinsicsToFile( int cameraIndex, const std::string& filename )
{
  if ( cameraIndex >= this->CVInternals->CameraCount() )
  {
    return;
  }

  cv::imwrite(filename, *CVInternals->GetInstrinsicMatrix(cameraIndex));
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::WriteDistortionToFile( int cameraIndex, const std::string& filename )
{
  if ( cameraIndex >= this->CVInternals->CameraCount() )
  {
    return;
  }

  cv::imwrite(filename, *CVInternals->GetDistortionCoeffs(cameraIndex));
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ComputeIntrinsicsAndDistortion( int cameraIndex, double& totalAvgErr, std::vector<float>& perViewErrors )
{
  if ( cameraIndex >= this->CVInternals->CameraCount() )
  {
    return;
  }

  if ( CaptureCount[cameraIndex] < MinBoardNeeded )
  {
    std::stringstream ss;
    ss << "Not enough board images recorded: " << CaptureCount[cameraIndex] << "/" << MinBoardNeeded;
    LOG_ERROR(ss.str());
    ShowStatusMessage(ss.str().c_str());
    return;
  }

  std::vector<cv::Mat> rotationsVector;
  std::vector<cv::Mat> translationsVector;

  std::vector<std::vector<cv::Point3d> > objectPoints(1);
  CalcBoardCornerPositions(GetBoardHeightCalib(), GetBoardWidthCalib(), GetBoardQuadSizeCalib(), objectPoints[0]);

  objectPoints.resize(image_points[cameraIndex].size(), objectPoints[0]);

  double reprojectionError = cv::calibrateCamera(objectPoints, image_points[cameraIndex], cv::Size(CameraImages[cameraIndex].size[0], CameraImages[cameraIndex].size[1]),
                             *CVInternals->GetInstrinsicMatrix(cameraIndex), *CVInternals->GetDistortionCoeffs(cameraIndex), rotationsVector, translationsVector);

  totalAvgErr = ComputeReprojectionErrors(objectPoints, image_points[cameraIndex], rotationsVector, translationsVector, *CVInternals->GetInstrinsicMatrix(cameraIndex),
                                          *CVInternals->GetDistortionCoeffs(cameraIndex), perViewErrors, true);

  LOG_INFO("Camera calibrated with reprojection error: " << reprojectionError);

  if ( cameraIndex == LeftCameraIndex )
  {
    LeftIntrinsicAvailable = LeftDistortionAvailable = true;
  }
  else if ( cameraIndex == RightCameraIndex )
  {
    RightIntrinsicAvailable = RightDistortionAvailable = true;
  }
}

//---------------------------------------------------------
// collect a single point using the sharp tool
void CameraCalibrationMainWidget::CollectRegPoint()
{
  if ( this->DataCollector->IsStarted() )
  {
    PlusTrackedFrame frame;
    TrackingDataChannel->GetTrackedFrame(frame);
    if( TransformRepository->SetTransforms(frame) != PLUS_SUCCESS)
    {
      LOG_ERROR("Unable to load transforms into repository. Aborting.");
      ShowStatusMessage("Unable to load transforms into repository. Aborting.");
      return;
    }
    bool isValid(false);
    PlusTransformName checkboardToReferenceName("Checkerboard", "Reference");
    vtkSmartPointer<vtkMatrix4x4> checkboardToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(checkboardToReferenceName, checkboardToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate checkerboard to reference transform.");
      ShowStatusMessage("Unable to locate checkerboard to reference transform.");
      return;
    }
    checkboardToReferenceTransform->Invert();

    PlusTransformName stylusToReferenceName("Stylus", "Reference");
    vtkSmartPointer<vtkMatrix4x4> stylusToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(stylusToReferenceName, stylusToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate stylus to reference transform.");
      ShowStatusMessage("Unable to locate stylus to reference transform.");
      return;
    }

    vtkSmartPointer<vtkTransform> tempTransform = vtkSmartPointer<vtkTransform>::New();
    tempTransform->Identity();
    tempTransform->Concatenate( stylusToReferenceTransform );
    tempTransform->Concatenate( checkboardToReferenceTransform );
    tempTransform->Update();

    double *pos = tempTransform->GetPosition();

    int nw = GetBoardWidthCalib();
    int nh = GetBoardHeightCalib();
    double size = GetBoardQuadSizeCalib();

    LOG_INFO("Board size is: " << nw << "x" << nh << "x" << size << ".");

    // the board registration is from the checkerboard's 2D coordinate
    ValidBoardSource->InsertNextPoint( (double)nw*size,
                                       (double)nh*size,
                                       0.0 );

    ValidBoardTarget->InsertNextPoint( pos[0], pos[1], pos[2] );
    LOG_INFO("Point: " << pos[0] << " " << pos[1] << " " << pos[2] << ".");
  }
  else
  {
    LOG_ERROR("Either the tracker is not initialized or it is not tracking.");
    return;
  }
}

//---------------------------------------------------------
// perform landmark registration of the validation checkerboard to reference tool
void CameraCalibrationMainWidget::PerformRegBoardRegistration()
{
  if ( !ValidBoardAvailable )
  {
    if ( ValidBoardSource->GetNumberOfPoints() < 4 )
    {
      LOG_INFO("Please collect more than 4 points from the checkerboard.");
      ShowStatusMessage("Please collect more than 4 points from the checkerboard.");
    }
    else
    {
      ValidBoardRegTransform->SetSourceLandmarks( ValidBoardSource );
      ValidBoardRegTransform->SetTargetLandmarks( ValidBoardTarget );
      ValidBoardRegTransform->SetModeToRigidBody();
      ValidBoardRegTransform->Update();

      // TODO : file dialog
      double *data = (double*)ValidBoardRegTransform->GetMatrix()->Element;
      cvSave( "validCheckerBoardReg.xml", &cvMat( 4, 4, CV_64FC1, data ) );

      ValidBoardAvailable = true;
    }
  }
}

//---------------------------------------------------------
// collect a single point using the sharp tool
void CameraCalibrationMainWidget::CollectStylusPoint()
{
  if ( this->DataCollector->IsStarted() )
  {
    PlusTrackedFrame frame;
    TrackingDataChannel->GetTrackedFrame(frame);
    if( TransformRepository->SetTransforms(frame) != PLUS_SUCCESS)
    {
      LOG_ERROR("Unable to load transforms into repository. Aborting.");
      ShowStatusMessage("Unable to load transforms into repository. Aborting.");
      return;
    }
    bool isValid(false);
    PlusTransformName checkboardToReferenceName("Checkerboard", "Reference");
    vtkSmartPointer<vtkMatrix4x4> checkboardToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(checkboardToReferenceName, checkboardToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate checkerboard to reference transform.");
      ShowStatusMessage("Unable to locate checkerboard to reference transform.");
      return;
    }
    checkboardToReferenceTransform->Invert();

    PlusTransformName stylusToReferenceName("Stylus", "Reference");
    vtkSmartPointer<vtkMatrix4x4> stylusToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(stylusToReferenceName, stylusToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate stylus to reference transform.");
      ShowStatusMessage("Unable to locate stylus to reference transform.");
      return;
    }

    vtkSmartPointer<vtkTransform> tempTransform = vtkSmartPointer<vtkTransform>::New();
    tempTransform->Identity();
    tempTransform->Concatenate( stylusToReferenceTransform );
    tempTransform->Concatenate( checkboardToReferenceTransform );
    tempTransform->Update();

    double *pos = tempTransform->GetPosition();
    int nw = GetBoardWidthCalib();
    int nh = GetBoardHeightCalib();
    double size = ui.doubleSpinBox_QuadSizeValid->value();

    LOG_INFO("Board size is: " << nw << "x" << nh << "x" << size << ".");

    // the board registration is from the checkerboard's 2D coordinate
    // to the reference tool's 3D coordinate
    BoardSource->InsertNextPoint( (double)nw*size, (double)nh*size, 0.0 );
    BoardTarget->InsertNextPoint( pos[0], pos[1], pos[2] );

    LOG_INFO("Point: " << pos[0] << " " << pos[1] << " " << pos[2] << ".");
  }
  else
  {
    LOG_ERROR("Either the tracker is not initialized or it is not tracking.");
    ShowStatusMessage("Either the tracker is not initialized or it is not tracking.");
    return;
  }
}

//---------------------------------------------------------
// perform landmark registration of the checkerboard to reference tool
void CameraCalibrationMainWidget::PerformBoardRegistration()
{
  if ( !BoardRegAvailable )
  {
    if ( BoardSource->GetNumberOfPoints() < 4 )
    {
      LOG_ERROR("Please collect more than 4 points from the checkerboard");
      ShowStatusMessage("Please collect more than 4 points from the checkerboard");
      return;
    }
    else
    {
      BoardRegTransform->SetSourceLandmarks( BoardSource );
      BoardRegTransform->SetTargetLandmarks( BoardTarget );
      BoardRegTransform->SetModeToRigidBody();
      BoardRegTransform->Update();

      // TODO : file dialog
      double *data = (double*)BoardRegTransform->GetMatrix()->Element;
      cvSave( "CheckerBoardReg.xml", &cvMat( 4, 4, CV_64FC1, data ) );

      BoardRegAvailable = true;
    }
  }
}

//---------------------------------------------------------
// for finding the landmark registration
// between the optical axis and the attached reference tool
void CameraCalibrationMainWidget::DrawLeftChessBoardCorners()
{
  int beforeAttempt = CaptureCount[LeftCameraIndex];
  int k = ProcessCheckerBoard( LeftCameraIndex,
                               GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
                               forLandmark,
                               BoardCornerLeftSource, BoardCornerLeftTarget, std::string( "Results" ) );

  if( CaptureCount[LeftCameraIndex] - beforeAttempt == 0 )
  {
    LOG_WARNING("Unable to locate checkerboard in camera image. Try again.");
    ShowStatusMessage("Unable to locate checkerboard in camera image. Try again.");
  }
  else
  {
    std::stringstream ss;
    ss << "Success! Captured " << k << " left calibration points thus far.";
    LOG_INFO(ss.str());
    ShowStatusMessage(ss.str().c_str());
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::DrawRightChessBoardCorners()
{
  int beforeAttempt = CaptureCount[RightCameraIndex];
  int k = ProcessCheckerBoard( RightCameraIndex,
                               GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
                               forLandmark,
                               BoardCornerRightSource, BoardCornerRightTarget, std::string( "Results" ) );

  if( CaptureCount[RightCameraIndex] - beforeAttempt == 0 )
  {
    LOG_WARNING("Unable to locate checkerboard in camera image. Try again.");
    ShowStatusMessage("Unable to locate checkerboard in camera image. Try again.");
  }
  else
  {
    std::stringstream ss;
    ss << "Success! Captured " << k << " right calibration points thus far.";
    LOG_INFO(ss.str());
    ShowStatusMessage(ss.str().c_str());
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ComputeHMDRegistration()
{
  if ( BoardCornerLeftSource->GetNumberOfPoints() != BoardCornerLeftTarget->GetNumberOfPoints() )
  {
    LOG_ERROR("Number of points do not match in HMD registration.");
    ShowStatusMessage("Number of points do not match in HMD registration.");
    return;
  }
  else
  {
    if ( !LeftLandmarkAvailable && BoardCornerLeftSource->GetNumberOfPoints() > 0 )
    {
      LeftLandmarkTransform->SetSourceLandmarks( BoardCornerLeftSource );
      LeftLandmarkTransform->SetTargetLandmarks( BoardCornerLeftTarget );
      LeftLandmarkTransform->SetModeToRigidBody();
      LeftLandmarkTransform->Update();
      LeftLandmarkAvailable = true;

      double *data = (double*)LeftLandmarkTransform->GetMatrix()->Element;

      // TODO : file dialog
      cvSave( "Left_Landmark.xml", &cvMat(4,4,CV_64FC1, data ) );
    }
  }

  if ( BoardCornerRightSource->GetNumberOfPoints() != BoardCornerRightTarget->GetNumberOfPoints() )
  {
    LOG_ERROR("Number of points do not match in HMD registration.");
    ShowStatusMessage("Number of points do not match in HMD registration.");
    return;
  }
  else
  {
    if ( !RightLandmarkAvailable && BoardCornerRightSource->GetNumberOfPoints() > 0 )
    {
      RightLandmarkTransform->SetSourceLandmarks( BoardCornerRightSource );
      RightLandmarkTransform->SetTargetLandmarks( BoardCornerRightTarget );
      RightLandmarkTransform->SetModeToRigidBody();
      RightLandmarkTransform->Update();
      RightLandmarkAvailable = true;

      double *data = (double*)RightLandmarkTransform->GetMatrix()->Element;

      // TODO : file dialog
      cvSave( "Right_Landmark.xml", &cvMat( 4, 4, CV_64FC1, data ) );
    }
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateChessVisualTimer( bool v)
{
  if ( v )
  {
    ValidateTrackingTimer->start( 0 );
  }
  else
  {
    ValidateTrackingTimer->stop();
  }
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
  cv::Mat rightImage;
  CVInternals->QueryFrame(LeftCameraIndex, leftImage);
  CVInternals->QueryFrame(RightCameraIndex, rightImage);

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
void CameraCalibrationMainWidget::ValidateValidChess()
{
  if ( this->DataCollector->IsStarted() )
  {
    if ( LeftLandmarkAvailable && LeftIntrinsicAvailable && LeftDistortionAvailable && ValidBoardAvailable )
    {
      ProcessCheckerBoard( LeftCameraIndex,
                           ui.spinBox_BoardWidthValid->value(), ui.spinBox_BoardHeightValid->value(), ui.doubleSpinBox_QuadSizeValid->value(),
                           forEvaluation, 0, 0, std::string( "Left Result" ) );
    }

    if ( RightLandmarkAvailable && RightIntrinsicAvailable && RightDistortionAvailable && ValidBoardAvailable )
    {
      ProcessCheckerBoard( RightCameraIndex,
                           ui.spinBox_BoardWidthValid->value(), ui.spinBox_BoardHeightValid->value(), ui.doubleSpinBox_QuadSizeValid->value(),
                           forEvaluation, 0, 0, std::string( "Right Result" ) );
    }
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateChessVisual()
{
  if ( this->DataCollector->IsStarted() )
  {
    CVInternals->QueryFrame(LeftCameraIndex, CameraImages[LeftCameraIndex]);
    CVInternals->QueryFrame(RightCameraIndex, CameraImages[RightCameraIndex]);

    PlusTrackedFrame frame;
    TrackingDataChannel->GetTrackedFrame(frame);
    if( TransformRepository->SetTransforms(frame) != PLUS_SUCCESS)
    {
      LOG_ERROR("Unable to load transforms into repository. Aborting.");
      ShowStatusMessage("Unable to load transforms into repository. Aborting.");
      return;
    }
    bool isValid(false);
    PlusTransformName checkboardToReferenceName("Checkerboard", "Reference");
    vtkSmartPointer<vtkMatrix4x4> checkboardToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(checkboardToReferenceName, checkboardToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate checkerboard to reference transform.");
      ShowStatusMessage("Unable to locate checkerboard to reference transform.");
      return;
    }

    PlusTransformName HMDToReferenceName("HMD", "Reference");
    vtkSmartPointer<vtkMatrix4x4> HMDToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(HMDToReferenceName, HMDToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate HMD to reference transform.");
      ShowStatusMessage("Unable to locate HMD to reference transform.");
      return;
    }
    HMDToReferenceTransform->Invert();

    vtkSmartPointer<vtkTransform> tempTransform = vtkSmartPointer<vtkTransform>::New();
    tempTransform->PostMultiply();

    double points[4];
    int nPoints = GetBoardWidthCalib() * GetBoardHeightCalib();

    cv::Mat MP( 1, 3, CV_64FC1 ); // the 3D point in CvMat
    cv::Mat Mres( 1, 2, CV_64FC1 ); // the projected points in pixel
    cv::Mat MR( 3, 1, CV_64FC1 ); // the rotation, set to 0
    cv::Mat Mt( 3, 1, CV_64FC1 ); // the translation, set to 0

    cv::Point text_origin;

    // set the rotation/translation to none
    MR.at<double>(0,0) = 0;
    MR.at<double>(1,0) = 0;
    MR.at<double>(2,0) = 0;
    Mt.at<double>(0,0) = 0;
    Mt.at<double>(1,0) = 0;
    Mt.at<double>(2,0) = 0;

    if ( LeftLandmarkAvailable && LeftIntrinsicAvailable && LeftDistortionAvailable && BoardRegAvailable )
    {
      tempTransform->Identity();
      tempTransform->Concatenate( BoardRegTransform );
      tempTransform->Concatenate( checkboardToReferenceTransform );
      tempTransform->Concatenate( HMDToReferenceTransform );
      tempTransform->Concatenate( LeftLandmarkTransform->GetMatrix() );
      tempTransform->Update();

      for ( int i = 0, j = 0; j < GetBoardHeightCalib(); j++ )
      {
        for ( int k = 0; k < GetBoardWidthCalib(); k++, i++ )
        {
          points[0] = (double)k * GetBoardQuadSizeCalib();
          points[1] = (double)j * GetBoardQuadSizeCalib();
          points[2] = 0.0;
          points[3] = 1.0;
          tempTransform->MultiplyPoint( points, points );

          MP.at<double>(0,0) = points[0];
          MP.at<double>(1,0) = points[1];
          MP.at<double>(2,0) = points[2];

          // projection
          cv::projectPoints( MP, MR, Mt, *CVInternals->GetInstrinsicMatrix(LeftCameraIndex), *CVInternals->GetDistortionCoeffs(LeftCameraIndex), Mres);

          // Mres is the projected 2D pixel of Phm
          text_origin.x = Mres.at<double>(0,0) = points[2];
          text_origin.y = Mres.at<double>(0,1) = points[2];
          cv::circle(CameraImages[LeftCameraIndex], text_origin, 3, cv::Scalar(255,0,0), 2);
        }
      }

      cv::imshow("Left Result", CameraImages[LeftCameraIndex]);
    }

    if ( RightLandmarkAvailable && RightIntrinsicAvailable && RightDistortionAvailable && BoardRegAvailable )
    {
      tempTransform->Identity();
      tempTransform->Concatenate( BoardRegTransform );
      tempTransform->Concatenate( checkboardToReferenceTransform );
      tempTransform->Concatenate( HMDToReferenceTransform );
      tempTransform->Concatenate( RightLandmarkTransform->GetMatrix() );
      tempTransform->Update();

      for ( int i = 0, j = 0; j < GetBoardHeightCalib(); j++ )
      {
        for ( int k = 0; k < GetBoardWidthCalib(); k++, i++ )
        {
          points[0] = (double)k * GetBoardQuadSizeCalib();
          points[1] = (double)j * GetBoardQuadSizeCalib();
          points[2] = 0.0;
          points[3] = 1.0;
          tempTransform->MultiplyPoint( points, points );

          MP.at<double>(0,0) = points[0];
          MP.at<double>(1,0) = points[1];
          MP.at<double>(2,0) = points[2];

          // projection
          cv::projectPoints( MP, MR, Mt, *CVInternals->GetInstrinsicMatrix(RightCameraIndex), *CVInternals->GetDistortionCoeffs(RightCameraIndex), Mres);

          // Mres is the projected 2D pixel of Phm
          text_origin.x = Mres.at<double>(0,0) = points[2];
          text_origin.y = Mres.at<double>(0,1) = points[2];
          cv::circle(CameraImages[RightCameraIndex], text_origin, 3, cv::Scalar(255,0,0), 2);
        }
      }
      cv::imshow( "Right Result", CameraImages[RightCameraIndex] );
    }
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateChess()
{
  if ( this->DataCollector->IsStarted() )
  {
    if ( LeftLandmarkAvailable && LeftIntrinsicAvailable && LeftDistortionAvailable && BoardRegAvailable )
    {
      ProcessCheckerBoard( LeftCameraIndex,
                           GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
                           forEvaluation, 0, 0, std::string( "Left Result" ) );
    }

    if ( RightLandmarkAvailable && RightIntrinsicAvailable && RightDistortionAvailable && BoardRegAvailable )
    {
      ProcessCheckerBoard( RightCameraIndex,
                           GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
                           forEvaluation, 0, 0, std::string( "Right Result" ) );
    }

    // perform triangulation
    // corners are stores at imagePointsL and imagePointsR
    // make sure both left and right images are processed and have equal number of corners
    if ( LeftLandmarkAvailable && LeftIntrinsicAvailable && LeftDistortionAvailable && BoardRegAvailable  &&
         RightLandmarkAvailable && RightIntrinsicAvailable && RightDistortionAvailable && BoardRegAvailable  &&
         ImagePointsLeft.size() == ImagePointsRight.size() )
    {
      int nPoints = ImagePointsLeft.size();

      // the projection matrix for the left camera is identity
      double leftExtrinsic[12] = { 1, 0, 0, 0,
                                   0, 1, 0, 0,
                                   0, 0, 1, 0
                                 };
      cv::Mat LeftExtrinsic( 3, 4, CV_64F, leftExtrinsic );

      // the projection matrix for the right camera is
      // Right_landmark * inverse( Left_landmark )
      vtkSmartPointer< vtkTransform > tTransform =
        vtkSmartPointer< vtkTransform >::New();
      tTransform->PostMultiply();
      tTransform->Identity();
      tTransform->Concatenate( LeftLandmarkTransform->GetLinearInverse() );
      tTransform->Concatenate( RightLandmarkTransform );
      tTransform->Update();

      double rightExtrinsic[12] = { tTransform->GetMatrix()->Element[0][0],
                                    tTransform->GetMatrix()->Element[0][1],
                                    tTransform->GetMatrix()->Element[0][2],
                                    tTransform->GetMatrix()->Element[0][3],
                                    tTransform->GetMatrix()->Element[1][0],
                                    tTransform->GetMatrix()->Element[1][1],
                                    tTransform->GetMatrix()->Element[1][2],
                                    tTransform->GetMatrix()->Element[1][3],
                                    tTransform->GetMatrix()->Element[2][0],
                                    tTransform->GetMatrix()->Element[2][1],
                                    tTransform->GetMatrix()->Element[2][2],
                                    tTransform->GetMatrix()->Element[2][3]
                                  };
      cv::Mat RightExtrinsic( 3, 4, CV_64F, rightExtrinsic );

      std::vector<cv::Point2d> leftImagePoints;
      std::vector<cv::Point2d> rightImagePoints;
      std::vector<cv::Vec4d> points4D;

      // copy the points over
      for ( int i = 0; i < nPoints; i++ )
      {
        leftImagePoints.push_back(cv::Point2d(ImagePointsLeft[i].x, ImagePointsLeft[i].y));
        rightImagePoints.push_back(cv::Point2d(ImagePointsRight[i].x, ImagePointsRight[i].y));
      }

      cv::triangulatePoints(LeftExtrinsic, RightExtrinsic, leftImagePoints, rightImagePoints, points4D);

      double x, y, z;
      for ( int i = 0; i < nPoints; i++ )
      {
        // output to file
        cv::Vec4d result = points4D[i]/points4D[i][3];
        x = result[0];
        y = result[1];
        z = result[2];
        std::cout << x << ", " << y << ", " << z << std::endl;
        *(fileOutput[2]) << x << ", "
                         << y << ", "
                         << z << std::endl;

        *(fileOutput[3]) << i << ", "
                         << ImagePointsLeft[i].x << ", " << ImagePointsLeft[i].y << ", "
                         << HomographyPointsLeft[i].x << ", "
                         << HomographyPointsLeft[i].y << ", "
                         << HomographyPointsLeft[i].z << ", "
                         << ReprojectionPointsLeft[i].x << ", "
                         << ReprojectionPointsLeft[i].y << ", "
                         << TrackedPointsLeft[i].x << ", "
                         << TrackedPointsLeft[i].y << ", "
                         << TrackedPointsLeft[i].z << ", , ,"
                         << i << ", "
                         << ImagePointsRight[i].x << ", " << ImagePointsRight[i].y << ", "
                         << HomographyPointsRight[i].x << ", "
                         << HomographyPointsRight[i].y << ", "
                         << HomographyPointsRight[i].z << ", "
                         << ReprojectionPointsRight[i].x << ", "
                         << ReprojectionPointsRight[i].y << ", "
                         << TrackedPointsRight[i].x << ", "
                         << TrackedPointsRight[i].y << ", "
                         << TrackedPointsRight[i].z << ", , ,"
                         << x << ", " << y << ", " << z << std::endl;
      }
    }
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateChessStartTimer( bool checked )
{
  if ( checked)
  {
    LeftCameraTimer->stop();
    RightCameraTimer->stop();
    ValidateStylusTimer->stop();
    ValidateChessTimer->start();
  }
  else
  {
    ValidateChessTimer->stop();
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateStylusStartTimer( bool checked )
{
  if ( checked )
  {
    LeftCameraTimer->stop();
    RightCameraTimer->stop();
    ValidateStylusTimer->start();
  }
  else
  {
    ValidateStylusTimer->stop();
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateStylus( )
{
  if ( this->DataCollector->IsStarted() )
  {
    double pos[4];
    PlusTrackedFrame frame;
    TrackingDataChannel->GetTrackedFrame(frame);
    if( TransformRepository->SetTransforms(frame) != PLUS_SUCCESS)
    {
      LOG_ERROR("Unable to load transforms into repository. Aborting.");
      ShowStatusMessage("Unable to load transforms into repository. Aborting.");
      return;
    }
    bool isValid(false);
    PlusTransformName stylusToReferenceName("Stylus", "Reference");
    vtkSmartPointer<vtkMatrix4x4> stylusToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(stylusToReferenceName, stylusToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate stylus to reference transform.");
      ShowStatusMessage("Unable to locate stylus to reference transform.");
      return;
    }

    PlusTransformName HMDToReferenceName("HMD", "Reference");
    vtkSmartPointer<vtkMatrix4x4> HMDToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(HMDToReferenceName, HMDToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      LOG_ERROR("Unable to locate HMD to reference transform.");
      ShowStatusMessage("Unable to locate HMD to reference transform.");
      return;
    }
    HMDToReferenceTransform->Invert();

    vtkSmartPointer<vtkTransform> tempTransform = vtkSmartPointer<vtkTransform>::New();
    tempTransform->Identity();
    tempTransform->Concatenate( stylusToReferenceTransform );
    tempTransform->Concatenate( HMDToReferenceTransform );
    tempTransform->Update();

    if ( LeftLandmarkAvailable && LeftIntrinsicAvailable && LeftDistortionAvailable )
    {
      vtkSmartPointer<vtkTransform> tempTransformLeft = vtkSmartPointer<vtkTransform>::New();
      tempTransformLeft->Identity();
      tempTransformLeft->Concatenate( tempTransform );
      tempTransformLeft->Concatenate( LeftLandmarkTransform->GetMatrix() );
      tempTransformLeft->Update();
      tempTransformLeft->GetPosition( pos );
      this->ValidateStylusVideo( LeftCameraIndex, "LeftCamera", pos );
    }

    if ( RightLandmarkAvailable && RightIntrinsicAvailable && RightDistortionAvailable )
    {
      vtkSmartPointer<vtkTransform> tempTransformRight = vtkSmartPointer<vtkTransform>::New();
      tempTransformRight->Identity();
      tempTransformRight->Concatenate( tempTransform );
      tempTransformRight->Concatenate( RightLandmarkTransform->GetMatrix() );
      tempTransformRight->Update();
      tempTransformRight->GetPosition( pos );
      this->ValidateStylusVideo( RightCameraIndex, "RightCamera", pos );
    }
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateStylusVideo( int cameraIndex, std::string videoTitle, double *pos )
{
  if ( cameraIndex >= this->CVInternals->CameraCount() )
  {
    return;
  }

  cv::Mat MP( 1, 3, CV_64FC1 ); // the 3D point in CvMat
  cv::Mat Mres( 1, 2, CV_64FC1 ); // the projected points in pixel
  cv::Mat MR( 3, 1, CV_64FC1 ); // the rotation, set to 0
  cv::Mat Mt( 3, 1, CV_64FC1 ); // the translation, set to 0

  cv::Point text_origin;

  // set the rotation/translation to none
  MP.at<double>(0,0) = 0;
  MP.at<double>(1,0) = 0;
  MP.at<double>(2,0) = 0;

  // projection
  cv::projectPoints(MP, MR, Mt, *CVInternals->GetInstrinsicMatrix(cameraIndex), *CVInternals->GetDistortionCoeffs(cameraIndex), Mres);

  // display a circle at there the tip of the stylus is
  // as tracked by the tracker
  text_origin.x = Mres.at<double>(0,0);
  text_origin.y = Mres.at<double>(0,0);
  CVInternals->QueryFrame(cameraIndex, CameraImages[cameraIndex]);

  cv::circle(CameraImages[cameraIndex], text_origin, 4, cv::Scalar(0, 255, 0), 2);
  cv::imshow(videoTitle, CameraImages[cameraIndex]);
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ResetCalibrationCheckerboards()
{
  CaptureCount[LeftCameraIndex] = 0;
  CaptureCount[RightCameraIndex] = 0;
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
  cv::Mat rightImage;
  CVInternals->QueryFrame(LeftCameraIndex, leftImage);
  CVInternals->QueryFrame(RightCameraIndex, rightImage);

  // make a copy
  cv::Mat leftCopy(leftImage);
  cv::Mat rightCopy(rightImage);

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
  p3 point;
  point.z = 0.0;
  for ( int i = 0, idx=(board_n-1); i < leftCorners.size(); i++, idx-- )
  {
    if ( !leftFlip )
    {
      point.x = leftCorners[i].x;
      point.y = leftCorners[i].y;
    }
    else
    {
      point.x = leftCorners[idx].x;
      point.y = leftCorners[idx].y;
    }
    ImagePointsLeft.push_back( point );
  }

  LOG_INFO("Collected: " << ImagePointsLeft.size() << " left points so far.");

  point.z = 0.0;
  for ( int i = 0, idx=(board_n-1); i < rightCorners.size(); i++, idx-- )
  {
    if ( !rightFlip )
    {
      point.x = rightCorners[i].x;
      point.y = rightCorners[i].y;
    }
    else
    {
      point.x = rightCorners[idx].x;
      point.y = rightCorners[idx].y;
    }
    ImagePointsRight.push_back( point );
  }
  LOG_INFO("Collected: " << ImagePointsRight.size() << " right points so far.");
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ComputeFundamentalMatrix()
{
  int nPoints = ImagePointsLeft.size();

  // make sure we have more than 8 points and same for both cameras
  if ( nPoints != ImagePointsRight.size() || nPoints < 8 )
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
    image_pointsL.at<float>(i,0) = ImagePointsLeft[i].x;
    image_pointsL.at<float>(i,1) = ImagePointsLeft[i].y;
    image_pointsR.at<float>(i,0) = ImagePointsLeft[i].x;
    image_pointsR.at<float>(i,1) = ImagePointsLeft[i].y;

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