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
#include <QByteArray>
#include <QCoreApplication>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QPalette>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
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

// PLUS includes
#include <vtkPlusChannel.h>
#include <vtkPlusDataCollector.h>
#include <vtkPlusDataSource.h>
#include <vtkPlusTransformRepository.h>
#include <PlusDeviceSetSelectorWidget.h>
#include <PlusToolStateDisplayWidget.h>

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

CameraCalibrationMainWidget::CameraCalibrationMainWidget()
  : TrackingDataChannel(NULL)
  , MinBoardNeeded(6)
  , TrackingDataChannelName("")
{
  // Set up UI
  ui.setupUi(this);

  internals = new OpenCVInternals();

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

  InitUI();

  // VTK related objects
  CreateVTKObjects();
  SetupVTKPipeline();

  for ( int i = 0; i < internals->numOfCamera(); i++ )
  {
    captureCount[i] = 0;
    if ( internals->isFeedAvailable( i ) )
    {
      std::cerr << "Camera " << i << " is available." << std::endl;
    }
    else
    {
      std::cerr << "Camera " << i << " is NOT available." << std::endl;
    }
  }

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
  DataCollector->Stop();

  delete internals;

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

  // set up all the signal/slots here
  connect( ui.pushButton_LeftVideo, SIGNAL( toggled( bool ) ),
    this, SLOT( StartLeftVideo( bool ) ) );
  connect( LeftCameraTimer, SIGNAL( timeout() ),
    this, SLOT( UpdateLeftVideo() ) );
  connect( ui.pushButton_RightVideo, SIGNAL( toggled( bool ) ),
    this, SLOT( StartRightVideo( bool ) ) );
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
    this, SLOT( ResetCalibrationCheckerboards() ) );
  connect( ui.spinBox_BoardHeightCalib, SIGNAL( valueChanged( int ) ),
    this, SLOT( ResetCalibrationCheckerboards() ) );
  connect( ui.doubleSpinBox_QuadSizeCalib, SIGNAL( valueChanged( double ) ),
    this, SLOT( ResetCalibrationCheckerboards() ) );

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

  if ( DataCollector->Start() == PLUS_SUCCESS )
  {
    std::cerr << "PLUS data collection initialized." << std::endl;
  }
  else
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

//---------------------------------------------------------
// show OpenCV videos
void CameraCalibrationMainWidget::ShowOpenCVVideo( int cameraIndex, const std::string& videoTitle )
{
  // make sure we have the correct video feed
  if ( cameraIndex >= internals->numOfCamera() )
  {
    return;
  }

  internals->cameraFeeds[cameraIndex]->grab();
  *internals->cameraFeeds[cameraIndex] >> cameraImages[cameraIndex]; 
  cv::imshow( videoTitle, cameraImages[cameraIndex] );
  cv::waitKey(25);
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

//---------------------------------------------------------
// start the video feeds
void CameraCalibrationMainWidget::StartLeftVideo( bool checked )
{
  if ( internals->isFeedAvailable( 0 ) )
  {
    if ( checked )
    {
      LeftCameraTimer->start( 0 );
    }
    else
    {
      LeftCameraTimer->stop();
    }
  }
  else
  {
    std::cerr << "Left video not available" << std::endl;
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::StartRightVideo( bool checked )
{
  if ( internals->isFeedAvailable( 1 ) )
  {
    if ( checked )
    {
      RightCameraTimer->start( 0 );
    }
    else
    {
      RightCameraTimer->stop();
    }
  }
  else
  {
    std::cerr << "Right video not available" << std::endl;
  }
}

//---------------------------------------------------------
// start opencv video
void CameraCalibrationMainWidget::UpdateLeftVideo()
{
  ShowOpenCVVideo( 0, "LeftCamera" );
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::UpdateRightVideo()
{
  ShowOpenCVVideo( 1, "RightCamera" );
}

//---------------------------------------------------------
// file operation
void CameraCalibrationMainWidget::LoadLeftIntrinsic()
{
  QString leftIntrincFileName = QFileDialog::getOpenFileName( this,
    tr( "Open Left Intrinsic" ),
    QDir::currentPath(),
    "OpenCV XML (*.xml *.XML)" );

  if ( leftIntrincFileName.size() == 0 )
  {
    return;
  }

  QByteArray bytearray = leftIntrincFileName.toUtf8();

  CvMat *cameraMatrix = (CvMat*)cvLoad( bytearray.constData() );


  if( internals->intrinsic_matrix[0] )
  {
    cvReleaseMat( &(internals->intrinsic_matrix[0]) );
  }

  internals->intrinsic_matrix[0] = cvCreateMat( 3, 3, CV_32FC1 );


  cvCopy( cameraMatrix, internals->intrinsic_matrix[0] );

  LeftIntrinsicAvailable = true;

  std::cerr << "Left Intrinsic: " << std::endl;

  for ( int i = 0; i < internals->intrinsic_matrix[0]->rows; i++ )
  {
    for ( int j = 0; j < internals->intrinsic_matrix[0]->cols; j++ )
    {
      std::cerr << cvmGet( internals->intrinsic_matrix[0], i, j ) << "\t";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl << std::endl;
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

  QByteArray bytearray = FileName.toUtf8();

  CvMat *cameraMatrix = (CvMat*)cvLoad( bytearray.constData() );


  if( internals->intrinsic_matrix[1] )
  {
    cvReleaseMat( &(internals->intrinsic_matrix[1]) );
  }

  internals->intrinsic_matrix[1] = cvCreateMat( 3, 3, CV_32FC1 );


  cvCopy( cameraMatrix, internals->intrinsic_matrix[1] );

  RightIntrinsicAvailable = true;

  std::cerr << "Right Intrinsic: " << std::endl;
  for ( int i = 0; i < internals->intrinsic_matrix[1]->rows; i++ )
  {
    for ( int j = 0; j < internals->intrinsic_matrix[1]->cols; j++ )
    {
      std::cerr << cvmGet( internals->intrinsic_matrix[1], i, j ) << "\t";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl << std::endl;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadLeftDistortion()
{
  QString filename = QFileDialog::getOpenFileName( this,
    tr( "Open left Distortion" ),
    QDir::currentPath(),
    "OpenCV XML (*.xml *.XML)" );

  if ( filename.size() == 0 )
  {
    return;
  }

  QByteArray bytearray = filename.toUtf8();
  CvMat *distortion = (CvMat*) cvLoad( bytearray.constData() );

  if ( internals->distortion_coeffs[0] )
  {
    cvReleaseMat( &(internals->distortion_coeffs[0] ) );
  }

  internals->distortion_coeffs[0] = cvCreateMat( 4, 1, CV_32FC1 );

  cvCopy( distortion, internals->distortion_coeffs[0] );

  LeftDistortionAvailable = true;

  std::cerr << "Left Distortion: " << std::endl;

  for ( int i = 0; i < internals->distortion_coeffs[0]->rows; i++ )
  {
    for ( int j = 0; j < internals->distortion_coeffs[0]->cols; j++ )
    {
      std::cerr << cvmGet( internals->distortion_coeffs[0], i, j ) << "\t";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl << std::endl;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadRightDistortion()
{
  QString filename = QFileDialog::getOpenFileName( this,
    tr( "Open right Distortion" ),
    QDir::currentPath(),
    "OpenCV XML (*.xml *.XML)" );

  if ( filename.size() == 0 )
  {
    return;
  }

  QByteArray bytearray = filename.toUtf8();
  CvMat *distortion = (CvMat*) cvLoad( bytearray.constData() );

  if ( internals->distortion_coeffs[1] )
  {
    cvReleaseMat( &(internals->distortion_coeffs[1] ) );
  }

  internals->distortion_coeffs[1] = cvCreateMat( 4, 1, CV_32FC1 );

  cvCopy( distortion, internals->distortion_coeffs[1] );

  RightDistortionAvailable = true;

  std::cerr << "Right Distortion: " << std::endl;

  for ( int i = 0; i < internals->distortion_coeffs[1]->rows; i++ )
  {
    for ( int j = 0; j < internals->distortion_coeffs[1]->cols; j++ )
    {
      std::cerr << cvmGet( internals->distortion_coeffs[1], i, j ) << "\t";
    }
    std::cerr << std::endl;
  }
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

  QByteArray bytearray = FileName.toUtf8();

  CvMat *transform = (CvMat*)cvLoad( bytearray.constData() );

  LeftLandmarkTransform->GetMatrix()->DeepCopy( transform->data.db );

  LeftLandmarkAvailable = true;

  std::cerr << "Left Landmark: " << std::endl;
  LeftLandmarkTransform->Print( std::cerr );

  std::cerr << std::endl << std::endl;
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

  QByteArray bytearray = FileName.toUtf8();

  CvMat *transform = (CvMat*)cvLoad( bytearray.constData() );

  RightLandmarkTransform->GetMatrix()->DeepCopy( transform->data.db );
  RightLandmarkAvailable = true;

  std::cerr << "Right Landmark: " << std::endl;
  RightLandmarkTransform->Print( std::cerr );

  std::cerr << std::endl << std::endl;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadValidChessRegistration()
{
  QString filename = QFileDialog::getOpenFileName( this,
    tr( "Open Validation Checkerboard Registration" ),
    QDir::currentPath(),
    "OpenCV XML (*.xml *.XML)" );

  if ( filename.size() == 0 )
  {
    return;
  }

  QByteArray bytearray = filename.toUtf8();
  CvMat *matrix = (CvMat*) cvLoad( bytearray.constData() );

  ValidBoardRegTransform->GetMatrix()->DeepCopy( matrix->data.db );

  ValidBoardAvailable = true;

  std::cerr << "Validation Chessboard Registration: " << std::endl;
  ValidBoardRegTransform->Print( std::cerr );
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::LoadChessRegistration()
{
  QString filename = QFileDialog::getOpenFileName( this,
    tr( "Open Calibration Checkerboard Registration" ),
    QDir::currentPath(),
    "OpenCV XML (*.xml *.XML)" );

  if ( filename.size() == 0 )
  {
    return;
  }

  QByteArray bytearray = filename.toUtf8();
  CvMat *matrix = (CvMat*) cvLoad( bytearray.constData() );

  BoardRegTransform->GetMatrix()->DeepCopy( matrix->data.db );

  BoardRegAvailable = true;

  std::cerr << "Calibration Chessboard Registration: " << std::endl;
  BoardRegTransform->Print( std::cerr );
}

//---------------------------------------------------------
// process a single checkerboard
int CameraCalibrationMainWidget::ProcessCheckerBoard( int cameraIndex, int width, int height, double size, processingMode mode,
                                                     vtkPoints *source, vtkPoints *target, std::string videoTitle )
{
  // double check if the video feed is available
  if ( cameraIndex >= internals->numOfCamera() && !internals->isFeedAvailable( cameraIndex ) )
  {
    return (-1);
  }

  std::cerr << "Checkerboard dimension: "
    << width << "x" << height << "x" << size << std::endl;

  internals->cameraFeeds[cameraIndex]->grab();
  *internals->cameraFeeds[cameraIndex] >> cameraImages[cameraIndex];

  int depth = cameraImages[cameraIndex].depth();
  int nchannels = cameraImages[cameraIndex].channels();

  // make a copy of the current feed
  // need to do so as the current feed may be self-updating
  cv::Mat copy(cameraImages[cameraIndex]);

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
    return captureCount[cameraIndex];
  }

  cv::Mat viewGray;
  cv::cvtColor(copy, viewGray, cv::COLOR_BGR2GRAY);
  cv::cornerSubPix( viewGray, corners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1 ));

  // sometimes OpenCV returned a flipped image
  bool flip = false;

  if ( corners[board_n-1].x < corners[0].x )
  {
    flip = true;
    std::cerr << "Image is flipped." << std::endl;
  }

  if ( mode == forCalibration )
  {
    image_points[cameraIndex].push_back(corners);
    captureCount[cameraIndex]++;
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
      std::cerr << "Unable to load transforms into repository. Aborting." << std::endl;
      return captureCount[cameraIndex];
    }
    bool isValid(false);
    PlusTransformName checkboardToReferenceName("Checkerboard", "Reference");
    vtkSmartPointer<vtkMatrix4x4> checkboardToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(checkboardToReferenceName, checkboardToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate checkerboard to reference transform. See error log." << std::endl;
      return captureCount[cameraIndex];
    }

    PlusTransformName HMDToReferenceName("HMD", "Reference");
    vtkSmartPointer<vtkMatrix4x4> HMDToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(HMDToReferenceName, HMDToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate HMD to reference transform. See error log." << std::endl;
      return captureCount[cameraIndex];
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
    cv::Mat rvec(3, 1, CV_64FC1 );
    cv::Mat tvec(3, 1, CV_64FC1 );
    cv::Mat rmat(3, 3, CV_64FC1 );
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
    cv::calibrateCamera(objectPoints, imagePoints, board_sz, internals->intrinsic_matrix[cameraIndex], internals->distortion_coeffs[cameraIndex], rvec, tvec);
    cv::Rodrigues(rvec, rmat);

    // double check the transforms
    for ( int i = 0; i < rmat.rows; i++ )
    {
      for ( int j = 0; j < rmat.cols; j++ )
      {
        std::cerr << rmat.at<double>(i,j) << "\t";
      }
      std::cerr << tvec.at<double>(i, 0) << std::endl;
    }

    tempTransform->Print( std::cerr );

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
      cv::gemm(rmat, pointofinterest, 1., tvec, 1., fproject);

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
        cv::projectPoints(MP, MR, Mt, internals->intrinsic_matrix[cameraIndex], internals->distortion_coeffs[cameraIndex], Mres);

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

  return captureCount[cameraIndex];
}

//---------------------------------------------------------
// process the recorded image
void CameraCalibrationMainWidget::CaptureAndProcessLeftImage()
{
  int beforeAttempt = captureCount[0];
  int n = ProcessCheckerBoard( 0,
    GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
    forCalibration, 0, 0, std::string( "Results" ) );

  if( captureCount[0] - beforeAttempt == 0 )
  {
    std::cerr << "Unable to locate checkerboard in camera image. Try again." << std::endl;
  }
  else
  {
    std::cerr << "Success! Captured " << n << " left calibration image thus far." << std::endl;
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::CaptureAndProcessRightImage()
{
  int beforeAttempt = captureCount[1];
  int n = ProcessCheckerBoard( 1,
    GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
    forCalibration, 0, 0, std::string( "Results" ) );

  if( captureCount[1] - beforeAttempt == 0 )
  {
    std::cerr << "Unable to locate checkerboard in camera image. Try again." << std::endl;
  }
  else
  {
    std::cerr << "Success! Captured " << n << " right calibration image thus far." << std::endl;
  }
}

//---------------------------------------------------------
// compute the intrinsics
void CameraCalibrationMainWidget::ComputeLeftIntrinsic()
{
  this->ComputeIntrinsicsAndDistortion( 0 );

  // TODO : file dialogs
  this->WriteIntrinsicsToFile( 0, "Left_Intrinsics.xml" );
  this->WriteDistortionToFile( 0, "Left_Distortion.xml" );

  std::cerr << "Left intrinsics/distortions computed and saved to: "
    << "Left_Intrinsics.xml" << std::endl;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ComputeRightIntrinsic()
{
  this->ComputeIntrinsicsAndDistortion( 1 );

  // TODO : file dialogs
  this->WriteIntrinsicsToFile( 1, "Right_Intrinsics.xml" );
  this->WriteDistortionToFile( 1, "Right_Distortion.xml" );

  std::cerr << "Right intrinsics/distortions computed and saved to: "
    << "Right_Intrinsics.xml" << std::endl;
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::WriteIntrinsicsToFile( int cameraIndex, const std::string& filename )
{
  if ( cameraIndex >= this->internals->numOfCamera() )
  {
    return;
  }

  cvSave( filename.c_str(), this->internals->intrinsic_matrix[cameraIndex] );
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::WriteDistortionToFile( int cameraIndex, const std::string& filename )
{
  if ( cameraIndex >= this->internals->numOfCamera() )
  {
    return;
  }

  cvSave( filename.c_str(), this->internals->distortion_coeffs[cameraIndex] );
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ComputeIntrinsicsAndDistortion( int cameraIndex )
{
  if ( cameraIndex >= this->internals->numOfCamera() )
  {
    return;
  }

  if ( captureCount[cameraIndex] < MinBoardNeeded )
  {
    std::cerr << "Not enough board recorded: "
      << captureCount[cameraIndex] << "/"
      << MinBoardNeeded << std::endl;
    return;
  }

  cv::calibrateCamera(object_points, image_points, cv::Size(cameraImages[cameraIndex].size[0], cameraImages[cameraIndex].size[1]), 
    internals->intrinsic_matrix[cameraIndex], internals->distortion_coeffs[cameraIndex]);

  if ( cameraIndex == 0 )
  {
    LeftIntrinsicAvailable = LeftDistortionAvailable = true;
  }
  else if ( cameraIndex == 1 )
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
      std::cerr << "Unable to load transforms into repository. Aborting." << std::endl;
      return;
    }
    bool isValid(false);
    PlusTransformName checkboardToReferenceName("Checkerboard", "Reference");
    vtkSmartPointer<vtkMatrix4x4> checkboardToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(checkboardToReferenceName, checkboardToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate checkerboard to reference transform. See error log." << std::endl;
      return;
    }
    checkboardToReferenceTransform->Invert();

    PlusTransformName stylusToReferenceName("Stylus", "Reference");
    vtkSmartPointer<vtkMatrix4x4> stylusToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(stylusToReferenceName, stylusToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate stylus to reference transform. See error log." << std::endl;
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

    std::cerr << "Board size is: " << nw << "x" << nh << "x" << size << "." << std::endl;

    // the board registration is from the checkerboard's 2D coordinate
    ValidBoardSource->InsertNextPoint( (double)nw*size,
      (double)nh*size,
      0.0 );

    ValidBoardTarget->InsertNextPoint( pos[0], pos[1], pos[2] );
    std::cerr << "Point: " << pos[0] << " " << pos[1] << " " << pos[2] << "." << std::endl;
  }
  else
  {
    std::cerr << "Either the tracker is not initialized or it is not tracking." << std::endl;
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
      std::cerr << "Please collect more than 4 points from the checkerboard." << std::endl;
    }
    else
    {
      ValidBoardRegTransform->SetSourceLandmarks( ValidBoardSource );
      ValidBoardRegTransform->SetTargetLandmarks( ValidBoardTarget );
      ValidBoardRegTransform->SetModeToRigidBody();
      ValidBoardRegTransform->Update();

      std::cerr << "Checkerboard registration: " << std::endl;
      ValidBoardRegTransform->Print( std::cerr );

      std::cerr << "Saving the registration to validCheckerBoardReg.xml" << std::endl;
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
      std::cerr << "Unable to load transforms into repository. Aborting." << std::endl;
      return;
    }
    bool isValid(false);
    PlusTransformName checkboardToReferenceName("Checkerboard", "Reference");
    vtkSmartPointer<vtkMatrix4x4> checkboardToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(checkboardToReferenceName, checkboardToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate checkerboard to reference transform. See error log." << std::endl;
      return;
    }
    checkboardToReferenceTransform->Invert();

    PlusTransformName stylusToReferenceName("Stylus", "Reference");
    vtkSmartPointer<vtkMatrix4x4> stylusToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(stylusToReferenceName, stylusToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate stylus to reference transform. See error log." << std::endl;
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

    std::cerr << "Board size is: " << nw << "x" << nh << "x" << size << "." << std::endl;

    // the board registration is from the checkerboard's 2D coordinate
    // to the reference tool's 3D coordinate
    BoardSource->InsertNextPoint( (double)nw*size, (double)nh*size, 0.0 );
    BoardTarget->InsertNextPoint( pos[0], pos[1], pos[2] );

    std::cerr << "Point: " << pos[0] << " " << pos[1] << " " << pos[2] << "." << std::endl;
  }
  else
  {
    std::cerr << "Either the tracker is not initialized or it is not tracking." << std::endl;
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
      std::cerr << "Please collect more than 4 points from the checkerboard"
        << std::endl;
    }
    else
    {
      BoardRegTransform->SetSourceLandmarks( BoardSource );
      BoardRegTransform->SetTargetLandmarks( BoardTarget );
      BoardRegTransform->SetModeToRigidBody();
      BoardRegTransform->Update();

      std::cerr << "Checkerboard registration: " << std::endl;
      BoardRegTransform->Print( std::cerr );

      std::cerr << "Saving the registration to CheckerBoardReg.xml" << std::endl;
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
  int beforeAttempt = captureCount[0];
  int k = ProcessCheckerBoard( 0,
    GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
    forLandmark,
    BoardCornerLeftSource, BoardCornerLeftTarget, std::string( "Results" ) );

  if( captureCount[0] - beforeAttempt == 0 )
  {
    std::cerr << "Unable to locate checkerboard in camera image. Try again." << std::endl;
  }
  else
  {
    std::cerr << "Success! Captured " << k << " left calibration points thus far." << std::endl;
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::DrawRightChessBoardCorners()
{
  int beforeAttempt = captureCount[1];
  int k = ProcessCheckerBoard( 1,
    GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
    forLandmark,
    BoardCornerRightSource, BoardCornerRightTarget, std::string( "Results" ) );

  if( captureCount[1] - beforeAttempt == 0 )
  {
    std::cerr << "Unable to locate checkerboard in camera image. Try again." << std::endl;
  }
  else
  {
    std::cerr << "Success! Captured " << k << " right calibration points thus far." << std::endl;
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ComputeHMDRegistration()
{
  if ( BoardCornerLeftSource->GetNumberOfPoints() != BoardCornerLeftTarget->GetNumberOfPoints() )
  {
    std::cerr << "Number of points do not match in HMD registration." << std::endl;
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
      LeftLandmarkTransform->Print( std::cerr );
    }
  }

  if ( BoardCornerRightSource->GetNumberOfPoints() != BoardCornerRightTarget->GetNumberOfPoints() )
  {
    std::cerr << "number of points do not match in HMD registration." << std::endl;
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
      RightLandmarkTransform->Print( std::cerr );
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
  if ( this->internals->numOfCamera() < 2 )
  {
    return;
  }

  // get both images
  IplImage *leftImage = cvQueryFrame( internals->cameraFeeds[0] );
  IplImage *rightImage = cvQueryFrame( internals->cameraFeeds[1] );

  // size of the images
  CvSize img_sz = cvGetSize( leftImage );

  // detect a red ball
  CvScalar hsv_min = cvScalar( 150, 50, 100, 0 );
  CvScalar hsv_max = cvScalar( 180, 255, 255, 0 );

  // create a threshold image
  IplImage * leftThresholded = cvCreateImage( img_sz, IPL_DEPTH_8U, 1 );
  IplImage * rightThresholded = cvCreateImage( img_sz, IPL_DEPTH_8U, 1 );
  IplImage * leftHSV = cvCreateImage( img_sz, IPL_DEPTH_8U, 3 );
  IplImage * rightHSV = cvCreateImage( img_sz, IPL_DEPTH_8U, 3 );

  // consult page 161 of the OpenCV book

  // convert color space to HSV as so we can segment the image
  // based on the color
  cvCvtColor( leftImage, leftHSV, CV_BGR2HSV );
  cvCvtColor( rightImage, rightHSV, CV_BGR2HSV );

  // threshold the HSV image
  cvInRangeS( leftHSV, hsv_min, hsv_max, leftThresholded );
  cvInRangeS( rightHSV, hsv_min, hsv_max, rightThresholded );

  // apply a gaussian filter to smooth the binary image
  cvSmooth( leftThresholded, leftThresholded, CV_GAUSSIAN, 5, 5 );
  cvSmooth( rightThresholded, rightThresholded, CV_GAUSSIAN, 5, 5 );

  CvMemStorage *leftStorage = cvCreateMemStorage( 0 );
  CvMemStorage *rightStorage = cvCreateMemStorage( 0 );

  // use Hough detector to find the sphere/circle
  CvSeq *leftCircles = cvHoughCircles(
    leftThresholded,
    leftStorage,
    CV_HOUGH_GRADIENT,
    2,
    leftThresholded->height/4 );

  CvSeq *rightCircles = cvHoughCircles(
    rightThresholded,
    rightStorage,
    CV_HOUGH_GRADIENT,
    2,
    rightThresholded->height/4 );

  for ( int i = 0; i < leftCircles->total; i++ )
  {
    float *p = (float*)cvGetSeqElem( leftCircles, i );
    cvCircle(
      leftImage,
      cvPoint( cvRound( p[0] ), cvRound( p[1] ) ),
      3,
      CV_RGB( 0, 255, 0 ),
      -1,
      8,
      0 );
    cvCircle(
      leftImage,
      cvPoint( cvRound( p[0] ), cvRound( p[1] ) ),
      cvRound( p[2] ),
      CV_RGB( 255, 0, 0 ),
      3,
      8,
      0 );
  }

  for ( int i = 0; i < rightCircles->total; i++ )
  {
    float *p = (float*)cvGetSeqElem( rightCircles, i );
    cvCircle(
      rightImage,
      cvPoint( cvRound( p[0] ), cvRound( p[1] ) ),
      3,
      CV_RGB( 0, 255, 0 ),
      -1,
      8,
      0 );
    cvCircle(
      rightImage,
      cvPoint( cvRound( p[0] ), cvRound( p[1] ) ),
      cvRound( p[2] ),
      CV_RGB( 255, 0, 0 ),
      3,
      8,
      0 );
  }
  cvShowImage( "Left Result", leftImage );
  cvShowImage( "Right Result", rightImage );

  // clean up
  cvReleaseMemStorage( &rightStorage );
  cvReleaseMemStorage( &leftStorage );
  cvReleaseImage( &rightHSV );
  cvReleaseImage( &leftHSV );
  cvReleaseImage( &rightThresholded );
  cvReleaseImage( &leftThresholded );
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateValidChess()
{
  if ( this->DataCollector->IsStarted() )
  {
    if ( LeftLandmarkAvailable && LeftIntrinsicAvailable && LeftDistortionAvailable && ValidBoardAvailable )
    {
      ProcessCheckerBoard( 0,
        ui.spinBox_BoardWidthValid->value(), ui.spinBox_BoardHeightValid->value(), ui.doubleSpinBox_QuadSizeValid->value(),
        forEvaluation, 0, 0, std::string( "Left Result" ) );
    }

    if ( RightLandmarkAvailable && RightIntrinsicAvailable && RightDistortionAvailable && ValidBoardAvailable )
    {
      ProcessCheckerBoard( 1,
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
    cameraImages[0] = cvQueryFrame( this->internals->cameraFeeds[0] );
    cameraImages[1] = cvQueryFrame( this->internals->cameraFeeds[1] );

    PlusTrackedFrame frame;
    TrackingDataChannel->GetTrackedFrame(frame);
    if( TransformRepository->SetTransforms(frame) != PLUS_SUCCESS)
    {
      std::cerr << "Unable to load transforms into repository. Aborting." << std::endl;
      return;
    }
    bool isValid(false);
    PlusTransformName checkboardToReferenceName("Checkerboard", "Reference");
    vtkSmartPointer<vtkMatrix4x4> checkboardToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(checkboardToReferenceName, checkboardToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate checkerboard to reference transform. See error log." << std::endl;
      return;
    }

    PlusTransformName HMDToReferenceName("HMD", "Reference");
    vtkSmartPointer<vtkMatrix4x4> HMDToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(HMDToReferenceName, HMDToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate HMD to reference transform. See error log." << std::endl;
      return;
    }
    HMDToReferenceTransform->Invert();



    vtkSmartPointer<vtkTransform> tempTransform = vtkSmartPointer<vtkTransform>::New();
    tempTransform->PostMultiply();

    double points[4];
    int nPoints = GetBoardWidthCalib() * GetBoardHeightCalib();
    CvMat *MP = cvCreateMat( 1, 3, CV_64FC1 ); // the 3D point in CvMat
    CvMat *MR = cvCreateMat( 3, 1, CV_64FC1 ); // the rotation, set to 0
    CvMat *Mt = cvCreateMat( 3, 1, CV_64FC1 );
    CvMat *Mres = cvCreateMat( 1, 2, CV_64FC1 ); // the projected points in pixel

    CvPoint text_origin;

    // set the rotation/translation to none
    cvmSet( MR, 0, 0, 0 );
    cvmSet( MR, 1, 0, 0 );
    cvmSet( MR, 2, 0, 0 );
    cvmSet( Mt, 0, 0, 0 );
    cvmSet( Mt, 1, 0, 0 );
    cvmSet( Mt, 2, 0, 0 );

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

          cvmSet( MP, 0, 0, points[0] );
          cvmSet( MP, 0, 1, points[1] );
          cvmSet( MP, 0, 2, points[2] );

          // projection
          cvProjectPoints2( MP, MR, Mt,
            this->internals->intrinsic_matrix[0], this->internals->distortion_coeffs[0],
            Mres );

          // Mres is the projected 2D pixel of Phm
          text_origin.x = cvmGet( Mres, 0, 0 );
          text_origin.y = cvmGet( Mres, 0, 1 );
          cvCircle( cameraImages[0], text_origin, 3, cvScalar( 255, 0, 0 ), 2 );
        }
      }

      cvShowImage( "Left Result", cameraImages[0] );
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

          cvmSet( MP, 0, 0, points[0] );
          cvmSet( MP, 0, 1, points[1] );
          cvmSet( MP, 0, 2, points[2] );

          // projection
          cvProjectPoints2( MP, MR, Mt,
            this->internals->intrinsic_matrix[1], this->internals->distortion_coeffs[1],
            Mres );

          // Mres is the projected 2D pixel of Phm
          text_origin.x = cvmGet( Mres, 0, 0 );
          text_origin.y = cvmGet( Mres, 0, 1 );
          cvCircle( cameraImages[0], text_origin, 3, cvScalar( 255, 0, 0 ), 2 );
        }
      }
      cvShowImage( "Right Result", cameraImages[1] );
    }

    cvReleaseMat( &MR );
    cvReleaseMat( &Mt );
    cvReleaseMat( &Mres );
    cvReleaseMat( &MP );
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateChess()
{
  if ( this->DataCollector->IsStarted() )
  {
    if ( LeftLandmarkAvailable && LeftIntrinsicAvailable && LeftDistortionAvailable && BoardRegAvailable )
    {
      ProcessCheckerBoard( 0,
        GetBoardWidthCalib(), GetBoardHeightCalib(), GetBoardQuadSizeCalib(),
        forEvaluation, 0, 0, std::string( "Left Result" ) );
    }

    if ( RightLandmarkAvailable && RightIntrinsicAvailable && RightDistortionAvailable && BoardRegAvailable )
    {
      ProcessCheckerBoard( 1,
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
      CvMat LeftExtrinsic = cvMat( 3, 4, CV_64F, leftExtrinsic );

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
      CvMat RightExtrinsic = cvMat( 3, 4, CV_64F, rightExtrinsic );

      CvMat *leftImagePoints = cvCreateMat( 2, nPoints, CV_64FC1 );
      CvMat *rightImagePoints = cvCreateMat( 2, nPoints, CV_64FC1 );
      CvMat *points4D = cvCreateMat( 4, nPoints, CV_64F );

      // copy the points over
      for ( int i = 0; i < nPoints; i++ )
      {
        CV_MAT_ELEM( *leftImagePoints, double, 0, i ) = ImagePointsLeft[i].x;
        CV_MAT_ELEM( *leftImagePoints, double, 1, i ) = ImagePointsLeft[i].y;
        CV_MAT_ELEM( *rightImagePoints, double, 0, i ) = ImagePointsRight[i].x;
        CV_MAT_ELEM( *rightImagePoints, double, 1, i ) = ImagePointsRight[i].y;
      }

      cvTriangulatePoints( &LeftExtrinsic, &RightExtrinsic,
        leftImagePoints, rightImagePoints,
        points4D );

      double x, y, z;
      for ( int i = 0; i < nPoints; i++ )
      {
        // output to file
        x = cvmGet( points4D, 0, i ) / cvmGet( points4D, 3, i );
        y = cvmGet( points4D, 1, i ) / cvmGet( points4D, 3, i );
        z = cvmGet( points4D, 2, i ) / cvmGet( points4D, 3, i );
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

      // clean up
      cvReleaseMat( &points4D );
      cvReleaseMat( &rightImagePoints );
      cvReleaseMat( &leftImagePoints );
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
    std::cerr << "Starting validating with the stylus" << std::endl;
    LeftCameraTimer->stop();
    RightCameraTimer->stop();
    ValidateStylusTimer->start();
  }
  else
  {
    std::cerr << "stop validating" << std::endl;
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
      std::cerr << "Unable to load transforms into repository. Aborting." << std::endl;
      return;
    }
    bool isValid(false);
    PlusTransformName stylusToReferenceName("Stylus", "Reference");
    vtkSmartPointer<vtkMatrix4x4> stylusToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(stylusToReferenceName, stylusToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate stylus to reference transform. See error log." << std::endl;
      return;
    }

    PlusTransformName HMDToReferenceName("HMD", "Reference");
    vtkSmartPointer<vtkMatrix4x4> HMDToReferenceTransform = vtkSmartPointer<vtkMatrix4x4>::New();
    if( TransformRepository->GetTransform(HMDToReferenceName, HMDToReferenceTransform, &isValid) != PLUS_SUCCESS || !isValid )
    {
      std::cerr << "Unable to locate HMD to reference transform. See error log." << std::endl;
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
      this->ValidateStylusVideo( 0, "LeftCamera", pos );
    }

    if ( RightLandmarkAvailable && RightIntrinsicAvailable && RightDistortionAvailable )
    {
      vtkSmartPointer<vtkTransform> tempTransformRight = vtkSmartPointer<vtkTransform>::New();
      tempTransformRight->Identity();
      tempTransformRight->Concatenate( tempTransform );
      tempTransformRight->Concatenate( RightLandmarkTransform->GetMatrix() );
      tempTransformRight->Update();
      tempTransformRight->GetPosition( pos );
      this->ValidateStylusVideo( 1, "RightCamera", pos );
    }
  }
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ValidateStylusVideo( int cameraIndex, std::string videoTitle, double *pos )
{
  if ( cameraIndex >= this->internals->numOfCamera() )
  {
    return;
  }

  CvMat *MP = cvCreateMat( 1, 3, CV_64FC1 ); // the 3D point in CvMat
  CvMat *Mres = cvCreateMat( 1, 2, CV_64FC1 ); // the projected points in pixel
  cvmSet( MP, 0, 0, pos[0] );
  cvmSet( MP, 0, 1, pos[1] );
  cvmSet( MP, 0, 2, pos[2] );

  CvMat *MR = cvCreateMat( 3, 1, CV_64FC1 ); // the rotation, set to 0
  CvMat *Mt = cvCreateMat( 3, 1, CV_64FC1 ); // the translation, set to 0

  // set the rotation/translation to none
  cvmSet( MR, 0, 0, 0 );
  cvmSet( MR, 1, 0, 0 );
  cvmSet( MR, 2, 0, 0 );
  cvmSet( Mt, 0, 0, 0 );
  cvmSet( Mt, 1, 0, 0 );
  cvmSet( Mt, 2, 0, 0 );

  // projection
  cvProjectPoints2( MP, MR, Mt,
    this->internals->intrinsic_matrix[cameraIndex], this->internals->distortion_coeffs[cameraIndex],
    Mres );

  // display a circle at there the tip of the stylus is
  // as tracked by the tracker
  CvPoint text_origin;
  text_origin.x = cvmGet( Mres, 0, 0 );
  text_origin.y = cvmGet( Mres, 0, 1 );

  cameraImages[cameraIndex] = cvQueryFrame( this->internals->cameraFeeds[cameraIndex] );

  cvCircle( cameraImages[cameraIndex], text_origin, 4, cvScalar( 0, 255, 0 ), 2 );
  cvShowImage( videoTitle.c_str(), cameraImages[cameraIndex] );

  std::cerr << text_origin.x << " " << text_origin.y << " " << pos[0] << " "
    << pos[1] << " " << pos[2] << std::endl;

  // clean up
  cvReleaseMat( &Mt );
  cvReleaseMat( &MR );
  cvReleaseMat( &Mres );
  cvReleaseMat( &MP );
}

//---------------------------------------------------------
void CameraCalibrationMainWidget::ResetCalibrationCheckerboards()
{
  for ( int i = 0; i < this->internals->numOfCamera(); i++ )
  {
    ResetCalibrationCheckerboards( i );
  }
}

//---------------------------------------------------------
// reset OpenCV variables if the checkerboard geometry has been changed
int CameraCalibrationMainWidget::ResetCalibrationCheckerboards( int cameraIndex )
{
  if ( cameraIndex >= this->internals->numOfCamera() )
  {
    return ( -1 );
  }

  captureCount[cameraIndex] = 0;
  if ( image_points[cameraIndex] )
  {
    cvReleaseMat( &(image_points[cameraIndex]) );
    image_points[cameraIndex] = 0;
  }

  if ( object_points[cameraIndex] )
  {
    cvReleaseMat( &(object_points[cameraIndex]) );
    object_points[cameraIndex] = 0;
  }

  if ( point_counts[cameraIndex] )
  {
    cvReleaseMat( &(point_counts[cameraIndex]) );
    point_counts[cameraIndex] = 0;
  }

  return 0;
}

//---------------------------------------------------------
// acquire images from both camera, find the corners, and store them
void CameraCalibrationMainWidget::StereoAcquire()
{
  // only do something when we have stereo
  if ( this->internals->numOfCamera() < 2 )
  {
    return;
  }

  // get both images
  IplImage *leftImage = cvQueryFrame( internals->cameraFeeds[0] );
  IplImage *rightImage = cvQueryFrame( internals->cameraFeeds[1] );

  // make a copy
  IplImage *leftCopy = cvCreateImage( cvGetSize( leftImage ), leftImage->depth, leftImage->nChannels );
  cvCopy( leftImage, leftCopy );

  IplImage *rightCopy = cvCreateImage( cvGetSize( rightImage ), rightImage->depth, rightImage->nChannels );
  cvCopy( rightImage, rightCopy );

  // BW image
  IplImage *leftGray = cvCreateImage( cvGetSize( leftImage ), 8, 1 );
  IplImage *rightGray = cvCreateImage( cvGetSize( rightImage ), 8, 1 );

  CvSize board_sz = cvSize( GetBoardWidthCalib(), GetBoardHeightCalib() );
  int board_n = GetBoardWidthCalib() * GetBoardHeightCalib();
  CvPoint2D32f *leftCorners = new CvPoint2D32f[ board_n ];
  CvPoint2D32f *rightCorners = new CvPoint2D32f[ board_n ];

  int leftCorner_count, rightCorner_count;
  int leftFound = cvFindChessboardCorners( leftCopy, board_sz, leftCorners, &leftCorner_count,
    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );
  int rightFound = cvFindChessboardCorners( rightCopy, board_sz, rightCorners, &rightCorner_count,
    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );

  // make sure we found equal number of corners from both images
  if ( !leftFound || !rightFound || ( leftCorner_count != rightCorner_count ) )
  {
    return;
  }

  // get subpixel accuracy
  cvCvtColor( leftCopy, leftGray, CV_BGR2GRAY );
  cvCvtColor( rightCopy, rightGray, CV_BGR2GRAY );

  cvFindCornerSubPix( leftGray, leftCorners, leftCorner_count,
    cvSize(11,11), cvSize(-1,-1), cvTermCriteria(
    CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ) );
  cvFindCornerSubPix( rightGray, rightCorners, rightCorner_count,
    cvSize(11,11), cvSize(-1,-1), cvTermCriteria(
    CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ) );

  cvDrawChessboardCorners( leftCopy, board_sz, leftCorners, leftCorner_count, leftFound );
  cvDrawChessboardCorners( rightCopy, board_sz, rightCorners, rightCorner_count, rightFound );

  cvShowImage( "LeftResult", leftCopy );
  cvShowImage( "RightResult", rightCopy );

  bool leftFlip = false;
  bool rightFlip = false;

  // sometimes OpenCV returns the corner in the wrong order
  if ( leftCorner_count >= 2 && ( leftCorners[board_n-1].x < leftCorners[0].x ) )
  {
    leftFlip = true;
    std::cerr << "left flipped" << std::endl;
  }

  if ( rightCorner_count >= 2 && ( rightCorners[board_n-1].x < rightCorners[0].x ) )
  {
    rightFlip = true;
    std::cerr << "right flipped" << std::endl;
  }

  // now, enter everything into the external storage
  p3 point;
  point.z = 0.0;
  for ( int i = 0, idx=(board_n-1); i < leftCorner_count; i++, idx-- )
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

  std::cerr << "Collected: " << ImagePointsLeft.size() << " points so far." << std::endl;

  // clean up
  delete [] rightCorners;
  delete [] leftCorners;
  cvReleaseImage( &rightGray );
  cvReleaseImage( &leftGray );
  cvReleaseImage( &rightCopy );
  cvReleaseImage( &leftCopy );
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

  CvMat *image_pointsL = cvCreateMat( nPoints, 2, CV_32FC1 );
  CvMat *image_pointsR = cvCreateMat( nPoints, 2, CV_32FC1 );
  CvMat *object_points = cvCreateMat( nPoints, 3, CV_32FC1 );

  // copy over
  int board_n = GetBoardWidthCalib() * GetBoardHeightCalib();

  for ( int i = 0; i < nPoints; i++ )
  {
    CV_MAT_ELEM( *(image_pointsL), float, i, 0 ) = ImagePointsLeft[i].x;
    CV_MAT_ELEM( *(image_pointsL), float, i, 1 ) = ImagePointsLeft[i].y;
    CV_MAT_ELEM( *(image_pointsR), float, i, 0 ) = ImagePointsRight[i].x;
    CV_MAT_ELEM( *(image_pointsR), float, i, 1 ) = ImagePointsRight[i].y;

    CV_MAT_ELEM( *(object_points), float, i, 0 ) = (float)((i%board_n)/GetBoardWidthCalib()) * 17.0;
    CV_MAT_ELEM( *(object_points), float, i, 1 ) = (float)((i%board_n)%GetBoardWidthCalib()) * 17.0;
    CV_MAT_ELEM( *(object_points), float, i, 2 ) = 0.0;

    std::cerr << i << " " << ImagePointsLeft[i].x << " "
      << ImagePointsLeft[i].y << " "
      << ImagePointsRight[i].x << " "
      << ImagePointsRight[i].y << " "
      << (float)((i%board_n)/GetBoardWidthCalib()) * 17.0 << " "
      << (float)((i%board_n)%GetBoardWidthCalib()) * 17.0 << std::endl;
  }

  CvMat *intrinsic_matrixL = cvCreateMat( 3, 3, CV_64FC1 );
  CvMat *intrinsic_matrixR = cvCreateMat( 3, 3, CV_64FC1 );
  CvMat *rmat =  cvCreateMat( 3, 3, CV_64FC1 );
  CvMat *tmat =  cvCreateMat( 3, 1, CV_64FC1 );
  CvMat *distortion_coeffsL =  cvCreateMat( 4, 1,CV_64FC1 );
  CvMat *distortion_coeffsR =  cvCreateMat( 4, 1,CV_64FC1 );
  CvMat *Essential_matrix = cvCreateMat( 3, 3, CV_64FC1 );
  CvMat *Fundamental_matrix = cvCreateMat( 3, 3, CV_64FC1 );
  CvMat *point_counts = cvCreateMat( nPoints / board_n, 1, CV_32SC1 );
  CvMat *fundamental_matrix = cvCreateMat( 3, 3, CV_32FC1 );
  CvMat *status = cvCreateMat( 1, nPoints, CV_8UC1 );

  for ( int i = 0; i < nPoints/board_n; i++ )
  {
    CV_MAT_ELEM( *point_counts, int, i, 0 ) = board_n;
  }

  double r;

  if ( LeftIntrinsicAvailable && RightIntrinsicAvailable )
  {
    // if intrinsics are available, use them
    std::cerr << "using pre-calibrated camera" << std::endl;
    r = cvStereoCalibrate(  object_points, 
      image_pointsL, 
      image_pointsR,
      point_counts,
      this->internals->intrinsic_matrix[0], 
      this->internals->distortion_coeffs[0],
      this->internals->intrinsic_matrix[1], 
      this->internals->distortion_coeffs[1],
      cvGetSize( cameraImages[0] ),
      rmat, 
      tmat,
      Essential_matrix, 
      Fundamental_matrix,
      CV_CALIB_FIX_INTRINSIC,
      cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 1e-6) 
      );
  }
  else
  {
    std::cerr << "Stereo calibration" << std::endl;
    r = cvStereoCalibrate( object_points, 
      image_pointsL, 
      image_pointsR,
      point_counts,
      intrinsic_matrixL, 
      distortion_coeffsL,
      intrinsic_matrixR, 
      distortion_coeffsR,
      cvGetSize( cameraImages[0] ),
      rmat, 
      tmat,
      Essential_matrix, 
      Fundamental_matrix,
      0, // no flags
      cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 1e-6)
      );
  }

  int fm_count = cvFindFundamentalMat( image_pointsL, image_pointsR,
    fundamental_matrix,
    CV_FM_RANSAC, 1.0, 0.99, status );

  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      std::cerr << cvmGet( intrinsic_matrixL, i, j ) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      std::cerr << cvmGet( intrinsic_matrixR, i, j ) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;

  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      std::cerr << cvmGet( Essential_matrix, i, j ) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      std::cerr << cvmGet( Fundamental_matrix, i, j ) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
  std::cerr << std::endl;
  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      std::cerr << cvmGet( fundamental_matrix, i, j ) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
  std::cerr << std::endl;
  for ( int i = 0; i < 3; i++ )
  {
    for ( int j = 0; j < 3; j++ )
    {
      std::cerr << cvmGet( rmat, i, j ) << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << std::endl;
  for ( int i = 0; i < 3; i++ )
  {
    std::cerr << cvmGet( rmat, i, 0 ) << " ";
  }
  std::cerr << std::endl;

  // clean up
  cvReleaseMat( &status);
  cvReleaseMat( &fundamental_matrix );
  cvReleaseMat( &point_counts );
  cvReleaseMat( &Fundamental_matrix );
  cvReleaseMat( &Essential_matrix );
  cvReleaseMat( &distortion_coeffsR );
  cvReleaseMat( &distortion_coeffsL );
  cvReleaseMat( &tmat );
  cvReleaseMat( &rmat );
  cvReleaseMat( &intrinsic_matrixR );
  cvReleaseMat( &intrinsic_matrixL );
  cvReleaseMat( &object_points );
  cvReleaseMat( &image_pointsR );
  cvReleaseMat( &image_pointsL );
}