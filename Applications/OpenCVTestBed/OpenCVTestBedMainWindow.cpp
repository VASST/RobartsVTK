/*====================================================================
Copyright(c) 2016 Adam Rankin

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files(the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and / or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
====================================================================*/

// Local includes
#include "OpenCVTestBedMainWindow.h"

// PlusLib includes
#include <MediaFoundationVideoCaptureApi.h>
#include <MediaFoundationVideoDevice.h>
#include <MediaFoundationVideoDevices.h>
#include <PlusCommon.h>
#include <PlusDeviceSetSelectorWidget.h>
#include <vtkPlusChannel.h>
#include <vtkPlusDataCollector.h>
#include <vtkPlusDataSource.h>
#include <vtkPlusMmfVideoSource.h>
#include <vtkPlusTrackedFrameList.h>

// QT includes
#include <QActionGroup>
#include <QComboBox>
#include <QMessageBox>
#include <QTimer>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

namespace
{
  // Let compiler perform return value optimization, copy is unlikely given modern compiler
  cv::Mat VTKImageToOpenCVMat(vtkImageData& image)
  {
    int cvType(0);
    // Determine type
    switch (image.GetScalarType())
    {
    case VTK_UNSIGNED_CHAR:
    {
      cvType = CV_8UC(image.GetNumberOfScalarComponents());
      break;
    }
    case VTK_SIGNED_CHAR:
    {
      cvType = CV_8SC(image.GetNumberOfScalarComponents());
      break;
    }
    case VTK_UNSIGNED_SHORT:
    {
      cvType = CV_16UC(image.GetNumberOfScalarComponents());
      break;
    }
    case VTK_SHORT:
    {
      cvType = CV_16SC(image.GetNumberOfScalarComponents());
      break;
    }
    case VTK_INT:
    {
      cvType = CV_32SC(image.GetNumberOfScalarComponents());
      break;
    }
    case VTK_FLOAT:
    {
      cvType = CV_32FC(image.GetNumberOfScalarComponents());
      break;
    }
    case VTK_DOUBLE:
    {
      cvType = CV_64FC(image.GetNumberOfScalarComponents());
      break;
    }
    }

    int* dimensions = image.GetDimensions();
    // VTK images are continuous, so no need to pass in step details (they are automatically computed)
    return cv::Mat((dimensions[2] > 1 ? 3 : 2), dimensions, cvType, image.GetScalarPointer());
  }
}

//----------------------------------------------------------------------------
OpenCVTestBedMainWindow::OpenCVTestBedMainWindow()
  : m_uiUpdateTimer(new QTimer())
  , m_dataCollector(vtkSmartPointer<vtkPlusDataCollector>::New())
  , m_videoDevice(nullptr)
  , m_trackedFrameList(vtkSmartPointer<vtkPlusTrackedFrameList>::New())
  , m_mostRecentFrameTimestamp(UNDEFINED_TIMESTAMP)
  , m_currentChannel(nullptr)
  , m_deviceSetSelectorWidget(nullptr)
{
  mainWindow.setupUi(this);

  std::vector<std::wstring> deviceNames;
  MfVideoCapture::MediaFoundationVideoCaptureApi::GetInstance().GetDeviceNames(deviceNames);

  mainWindow.comboBox_device->clear();
  int i = 0;
  for (auto& deviceWName : deviceNames)
  {
    mainWindow.comboBox_device->addItem(QString::fromStdWString(deviceWName), QVariant(i));
    i++;
  }

  m_deviceSetSelectorWidget = new PlusDeviceSetSelectorWidget(mainWindow.widget_controlContainer);
  QVBoxLayout* layout = dynamic_cast<QVBoxLayout*>(mainWindow.widget_controlContainer->layout());
  layout->insertWidget(1, m_deviceSetSelectorWidget);

  CreateActions();

  connect(mainWindow.comboBox_channel, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &OpenCVTestBedMainWindow::OnChannelComboBoxChanged);
  connect(m_deviceSetSelectorWidget, &PlusDeviceSetSelectorWidget::ConnectToDevicesByConfigFileInvoked, this, &OpenCVTestBedMainWindow::OnConnectToDevicesByConfigFileInvoked);
  connect(mainWindow.pushButton_startStop, &QPushButton::clicked, this, &OpenCVTestBedMainWindow::OnStartStopButtonClicked);
  connect(mainWindow.comboBox_device, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &OpenCVTestBedMainWindow::OnDeviceComboBoxChanged);
  connect(m_uiUpdateTimer, &QTimer::timeout, this, &OpenCVTestBedMainWindow::OnUpdateTimerTimeout);

  m_uiUpdateTimer->start(16);
  OnDeviceComboBoxChanged(0);
}

//----------------------------------------------------------------------------
OpenCVTestBedMainWindow::~OpenCVTestBedMainWindow()
{
  disconnect(mainWindow.comboBox_channel, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &OpenCVTestBedMainWindow::OnChannelComboBoxChanged);
  disconnect(m_deviceSetSelectorWidget, &PlusDeviceSetSelectorWidget::ConnectToDevicesByConfigFileInvoked, this, &OpenCVTestBedMainWindow::OnConnectToDevicesByConfigFileInvoked);
  disconnect(mainWindow.pushButton_startStop, &QPushButton::clicked, this, &OpenCVTestBedMainWindow::OnStartStopButtonClicked);
  disconnect(mainWindow.comboBox_device, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &OpenCVTestBedMainWindow::OnDeviceComboBoxChanged);
  disconnect(m_uiUpdateTimer, &QTimer::timeout, this, &OpenCVTestBedMainWindow::OnUpdateTimerTimeout);
  delete m_uiUpdateTimer;
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::CreateActions()
{
  m_exitAction = new QAction(tr("E&xit"), this);
  m_exitAction->setShortcuts(QKeySequence::Quit);
  m_exitAction->setStatusTip(tr("Exit the application"));
  connect(m_exitAction, SIGNAL(triggered()), this, SLOT(close()));

  m_aboutAppAction = new QAction(tr("&About this App"), this);
  m_aboutAppAction->setStatusTip(tr("About Camera Calibration"));
  connect(m_aboutAppAction, SIGNAL(triggered()), this, SLOT(AboutApp()));

  m_aboutRobartsAction = new QAction(tr("About &Robarts"), this);
  m_aboutAppAction->setStatusTip(tr("About Robarts Research Institute"));
  connect(m_aboutRobartsAction, SIGNAL(triggered()), this, SLOT(AboutRobarts()));

  m_fileMenu = menuBar()->addMenu(tr("&File"));
  m_fileMenu->addSeparator();
  m_fileMenu->addAction(m_exitAction);

  menuBar()->addSeparator();

  m_helpMenu = menuBar()->addMenu(tr("&Help"));
  m_helpMenu->addAction(m_aboutAppAction);
  m_helpMenu->addAction(m_aboutRobartsAction);
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::PopulateChannelList()
{
  mainWindow.comboBox_channel->clear();
  if (m_dataCollector != nullptr && m_dataCollector->IsStarted())
  {
    for (auto iter = m_dataCollector->GetDeviceConstIteratorBegin(); iter != m_dataCollector->GetDeviceConstIteratorEnd(); ++iter)
    {
      for (auto chanIter = (*iter)->GetOutputChannelsStart(); chanIter != (*iter)->GetOutputChannelsEnd(); ++chanIter)
      {
        mainWindow.comboBox_channel->addItem(QString::fromLatin1((*iter)->GetDeviceId()) + QString(" - ") + QString::fromLatin1((*chanIter)->GetChannelId()));
      }
    }
  }
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::OnUpdateTimerTimeout()
{
  if (m_currentChannel != nullptr)
  {
    m_trackedFrameList->Clear();
    m_currentChannel->GetTrackedFrameList(m_mostRecentFrameTimestamp, m_trackedFrameList, 50);

    for (auto& frame : *m_trackedFrameList)
    {
      vtkImageData* image = frame->GetImageData()->GetImage();
      cv::Mat cvImage = VTKImageToOpenCVMat(*image);

      // TODO : your code here!
    }
  }
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::AboutApp()
{
  QMessageBox::about(this, tr("About OpenCVTestBed"), tr("This application is for developers to prototype OpenCV, PLUS, & C++ features:\nAdam Rankin\narankin@robarts.ca"));
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::AboutRobarts()
{
  QMessageBox::about(this, tr("About Robarts Research Institute"), tr("Robarts Research Institute\nLondon, Ontario\nCanada, N6A 5K8"));
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::OnDeviceComboBoxChanged(int deviceId)
{
  mainWindow.pushButton_startStop->setEnabled(false);
  mainWindow.pushButton_startStop->setText(tr("Start"));

  if (m_videoDevice != nullptr)
  {
    m_videoDevice->StopRecording();
    m_videoDevice->Disconnect();
    m_videoDevice->Delete();
    m_videoDevice = nullptr;
  }

  if (deviceId < 0 || deviceId >= MfVideoCapture::MediaFoundationVideoDevices::GetInstance().GetCount())
  {
    return;
  }

  mainWindow.comboBox_stream->clear();

  auto streamCount = MfVideoCapture::MediaFoundationVideoCaptureApi::GetInstance().GetNumberOfStreams(deviceId);
  for (unsigned int streamIndex = 0; streamIndex < streamCount; ++streamIndex)
  {
    auto formatCount = MfVideoCapture::MediaFoundationVideoCaptureApi::GetInstance().GetNumberOfFormats(deviceId, streamIndex);
    for (unsigned int formatIndex = 0; formatIndex < formatCount; ++formatIndex)
    {
      auto mediaType = MfVideoCapture::MediaFoundationVideoCaptureApi::GetInstance().GetFormat(deviceId, streamIndex, formatIndex);
      std::wstringstream ss;
      ss << mediaType.MF_MT_SUBTYPEName << ": " << mediaType.width << "x" << mediaType.height << "@" << mediaType.MF_MT_FRAME_RATE;
      mainWindow.comboBox_stream->addItem(QString::fromStdWString(ss.str()), QVariant(streamIndex << 16 | (formatIndex & 0x0000FFFF)));
    }
  }

  if (mainWindow.comboBox_stream->count() > 0)
  {
    mainWindow.pushButton_startStop->setEnabled(true);
  }
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::OnChannelComboBoxChanged(int index)
{
  m_currentChannel = nullptr;

  // Parse out device, and channel
  QString text = mainWindow.comboBox_channel->currentText();
  auto deviceId = text.left(text.indexOf(tr("-"))).trimmed();
  auto channelId = text.mid(text.indexOf(tr("-")) + 1).trimmed();

  vtkPlusDevice* device(nullptr);
  if (m_dataCollector->GetDevice(device, deviceId.toStdString()) != PLUS_SUCCESS)
  {
    LOG_ERROR("Device not found.");
    return;
  }

  vtkPlusChannel* channel(nullptr);
  if (device->GetOutputChannelByName(channel, channelId.toStdString()) != PLUS_SUCCESS)
  {
    LOG_ERROR("Output channel not found.");
    return;
  }

  m_currentChannel = channel;
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::OnStartStopButtonClicked()
{
  m_currentChannel = nullptr;

  if (mainWindow.pushButton_startStop->text() == QString("Start"))
  {
    m_deviceSetSelectorWidget->setEnabled(false);
    mainWindow.comboBox_device->setEnabled(false);
    mainWindow.comboBox_stream->setEnabled(false);
    mainWindow.comboBox_channel->setEnabled(false);
    mainWindow.pushButton_startStop->setEnabled(false);

    m_videoDevice = vtkSmartPointer<vtkPlusMmfVideoSource>::New();
    m_videoDevice->SetDeviceId("WebCam1");
    m_videoDevice->SetRequestedDeviceId(mainWindow.comboBox_device->currentIndex());
    auto streamIndex = mainWindow.comboBox_stream->currentData().toUInt() >> 16;
    auto formatIndex = mainWindow.comboBox_stream->currentData().toUInt() & 0x0000FFFF;
    m_videoDevice->SetRequestedStreamIndex(streamIndex);
    m_videoDevice->SetRequestedFormatIndex(formatIndex);

    auto mediaType = MfVideoCapture::MediaFoundationVideoDevices::GetInstance().GetDevice(mainWindow.comboBox_device->currentIndex())->GetFormat(streamIndex, formatIndex);
    vtkPlusDataSource* source = vtkPlusDataSource::New();
    source->SetSourceId("videoSource");
    source->SetInputFrameSize(mediaType.width, mediaType.height, 1U);
    source->SetInputImageOrientation(US_IMG_ORIENT_MF);
    source->SetOutputImageOrientation(US_IMG_ORIENT_MF);
    source->SetNumberOfScalarComponents(3); // Default to colour
    source->SetPixelType(VTK_UNSIGNED_CHAR);
    source->SetImageType(US_IMG_BRIGHTNESS);
    m_videoDevice->AddVideoSource(source);

    vtkPlusChannel* channel = vtkPlusChannel::New();
    channel->SetChannelId("VideoChannel");
    channel->SetOwnerDevice(m_videoDevice);
    channel->SetVideoSource(source);
    m_videoDevice->AddOutputChannel(channel);

    m_dataCollector->AddDevice(m_videoDevice);
    m_dataCollector->Connect();
    m_dataCollector->Start();

    m_currentChannel = channel;

    mainWindow.pushButton_startStop->setText(tr("Stop"));
    mainWindow.pushButton_startStop->setEnabled(true);
  }
  else
  {
    m_currentChannel = nullptr;
    mainWindow.pushButton_startStop->setEnabled(false);
    vtkPlusDevice* device(nullptr);
    m_dataCollector->GetDevice(device, "WebCam1");
    vtkPlusChannel* channel(nullptr);
    device->GetOutputChannelByName(channel, "VideoChannel");
    vtkPlusDataSource* source(nullptr);
    channel->GetVideoSource(source);
    m_dataCollector->Stop();
    m_dataCollector->Disconnect();
    m_dataCollector = vtkSmartPointer<vtkPlusDataCollector>::New();
    source->Delete();
    channel->Delete();
    device->Delete();
    mainWindow.pushButton_startStop->setText(tr("Start"));
    m_deviceSetSelectorWidget->setEnabled(true);
    mainWindow.comboBox_channel->setEnabled(true);
    mainWindow.comboBox_device->setEnabled(true);
    mainWindow.comboBox_stream->setEnabled(true);
    mainWindow.pushButton_startStop->setEnabled(true);
  }
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::OnConnectToDevicesByConfigFileInvoked(std::string configFile)
{
  m_currentChannel = nullptr;
  mainWindow.pushButton_startStop->setEnabled(false);
  mainWindow.comboBox_device->setEnabled(false);
  mainWindow.comboBox_stream->setEnabled(false);

  QApplication::setOverrideCursor(QCursor(Qt::BusyCursor));

  // If not empty, then try to connect; empty parameter string means disconnect
  if (STRCASECMP(configFile.c_str(), "") != 0)
  {
    // Read configuration
    vtkSmartPointer<vtkXMLDataElement> configRootElement = vtkSmartPointer<vtkXMLDataElement>::Take(vtkXMLUtilities::ReadElementFromFile(configFile.c_str()));
    if (configRootElement == NULL)
    {
      LOG_ERROR("Unable to read configuration from file " << configFile);
      mainWindow.statusbar->showMessage(tr("Unable to read configuration from file "));

      m_deviceSetSelectorWidget->SetConnectionSuccessful(false);
      QApplication::restoreOverrideCursor();

      return;
    }

    LOG_INFO("Device set configuration is read from file: " << configFile);
    mainWindow.statusbar->showMessage(tr("Device set configuration is read from file: ") + QString::fromStdString(configFile));

    vtkPlusConfig::GetInstance()->SetDeviceSetConfigurationData(configRootElement);

    // If connection has been successfully created then start data collection
    if (!m_deviceSetSelectorWidget->GetConnectionSuccessful())
    {
      // Disable main window
      this->setEnabled(false);

      mainWindow.statusbar->showMessage(tr("Connecting to devices, please wait..."));

      // Connect to devices
      if (m_dataCollector->ReadConfiguration(vtkPlusConfig::GetInstance()->GetDeviceSetConfigurationData()) != PLUS_SUCCESS)
      {
        m_dataCollector = vtkSmartPointer<vtkPlusDataCollector>::New();
        LOG_ERROR("Unable to read configuration for data collector.");
        mainWindow.statusbar->showMessage(tr("Unable to read configuration for data collector."));
        m_deviceSetSelectorWidget->setEnabled(true);
        mainWindow.pushButton_startStop->setEnabled(true);
        m_deviceSetSelectorWidget->SetConnectionSuccessful(false);
        this->setEnabled(true);
        return;
      }

      if (m_dataCollector->Start() != PLUS_SUCCESS)
      {
        m_dataCollector = vtkSmartPointer<vtkPlusDataCollector>::New();
        LOG_ERROR("Unable to start data collector.");
        mainWindow.statusbar->showMessage(tr("Unable to start data collector."));
        m_deviceSetSelectorWidget->setEnabled(true);
        mainWindow.pushButton_startStop->setEnabled(true);
        m_deviceSetSelectorWidget->SetConnectionSuccessful(false);
        this->setEnabled(true);
        return;
      }

      mainWindow.statusbar->showMessage(tr("Connection successful."));

      PopulateChannelList();

      // Re-enable main window
      this->setEnabled(true);
      m_deviceSetSelectorWidget->SetConnectionSuccessful(true);
    }
  }
  else // Disconnect
  {
    m_dataCollector->Stop();
    m_dataCollector->Disconnect();
    m_dataCollector = vtkSmartPointer<vtkPlusDataCollector>::New();

    mainWindow.comboBox_channel->clear();
    mainWindow.pushButton_startStop->setEnabled(true);
    m_deviceSetSelectorWidget->SetConnectionSuccessful(false);
    mainWindow.comboBox_device->setEnabled(true);
    mainWindow.comboBox_stream->setEnabled(true);

    mainWindow.statusbar->showMessage(tr("Disconnection successful."));
  }

  QApplication::restoreOverrideCursor();
}