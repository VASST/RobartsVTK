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
#include <PlusCommon.h>
#include <PlusDeviceSetSelectorWidget.h>
#include <vtkPlusDataCollector.h>
#include <vtkPlusMmfVideoSource.h>
#include <MediaFoundationVideoCaptureApi.h>

// QT includes
#include <QActionGroup>
#include <QComboBox>
#include <QMessageBox>
#include <QTimer>

//----------------------------------------------------------------------------
OpenCVTestBedMainWindow::OpenCVTestBedMainWindow()
  : m_dataCollector(vtkSmartPointer<vtkPlusDataCollector>::New())
  , m_videoDevice(nullptr)
  , m_uiUpdateTimer(new QTimer())
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

  m_deviceSetSelectorWidget = new PlusDeviceSetSelectorWidget(mainWindow.centralwidget);
  QGridLayout* gridLayout = dynamic_cast<QGridLayout*>(mainWindow.centralwidget->layout());
  auto layoutItem = gridLayout->takeAt(1);
  gridLayout->addWidget(m_deviceSetSelectorWidget, 1, 0);
  gridLayout->addItem(layoutItem, 2, 0);

  CreateActions();

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
  if (!m_channelActive && m_dataCollector != nullptr && m_dataCollector->IsStarted() && mainWindow.comboBox_channel->currentText() != QString(""))
  {
    // Parse out device, and channel
    QString text = mainWindow.comboBox_channel->currentText();
    auto deviceId = text.left(text.indexOf(tr("-"))).trimmed();
    auto channelId = text.mid(text.indexOf(tr("-")) + 1).trimmed();

    vtkPlusDevice* device(nullptr);
    if (m_dataCollector->GetDevice(device, deviceId.toStdString()) != PLUS_SUCCESS)
    {
      LOG_ERROR("Device not found.");
      m_channelActive = false;
      return;
    }

    vtkPlusChannel* channel(nullptr);
    if (device->GetOutputChannelByName(channel, channelId.toStdString()) != PLUS_SUCCESS)
    {
      LOG_ERROR("Output channel not found.");
      m_channelActive = false;
      return;
    }

    m_currentChannel = channel;
  }

  if (m_channelActive)
  {

  }
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::AboutApp()
{
  QMessageBox::about(this, tr("About OpenCVTestBed"), tr("This application is for developers to prototype OpenCV & C++ features:\nAdam Rankin\narankin@robarts.ca"));
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
void OpenCVTestBedMainWindow::OnStartStopButtonClicked()
{
  m_channelActive = false;

  if (mainWindow.pushButton_startStop->text() == QString("Start"))
  {
    m_deviceSetSelectorWidget->setEnabled(false);
    mainWindow.pushButton_startStop->setEnabled(false);
    mainWindow.pushButton_startStop->setText(tr("Stop"));

    m_videoDevice = vtkSmartPointer<vtkPlusMmfVideoSource>::New();
    m_videoDevice->SetDeviceId("WebCam1");
    m_videoDevice->SetRequestedDeviceId(mainWindow.comboBox_device->currentIndex());
    auto streamIndex = mainWindow.comboBox_stream->currentData().toUInt() >> 16;
    auto formatIndex = mainWindow.comboBox_stream->currentData().toUInt() & 0x0000FFFF;
    m_videoDevice->SetRequestedStreamIndex(streamIndex);

    mainWindow.pushButton_startStop->setEnabled(true);
  }
  else
  {
    mainWindow.pushButton_startStop->setEnabled(false);
    if (m_videoDevice != nullptr)
    {
      m_videoDevice->StopRecording();
      m_videoDevice->Disconnect();
      m_videoDevice->Delete();
      m_videoDevice = nullptr;
    }
    mainWindow.pushButton_startStop->setText(tr("Start"));
    m_deviceSetSelectorWidget->setEnabled(true);
    mainWindow.pushButton_startStop->setEnabled(true);
  }
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::OnConnectToDevicesByConfigFileInvoked(std::string configFile)
{
  m_channelActive = false;
  mainWindow.pushButton_startStop->setEnabled(false);

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

    mainWindow.statusbar->showMessage(tr("Disconnection successful."));
  }

  QApplication::restoreOverrideCursor();
}