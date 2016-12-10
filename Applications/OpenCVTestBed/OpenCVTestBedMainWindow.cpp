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
  : collector(vtkSmartPointer<vtkPlusDataCollector>::New())
  , videoDevice(nullptr)
  , uiUpdateTimer(new QTimer())
{
  mainWindow.setupUi(this);

  std::vector<std::wstring> deviceNames;
  MfVideoCapture::MediaFoundationVideoCaptureApi::GetInstance().GetDeviceNames(deviceNames);
  MfVideoCapture::MediaFoundationVideoCaptureApi::GetInstance().GetDeviceNames(deviceNames);

  mainWindow.comboBox_device->clear();
  int i = 0;
  for (auto& deviceWName : deviceNames)
  {
    mainWindow.comboBox_device->addItem(QString::fromStdWString(deviceWName), QVariant(i));
    i++;
  }

  CreateActions();

  connect(mainWindow.pushButton_startStop, &QPushButton::clicked, this, &OpenCVTestBedMainWindow::OnStartStopButtonClicked);
  connect(mainWindow.comboBox_device, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &OpenCVTestBedMainWindow::OnDeviceComboBoxChanged);
  connect(uiUpdateTimer, &QTimer::timeout, this, &OpenCVTestBedMainWindow::OnUpdateTimerTimeout);

  uiUpdateTimer->start(16);
  OnDeviceComboBoxChanged(0);
}

//----------------------------------------------------------------------------
OpenCVTestBedMainWindow::~OpenCVTestBedMainWindow()
{
  disconnect(mainWindow.pushButton_startStop, &QPushButton::clicked, this, &OpenCVTestBedMainWindow::OnStartStopButtonClicked);
  disconnect(mainWindow.comboBox_device, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &OpenCVTestBedMainWindow::OnDeviceComboBoxChanged);
  disconnect(uiUpdateTimer, &QTimer::timeout, this, &OpenCVTestBedMainWindow::OnUpdateTimerTimeout);
  delete uiUpdateTimer;
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::CreateActions()
{
  exitAct = new QAction(tr("E&xit"), this);
  exitAct->setShortcuts(QKeySequence::Quit);
  exitAct->setStatusTip(tr("Exit the application"));
  connect(exitAct, SIGNAL(triggered()), this, SLOT(close()));

  aboutAppAct = new QAction(tr("&About this App"), this);
  aboutAppAct->setStatusTip(tr("About Camera Calibration"));
  connect(aboutAppAct, SIGNAL(triggered()), this, SLOT(AboutApp()));

  aboutRobartsAct = new QAction(tr("About &Robarts"), this);
  aboutAppAct->setStatusTip(tr("About Robarts Research Institute"));
  connect(aboutRobartsAct, SIGNAL(triggered()), this, SLOT(AboutRobarts()));

  fileMenu = menuBar()->addMenu(tr("&File"));
  fileMenu->addSeparator();
  fileMenu->addAction(exitAct);

  menuBar()->addSeparator();

  helpMenu = menuBar()->addMenu(tr("&Help"));
  helpMenu->addAction(aboutAppAct);
  helpMenu->addAction(aboutRobartsAct);
}

//----------------------------------------------------------------------------
void OpenCVTestBedMainWindow::OnUpdateTimerTimeout()
{

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

  if (videoDevice != nullptr)
  {
    videoDevice->StopRecording();
    videoDevice->Disconnect();
    videoDevice->Delete();
    videoDevice = nullptr;
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
  if (mainWindow.pushButton_startStop->text() == QString("Start"))
  {
    mainWindow.pushButton_startStop->setEnabled(false);
    mainWindow.pushButton_startStop->setText(tr("Stop"));

    videoDevice = vtkSmartPointer<vtkPlusMmfVideoSource>::New();
    videoDevice->SetDeviceId("WebCam1");
    videoDevice->SetRequestedDeviceId(mainWindow.comboBox_device->currentIndex());
    auto streamIndex = mainWindow.comboBox_stream->currentData().toUInt() >> 16;
    auto formatIndex = mainWindow.comboBox_stream->currentData().toUInt() & 0x0000FFFF;
    videoDevice->SetRequestedStreamIndex(streamIndex);

    mainWindow.pushButton_startStop->setEnabled(true);
  }
  else
  {
    mainWindow.pushButton_startStop->setEnabled(false);
    if (videoDevice != nullptr)
    {
      videoDevice->StopRecording();
      videoDevice->Disconnect();
      videoDevice->Delete();
      videoDevice = nullptr;
    }
    mainWindow.pushButton_startStop->setText(tr("Start"));
    mainWindow.pushButton_startStop->setEnabled(true);
  }
}