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

#ifndef __OpenCVTestBedMainWindow_H__
#define __OpenCVTestBedMainWindow_H__

// Local includes
#include "ui_OpenCVTestBedMainWindow.h"

// QT includes
#include <QMainWindow>

// VTK includes
#include <vtkSmartPointer.h>

// STL includes
#include <atomic>

class PlusDeviceSetSelectorWidget;
class QAction;
class QMenu;
class QTimer;
class vtkPlusChannel;
class vtkPlusDataCollector;
class vtkPlusMmfVideoSource;
class vtkPlusTrackedFrameList;

class OpenCVTestBedMainWindow : public QMainWindow
{
  Q_OBJECT

public:
  OpenCVTestBedMainWindow();
  ~OpenCVTestBedMainWindow();

protected slots:
  void OnUpdateTimerTimeout();
  void AboutApp();
  void AboutRobarts();

  void OnDeviceComboBoxChanged(int index);
  void OnChannelComboBoxChanged(int index);
  void OnStartStopButtonClicked();
  void OnConnectToDevicesByConfigFileInvoked(std::string configFile);

protected:
  void CreateActions();
  void PopulateChannelList();

protected:
  QAction*                                  m_aboutAppAction;
  QAction*                                  m_aboutRobartsAction;
  QAction*                                  m_exitAction;
  QMenu*                                    m_fileMenu;
  QMenu*                                    m_helpMenu;
  QTimer*                                   m_uiUpdateTimer;

  vtkSmartPointer<vtkPlusDataCollector>     m_dataCollector;
  vtkSmartPointer<vtkPlusMmfVideoSource>    m_videoDevice;
  vtkSmartPointer<vtkPlusTrackedFrameList>  m_trackedFrameList;
  double                                    m_mostRecentFrameTimestamp;

  vtkPlusChannel*                           m_currentChannel;
  PlusDeviceSetSelectorWidget*              m_deviceSetSelectorWidget;

private:
  Ui::MainWindow mainWindow;
};

#endif