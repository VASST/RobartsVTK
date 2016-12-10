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

class QAction;
class QMenu;
class QTimer;
class vtkPlusDataCollector;
class vtkPlusMmfVideoSource;

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
  void OnStartStopButtonClicked();

protected:
  void CreateActions();

protected:
  QAction*    aboutAppAct;
  QAction*    aboutRobartsAct;
  QAction*    exitAct;
  QMenu*      fileMenu;
  QMenu*      patternMenu;
  QMenu*      helpMenu;
  QTimer*     uiUpdateTimer;

  vtkSmartPointer<vtkPlusDataCollector> collector;
  vtkSmartPointer<vtkPlusMmfVideoSource> videoDevice;

private:
  Ui::MainWindow mainWindow;
};

#endif