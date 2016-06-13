/*=========================================================================

Program:   Main window
Module:    $RCSfile: CameraCalibrationMainWindow.cpp,v $
Creator:   Elvis C. S. Chen <chene@robarts.ca>
Language:  C++
Author:    $Author: Elvis Chen $
Date:      $Date: 2011/07/4 15:28:30 $
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

#include "CameraCalibrationMainWindow.h"
#include <QAction>
#include <QFileDialog>
#include <QMenuBar>
#include <QMessageBox>
#include <QStatusBar>

//----------------------------------------------------------------------------
CameraCalibrationMainWindow::CameraCalibrationMainWindow()
{
  CCMainWidget = new CameraCalibrationMainWidget(this);
  setCentralWidget( CCMainWidget );

  CreateActions();
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWindow::SetPLUSTrackingChannel(const std::string& trackingChannel)
{
  this->CCMainWidget->SetPLUSTrackingChannel(trackingChannel);
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWindow::CreateActions()
{
  exitAct = new QAction( tr( "E&xit" ), this );
  exitAct->setShortcuts( QKeySequence::Quit );
  exitAct->setStatusTip( tr( "Exit the application" ) );
  connect( exitAct, SIGNAL( triggered() ), this, SLOT( close() ) );

  aboutAppAct = new QAction( tr( "&About this App" ), this );
  aboutAppAct->setStatusTip( tr("About Camera Calibration" ) );
  connect( aboutAppAct, SIGNAL( triggered() ), this, SLOT( AboutApp() ) );

  aboutRobartsAct = new QAction( tr( "About &Robarts" ), this );
  aboutAppAct->setStatusTip( tr("About Robarts Research Institute" ) );
  connect( aboutRobartsAct, SIGNAL( triggered() ), this, SLOT( AboutRobarts() ) );

  QAction* leftIntrinsicAct = new QAction( tr( "Load Left Intrinsic" ), this );
  leftIntrinsicAct->setStatusTip( tr( "Load Left Intrinsic" ) );
  connect( leftIntrinsicAct, SIGNAL( triggered() ), this, SLOT( LoadLeftIntrinsic() ) );

  QAction* rightIntrinsicAct = new QAction( tr( "Load Right Intrinsic" ), this );
  rightIntrinsicAct->setStatusTip( tr( "Load Right Intrinsic" ) );
  connect( rightIntrinsicAct, SIGNAL( triggered() ), this, SLOT( LoadRightIntrinsic() ) );

  QAction* leftDistortionAct = new QAction( tr( "Load Left Distortion" ), this );
  leftDistortionAct->setStatusTip( tr( "Load Left Distortion" ) );
  connect( leftDistortionAct, SIGNAL( triggered() ), this, SLOT( LoadLeftDistortion() ) );

  QAction* rightDistortionAct = new QAction( tr( "Load Right Distortion" ), this );
  rightDistortionAct->setStatusTip( tr( "Load Right Distortion" ) );
  connect( rightDistortionAct, SIGNAL( triggered() ), this, SLOT( LoadRightDistortion() ) );

  fileMenu = menuBar()->addMenu( tr( "&File" ) );
  fileMenu->addSeparator();
  fileMenu->addAction( leftIntrinsicAct );
  fileMenu->addAction( leftDistortionAct );
  fileMenu->addAction( rightIntrinsicAct );
  fileMenu->addAction( rightDistortionAct );
  fileMenu->addSeparator();
  fileMenu->addAction( exitAct );

  menuBar()->addSeparator();

  helpMenu = menuBar()->addMenu( tr( "&Help" ) );
  helpMenu->addAction( aboutAppAct );
  helpMenu->addAction( aboutRobartsAct );
}

//---------------------------------------------------------
// file operation
void CameraCalibrationMainWindow::LoadLeftIntrinsic()
{
  QString FileName = QFileDialog::getOpenFileName( this,
    tr( "Open Left Intrinsic" ),
    QDir::currentPath(),
    "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  this->CCMainWidget->LoadLeftIntrinsic(FileName.toStdString());
}

//---------------------------------------------------------
void CameraCalibrationMainWindow::LoadRightIntrinsic()
{
  QString FileName = QFileDialog::getOpenFileName( this,
    tr( "Open Right Intrinsic" ),
    QDir::currentPath(),
    "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  this->CCMainWidget->LoadRightIntrinsic(FileName.toStdString());
}

//---------------------------------------------------------
void CameraCalibrationMainWindow::LoadLeftDistortion()
{
  QString FileName = QFileDialog::getOpenFileName( this,
    tr( "Open left Distortion" ),
    QDir::currentPath(),
    "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  this->CCMainWidget->LoadLeftDistortion(FileName.toStdString());
}

//---------------------------------------------------------
void CameraCalibrationMainWindow::LoadRightDistortion()
{
  QString FileName = QFileDialog::getOpenFileName( this,
    tr( "Open right Distortion" ),
    QDir::currentPath(),
    "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  this->CCMainWidget->LoadRightDistortion(FileName.toStdString());
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWindow::AboutApp()
{
  QMessageBox::about( this, tr( "About Camera Calibration" ),
    tr( "This camera calibration application is brought to you by:\n\n"
    "Elvis C.S. Chen\n\n"
    "chene@robarts.ca"
    ) );
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWindow::AboutRobarts()
{
  QMessageBox::about( this, tr( "About Robarts Research Institute" ),
    tr( "Robarts Research Institute\n\n"
    "London, Ontario\n"
    "Canada, N6A 5K8"
    ) );
}