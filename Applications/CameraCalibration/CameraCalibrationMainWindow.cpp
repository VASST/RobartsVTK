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
#include "QComputeThread.h"
#include <PlusCommon.h>
#include <QAction>
#include <QActionGroup>
#include <QFileDialog>
#include <QMenuBar>
#include <QMessageBox>
#include <QStatusBar>

//----------------------------------------------------------------------------
CameraCalibrationMainWindow::CameraCalibrationMainWindow()
{
  CCMainWidget = new CameraCalibrationMainWidget(this);
  setCentralWidget( CCMainWidget );

  this->layout()->setSizeConstraint(QLayout::SetFixedSize);

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

  QAction* leftIntrinsicAct = new QAction( tr( "Load Left Camera Parameters" ), this );
  leftIntrinsicAct->setStatusTip( tr( "Load Left Camera Parameters" ) );
  connect( leftIntrinsicAct, SIGNAL( triggered() ), this, SLOT( LoadLeftCameraParameters() ) );

  QAction* rightIntrinsicAct = new QAction( tr( "Load Right Camera Parameters" ), this );
  rightIntrinsicAct->setStatusTip( tr( "Load Right Camera Parameters" ) );
  connect( rightIntrinsicAct, SIGNAL( triggered() ), this, SLOT( LoadRightCameraParameters() ) );

  fileMenu = menuBar()->addMenu( tr( "&File" ) );
  fileMenu->addAction( leftIntrinsicAct );
  fileMenu->addAction( rightIntrinsicAct );
  fileMenu->addSeparator();
  fileMenu->addAction( exitAct );

  menuBar()->addSeparator();

  QAction* chessboardAction = new QAction( tr( "Chessboard" ), this );
  chessboardAction->setStatusTip( tr( "Chessboard pattern" ) );
  chessboardAction->setCheckable(true);
  connect( chessboardAction, SIGNAL( triggered() ), this, SLOT( SetCalibrationPatternChessboard() ) );

  QAction* circlesAction = new QAction( tr( "Circles" ), this );
  circlesAction->setStatusTip( tr( "Circles pattern" ) );
  circlesAction->setCheckable(true);
  connect( circlesAction, SIGNAL( triggered() ), this, SLOT( SetCalibrationPatternCircles() ) );

  QAction* asymCirclesAction = new QAction( tr( "Asymmetric Circles" ), this );
  asymCirclesAction->setStatusTip( tr( "Asymmetric circles pattern" ) );
  asymCirclesAction->setCheckable(true);
  connect( asymCirclesAction, SIGNAL( triggered() ), this, SLOT( SetCalibrationPatternAsymCircles() ) );

  QActionGroup* patternGroup = new QActionGroup(this);
  patternGroup->addAction(chessboardAction);
  patternGroup->addAction(circlesAction);
  patternGroup->addAction(asymCirclesAction);
  chessboardAction->setChecked(true);

  patternMenu = menuBar()->addMenu( tr( "&Pattern") );
  patternMenu->addAction(chessboardAction);
  patternMenu->addAction(circlesAction);
  patternMenu->addAction(asymCirclesAction);

  helpMenu = menuBar()->addMenu( tr( "&Help" ) );
  helpMenu->addAction( aboutAppAct );
  helpMenu->addAction( aboutRobartsAct );
}

//---------------------------------------------------------
// file operation
void CameraCalibrationMainWindow::LoadLeftCameraParameters()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open Left Camera Parameters" ),
                     QString(vtkPlusConfig::GetInstance()->GetOutputDirectory().c_str()),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  this->CCMainWidget->LoadLeftCameraParameters(FileName.toStdString());
}

//---------------------------------------------------------
void CameraCalibrationMainWindow::LoadRightCameraParameters()
{
  QString FileName = QFileDialog::getOpenFileName( this,
                     tr( "Open Right Camera Parameters" ),
                     QString(vtkPlusConfig::GetInstance()->GetOutputDirectory().c_str()),
                     "OpenCV XML (*.xml *.XML)" );

  if ( FileName.size() == 0 )
  {
    return;
  }

  this->CCMainWidget->LoadRightCameraParameters(FileName.toStdString());
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWindow::SetCalibrationPatternChessboard()
{
  this->CCMainWidget->SetCalibrationPattern(QComputeThread::CHESSBOARD);
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWindow::SetCalibrationPatternCircles()
{
  this->CCMainWidget->SetCalibrationPattern(QComputeThread::CIRCLES_GRID);
}

//----------------------------------------------------------------------------
void CameraCalibrationMainWindow::SetCalibrationPatternAsymCircles()
{
  this->CCMainWidget->SetCalibrationPattern(QComputeThread::ASYMMETRIC_CIRCLES_GRID);
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