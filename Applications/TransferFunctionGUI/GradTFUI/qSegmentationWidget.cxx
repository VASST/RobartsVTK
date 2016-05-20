/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "qSegmentationWidget.h"
#include "qTransferFunctionWindowWidget.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCuda2DVolumeMapper.h"
#include "vtkCudaVoxelClassifier.h"
#include "vtkImageAppendComponents.h"
#include "vtkImageData.h"
#include "vtkImageGradientMagnitude.h"
#include "vtkMetaImageWriter.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkSystemIncludes.h"
#include "vtksys/SystemTools.hxx"
#include <QFileDialog>
#include <QMenu>
#include <QPushButton>
#include <QVBoxLayout>

// ---------------------------------------------------------------------------------------
// Construction and destruction code
qSegmentationWidget::qSegmentationWidget( qTransferFunctionWindowWidget* p ) 
  : QWidget(p)
{
  parent = p;
  window = 0;
  renderer = 0;
  mapper = 0;

  segmentationMenu = 0;
  setupMenu();
}

qSegmentationWidget::~qSegmentationWidget( )
{
}

void qSegmentationWidget::setupMenu()
{
  segmentationMenu = new QMenu("Segmentation",this);
  segmentNowOption = new QAction("Segment Now",this);
  segmentNowOption->setEnabled(true);
  segmentationMenu->addAction(segmentNowOption);

  connect(segmentNowOption,SIGNAL(triggered()),this,SLOT(segment()));
}

QMenu* qSegmentationWidget::getMenuOptions()
{
  return segmentationMenu;
}

void qSegmentationWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCuda2DVolumeMapper* c )
{
  window = w;
  renderer = r;
  mapper = c;
}

// ---------------------------------------------------------------------------------------
// Code to interface with the slots and user
void qSegmentationWidget::segment()
{
  mapper->GetInput();

  vtkImageGradientMagnitude* gradient = vtkImageGradientMagnitude::New();
  gradient->SetInputData( mapper->GetInput(mapper->GetCurrentFrame() ) );

  vtkImageAppendComponents* appender = vtkImageAppendComponents::New();
  appender->SetInputData(0, mapper->GetInput( mapper->GetCurrentFrame() ) );
  appender->SetInputData(1, gradient->GetOutput() );

  vtkCudaVoxelClassifier* classifier = vtkCudaVoxelClassifier::New();
  classifier->SetInputData( appender->GetOutput() );
  classifier->SetClippingPlanes( mapper->GetClippingPlanes() );
  classifier->SetKeyholePlanes( mapper->GetKeyholePlanes() );
  classifier->SetFunction( mapper->GetFunction() );
  classifier->SetKeyholeFunction( mapper->GetKeyholeFunction() );
  classifier->Update();

  QString filename = QFileDialog::getSaveFileName(this, tr("Open File"), QDir::currentPath(),"Meta Image Files (*.mhd)" );

  if( filename.size() != 0 )
  {
    std::string rawfilename = vtksys::SystemTools::GetFilenameWithoutExtension( std::string(filename.toLatin1().data()) );
    rawfilename.append( ".raw" );
    vtkMetaImageWriter* writer = vtkMetaImageWriter::New();
    writer->SetCompression(false);
    writer->SetFileName( filename.toLatin1().data() );
    writer->SetRAWFileName( rawfilename.c_str() );
    writer->SetInputData( classifier->GetOutput() );
    writer->Write();
    writer->Delete();
  }
  classifier->Delete();
  gradient->Delete();
  appender->Delete();
}
