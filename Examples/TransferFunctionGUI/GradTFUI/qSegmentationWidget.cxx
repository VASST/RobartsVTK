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
#include "vtksys\SystemTools.hxx"
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
#if ( VTK_MAJOR_VERSION < 6 )
  gradient->SetInput( mapper->GetInput( mapper->GetCurrentFrame() ) );
  gradient->Update();
#else
  gradient->SetInputData( mapper->GetInput(mapper->GetCurrentFrame() ) );
#endif

  vtkImageAppendComponents* appender = vtkImageAppendComponents::New();
#if ( VTK_MAJOR_VERSION < 6 )
  appender->SetInput(0, mapper->GetInput( mapper->GetCurrentFrame() ) );
  appender->SetInput(1, gradient->GetOutput() );
#else
  appender->SetInputData(0, mapper->GetInput( mapper->GetCurrentFrame() ) );
  appender->SetInputData(1, gradient->GetOutput() );
#endif

  vtkCudaVoxelClassifier* classifier = vtkCudaVoxelClassifier::New();
#if ( VTK_MAJOR_VERSION < 6 )
  classifier->SetInput( appender->GetOutput() );
#else
  classifier->SetInputData( appender->GetOutput() );
#endif
  classifier->SetClippingPlanes( mapper->GetClippingPlanes() );
  classifier->SetKeyholePlanes( mapper->GetKeyholePlanes() );
  classifier->SetFunction( mapper->GetFunction() );
  classifier->SetKeyholeFunction( mapper->GetKeyholeFunction() );
  classifier->Update();

  QString filename = QFileDialog::getSaveFileName(this, tr("Open File"), QDir::currentPath(),"Meta Image Files (*.mhd)" );

  if( filename.size() != 0 )
  {
    std::string rawfilename = vtksys::SystemTools::GetFilenameWithoutExtension( filename.toStdString() );
    rawfilename.append( ".raw" );
    vtkMetaImageWriter* writer = vtkMetaImageWriter::New();
    writer->SetCompression(false);
    writer->SetFileName( filename.toStdString().c_str() );
    writer->SetRAWFileName( rawfilename.c_str() );
#if ( VTK_MAJOR_VERSION < 6 )
    writer->SetInput( classifier->GetOutput() );
#else
    writer->SetInputData( classifier->GetOutput() );
#endif
    writer->Write();
    writer->Delete();
  }
  classifier->Delete();
  gradient->Delete();
  appender->Delete();
}