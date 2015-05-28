#include "DUALTF_segmentationWidget.h"

#include "vtkCudaVoxelClassifier.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkImageGradientMagnitude.h"
#include "vtkImageAppendComponents.h"
#include "vtkMetaImageWriter.h"

#include "vtkSystemIncludes.h"
#include "vtksys\SystemTools.hxx"

#include <QVBoxLayout>
#include <QFileDialog>
// ---------------------------------------------------------------------------------------
// Construction and destruction code
DUALTF_segmentationWidget::DUALTF_segmentationWidget( DUALTF_transferFunctionWindowWidget* p ) :
  QWidget(p)
{
  parent = p;
  window = 0;
  renderer = 0;
  mapper = 0;

  segmentationMenu = 0;
  setupMenu();
}

DUALTF_segmentationWidget::~DUALTF_segmentationWidget( ) {
}

void DUALTF_segmentationWidget::setupMenu(){
  
  segmentationMenu = new QMenu("Segmentation",this);
  segmentNowOption = new QAction("Segment Now",this);
  segmentNowOption->setEnabled(true);
  segmentationMenu->addAction(segmentNowOption);
  
  connect(segmentNowOption,SIGNAL(triggered()),this,SLOT(segment()));

}

QMenu* DUALTF_segmentationWidget::getMenuOptions(){
  return segmentationMenu;
}

void DUALTF_segmentationWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCudaDualImageVolumeMapper* c ){
  window = w;
  renderer = r;
  mapper = c;
}
// ---------------------------------------------------------------------------------------
// Code to interface with the slots and user

void DUALTF_segmentationWidget::segment(){

  vtkCudaVoxelClassifier* classifier = vtkCudaVoxelClassifier::New();
  classifier->SetInput( mapper->GetInput( mapper->GetCurrentFrame() ) );
  classifier->SetClippingPlanes( mapper->GetClippingPlanes() );
  classifier->SetKeyholePlanes( mapper->GetKeyholePlanes() );
  classifier->SetFunction( mapper->GetFunction() );
  classifier->SetKeyholeFunction( mapper->GetKeyholeFunction() );
  classifier->Update();

  QString filename = QFileDialog::getSaveFileName(this, tr("Open File"), QDir::currentPath(),"Meta Image Files (*.mhd)" );

  if( filename.size() != 0 ){
    std::string rawfilename = vtksys::SystemTools::GetFilenameWithoutExtension( filename.toStdString() );
    rawfilename.append( ".raw" );
    vtkMetaImageWriter* writer = vtkMetaImageWriter::New();
    writer->SetCompression(false);
    writer->SetFileName( filename.toStdString().c_str() );
    writer->SetRAWFileName( rawfilename.c_str() );
    writer->SetInput( classifier->GetOutput() );
    writer->Write();
    writer->Delete();
  }
  classifier->Delete();
}
