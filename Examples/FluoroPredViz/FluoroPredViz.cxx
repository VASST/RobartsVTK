#define _USE_MATH_DEFINES // for C++
#include <cmath>

#include "FluoroPredViz.h"
#include "ResizableQVTKWidget.h"
#include "qboxlayout.h"
#include "qfiledialog.h"
#include "qframe.h"
#include "qgridlayout.h"
#include "qgroupbox.h"
#include "qlabel.h"
#include "qlineedit.h"
#include "qpushbutton.h"
#include "qsplitter.h"
#include "vtkActor.h"
#include "vtkArrowSource.h"
#include "vtkBoxWidget.h"
#include "vtkCamera.h"
#include "vtkConeSource.h"
#include "vtkCubeSource.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaDRRImageVolumeMapper.h"
#include "vtkCudaDualImageVolumeMapper.h"
#include "vtkCudaFunctionObject.h"
#include "vtkCudaFunctionPolygonReader.h"
#include "vtkCudaVolumeMapper.h"
#include "vtkDICOMImageReader.h"
#include "vtkImageData.h"
#include "vtkImageExtractComponents.h"
#include "vtkImagePlaneWidget.h"
#include "vtkMINCImageReader.h"
#include "vtkMetaImageReader.h"
#include "vtkPlanes.h"
#include "vtkPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkRendererCollection.h"
#include "vtkSphereSource.h"
#include "vtkTransform.h"
#include "vtkVolume.h"

FluoroPredViz::FluoroPredViz( QWidget* parent )
  : QWidget(0)
  , DegreeMarkers(NULL)
  , NumMarkers(0)
  , XrayMarker(NULL)
  , XraySource(NULL)
  , DVRSource(NULL)
  , ImageBoundsMarkerActor(NULL)
  , SuccessInit(false)
  , Paused(false)
{
  //get initial image
  Reader = 0;
  Extractor = 0;

  SuccessInit = SetUpReader(RequestFilename());
  if(SuccessInit != 0)
  {
    return;
  }

  //create main layout
  QHBoxLayout* MainLayout = new QHBoxLayout();
  this->setLayout(MainLayout);
  QSplitter* WindowSplitter = new QSplitter(Qt::Orientation::Vertical);
  QSplitter* WindowSplitter2 = new QSplitter(Qt::Orientation::Horizontal);
  QSplitter* WindowSplitter3 = new QSplitter(Qt::Orientation::Horizontal);
  //MainLayout->addWidget(Params);
  MainLayout->addWidget(WindowSplitter);
  WindowSplitter->addWidget(WindowSplitter2);
  WindowSplitter->addWidget(WindowSplitter3);
  WindowSplitter->setSizePolicy(QSizePolicy::Policy::Expanding,QSizePolicy::Policy::Expanding);

  //set up fluoroscope parameters bar
  QWidget* FluoroParams = new QWidget(0);
  QVBoxLayout* FluoroParamsLayout = new QVBoxLayout();
  FluoroParams->setLayout(FluoroParamsLayout);
  SetupFluoroParams(FluoroParamsLayout);
  WindowSplitter3->addWidget(FluoroParams);

  //set up object parameters bar
  QWidget* ObjectParams = new QWidget(0);
  QVBoxLayout* ObjectParamsLayout = new QVBoxLayout();
  ObjectParams->setLayout(ObjectParamsLayout);
  SetupObjectParams(ObjectParamsLayout);
  WindowSplitter3->addWidget(ObjectParams);

  //set up screens
  SetupDVRScreen(WindowSplitter2);
  SetupDRRScreen(WindowSplitter2);
  SetupSchematicScreen(WindowSplitter3);

  //set up pipeline
  ConnectUpPipeline();
  UpdateDegreeMarkers();
  UpdateXrayMarker();

  //final prep on the screens
  DRRScreen->ready = true;
  for(int i = 0; i < 3; i++)
  {
    SchematicScreen[i]->ready = true;
  }

  //do final prep on the cameras
  Angle->setValue(Angle->maximum()/2);
  this->DVRSource->SetFocalPoint( this->XraySource->GetFocalPoint() );
  this->DVRSource->SetViewUp( this->XraySource->GetViewUp() );
  this->DVRSource->SetPosition( -this->XraySource->GetPosition()[0],
                                -this->XraySource->GetPosition()[1],
                                -this->XraySource->GetPosition()[2] );

  UpdateViz();

  //finalize clipping planes
  ClippingPlanes->PlaceWidget();
  ClippingPlanes->EnabledOn();
  ClippingPlanes->On();
  ClippingPlanesPosition = vtkTransform::New();
  ClippingPlanesPosition->PostMultiply();
  UpdateImageRegistration();

}

bool FluoroPredViz::GetSuccessInit()
{
  return SuccessInit;
}

FluoroPredViz::~FluoroPredViz()
{

  //fluoroscope params sliders
  delete FocusX;
  delete FocusY;
  delete PrincipleX;
  delete PrincipleY;
  delete DetectorDistance;
  delete Angle;

  //object params sliders
  delete TranslationX;
  delete TranslationY;
  delete TranslationZ;
  delete OrientationX;
  delete OrientationY;
  delete OrientationZ;

  //screens/pipelines
  for(int i = 0; i < 3; i++)
  {
    delete SchematicScreen[i];
  }
  delete DRRScreen;
  if(Reader)
  {
    Reader->Delete();
  }
  if(Extractor)
  {
    Extractor->Delete();
  }
  TransferFunction->Delete();
  ClippingCallback->Delete();

}

//-------------------------------------------------------------------------------//
// Manage fluoroscope parameters
//-------------------------------------------------------------------------------//

class FluoroPredViz::MimicViewCallback : public vtkCommand
{
public:
  MimicViewCallback(FluoroPredViz* w) : window(w) {};
  FluoroPredViz* window;
  void Execute(vtkObject* caller, unsigned long, void*)
  {
    window->MimicView();
  }
};

void FluoroPredViz::SetupFluoroParams(QBoxLayout* ParamsLayout)
{

  //fluoroscope params tab labels
  QGroupBox* FluoroTab = new QGroupBox("Fluoro Parameters",this);
  QPushButton* MimicView = new QPushButton("Mimic 3D View on Fluoro");
  QGridLayout* FluoroTabLayout = new QGridLayout();
  FluoroTabLayout->addWidget(MimicView,0,0,1,5);
  FluoroTabLayout->addWidget(new QLabel("Focus (x)"),1,0);
  FluoroTabLayout->addWidget(new QLabel("Focus (y)"),3,0);
  FluoroTabLayout->addWidget(new QLabel("Detector Distance (d)"),5,0);
  FluoroTabLayout->addWidget(new QLabel("Principle Point (x)"),1,3);
  FluoroTabLayout->addWidget(new QLabel("Principle Point (y)"),3,3);
  FluoroTabLayout->addWidget(new QLabel("Width (w)"),5,3);
  QString ThetaName = "Left-Right Angle (";
  ThetaName.append( QChar(0x98, 0x03) );
  ThetaName.append( ")" );
  FluoroTabLayout->addWidget(new QLabel( ThetaName ),7,0);
  ThetaName = "Cranial-Caudal Angle (";
  ThetaName.append( QChar(0x98, 0x03) );
  ThetaName.append( ")" );
  FluoroTabLayout->addWidget(new QLabel( ThetaName ),9,0);

  //fluoroscope params sliders
  WidthVal = 500.0;
  Width = new QSlider(Qt::Orientation::Horizontal);
  Width->setMaximum(1000);
  Width->setValue(500);
  FocusXVal = 1000.0;
  FocusX = new QSlider(Qt::Orientation::Horizontal);
  FocusX->setMaximum(2000);
  FocusX->setValue(1000);
  FocusYVal = 1000.0;
  FocusY = new QSlider(Qt::Orientation::Horizontal);
  FocusY->setMaximum(2000);
  FocusY->setValue(1000);
  PrincipleXVal = 1000.0;
  PrincipleX = new QSlider(Qt::Orientation::Horizontal);
  PrincipleX->setMaximum(2000);
  PrincipleX->setValue(1000);
  PrincipleYVal = 0.0;
  PrincipleY = new QSlider(Qt::Orientation::Horizontal);
  PrincipleY->setMaximum(2000);
  PrincipleY->setValue(1000);
  DetectorDistanceVal = 1000.0;
  DetectorDistance = new QSlider(Qt::Orientation::Horizontal);
  DetectorDistance->setMaximum(2000);
  DetectorDistance->setValue(1000);
  Angle = new QSlider(Qt::Orientation::Horizontal);
  Angle->setMaximum(360);
  Angle->setValue(180);
  AngleVal = 0.0;
  CrAngle = new QSlider(Qt::Orientation::Horizontal);
  CrAngle->setMaximum(110);
  CrAngle->setValue(65);
  CrAngleVal = 0.0;
  FluoroTabLayout->addWidget(FocusX,2,1);
  FluoroTabLayout->addWidget(FocusY,4,1);
  FluoroTabLayout->addWidget(DetectorDistance,6,1);
  FluoroTabLayout->addWidget(PrincipleX,2,4);
  FluoroTabLayout->addWidget(PrincipleY,4,4);
  FluoroTabLayout->addWidget(Width,6,4);
  FluoroTabLayout->addWidget(Angle,8,1,1,4);
  FluoroTabLayout->addWidget(CrAngle,10,1,1,4);

  //add line edits
  FocusXValBox = new QLineEdit();
  FocusXValBox->setText(QString::number(FocusXVal));
  FocusYValBox = new QLineEdit();
  FocusYValBox->setText(QString::number(FocusYVal));
  PrincipleXValBox = new QLineEdit();
  PrincipleXValBox->setText(QString::number(PrincipleXVal));
  PrincipleYValBox = new QLineEdit();
  PrincipleYValBox->setText(QString::number(PrincipleYVal));
  FocusXValBox->setEnabled(false);
  FocusYValBox->setEnabled(false);
  PrincipleXValBox->setEnabled(false);
  PrincipleYValBox->setEnabled(false);
  FluoroTabLayout->addWidget(FocusXValBox,1,1);
  FluoroTabLayout->addWidget(FocusYValBox,3,1);
  FluoroTabLayout->addWidget(PrincipleXValBox,1,4);
  FluoroTabLayout->addWidget(PrincipleYValBox,3,4);
  DetectorDistanceValBox = new QLineEdit();
  DetectorDistanceValBox->setText(QString::number(DetectorDistanceVal));
  WidthValBox = new QLineEdit();
  WidthValBox->setText(QString::number(WidthVal));
  AngleValBox = new QLineEdit();
  AngleValBox->setText(QString::number(AngleVal));
  CrAngleValBox = new QLineEdit();
  CrAngleValBox->setText(QString::number(CrAngleVal));
  DetectorDistanceValBox->setEnabled(false);
  WidthValBox->setEnabled(false);
  AngleValBox->setEnabled(false);
  CrAngleValBox->setEnabled(false);
  FluoroTabLayout->addWidget(DetectorDistanceValBox,5,1);
  FluoroTabLayout->addWidget(WidthValBox,5,4);
  FluoroTabLayout->addWidget(AngleValBox,7,1,1,4);
  FluoroTabLayout->addWidget(CrAngleValBox,9,1,1,4);

  //fluoroscope slider slots
  connect( MimicView, SIGNAL(pressed()), this, SLOT(MimicView()) );
  connect( FocusX, SIGNAL(valueChanged(int)), this, SLOT(SetFocusX(int)) );
  connect( FocusY, SIGNAL(valueChanged(int)), this, SLOT(SetFocusY(int)) );
  connect( PrincipleX, SIGNAL(valueChanged(int)), this, SLOT(SetPrincipleX(int)) );
  connect( PrincipleY, SIGNAL(valueChanged(int)), this, SLOT(SetPrincipleY(int)) );
  connect( DetectorDistance, SIGNAL(valueChanged(int)), this, SLOT(SetDetectorDistance(int)) );
  connect( Width, SIGNAL(valueChanged(int)), this, SLOT(SetWidth(int)) );
  connect( Angle, SIGNAL(valueChanged(int)), this, SLOT(SetAngle(int)) );
  connect( CrAngle, SIGNAL(valueChanged(int)), this, SLOT(SetCrAngle(int)) );

  //add fluoroscope params to main screen
  FluoroTab->setLayout(FluoroTabLayout);
  FluoroTab->setMaximumWidth(500);
  FluoroTab->setMinimumWidth(300);
  ParamsLayout->addWidget(FluoroTab);
}

void FluoroPredViz::MimicView()
{
  //get DVR camera orientation
  double DesiredOrientation[3] =
  {
    this->DVRSource->GetFocalPoint()[0]-this->DVRSource->GetPosition()[0],
    this->DVRSource->GetFocalPoint()[1]-this->DVRSource->GetPosition()[1],
    this->DVRSource->GetFocalPoint()[2]-this->DVRSource->GetPosition()[2]
  };

  double norm = std::sqrt( DesiredOrientation[0]*DesiredOrientation[0] +
                           DesiredOrientation[1]*DesiredOrientation[1] +
                           DesiredOrientation[2]*DesiredOrientation[2] );
  DesiredOrientation[0] /= norm;
  DesiredOrientation[1] /= norm;
  DesiredOrientation[2] /= norm;

  //no need to correct for object rotation since DVR and DRR are rendered in the same co-ordinate frame

  //find the angle for left-right
  bool truncated = false;
  double DesiredAngle = std::atan2(DesiredOrientation[1],-DesiredOrientation[0]);
  DesiredAngle += 0.5*M_PI;
  if(DesiredAngle > M_PI)
  {
    DesiredAngle -= 2*M_PI;
  }
  if(DesiredAngle < -M_PI)
  {
    DesiredAngle += 2*M_PI;
  }
  DesiredAngle = 180 * DesiredAngle / M_PI;

  //find closest achievable cranial-caudal angle
  double DesiredCrAngle = std::acos( DesiredOrientation[2] ) - 0.5 * M_PI ;
  if(DesiredCrAngle > M_PI)
  {
    DesiredCrAngle -= 2*M_PI;
  }
  if(DesiredCrAngle < -M_PI)
  {
    DesiredCrAngle += 2*M_PI;
  }
  DesiredCrAngle = 180 * DesiredCrAngle / M_PI;
  double MaxCranialAngle = 45.0;
  double MinCranialAngle = -65.0;
  truncated = (DesiredCrAngle > MaxCranialAngle) || (DesiredCrAngle < MinCranialAngle);
  DesiredCrAngle = std::min( MaxCranialAngle, std::max( MinCranialAngle, DesiredCrAngle ) );

  //tint screen if required
  if(truncated)
  {
    unsigned char tint[4] = {255,0,0,128};
    DRRMapper->SetTint(tint);
  }
  else
  {
    unsigned char tint[4] = {255,0,0,0};
    DRRMapper->SetTint(tint);
  }

  //apply angles
  this->Paused = true;
  this->Angle->setValue(180+DesiredAngle);
  this->CrAngle->setValue(65+DesiredCrAngle);
  this->Paused = false;
  UpdateXrayMarker();
  UpdateViz();
}

void FluoroPredViz::SetFocusX(int v)
{
  double Range = 2000.0;
  double Offset = 0.0;
  double aV = Offset + Range*((double) v / (double) (FocusX->maximum() - FocusX->minimum()));
  SetFocusX(aV);
}

void FluoroPredViz::SetFocusY(int v)
{
  double Range = 2000.0;
  double Offset = 0.0;
  double aV = Offset + Range*((double) v / (double) (FocusY->maximum() - FocusY->minimum()));
  SetFocusY(aV);
}

void FluoroPredViz::SetPrincipleX(int v)
{
  double Range = 2000.0;
  double Offset = -1000.0;
  double aV = Offset + Range*((double) v / (double) (PrincipleX->maximum() - PrincipleX->minimum()));
  SetPrincipleX(aV);
}

void FluoroPredViz::SetPrincipleY(int v)
{
  double Range = 2000.0;
  double Offset = -1000.0;
  double aV = Offset + Range*((double) v / (double) (PrincipleY->maximum() - PrincipleY->minimum()));
  SetPrincipleY(aV);
}

void FluoroPredViz::SetDetectorDistance(int v)
{
  double Range = 300.0;
  double Offset = 900.0;
  double aV = Offset + Range*((double) v / (double) (DetectorDistance->maximum() - DetectorDistance->minimum()));
  SetDetectorDistance(aV);
}

void FluoroPredViz::SetAngle(int v)
{
  double Range = 360;
  double Offset = -180.0;
  double aV = Offset + Range*((double) v / (double) (Angle->maximum() - Angle->minimum()));
  SetAngle(aV);
}

void FluoroPredViz::SetCrAngle(int v)
{
  double Range = 110.0;
  double Offset = -65.0;
  double aV = Offset + Range*((double) v / (double) (CrAngle->maximum() - CrAngle->minimum()));
  SetCrAngle(aV);
}

void FluoroPredViz::SetWidth(int v)
{
  double Range = 1000;
  double Offset = 0.0;
  double aV = Offset + Range*((double) v / (double) (Width->maximum() - Width->minimum()));
  SetWidth(aV);
}

void FluoroPredViz::SetFocusX(double v)
{
  FocusXVal = v;
  FocusXValBox->setText(QString::number(v));
  UpdateXrayMarker();
  UpdateViz();
}

void FluoroPredViz::SetFocusY(double v)
{
  FocusYVal = v;
  FocusYValBox->setText(QString::number(v));
  UpdateXrayMarker();
  UpdateViz();
}

void FluoroPredViz::SetPrincipleX(double v)
{
  PrincipleXVal = v;
  PrincipleXValBox->setText(QString::number(v));
  UpdateViz();
}

void FluoroPredViz::SetPrincipleY(double v)
{
  PrincipleYVal = v;
  PrincipleYValBox->setText(QString::number(v));
  UpdateViz();
}

void FluoroPredViz::SetDetectorDistance(double v)
{
  DetectorDistanceVal = v;
  DetectorDistanceValBox->setText(QString::number(v));
  UpdateDegreeMarkers();
  UpdateXrayMarker();
  UpdateViz();
}

void FluoroPredViz::SetAngle(double v)
{
  AngleVal = v;

  if(v > 0)
    if(v < 90)
    {
      AngleValBox->setText("Anterior Left Oblique " + QString::number(v) );
    }

    else if( v == 90 )
    {
      AngleValBox->setText("Left Projection");
    }
    else if( v == 180 )
    {
      AngleValBox->setText("Posterior Projection");
    }
    else
    {
      AngleValBox->setText("Posterior Left Oblique " + QString::number(180-v) );
    }

  else if(v < 0)
    if(v > -90)
    {
      AngleValBox->setText("Anterior Right Oblique " + QString::number(-v) );
    }
    else if( v == -90 )
    {
      AngleValBox->setText("Right Projection");
    }
    else if( v == -180 )
    {
      AngleValBox->setText("Posterior Projection");
    }
    else
    {
      AngleValBox->setText("Posterior Right Oblique " + QString::number(v+180) );
    }

  else
  {
    AngleValBox->setText("Anterior Projection");
  }

  //AngleValBox->setText(QString::number(v));
  UpdateXrayMarker();
  UpdateViz();
}

void FluoroPredViz::SetCrAngle(double v)
{
  CrAngleVal = v;


  if(v > 0)
  {
    CrAngleValBox->setText("Cranial " + QString::number(v) );
  }
  else if(v < 0)
  {
    CrAngleValBox->setText("Caudal " + QString::number(-v) );
  }
  else
  {
    CrAngleValBox->setText("Cranial 0" );
  }

  UpdateXrayMarker();
  UpdateViz();
}

void FluoroPredViz::SetWidth(double v)
{
  WidthVal = v;
  WidthValBox->setText(QString::number(v));
  UpdateXrayMarker();
  UpdateViz();
}

//-------------------------------------------------------------------------------//
// Manage object parameters
//-------------------------------------------------------------------------------//

void FluoroPredViz::SetupObjectParams(QBoxLayout* ParamsLayout)
{

  //clear transform
  ObjectParams = vtkTransform::New();
  ObjectParams->Identity();
  ObjectParams->PostMultiply();

  //Object params tab labels
  QGroupBox* ObjectTab = new QGroupBox("Object Parameters",this);
  QGridLayout* ObjectTabLayout = new QGridLayout();
  ObjectTabLayout->addWidget(new QLabel("Position (x)"),0,0);
  ObjectTabLayout->addWidget(new QLabel("Position (y)"),2,0);
  ObjectTabLayout->addWidget(new QLabel("Position (z)"),4,0);
  ObjectTabLayout->addWidget(new QLabel("Orientation (x)"),0,3);
  ObjectTabLayout->addWidget(new QLabel("Orientation (y)"),2,3);
  ObjectTabLayout->addWidget(new QLabel("Orientation (z)"),4,3);

  //Object params sliders
  TranslationX = new QSlider(Qt::Orientation::Horizontal);
  TranslationXVal = -0.5*this->Reader->GetOutput()->GetSpacing()[0]*
                    (this->Reader->GetOutput()->GetExtent()[0]+this->Reader->GetOutput()->GetExtent()[1])
                    - this->Reader->GetOutput()->GetOrigin()[0];
  TranslationX->setMaximum(2000);
  TranslationX->setValue(1000+TranslationXVal);
  TranslationY = new QSlider(Qt::Orientation::Horizontal);
  TranslationYVal = -0.5*this->Reader->GetOutput()->GetSpacing()[1]*
                    (this->Reader->GetOutput()->GetExtent()[2]+this->Reader->GetOutput()->GetExtent()[3])
                    - this->Reader->GetOutput()->GetOrigin()[1];
  TranslationY->setMaximum(2000);
  TranslationY->setValue(1000+TranslationYVal);
  TranslationZ = new QSlider(Qt::Orientation::Horizontal);
  TranslationZVal = -0.5*this->Reader->GetOutput()->GetSpacing()[2]*
                    (this->Reader->GetOutput()->GetExtent()[4]+this->Reader->GetOutput()->GetExtent()[5])
                    - this->Reader->GetOutput()->GetOrigin()[2];
  TranslationZ->setMaximum(2000);
  TranslationZ->setValue(1000+TranslationZVal);
  OrientationX = new QSlider(Qt::Orientation::Horizontal);
  OrientationX->setMaximum(720);
  OrientationX->setValue(360);
  OrientationXVal = 0.0;
  OrientationY = new QSlider(Qt::Orientation::Horizontal);
  OrientationY->setMaximum(720);
  OrientationY->setValue(360);
  OrientationYVal = 0.0;
  OrientationZ = new QSlider(Qt::Orientation::Horizontal);
  OrientationZ->setMaximum(720);
  OrientationZ->setValue(360);
  OrientationZVal = 0.0;
  ObjectTabLayout->addWidget(TranslationX,1,0,1,2);
  ObjectTabLayout->addWidget(TranslationY,3,0,1,2);
  ObjectTabLayout->addWidget(TranslationZ,5,0,1,2);
  ObjectTabLayout->addWidget(OrientationX,1,3,1,2);
  ObjectTabLayout->addWidget(OrientationY,3,3,1,2);
  ObjectTabLayout->addWidget(OrientationZ,5,3,1,2);

  //add line edits
  TranslationXValBox = new QLineEdit();
  TranslationXValBox->setText(QString::number(TranslationXVal));
  TranslationYValBox = new QLineEdit();
  TranslationYValBox->setText(QString::number(TranslationYVal));
  TranslationZValBox = new QLineEdit();
  TranslationZValBox->setText(QString::number(TranslationZVal));
  TranslationXValBox->setEnabled(false);
  TranslationYValBox->setEnabled(false);
  TranslationZValBox->setEnabled(false);
  ObjectTabLayout->addWidget(TranslationXValBox,0,1);
  ObjectTabLayout->addWidget(TranslationYValBox,2,1);
  ObjectTabLayout->addWidget(TranslationZValBox,4,1);
  OrientationXValBox = new QLineEdit();
  OrientationXValBox->setText(QString::number(OrientationXVal));
  OrientationYValBox = new QLineEdit();
  OrientationYValBox->setText(QString::number(OrientationYVal));
  OrientationZValBox = new QLineEdit();
  OrientationZValBox->setText(QString::number(OrientationZVal));
  ObjectTabLayout->addWidget(OrientationXValBox,0,4);
  ObjectTabLayout->addWidget(OrientationYValBox,2,4);
  ObjectTabLayout->addWidget(OrientationZValBox,4,4);
  OrientationXValBox->setEnabled(false);
  OrientationYValBox->setEnabled(false);
  OrientationZValBox->setEnabled(false);

  //Object slider slots
  connect( TranslationX, SIGNAL(valueChanged(int)), this, SLOT(SetTranslationX(int)) );
  connect( TranslationY, SIGNAL(valueChanged(int)), this, SLOT(SetTranslationY(int)) );
  connect( TranslationZ, SIGNAL(valueChanged(int)), this, SLOT(SetTranslationZ(int)) );
  connect( OrientationX, SIGNAL(valueChanged(int)), this, SLOT(SetOrientationX(int)) );
  connect( OrientationY, SIGNAL(valueChanged(int)), this, SLOT(SetOrientationY(int)) );
  connect( OrientationZ, SIGNAL(valueChanged(int)), this, SLOT(SetOrientationZ(int)) );

  //add Object params to main screen
  ObjectTab->setLayout(ObjectTabLayout);
  ObjectTab->setMaximumWidth(500);
  ObjectTab->setMinimumWidth(300);
  ParamsLayout->addWidget(ObjectTab);

}

void FluoroPredViz::SetTranslationX(int v)
{
  double Range = 2000.0;
  double Offset = -1000.0;
  double aV = Offset + Range*((double) v / (double) (TranslationX->maximum() - TranslationX->minimum()));
  SetTranslationX(aV);
}

void FluoroPredViz::SetTranslationY(int v)
{
  double Range = 2000.0;
  double Offset = -1000.0;
  double aV = Offset + Range*((double) v / (double) (TranslationY->maximum() - TranslationY->minimum()));
  SetTranslationY(aV);
}

void FluoroPredViz::SetTranslationZ(int v)
{
  double Range = 2000.0;
  double Offset = -1000.0;
  double aV = Offset + Range*((double) v / (double) (TranslationZ->maximum() - TranslationZ->minimum()));
  SetTranslationZ(aV);
}

void FluoroPredViz::SetOrientationX(int v)
{
  double Range = 360.0;
  double Offset = -180.0;
  double aV = Offset + Range*((double) v / (double) (OrientationX->maximum() - OrientationX->minimum()));
  SetOrientationX(aV);
}

void FluoroPredViz::SetOrientationY(int v)
{
  double Range = 360.0;
  double Offset = -180.0;
  double aV = Offset + Range*((double) v / (double) (OrientationY->maximum() - OrientationY->minimum()));
  SetOrientationY(aV);
}

void FluoroPredViz::SetOrientationZ(int v)
{
  double Range = 360.0;
  double Offset = -180.0;
  double aV = Offset + Range*((double) v / (double) (OrientationZ->maximum() - OrientationZ->minimum()));
  SetOrientationZ(aV);
}

void FluoroPredViz::SetTranslationX(double v)
{
  TranslationXVal = v;
  TranslationXValBox->setText(QString::number(v));
  UpdateImageRegistration();
  UpdateViz();
}

void FluoroPredViz::SetTranslationY(double v)
{
  TranslationYVal = v;
  TranslationYValBox->setText(QString::number(v));
  UpdateImageRegistration();
  UpdateViz();
}

void FluoroPredViz::SetTranslationZ(double v)
{
  TranslationZVal = v;
  TranslationZValBox->setText(QString::number(v));
  UpdateImageRegistration();
  UpdateViz();
}

void FluoroPredViz::SetOrientationX(double v)
{
  OrientationXVal = v;
  OrientationXValBox->setText(QString::number(v));
  UpdateImageRegistration();
  UpdateViz();
}

void FluoroPredViz::SetOrientationY(double v)
{
  OrientationYVal = v;
  OrientationYValBox->setText(QString::number(v));
  UpdateImageRegistration();
  UpdateViz();
}

void FluoroPredViz::SetOrientationZ(double v)
{
  OrientationZVal = v;
  OrientationZValBox->setText(QString::number(v));
  UpdateImageRegistration();
  UpdateViz();
}


//-------------------------------------------------------------------------------//
// Manage Image
//-------------------------------------------------------------------------------//
#include "vtkStringArray.h"

QString FluoroPredViz::RequestFilename()
{
  //get file name
  QString filename = QFileDialog::getOpenFileName(this,"Open Image","", tr("DICOM (*.dcm);; Meta (*.mhd *.mha);; MINC (*.mnc *.minc)") );
  return filename;
}

bool FluoroPredViz::SetUpReader(QString filename)
{

  //return if there is no filename
  if( filename.isNull() || filename.isEmpty() )
  {
    return false;
  }

  //create reader based on file name extension
  if( filename.endsWith(".mhd",Qt::CaseInsensitive) || filename.endsWith(".mha",Qt::CaseInsensitive) )
  {
    if( this->Reader )
    {
      this->Reader->Delete();
    }
    this->Reader = vtkMetaImageReader::New();
    if( !this->Reader->CanReadFile( filename.toStdString().c_str() ) )
    {
      return false;
    }
    this->Reader->SetFileName( filename.toStdString().c_str() );

  }
  else if( filename.endsWith(".mnc",Qt::CaseInsensitive) || filename.endsWith(".minc",Qt::CaseInsensitive) )
  {
    if( this->Reader )
    {
      this->Reader->Delete();
    }
    this->Reader = vtkMINCImageReader::New();
    if( !this->Reader->CanReadFile( filename.toStdString().c_str() ) )
    {
      return false;
    }
    this->Reader->SetFileName( filename.toStdString().c_str() );

  }
  else if( filename.endsWith(".dcm",Qt::CaseInsensitive) )
  {
    if( this->Reader )
    {
      this->Reader->Delete();
    }
    vtkDICOMImageReader* dicomReader = vtkDICOMImageReader::New();
    this->Reader =  dicomReader;

    //find directory name
    QFileInfo file(filename);
    int length = file.fileName().length();
    filename.chop(length+1);

    //list files in directory ending in .dcm
    QStringList nameFilter("*.dcm");
    QDir directory(filename);
    QStringList txtFilesAndDirectories = directory.entryList(nameFilter);

    //load into array for the reader
    vtkStringArray* filenameArray = vtkStringArray::New();
    for(int i = 0; i < txtFilesAndDirectories.length(); i++)
    {
      filenameArray->InsertNextValue( txtFilesAndDirectories[i].toStdString() );
      //if( !Reader->CanReadFile(  txtFilesAndDirectories[i].toStdString().c_str() ) )
      //  return -1;
    }
    Reader->SetFileNames( filenameArray );
    dicomReader->SetDirectoryName(filename.toStdString().c_str());

    filenameArray->Delete();

  }
  else
  {
    return false;
  }

  Reader->Update();
  Extractor = vtkImageExtractComponents::New();
  Extractor->SetInputConnection(Reader->GetOutputPort());
  Extractor->SetComponents(0);
  Extractor->Update();

  return true;
}

int FluoroPredViz::RequestImage()
{

  //get file name and set up reader
  int failed = SetUpReader(RequestFilename());
  if( failed )
  {
    return failed;
  }

  //connect up remainder of the pipeline
  ConnectUpPipeline();

  //update viz
  UpdateViz();

  return 0;

}

//-------------------------------------------------------------------------------//
// Update Visualization
//-------------------------------------------------------------------------------//

// ---------------------------------------------------------------------------------------
//Callbacks for the clipping plane widgets

class vtkPlaneWidgetCallback : public vtkCommand
{
public:
  static vtkPlaneWidgetCallback *New()
  {
    return new vtkPlaneWidgetCallback;
  }

  virtual void Execute(vtkObject *caller, unsigned long, void*)
  {
    if(window)
    {
      window->Render();
    }
  }

  void SetWindow(vtkRenderWindow* w)
  {
    this->window = w;
  }

private:
  vtkRenderWindow* window;
};

// Callback for moving the planes from the box widget to the mapper
class vtkClippingBoxWidgetCallback : public vtkCommand
{
public:
  static vtkClippingBoxWidgetCallback *New()
  {
    return new vtkClippingBoxWidgetCallback;
  }

  virtual void Execute(vtkObject *caller, unsigned long, void*)
  {
    vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
    if (this->Mapper1 || this->Mapper2)
    {
      vtkPlanes *planes = vtkPlanes::New();
      widget->GetPlanes(planes);
      if(this->Mapper1)
      {
        this->Mapper1->SetClippingPlanes(planes);
      }
      if(this->Mapper2)
      {
        this->Mapper2->SetClippingPlanes(planes);
      }
      planes->Delete();
    }
  }

  void SetMapper(vtkCudaVolumeMapper* m1, vtkCudaVolumeMapper* m2)
  {
    this->Mapper1 = m1;
    this->Mapper2 = m2;
  }

protected:
  vtkClippingBoxWidgetCallback()
  {
    this->Mapper1 = this->Mapper2 = 0;
  }

  vtkCudaVolumeMapper *Mapper1;
  vtkCudaVolumeMapper *Mapper2;
};

void FluoroPredViz::UpdateViz()
{
  if(Paused)
  {
    return;
  }
  DRRScreen->GetRenderWindow()->Render();
  DVRScreen->GetRenderWindow()->Render();
  for(int i = 0; i < 3; i++)
  {
    SchematicScreen[i]->GetRenderWindow()->Render();
  }
}

void FluoroPredViz::ConnectUpPipeline()
{
  //build remaining DRR pipeline
  DRRMapper = vtkCudaDRRImageVolumeMapper::New();
  DRRMapper->SetImageFlipped(true);
#if (VTK_MAJOR_VERSION < 6)
  DRRMapper->SetInput(Extractor->GetOutput());
#else
  DRRMapper->SetInputConnection(Extractor->GetOutputPort());
#endif
  //MapperDRR->SetCTOffset(16.0);
  ImageVolumeDRR = vtkVolume::New();
  ImageVolumeDRR->SetMapper(DRRMapper);
  vtkRenderer* DRR_Renderer = vtkRenderer::New();
  DRR_Renderer->SetBackground(1,1,1);
  DRR_Renderer->AddVolume(ImageVolumeDRR);
  DRRScreen->GetRenderWindow()->AddRenderer(DRR_Renderer);
  XraySource = DRR_Renderer->GetActiveCamera();
  DRR_Renderer->Delete();
  DRRScreen->GetInteractor()->Disable();

  //buidl remaining DVR pipeline
  this->DVRMapper = vtkCudaDualImageVolumeMapper::New();
#if (VTK_MAJOR_VERSION < 6)
  DRRMapper->SetInput(Reader->GetOutput());
#else
  DRRMapper->SetInputConnection(Reader->GetOutputPort());
#endif
  //DVRMapper->SetInput(Extractor->GetOutput());
  DVRMapper->SetImageFlipped(false);
  ImageVolumeDVR = vtkVolume::New();
  ImageVolumeDVR->SetMapper(DVRMapper);
  vtkRenderer* DVR_Renderer = vtkRenderer::New();
  DVR_Renderer->SetBackground(0,0,0);
  DVR_Renderer->AddVolume(ImageVolumeDVR);
  DVRScreen->GetRenderWindow()->AddRenderer(DVR_Renderer);
  DVR_Renderer->ResetCamera();
  DVRSource = DVR_Renderer->GetActiveCamera();
  TransferFunction = vtkCuda2DTransferFunction::New();
  DVRMapper->SetFunction(TransferFunction);
  DVRScreen->GetInteractor()->AddObserver(vtkCommand::StartInteractionEvent,new MimicViewCallback(this));
  DVRScreen->GetInteractor()->AddObserver(vtkCommand::EndInteractionEvent,new MimicViewCallback(this));
  //DVRScreen->GetInteractor()->AddObserver(vtkCommand::AnyEvent,new MimicViewCallback(this));

  //add clipping planes
  ClippingPlanes = vtkBoxWidget::New();
  ClippingPlanes->SetInteractor(DVRScreen->GetInteractor());
  ClippingPlanes->SetPlaceFactor(1.01);
  ClippingPlanes->SetDefaultRenderer(DVR_Renderer);
  ClippingPlanes->InsideOutOn();
  ClippingCallback = vtkClippingBoxWidgetCallback::New();
  ((vtkClippingBoxWidgetCallback*)ClippingCallback)->SetMapper(DVRMapper,DRRMapper);
  ClippingPlanes->AddObserver(vtkCommand::InteractionEvent, ClippingCallback);
  ClippingPlanes->GetSelectedFaceProperty()->SetOpacity(0.05);
#if (VTK_MAJOR_VERSION < 6)
  ClippingPlanes->SetInput( Extractor->GetOutput() );
#else
  ClippingPlanes->SetInputConnection( Extractor->GetOutputPort() );
#endif

  //start cleaning
  DVRMapper->Delete();
  DVR_Renderer->Delete();
  DRRMapper->Delete();

  //create schematic renderer
  vtkRenderer* Schematic_Renderer[3];
  for(int i = 0; i< 3; i++)
  {
    Schematic_Renderer[i] = vtkRenderer::New();
    Schematic_Renderer[i]->SetBackground(0,0,0);
    SchematicScreen[i]->GetRenderWindow()->AddRenderer(Schematic_Renderer[i]);
    SchematicScreen[i]->GetInteractor()->Disable();
  }

  //build angular trajectory markers
  double DegreeIncrements = 10.0;
  NumMarkers = (int)(360.0/DegreeIncrements + 0.5);
  DegreeMarkers = new vtkSphereSource*[NumMarkers];
  for(int i = 0; i < NumMarkers; i++)
  {
    DegreeMarkers[i] = vtkSphereSource::New();
    DegreeMarkers[i]->SetRadius(10);
    vtkPolyDataMapper* Mapper = vtkPolyDataMapper::New();
    Mapper->SetInputConnection(DegreeMarkers[i]->GetOutputPort());
    vtkActor* Actor = vtkActor::New();
    Actor->SetMapper(Mapper);
    for(int i = 0; i < 3; i++)
    {
      Schematic_Renderer[i]->AddActor(Actor);
    }
    Mapper->Delete();
    Actor->Delete();

  }
  UpdateDegreeMarkers();

  //build center point
  vtkSphereSource* CenterPoint = vtkSphereSource::New();
  CenterPoint->SetCenter(0,0,0);
  CenterPoint->SetRadius(20);
  vtkPolyDataMapper* CenterPointMapper = vtkPolyDataMapper::New();
  CenterPointMapper->SetInputConnection(CenterPoint->GetOutputPort());
  CenterPoint->Delete();
  vtkActor* CenterPointActor = vtkActor::New();
  CenterPointActor->SetMapper(CenterPointMapper);
  CenterPointActor->GetProperty()->SetColor(0,1,1);
  CenterPointMapper->Delete();
  for(int i = 0; i < 3; i++)
  {
    Schematic_Renderer[i]->AddActor(CenterPointActor);
  }
  CenterPointActor->Delete();

  //build axis
  vtkArrowSource* AxisXSource = vtkArrowSource::New();
  vtkPolyDataMapper* AxisXMapper = vtkPolyDataMapper::New();
  AxisXMapper->SetInputConnection(AxisXSource->GetOutputPort());
  AxisXSource->Delete();
  vtkActor* AxisXActor = vtkActor::New();
  AxisXActor->SetMapper(AxisXMapper);
  AxisXMapper->Delete();
  AxisXActor->SetScale(500);
  AxisXActor->GetProperty()->SetColor(1,0,0);
  for(int i = 0; i < 3; i++)
    if(i != 2)
    {
      Schematic_Renderer[i]->AddActor(AxisXActor);
    }
  AxisXActor->Delete();
  vtkArrowSource* AxisYSource = vtkArrowSource::New();
  vtkPolyDataMapper* AxisYMapper = vtkPolyDataMapper::New();
  AxisYMapper->SetInputConnection(AxisYSource->GetOutputPort());
  AxisYSource->Delete();
  vtkActor* AxisYActor = vtkActor::New();
  AxisYActor->SetMapper(AxisYMapper);
  AxisYMapper->Delete();
  AxisYActor->SetScale(500);
  AxisYActor->SetOrientation(0,0,90);
  AxisYActor->GetProperty()->SetColor(0,1,0);
  for(int i = 0; i < 3; i++)
    if(i != 1)
    {
      Schematic_Renderer[i]->AddActor(AxisYActor);
    }
  AxisYActor->Delete();
  vtkArrowSource* AxisZSource = vtkArrowSource::New();
  vtkPolyDataMapper* AxisZMapper = vtkPolyDataMapper::New();
  AxisZMapper->SetInputConnection(AxisZSource->GetOutputPort());
  AxisZSource->Delete();
  vtkActor* AxisZActor = vtkActor::New();
  AxisZActor->SetMapper(AxisZMapper);
  AxisZMapper->Delete();
  AxisZActor->SetScale(500);
  AxisZActor->GetProperty()->SetColor(0,0,1);
  AxisZActor->SetOrientation(0,-90,0);
  for(int i = 0; i < 3; i++)
    if(i != 0)
    {
      Schematic_Renderer[i]->AddActor(AxisZActor);
    }
  AxisZActor->Delete();

  //build fluoro location
  XrayMarker = vtkConeSource::New();
  XrayMarker->SetResolution(100);
  vtkPolyDataMapper* XrayMarkerMapper = vtkPolyDataMapper::New();
  XrayMarkerMapper->SetInputConnection(XrayMarker->GetOutputPort());
  vtkActor* XrayMarkerActor = vtkActor::New();
  XrayMarkerActor->SetMapper(XrayMarkerMapper);
  XrayMarkerActor->GetProperty()->SetColor(1,0.75,0);
  XrayMarkerActor->GetProperty()->SetOpacity(0.5);
  XrayMarkerMapper->Delete();
  for(int i = 0; i < 3; i++)
  {
    Schematic_Renderer[i]->AddActor(XrayMarkerActor);
  }
  XrayMarkerActor->Delete();
  UpdateXrayMarker();

  //build object location
  int Extent[6];
  double Spacing[3];
  double Origin[3];
  Reader->GetOutput()->GetExtent(Extent);
  Reader->GetOutput()->GetSpacing(Spacing);
  Reader->GetOutput()->GetOrigin(Origin);
  vtkCubeSource* ImageBoundsMarker = vtkCubeSource::New();
  ImageBoundsMarker->SetXLength( abs( (double)(Extent[1]-Extent[0]+1) * Spacing[0]) );
  ImageBoundsMarker->SetYLength( abs( (double)(Extent[3]-Extent[2]+1) * Spacing[1]) );
  ImageBoundsMarker->SetZLength( abs( (double)(Extent[5]-Extent[4]+1) * Spacing[2]) );
  ImageBoundsMarker->SetCenter(  Origin[0] + (double)(Extent[1]+Extent[0]) * Spacing[0] / 2.0,
                                 Origin[1] + (double)(Extent[3]+Extent[2]) * Spacing[1] / 2.0,
                                 Origin[2] + (double)(Extent[5]+Extent[4]) * Spacing[2] / 2.0 );
  vtkPolyDataMapper* ImageBoundsMarkerMapper = vtkPolyDataMapper::New();
  ImageBoundsMarkerMapper->SetInputConnection(ImageBoundsMarker->GetOutputPort());
  ImageBoundsMarker->Delete();
  ImageBoundsMarkerActor = vtkActor::New();
  ImageBoundsMarkerActor->SetMapper(ImageBoundsMarkerMapper);
  ImageBoundsMarkerMapper->Delete();
  for(int i = 0; i < 3; i++)
  {
    Schematic_Renderer[i]->AddActor(ImageBoundsMarkerActor);
  }
  ImageBoundsMarkerActor->GetProperty()->SetColor(1,1,1);
  ImageBoundsMarkerActor->GetProperty()->SetAmbient(1);
  ImageBoundsMarkerActor->GetProperty()->SetDiffuse(0);
  ImageBoundsMarkerActor->GetProperty()->SetRepresentationToWireframe();

  //turn on the camera
  for(int i = 0; i < 3; i++)
  {
    vtkCamera* schemCamera = Schematic_Renderer[i]->GetActiveCamera();
    double pos[3] = {0,DetectorDistanceVal/2,0};
    if(i==0)
    {
      pos[2] = 5*DetectorDistanceVal;
    }
    if(i==1)
    {
      pos[1] = -5*DetectorDistanceVal;
    }
    if(i==2)
    {
      pos[0] = 5*DetectorDistanceVal;
    }
    schemCamera->SetPosition(pos);
    schemCamera->SetFocalPoint(0,0,0);
    if(i==0)
    {
      schemCamera->SetFocalPoint(0,0,0);
      schemCamera->SetViewUp(0,0,-1);
    }
    if(i==1)
    {
      schemCamera->SetViewUp(0,0,1);
    }
    if(i==2)
    {
      schemCamera->SetViewUp(0,0,1);
    }

    schemCamera->SetClippingRange(DetectorDistanceVal,8*DetectorDistanceVal);
    Schematic_Renderer[i]->Delete();
    SchematicScreen[i]->GetRenderWindow()->Render();
  }

  Schematic_Renderer[0]->ResetCamera();
  Schematic_Renderer[0]->GetActiveCamera()->Zoom(1.5);
}


void FluoroPredViz::UpdateDegreeMarkers()
{
  if(Paused)
  {
    return;
  }
  //set location of markers based on angle and C-arm Detector Distance
  for(int i = 0; i < NumMarkers; i++)
  {
    double DegreeLocation = 2.0 * M_PI * (double)i / (double) (NumMarkers-1);
    double LX = DetectorDistanceVal*std::sin(DegreeLocation)/2;
    double LY = DetectorDistanceVal*std::cos(DegreeLocation)/2;
    DegreeMarkers[i]->SetCenter(LX,LY,0);
    DegreeMarkers[i]->Update();
  }

}

void FluoroPredViz::UpdateXrayMarker()
{
  if(Paused)
  {
    return;
  }

  //set location and orientation of x-ray schematic marker
  double focus = 0.5*FocusXVal + 0.5*FocusYVal;
  double DegreeLocation = 3.141592 * AngleVal / 180;
  double ElevationLocation = 3.141592 * CrAngleVal / 180;

  double LX = 0.5*DetectorDistanceVal*std::sin(DegreeLocation)*std::cos(ElevationLocation);
  double LY = 0.5*DetectorDistanceVal*std::cos(DegreeLocation)*std::cos(ElevationLocation);
  double LZ = 0.5*DetectorDistanceVal*std::sin(ElevationLocation);

  XrayMarker->SetCenter(0,0,0);
  XrayMarker->SetDirection(-LX,-LY,-LZ);
  XrayMarker->SetHeight(DetectorDistanceVal);
  XrayMarker->SetRadius(WidthVal/2);

  //set location and orientation of x-ray source
  double aspect = (double) this->DRRScreen->width() /  (double) this->DRRScreen->height();
  XraySource->SetPosition(-LX,-LY,-LZ);
  XraySource->SetFocalPoint(0,0,0);
  XraySource->SetViewUp(0,0,-1);
  XraySource->SetClippingRange(DetectorDistanceVal/16,DetectorDistanceVal);
  double angle = 360 * atan((WidthVal*DetectorDistanceVal/focus)/(2*DetectorDistanceVal)) / M_PI;
  XraySource->SetViewAngle(angle/aspect);
}

void FluoroPredViz::UpdateImageRegistration()
{
  if(Paused)
  {
    return;
  }

  //collect old transform and invert it
  vtkTransform* OldRegistration = vtkTransform::New();
  OldRegistration->DeepCopy(ObjectParams);
  OldRegistration->Inverse();
  OldRegistration->Modified();
  OldRegistration->Update();

  //get clipping planes pose sans registration
  vtkTransform* OldPosition = vtkTransform::New();
  ClippingPlanes->GetTransform(OldPosition);
  ClippingPlanesPosition->DeepCopy(OldPosition);
  ClippingPlanesPosition->PostMultiply();
  ClippingPlanesPosition->Concatenate(OldRegistration);

  //find centering translation
  double Centering[3];
  Centering[0] = -0.5*this->Reader->GetOutput()->GetSpacing()[0]*
                 (this->Reader->GetOutput()->GetExtent()[0]+this->Reader->GetOutput()->GetExtent()[1])
                 - this->Reader->GetOutput()->GetOrigin()[0];
  Centering[1] = -0.5*this->Reader->GetOutput()->GetSpacing()[1]*
                 (this->Reader->GetOutput()->GetExtent()[2]+this->Reader->GetOutput()->GetExtent()[3])
                 - this->Reader->GetOutput()->GetOrigin()[1];
  Centering[2] = -0.5*this->Reader->GetOutput()->GetSpacing()[2]*
                 (this->Reader->GetOutput()->GetExtent()[4]+this->Reader->GetOutput()->GetExtent()[5])
                 - this->Reader->GetOutput()->GetOrigin()[2];
  double AntiCentering[3] = {-Centering[0],-Centering[1],-Centering[2]};

  //reset transform for objection
  ObjectParams->Identity();
  ObjectParams->Translate(Centering);
  ObjectParams->RotateZ(OrientationZVal);
  ObjectParams->RotateX(OrientationXVal);
  ObjectParams->RotateY(OrientationYVal);
  ObjectParams->Translate(AntiCentering);
  ObjectParams->Translate(TranslationXVal,TranslationYVal,TranslationZVal);
  ObjectParams->Update();

  //update clipping planes
  ClippingPlanesPosition->Concatenate(ObjectParams);
  ClippingPlanesPosition->Modified();
  ClippingPlanesPosition->Update();
  ClippingPlanes->SetTransform(ClippingPlanesPosition);
  ClippingPlanes->Modified();
  ClippingCallback->Execute(ClippingPlanes,0,0);

  //update image bounding box
  ImageBoundsMarkerActor->SetUserTransform(ObjectParams);
  ImageVolumeDRR->SetUserTransform(ObjectParams);
  ImageVolumeDVR->SetUserTransform(ObjectParams);
  UpdateViz();

  //cleanup
  OldRegistration->Delete();
  OldPosition->Delete();
}

void FluoroPredViz::SetupDRRScreen(QSplitter* WindowsLayout)
{
  DRRScreen = new ResizableQVTKWidget(0);
  DRRScreen->setMinimumHeight(500);
  DRRScreen->setMinimumWidth(500);
  DRRScreen->setSizePolicy( QSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum) );
  WindowsLayout->addWidget(DRRScreen);
}

void FluoroPredViz::SetTFName()
{

  //get filename and apply to view
  QString filename = QFileDialog::getOpenFileName(this,"Open Transfer Function","", tr("2D TF (*.2tf)") );
  if(filename.isNull() || filename.isEmpty() )
  {
    return;
  }
  TFName->setText(filename);

  //set TF in renderer
  size_t n = TransferFunction->GetNumberOfFunctionObjects();
  for(size_t i = 0; i < n; i++)
  {
    vtkCudaFunctionObject* object = TransferFunction->GetFunctionObject(n-i-1);
    TransferFunction->RemoveFunctionObject(object);
    object->Delete();
  }
  vtkCudaFunctionPolygonReader* tfReader = vtkCudaFunctionPolygonReader::New();
  tfReader->SetFileName(filename.toStdString());
  tfReader->Read();
  for(int i = 0; i < tfReader->GetNumberOfOutputs(); i++)
  {
    TransferFunction->AddFunctionObject(tfReader->GetOutput(i));
  }
  TransferFunction->Modified();

  UpdateViz();
}

void FluoroPredViz::SetupDVRScreen(QSplitter* WindowsLayout)
{

  //set up screen
  DVRScreen = new QVTKWidget(0);
  DVRScreen->setMinimumHeight(500);
  DVRScreen->setMinimumWidth(500);
  DVRScreen->setSizePolicy( QSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum) );
  QVBoxLayout* l1 = new QVBoxLayout();
  DVRScreen->setLayout(l1);
  //l1->addWidget(DVRScreen);

  //set up line edit
  TFName = new QLineEdit(0);
  TFName->setEnabled(false);
  QPushButton* button = new QPushButton("Select File",0);
  connect(button,SIGNAL(pressed()),this,SLOT(SetTFName()));
  l1->addStretch();
  l1->addWidget(TFName);
  l1->addWidget(button);

  WindowsLayout->addWidget(DVRScreen);
}

void FluoroPredViz::SetupSchematicScreen(QSplitter* WindowsLayout)
{

  QVBoxLayout* layout1 = new QVBoxLayout();
  QWidget* widget1 = new QWidget();
  widget1->setLayout(layout1);
  widget1->setSizePolicy( QSizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding) );
  QHBoxLayout* layout2 = new QHBoxLayout();
  QWidget* widget2 = new QWidget();
  widget2->setLayout(layout2);
  widget2->setSizePolicy( QSizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding) );
  for(int i = 0; i < 3; i++)
  {
    SchematicScreen[i] = new ResizableQVTKWidget(0);
    SchematicScreen[i]->setMinimumWidth(200);
    SchematicScreen[i]->setMinimumHeight(150);
    SchematicScreen[i]->setMaximumHeight(150);
    SchematicScreen[i]->setSizePolicy( QSizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding) );
  }
  layout1->addWidget(SchematicScreen[0]);
  layout1->addWidget(widget2);
  layout2->addWidget(SchematicScreen[1]);
  layout2->addWidget(SchematicScreen[2]);

  WindowsLayout->addWidget(widget1);

}
