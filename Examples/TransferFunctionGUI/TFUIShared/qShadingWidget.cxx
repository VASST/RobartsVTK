#include "qShadingWidget.h"

#include <QVBoxLayout>

// ---------------------------------------------------------------------------------------
// Construction and destruction code

#define SLIDERMAX 1000

qShadingWidget::qShadingWidget( qTransferFunctionWindowWidgetInterface* p ) :
  QWidget(p)
{
  parent = p;
  window = 0;
  renderer = 0;
  mapper = 0;
  
  //create and arrange the sliders
  QVBoxLayout* shadeLayout = new QVBoxLayout();
  CelDarknessSliderLabel = new QLabel("Cel Shading Darkness:",this);
  CelASliderLabel = new QLabel("Cel Shading Start Value:",this);
  CelBSliderLabel = new QLabel("Cel Shading Stop Value:",this);
  CelDarknessSliderValue = new QLabel("0.688",this);
  CelASliderValue = new QLabel("0.0181",this);
  CelBSliderValue = new QLabel("0.0423",this);
  CelDarknessSlider = new QSlider(Qt::Orientation::Horizontal, this);
  CelDarknessSlider->setRange(0,SLIDERMAX);
  CelDarknessSlider->setValue(688);
  CelASlider = new QSlider(Qt::Orientation::Horizontal, this);
  CelASlider->setRange(0,SLIDERMAX);
  CelASlider->setValue(181);
  CelBSlider = new QSlider(Qt::Orientation::Horizontal, this);
  CelBSlider->setRange(0,SLIDERMAX);
  CelBSlider->setValue(423);
  
  DistDarknessSliderLabel = new QLabel("Distance Shading Darkness:",this);
  DistASliderLabel = new QLabel("Distance Shading Start Value:",this);
  DistBSliderLabel = new QLabel("Distance Shading Stop Value:",this);
  DistDarknessSliderValue = new QLabel("0.25",this);
  DistASliderValue = new QLabel("0.45",this);
  DistBSliderValue = new QLabel("0.6",this);
  DistDarknessSlider = new QSlider(Qt::Orientation::Horizontal, this);
  DistDarknessSlider->setRange(0,SLIDERMAX);
  DistDarknessSlider->setValue(250);
  DistASlider = new QSlider(Qt::Orientation::Horizontal, this);
  DistASlider->setRange(0,SLIDERMAX);
  DistASlider->setValue(450);
  DistBSlider = new QSlider(Qt::Orientation::Horizontal, this);
  DistBSlider->setRange(0,SLIDERMAX);
  DistBSlider->setValue(600);
  
  shadeLayout->addWidget(CelDarknessSliderLabel);
  shadeLayout->addWidget(CelDarknessSliderValue);
  shadeLayout->addWidget(CelDarknessSlider);
  shadeLayout->addWidget(CelASliderLabel);
  shadeLayout->addWidget(CelASliderValue);
  shadeLayout->addWidget(CelASlider);
  shadeLayout->addWidget(CelBSliderLabel);
  shadeLayout->addWidget(CelBSliderValue);
  shadeLayout->addWidget(CelBSlider);
  shadeLayout->addWidget(DistDarknessSliderLabel);
  shadeLayout->addWidget(DistDarknessSliderValue);
  shadeLayout->addWidget(DistDarknessSlider);
  shadeLayout->addWidget(DistASliderLabel);
  shadeLayout->addWidget(DistASliderValue);
  shadeLayout->addWidget(DistASlider);
  shadeLayout->addWidget(DistBSliderLabel);
  shadeLayout->addWidget(DistBSliderValue);
  shadeLayout->addWidget(DistBSlider);
  this->setLayout(shadeLayout);

  //setup the hooks for the shading sliders
  connect(CelDarknessSlider, SIGNAL(valueChanged(int)), this, SLOT(changeShading()) );
  connect(CelASlider, SIGNAL(valueChanged(int)), this, SLOT(changeShading()) );
  connect(CelBSlider, SIGNAL(valueChanged(int)), this, SLOT(changeShading()) );
  connect(DistDarknessSlider, SIGNAL(valueChanged(int)), this, SLOT(changeShading()) );
  connect(DistASlider, SIGNAL(valueChanged(int)), this, SLOT(changeShading()) );
  connect(DistBSlider, SIGNAL(valueChanged(int)), this, SLOT(changeShading()) );

  //show the sliders
  CelDarknessSlider->show();
  CelASlider->show();
  CelBSlider->show();
  DistDarknessSlider->show();
  DistASlider->show();
  DistBSlider->show();
}

qShadingWidget::~qShadingWidget( ) {
  delete CelDarknessSliderLabel;
  delete CelASliderLabel;
  delete CelBSliderLabel;
  delete DistDarknessSliderLabel;
  delete DistASliderLabel;
  delete DistBSliderLabel;
  delete CelDarknessSlider;
  delete CelASlider;
  delete CelBSlider;
  delete DistDarknessSlider;
  delete DistASlider;
  delete DistBSlider;
}


void qShadingWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCudaVolumeMapper* c ){
  window = w;
  renderer = r;
  mapper = c;

  //set reasonable default shading values
  float darkness = (double) CelDarknessSlider->value() / (double) CelDarknessSlider->maximum();
  float a = 0.1 * (double) CelASlider->value() / (double) CelASlider->maximum();
  float b = 0.1 * (double) CelBSlider->value() / (double) CelBSlider->maximum();
  this->CelDarknessSliderValue->setText( QString::number(darkness) );
  this->CelDarknessSliderValue->repaint();
  this->CelASliderValue->setText( QString::number(a) );
  this->CelBSliderValue->setText( QString::number(b) );
  setCelShadingConstants(darkness,a,b);
  
  darkness = (double) DistDarknessSlider->value() / (double) DistDarknessSlider->maximum();
  a = 0.1 * (double) DistASlider->value() / (double) DistASlider->maximum();
  b = 0.1 * (double) DistBSlider->value() / (double) DistBSlider->maximum();
  this->DistDarknessSliderValue->setText( QString::number(darkness) );
  this->DistASliderValue->setText( QString::number(a) );
  this->DistBSliderValue->setText( QString::number(b) );
  setDistanceShadingConstants(darkness,a,b);
}
// ---------------------------------------------------------------------------------------
// Code to interface with the slots and user

void qShadingWidget::changeShading(){
  

  //set reasonable default shading values
  float darkness = (double) CelDarknessSlider->value() / (double) CelDarknessSlider->maximum();
  float a = 0.1 * (double) CelASlider->value() / (double) CelASlider->maximum();
  float b = 0.1 * (double) CelBSlider->value() / (double) CelBSlider->maximum();
  this->CelDarknessSliderValue->setText( QString::number(darkness) );
  this->CelDarknessSliderValue->repaint();
  this->CelASliderValue->setText( QString::number(a) );
  this->CelBSliderValue->setText( QString::number(b) );
  setCelShadingConstants(darkness,a,b);
  
  darkness = (double) DistDarknessSlider->value() / (double) DistDarknessSlider->maximum();
  a = (double) DistASlider->value() / (double) DistASlider->maximum();
  b = (double) DistBSlider->value() / (double) DistBSlider->maximum();
  this->DistDarknessSliderValue->setText( QString::number(darkness) );
  this->DistASliderValue->setText( QString::number(a) );
  this->DistBSliderValue->setText( QString::number(b) );
  setDistanceShadingConstants(darkness,a,b);

  //update the image
  parent->UpdateScreen();
}

// ---------------------------------------------------------------------------------------
// Code to interface with the model

void qShadingWidget::setCelShadingConstants(float d, float a, float b){
  mapper->SetCelShadingConstants(d,a,b);
}

void qShadingWidget::setDistanceShadingConstants(float d, float a, float b){
  mapper->SetDistanceShadingConstants(d,a,b);
}