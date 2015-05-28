#include "stereoRenderingWidget.h"

#include "vtkCamera.h"
// ---------------------------------------------------------------------------------------
// Construction and destruction code

stereoRenderingWidget::stereoRenderingWidget( transferFunctionWindowWidgetInterface* p ) :
  QWidget(p)
{
  parent = p;
  window = 0;
  renderer = 0;
  mapper = 0;
  old = 0;
  
  setupMenu();

}
  
void stereoRenderingWidget::setupMenu(){
  
  stereoMenu = new QMenu("Stereo",this);
  stereoOnMenuOption = new QAction("Turn Stereo On",this);
  stereoOffMenuOption = new QAction("Turn Stereo Off",this);
  stereoEyeToggleMenuOption = new QAction("Swap Eyes",this);
  stereoOffMenuOption->setEnabled(false);
  stereoOnMenuOption->setEnabled(true);
  stereoEyeToggleMenuOption->setEnabled(false);
  stereoMenu->addAction(stereoOnMenuOption);
  stereoMenu->addAction(stereoOffMenuOption);
  stereoMenu->addAction(stereoEyeToggleMenuOption);
  
  connect(stereoOnMenuOption,SIGNAL(triggered()),this,SLOT(setStereoOn()));
  connect(stereoOffMenuOption,SIGNAL(triggered()),this,SLOT(setStereoOff()));
  connect(stereoEyeToggleMenuOption,SIGNAL(triggered()),this,SLOT(toggleEyes()));

}

QMenu* stereoRenderingWidget::getMenuOptions(){
  return stereoMenu;
}

stereoRenderingWidget::~stereoRenderingWidget( ) {
  
  delete stereoOffMenuOption;
  delete stereoOnMenuOption;
  delete stereoEyeToggleMenuOption;
  delete stereoMenu;
}


void stereoRenderingWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCudaVolumeMapper* c ){
  window = w;
  renderer = r;
  mapper = c;

}
// ---------------------------------------------------------------------------------------
// Code to interface with the slots and user

void stereoRenderingWidget::setStereoOn(){
  this->setStereoEnabled(true);
  this->stereoOnMenuOption->setEnabled(false);
  this->stereoOffMenuOption->setEnabled(true);
  this->stereoEyeToggleMenuOption->setEnabled(true);
}

void stereoRenderingWidget::setStereoOff(){
  this->setStereoEnabled(false);
  this->stereoOnMenuOption->setEnabled(true);
  this->stereoOffMenuOption->setEnabled(false);
  this->stereoEyeToggleMenuOption->setEnabled(false);
}

void stereoRenderingWidget::toggleEyes(){
  this->toggleStereoEyes();
}

// ---------------------------------------------------------------------------------------
// Code to interface with the model

void stereoRenderingWidget::setStereoEnabled(bool b){

  //set the window to render in stereo
  if(b){
    window->SetStereoType(3);
    window->SetStereoRender(1);
    old = 0;
  }else{
    window->SetStereoRender(0);
  }
  window->Render();
}

void stereoRenderingWidget::toggleStereoEyes(){

  //TODO This area is buggy in general - no clue why
  old = 1 - old;
  renderer->GetActiveCamera()->SetLeftEye(old);
  renderer->GetActiveCamera()->Modified();
  renderer->Modified();
  window->Render();
  

}