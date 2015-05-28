#include "transferFunctionWindowWidget.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkStringArray.h"

#include "vtkMetaImageReader.h"

#include <QString>
#include <QStringList>
#include <QFileDialog>
#include <QColorDialog>
#include <QInputDialog>
#include <QTabWidget>

#include <fstream>

#include "vtkPointData.h"

transferFunctionWindowWidget::transferFunctionWindowWidget(QWidget *parent) :
  transferFunctionWindowWidgetInterface(parent)
{

  //initialize communal elements
  window = vtkRenderWindow::New();
  renderer = vtkRenderer::New();
  mapper = vtkCudaDualImageVolumeMapper::New();
  mapper->UseFullVTKCompatibility();
  
  //assemble the communal pipeline
  renderer->SetBackground(0,0,0);
  window->AddRenderer(renderer);
  window->SetStereoCapableWindow(1);

  //initialize overall layout
  menuLayout = new QVBoxLayout();
  setLayout(menuLayout);
  mainLayout = new QHBoxLayout();
  tabbar = new QTabWidget(this);
  tabbar->setMinimumWidth(100);
  tabbar->setMaximumWidth(250);
  connect(tabbar, SIGNAL(currentChanged(int)), this, SLOT(changeTab()) );
  mainLayout->addWidget(tabbar);

  //get KSOM filename
  QString filename = QFileDialog::getOpenFileName(0, "Open File", QDir::currentPath(),"Meta Image Files (*.mhd)" );
  bool loaded = !( filename.isEmpty() || filename.isEmpty() );
  if( !loaded ) std::exit(1);
  vtkMetaImageReader* mapReader = vtkMetaImageReader::New();
  if( !mapReader->CanReadFile(filename.toStdString().c_str()) ) std::exit(1);

  //load KSOM
  mapReader->SetFileName( filename.toStdString().c_str() );
  mapReader->Update();

  //set up the file widget and menu
  fmWidget = new kFileManagementWidget( this );
  fmWidget->setStandardWidgets( window, renderer, mapper );
  fileMenu = fmWidget->getMenuOptions();
  tabbar->addTab(fmWidget,"Files");
  fmWidget->SetMap(mapReader->GetOutput());
  mapReader->Delete();
  
  //set up the widgets GUI
  vtWidget = new virtualToolWidget( this );
  vtWidget->setStandardWidgets( window, renderer, mapper );
  widgetMenu = vtWidget->getMenuOptions();
  widgetMenu->setEnabled( false );
  
  //set up the transfer function widget
  vtkCuda2DTransferFunction* function = vtkCuda2DTransferFunction::New();
  mapper->SetFunction( function );
  function->Delete();
  tfWidget = new transferFunctionDefinitionWidget( this, mapper->GetFunction(),
              fmWidget->GetMap()->GetDimensions()[0],
              fmWidget->GetMap()->GetDimensions()[1] );
  tfWidget->setStandardWidgets( window, renderer, mapper );
  transferFunctionMenu = tfWidget->getMenuOptions();
  transferFunctionMenu->setEnabled(false);
  tfWidget->selectImage( fmWidget->GetMap() );
  tfWidget->computeHistogram( );
  tabbar->addTab(tfWidget,"Histogram");
  connect(tfWidget,SIGNAL(SignalRed(int)),this,SLOT(changeColouring()));
  connect(tfWidget,SIGNAL(SignalGreen(int)),this,SLOT(changeColouring()));
  connect(tfWidget,SIGNAL(SignalBlue(int)),this,SLOT(changeColouring()));
  
  //set up the keyhole transfer function widget
  function = vtkCuda2DTransferFunction::New();
  mapper->SetKeyholeFunction( function );
  function->Delete();
  ktfWidget = new transferFunctionDefinitionWidget( this, mapper->GetFunction(),
              fmWidget->GetMap()->GetDimensions()[0],
              fmWidget->GetMap()->GetDimensions()[1] );
  ktfWidget->setStandardWidgets( window, renderer, mapper );
  keyholetransferFunctionMenu = ktfWidget->getMenuOptions();
  keyholetransferFunctionMenu->setEnabled(false);
  ktfWidget->selectImage( fmWidget->GetMap() );
  ktfWidget->computeHistogram( );
  tabbar->addTab(ktfWidget,"Histogram x2");

  //set up the shading widget
  shWidget = new shadingWidget(this);
  shWidget->setStandardWidgets( window, renderer, mapper );
  tabbar->addTab(shWidget,"Shading");

  //set up the device management widget
  devWidget = new deviceManagementWidget(this);
  devWidget->setStandardWidgets( window, renderer, mapper );
  tabbar->addTab(devWidget,"Device");

  //set up the stereo rendering menu
  srWidget = new stereoRenderingWidget( this );
  srWidget->setStandardWidgets( window, renderer, mapper );
  stereoRenderingMenu = srWidget->getMenuOptions();

  //set up the segmentation menu
  segWidget = new segmentationWidget(this);
  segWidget->setStandardWidgets( window, renderer, mapper );
  segmentationMenu = segWidget->getMenuOptions();
  segmentationMenu->setEnabled(false);
  
  //create the menu from the submenus
  menubar = new QMenuBar(this);
  menubar->addMenu(fileMenu);
  menubar->addMenu(stereoRenderingMenu);
  menubar->addMenu(transferFunctionMenu);
  menubar->addMenu(keyholetransferFunctionMenu);
  menubar->addMenu(segmentationMenu);
  menubar->addMenu( widgetMenu );
  menubar->show();

  //set up the screen
  screenMainLayout =  new QVBoxLayout();
  screen = new QVTKWidget(this);
  actorManipulationStyle = vtkInteractorStyleTrackballActor::New();
  cameraManipulationStyle = vtkInteractorStyleTrackballCamera::New();
  screen->SetRenderWindow(window);
  screen->GetInteractor()->SetInteractorStyle( cameraManipulationStyle );
  usingCamera = true;
  screen->setMinimumSize(512,512);
  screenMainLayout->addWidget(screen);
  mainLayout->addLayout(screenMainLayout);

  //set up the plane screens
  planeXScreen = new QVTKWidget(this);
  planeYScreen = new QVTKWidget(this);
  planeZScreen = new QVTKWidget(this);
  planeXScreen->setMinimumSize(128,128);
  planeYScreen->setMinimumSize(128,128);
  planeZScreen->setMinimumSize(128,128);
  planeXScreen->setMaximumSize(128,128);
  planeYScreen->setMaximumSize(128,128);
  planeZScreen->setMaximumSize(128,128);
  screenPlanesLayout =  new QHBoxLayout();
  screenPlanesLayout->addWidget(planeXScreen);
  screenPlanesLayout->addWidget(planeYScreen);
  screenPlanesLayout->addWidget(planeZScreen);
  screenMainLayout->addLayout(screenPlanesLayout);
  planeXScreen->SetRenderWindow(vtWidget->GetXPlaneRenderWindow());
  planeYScreen->SetRenderWindow(vtWidget->GetYPlaneRenderWindow());
  planeZScreen->SetRenderWindow(vtWidget->GetZPlaneRenderWindow());

  this->grabKeyboard();
  

  //main Layout manages the entire screen
  menuLayout->addWidget(menubar);
  menuLayout->addLayout(mainLayout);

}

transferFunctionWindowWidget::~transferFunctionWindowWidget(){

  //remove the interaction components
  this->cameraManipulationStyle->Delete();
  this->actorManipulationStyle->Delete();

  //remove the screen components
  delete screen;
  delete planeXScreen;
  delete planeYScreen;
  delete planeZScreen;

  //delete associated widgets
  delete fmWidget;
  delete vtWidget;
  delete srWidget;
  delete shWidget;
  delete segWidget;
  delete tfWidget;
  delete ktfWidget;

  //remove the menu system
  delete menubar;

  //delete the tab bar
  delete tabbar;

  //remove the layout managers
  delete screenPlanesLayout;
  delete screenMainLayout;
  delete mainLayout;
  delete menuLayout;
  
  //clear the pipeline
  window->Delete();
  renderer->Delete();
  mapper->Delete();

  //clear the Kohonen data
}

void transferFunctionWindowWidget::UpdateScreen(){
  this->screen->repaint();
  this->planeXScreen->repaint();
  this->planeYScreen->repaint();
  this->planeZScreen->repaint();
}

void transferFunctionWindowWidget::LoadedImageData(){

  //conservative update on colouring
  int scalars = fmWidget->getCurrentImage()->GetNumberOfScalarComponents();
  for(int i = 0; i < scalars; i++){
    tfWidget->SetMin(fmWidget->getCurrentImage()->GetPointData()->GetScalars()->GetRange(i)[0],i);
    tfWidget->SetMax(fmWidget->getCurrentImage()->GetPointData()->GetScalars()->GetRange(i)[1],i);
    ktfWidget->SetMin(fmWidget->getCurrentImage()->GetPointData()->GetScalars()->GetRange(i)[0],i);
    ktfWidget->SetMax(fmWidget->getCurrentImage()->GetPointData()->GetScalars()->GetRange(i)[1],i);
  }
  this->tfWidget->computeHistogram();
  this->ktfWidget->computeHistogram();
  
  //inform the virtual tool handler
  vtWidget->selectImage( fmWidget->getCurrentImage() );

  //if this is the first image, enable the menu bars
  if( fmWidget->getNumFrames() == 1 ){
    transferFunctionMenu->setEnabled(true);
    keyholetransferFunctionMenu->setEnabled(true);
    segmentationMenu->setEnabled(true);
    widgetMenu->setEnabled(true);
    renderer->ResetCamera();
  }

  this->UpdateScreen();

}

void transferFunctionWindowWidget::changeTab(){

  QWidget* currTab = tabbar->currentWidget();

  //change the width of the tab bar according to which tab is selected
  //if no valid tab is found, don't change the size
  if(currTab == fmWidget){
    tabbar->setMinimumWidth(100);
    tabbar->setMaximumWidth(250);
  }else if(currTab == tfWidget){
    tabbar->setMinimumWidth(tfWidget->getHistoSize(1) + 80 );
    tabbar->setMaximumWidth(tfWidget->getHistoSize(1) + 80 );
  }else if(currTab == ktfWidget){
    tabbar->setMinimumWidth(ktfWidget->getHistoSize(1) + 80 );
    tabbar->setMaximumWidth(ktfWidget->getHistoSize(1) + 80 );
  }else if(currTab == shWidget){
    tabbar->setMinimumWidth(100);
    tabbar->setMaximumWidth(250);
  }else{
    std::cerr << "Tab error: cannot find tab." << std::endl;
    return;
  }

  //if the window is still visible, redraw all the screens (just in case)
  if( this->isVisible() ) this->UpdateScreen();

}

void transferFunctionWindowWidget::keyPressEvent(QKeyEvent* e){

  if( e->key() == Qt::Key::Key_Alt ){
    screen->GetInteractor()->SetInteractorStyle( actorManipulationStyle );
    screen->GetInteractor()->ReInitialize();
  }
  tfWidget->keyPressEvent(e);
  ktfWidget->keyPressEvent(e);
}

void transferFunctionWindowWidget::keyReleaseEvent(QKeyEvent* e){

  if( e->key() == Qt::Key::Key_Alt ){
    screen->GetInteractor()->SetInteractorStyle( cameraManipulationStyle );
    screen->GetInteractor()->ReInitialize();
  }
  tfWidget->keyReleaseEvent(e);
  ktfWidget->keyReleaseEvent(e);
}

int transferFunctionWindowWidget::GetRComponent(){
  return this->tfWidget->GetRed();
}

int transferFunctionWindowWidget::GetGComponent(){
  return this->tfWidget->GetGreen();
}

int transferFunctionWindowWidget::GetBComponent(){
  return this->tfWidget->GetBlue();
}

double transferFunctionWindowWidget::GetRMax(){
  return this->tfWidget->GetRMax();
}

double transferFunctionWindowWidget::GetGMax(){
  return this->tfWidget->GetGMax();
}

double transferFunctionWindowWidget::GetBMax(){
  return this->tfWidget->GetBMax();
}

double transferFunctionWindowWidget::GetRMin(){
  return this->tfWidget->GetRMin();
}

double transferFunctionWindowWidget::GetGMin(){
  return this->tfWidget->GetGMin();
}

double transferFunctionWindowWidget::GetBMin(){
  return this->tfWidget->GetBMin();
}

void transferFunctionWindowWidget::changeColouring(){
  this->vtWidget->selectImage(this->fmWidget->getCurrentImage());
  this->screen->repaint();
}