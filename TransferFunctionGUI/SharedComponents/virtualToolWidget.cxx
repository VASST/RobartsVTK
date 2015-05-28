#include "virtualToolWidget.h"

#include <QFileDialog>
#include <QColorDialog>
#include <QInputDialog>

#include "vtkTransform.h"
#include "vtkProperty.h"
#include "vtkCommand.h"
#include "vtkActor2D.h"
#include "vtkImageExtractComponents.h"
#include "vtkColorTransferFunction.h"
#include "vtkImageMapToColors.h"

// ---------------------------------------------------------------------------------------
//Callbacks for the box and plane widgets

class vtkPlaneWidgetCallback : public vtkCommand {
public:
  static vtkPlaneWidgetCallback *New()
    { return new vtkPlaneWidgetCallback; }
  virtual void Execute(vtkObject *caller, unsigned long, void*){
    if(window) window->Render();
  }
  void SetWindow(vtkRenderWindow* w) 
    { this->window = w; }
private:

  vtkRenderWindow* window;
};

// Callback for moving the planes from the box widget to the mapper
class vtkClippingBoxWidgetCallback : public vtkCommand
{
public:
  static vtkClippingBoxWidgetCallback *New()
    { return new vtkClippingBoxWidgetCallback; }
  virtual void Execute(vtkObject *caller, unsigned long, void*)
    {
      vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
      if (this->Mapper)
        {
        vtkPlanes *planes = vtkPlanes::New();
        widget->GetPlanes(planes);
        this->Mapper->SetClippingPlanes(planes);
        planes->Delete();
        }
    }
  void SetMapper(vtkCudaVolumeMapper* m) 
    { this->Mapper = m; }

protected:
  vtkClippingBoxWidgetCallback() 
    { this->Mapper = 0; }

  vtkCudaVolumeMapper *Mapper;
};

// Callback for moving the planes from the box widget to the mapper
class vtkKeyholeBoxWidgetCallback : public vtkCommand
{
public:
  static vtkKeyholeBoxWidgetCallback *New()
    { return new vtkKeyholeBoxWidgetCallback; }
  virtual void Execute(vtkObject *caller, unsigned long, void*)
    {
      vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
      if (this->Mapper)
        {
        vtkPlanes *planes = vtkPlanes::New();
        widget->GetPlanes(planes);
        this->Mapper->SetKeyholePlanes(planes);
        planes->Delete();
        }
    }
  void SetMapper(vtkCudaVolumeMapper* m) 
    { this->Mapper = m; }

protected:
  vtkKeyholeBoxWidgetCallback() 
    { this->Mapper = 0; }

  vtkCudaVolumeMapper *Mapper;
};

// ---------------------------------------------------------------------------------------
// Construction and destruction code

virtualToolWidget::virtualToolWidget( transferFunctionWindowWidgetInterface* p ) :
  QWidget(p)
{
  parent = p;
  window = 0;
  renderer = 0;
  clippingPlanes = 0;
  keyholePlanes = 0;
  xPlaneReslice = 0;
  yPlaneReslice = 0;
  zPlaneReslice = 0;
  
  widgetMenu = new QMenu("&Virtual Tools",this);
  availableWidgetsMenu = new QMenu("&Current Tools",this);
  setupMenu();
}

virtualToolWidget::~virtualToolWidget(){
  
  //remove any added actions
  for(std::vector<QAction*>::iterator it = availableWidgetsVisibility.begin(); it != availableWidgetsVisibility.end(); it++){
    delete *it;
  }
  for(std::vector<QAction*>::iterator it = availableWidgetsReset.begin(); it != availableWidgetsReset.end(); it++){
    delete *it;
  }
  for(std::vector<QMenu*>::iterator it = availableWidgetMenus.begin(); it != availableWidgetMenus.end(); it++){
    delete *it;
  }
  availableWidgetMenus.clear();
  availableWidgetsVisibility.clear();
  availableWidgetsReset.clear();
  availableWidgetStatus.clear();

  //delete plane pipeline elements
  windowXPlane->Delete();
  windowYPlane->Delete();
  windowZPlane->Delete();
  xPlaneReslice->Delete();
  yPlaneReslice->Delete();
  zPlaneReslice->Delete();
  xPlaneMapper->Delete();
  yPlaneMapper->Delete();
  zPlaneMapper->Delete();
  xPlaneActor->Delete();
  yPlaneActor->Delete();
  zPlaneActor->Delete();
  xPlaneRenderer->Delete();
  yPlaneRenderer->Delete();
  zPlaneRenderer->Delete();
  clippingPlanes->Delete();
  keyholePlanes->Delete();

  //clear the nonstandard virtual tools
  for(std::vector<vtkActor*>::iterator it = virtualToolActors.begin();
    it != virtualToolActors.end(); it++){
    (*it)->Delete();
  }
  virtualToolActors.clear();
  for(std::vector<vtkPolyDataMapper*>::iterator it = virtualToolMappers.begin();
    it != virtualToolMappers.end(); it++){
    (*it)->Delete();
  }
  virtualToolMappers.clear();
  for(std::vector<vtkPolyDataReader*>::iterator it = virtualToolReaders.begin();
    it != virtualToolReaders.end(); it++){
    (*it)->Delete();
  }
  virtualToolReaders.clear();
}

void virtualToolWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCudaVolumeMapper* c ){
  
  //load up the shared bits of the pipeline
  this->window = w;
  this->renderer = r;
  this->mapper = c;

  //declare plane pipeline elements
  windowXPlane = vtkRenderWindow::New();
  windowYPlane = vtkRenderWindow::New();
  windowZPlane = vtkRenderWindow::New();
  xPlaneReslice = vtkImageHackedPlaneWidget::New();
  yPlaneReslice = vtkImageHackedPlaneWidget::New();
  zPlaneReslice = vtkImageHackedPlaneWidget::New();
  xPlaneMapper = vtkImageMapper::New();
  yPlaneMapper = vtkImageMapper::New();
  zPlaneMapper = vtkImageMapper::New();
  xPlaneActor = vtkActor2D::New();
  yPlaneActor = vtkActor2D::New();
  zPlaneActor = vtkActor2D::New();
  xPlaneRenderer = vtkRenderer::New();
  yPlaneRenderer = vtkRenderer::New();
  zPlaneRenderer = vtkRenderer::New();

  //set up orthogonal planes pipeline
  xPlaneMapper->SetInput(xPlaneReslice->GetResliceOutput());
  yPlaneMapper->SetInput(yPlaneReslice->GetResliceOutput());
  zPlaneMapper->SetInput(zPlaneReslice->GetResliceOutput());
  xPlaneActor->SetMapper(xPlaneMapper);
  yPlaneActor->SetMapper(yPlaneMapper);
  zPlaneActor->SetMapper(zPlaneMapper);
  xPlaneRenderer->AddActor(xPlaneActor);
  yPlaneRenderer->AddActor(yPlaneActor);
  zPlaneRenderer->AddActor(zPlaneActor);
  windowXPlane->AddRenderer(xPlaneRenderer);
  windowYPlane->AddRenderer(yPlaneRenderer);
  windowZPlane->AddRenderer(zPlaneRenderer);
  xPlaneReslice->SetInteractor( window->GetInteractor() );
  yPlaneReslice->SetInteractor( window->GetInteractor() );
  zPlaneReslice->SetInteractor( window->GetInteractor() );
  xPlaneReslice->SetPlaneOrientationToXAxes();
  yPlaneReslice->SetPlaneOrientationToYAxes();
  zPlaneReslice->SetPlaneOrientationToZAxes();
  xPlaneReslice->SetMarginSizeX(0.0);
  xPlaneReslice->SetMarginSizeY(0.0);
  yPlaneReslice->SetMarginSizeX(0.0);
  yPlaneReslice->SetMarginSizeY(0.0);
  zPlaneReslice->SetMarginSizeX(0.0);
  zPlaneReslice->SetMarginSizeY(0.0);
  xPlaneReslice->Off();
  yPlaneReslice->Off();
  zPlaneReslice->Off();
  vtkPlaneWidgetCallback* xPlaneCommand = vtkPlaneWidgetCallback::New();
  xPlaneCommand->SetWindow(windowXPlane);
  vtkPlaneWidgetCallback* yPlaneCommand = vtkPlaneWidgetCallback::New();
  yPlaneCommand->SetWindow(windowYPlane);
  vtkPlaneWidgetCallback* zPlaneCommand = vtkPlaneWidgetCallback::New();
  zPlaneCommand->SetWindow(windowZPlane);
  xPlaneReslice->AddObserver(vtkCommand::InteractionEvent, xPlaneCommand);
  yPlaneReslice->AddObserver(vtkCommand::InteractionEvent, yPlaneCommand);
  zPlaneReslice->AddObserver(vtkCommand::InteractionEvent, zPlaneCommand);
  xPlaneCommand->Delete();
  yPlaneCommand->Delete();
  zPlaneCommand->Delete();

  //set up clipping planes
  clippingPlanes = vtkBoxWidget::New();
  clippingPlanes->SetInteractor(window->GetInteractor());
  clippingPlanes->SetPlaceFactor(1.01);
  clippingPlanes->SetDefaultRenderer(renderer);
  clippingPlanes->InsideOutOn();
  vtkClippingBoxWidgetCallback *clippingCallback = vtkClippingBoxWidgetCallback::New();
  clippingCallback->SetMapper(mapper);
  clippingPlanes->AddObserver(vtkCommand::InteractionEvent, clippingCallback);
  clippingCallback->Delete();
  clippingPlanes->GetSelectedFaceProperty()->SetOpacity(0.05);
  clippingPlanes->Off();

  //set up the keyhole planes
  keyholePlanes = vtkBoxWidget::New();
  keyholePlanes->SetInteractor(window->GetInteractor());
  keyholePlanes->SetPlaceFactor(1.01);
  keyholePlanes->SetDefaultRenderer(renderer);
  keyholePlanes->InsideOutOn();
  vtkKeyholeBoxWidgetCallback *keyholeCallback = vtkKeyholeBoxWidgetCallback::New();
  keyholeCallback->SetMapper(mapper);
  keyholePlanes->AddObserver(vtkCommand::InteractionEvent, keyholeCallback);
  keyholeCallback->Delete();
  keyholePlanes->GetSelectedFaceProperty()->SetOpacity(0.05);
  keyholePlanes->Off();

}

void virtualToolWidget::setupMenu(){

  QAction* newVTKFileMenuOption = new QAction("Add VTK Virtual Tool",this);
  connect(newVTKFileMenuOption, SIGNAL(triggered()), this, SLOT(addVTKFile()) );

  //add the first two main widgets (clipping box, keyhole box and ortho planes) and connect them
  this->numberOfVirtualTools = 3;

  //clipping planes
  QMenu* clippingPlanesMenuOption = new QMenu("Clipping Planes",this);
  QAction* clippingPlanesToggleMenuOption = new QAction("Toggle Visibility",this);
  connect(clippingPlanesToggleMenuOption, SIGNAL(triggered()), this, SLOT(toggleVirtualTool()) );
  QAction* clippingPlanesResetMenuOption = new QAction("Reset Position",this);
  connect(clippingPlanesResetMenuOption, SIGNAL(triggered()), this, SLOT(resetVirtualTool()) );
  clippingPlanesMenuOption->addAction(clippingPlanesToggleMenuOption);
  clippingPlanesMenuOption->addAction(clippingPlanesResetMenuOption);
  availableWidgetStatus.push_back(0);
  availableWidgetsMenu->addMenu(clippingPlanesMenuOption);
  availableWidgetMenus.push_back(clippingPlanesMenuOption);
  availableWidgetsVisibility.push_back(clippingPlanesToggleMenuOption);
  availableWidgetsReset.push_back(clippingPlanesResetMenuOption);

  //keyhole planes
  QMenu* keyholePlanesMenuOption = new QMenu("Keyhole Planes",this);
  QAction* keyholePlanesToggleMenuOption = new QAction("Toggle Visibility",this);
  connect(keyholePlanesToggleMenuOption, SIGNAL(triggered()), this, SLOT(toggleVirtualTool()) );
  QAction* keyholePlanesResetMenuOption = new QAction("Reset Position",this);
  connect(keyholePlanesResetMenuOption, SIGNAL(triggered()), this, SLOT(resetVirtualTool()) );
  keyholePlanesMenuOption->addAction(keyholePlanesToggleMenuOption);
  keyholePlanesMenuOption->addAction(keyholePlanesResetMenuOption);
  availableWidgetStatus.push_back(0);
  availableWidgetsMenu->addMenu(keyholePlanesMenuOption);
  availableWidgetMenus.push_back(keyholePlanesMenuOption);
  availableWidgetsVisibility.push_back(keyholePlanesToggleMenuOption);
  availableWidgetsReset.push_back(keyholePlanesResetMenuOption);

  //orthoplanes
  QMenu* orthoPlanesMenuOption = new QMenu("Orthogonal Planes",this);
  QAction* orthoPlanesToggleMenuOption = new QAction("Toggle Visibility",this);
  connect(orthoPlanesToggleMenuOption, SIGNAL(triggered()), this, SLOT(toggleVirtualTool()) );
  QAction* orthoPlanesResetMenuOption = new QAction("Reset Position",this);
  connect(orthoPlanesResetMenuOption, SIGNAL(triggered()), this, SLOT(resetVirtualTool()) );
  availableWidgetMenus.push_back(orthoPlanesMenuOption);
  availableWidgetsVisibility.push_back(orthoPlanesToggleMenuOption);
  availableWidgetsReset.push_back(orthoPlanesResetMenuOption);
  availableWidgetStatus.push_back(0);
  orthoPlanesMenuOption->addAction(orthoPlanesToggleMenuOption);
  orthoPlanesMenuOption->addAction(orthoPlanesResetMenuOption);
  availableWidgetsMenu->addMenu(orthoPlanesMenuOption);
  availableWidgetsMenu->addSeparator();
  
  widgetMenu->addAction(newVTKFileMenuOption);
  widgetMenu->addSeparator();
  widgetMenu->addMenu(availableWidgetsMenu);

}

QMenu* virtualToolWidget::getMenuOptions(){
  return widgetMenu;
}

// ---------------------------------------------------------------------------------------
//Code that interacts with the slots and interface

void virtualToolWidget::changeVirtualToolColour(){
  
  //find the appropriate sender
  QAction* sender = dynamic_cast<QAction*>(QObject::sender());
  unsigned int tool = 0;
  for(std::vector<QAction*>::iterator it = availableWidgetsColour.begin(); it != availableWidgetsColour.end(); it++){
    if( (*it) == sender ) break;
    tool++;
  }
  if(tool >= numberOfVirtualTools - 3) return;
  
  //request a colour
  QColor org;
  org.setHsl(0,255,255);
  parent->releaseKeyboard();
  QColor colour = QColorDialog::getColor(org,this,"TF Colour",QColorDialog::ShowAlphaChannel);
  parent->grabKeyboard();

  //apply the value to the virtual tool
  if(!colour.isValid()) return;
  this->changeCustomVirtualToolColour( tool, colour.redF(), colour.greenF(), colour.blueF() );

}


void virtualToolWidget::addVTKFile(){
  //find the requisite filename
  QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath(),"VTK Polydata Files (*.vtk)" );

  //if we cancel the window
  if(filename.isNull()) return;

  //add image to the model
  bool result = this->addVTKFile(filename.toStdString());

  //if an error occured, print message and do not continue
  if(result){
    std::cerr << "Could not load tool " << filename.toStdString() << "." << std::endl;
    return;
  }
  
  //query for a tool name (defaulting to "Tool #")
  QString toolname = QString::Null();
  parent->releaseKeyboard();
  while(toolname.isNull()) toolname = QInputDialog::getText(this,"Add Virtual Tools","Enter tool name.",QLineEdit::Normal,"New Tool");
  parent->grabKeyboard();

  //add the tool to the menu
  numberOfVirtualTools++;

  QMenu* newToolMenu = new QMenu(toolname,this);
  QAction* newToolMenuVisibilityOption = new QAction("Toggle Visibility", this);
  connect(newToolMenuVisibilityOption, SIGNAL(triggered()), this, SLOT(toggleVirtualTool()) );
  QAction* newToolMenuChangeColourOption = new QAction("Change Colour", this);
  connect(newToolMenuChangeColourOption, SIGNAL(triggered()), this, SLOT(changeVirtualToolColour()) );
  
  availableWidgetsColour.push_back( newToolMenuChangeColourOption );
  availableWidgetsVisibility.push_back(newToolMenuVisibilityOption);
  availableWidgetStatus.push_back(1);
  
  newToolMenu->addAction( newToolMenuVisibilityOption );
  newToolMenu->addAction( newToolMenuChangeColourOption );
  availableWidgetsMenu->addMenu(newToolMenu);

}


void virtualToolWidget::toggleVirtualTool(){

  //find the appropriate sender
  QAction* sender = dynamic_cast<QAction*>(QObject::sender());
  unsigned int tool = 0;
  for(std::vector<QAction*>::iterator it = availableWidgetsVisibility.begin(); it != availableWidgetsVisibility.end(); it++){
    if( (*it) == sender ) break;
    tool++;
  }
  if(tool >= numberOfVirtualTools) return;

  //toggle the tool status
  unsigned int status = availableWidgetStatus[tool];
  availableWidgetStatus[tool] = (status==0) ? 1 : 0;

  //if the tool == 0, this is the clipping planes
  //       tool == 1, this is the keyhole planes
  //       tool == 2, this is the orthogonal planes
  //else, custom tool
  switch(tool){
    case 0:
      this->setClippingPlanesVisibility( (status==0) );
      break;
    case 1:
      this->setKeyholePlanesVisibility( (status==0) );
      break;
    case 2:
      this->setOrthoPlanesVisibility( (status==0) );
      break;
    default:
      this->setCustomVirtualToolVisibility(tool-3,(status==0));
      break;
  }
}


void virtualToolWidget::resetVirtualTool(){

  //find the appropriate sender
  QAction* sender = dynamic_cast<QAction*>(QObject::sender());
  unsigned int tool = 0;
  for(std::vector<QAction*>::iterator it = availableWidgetsReset.begin(); it != availableWidgetsReset.end(); it++){
    if( (*it) == sender ) break;
    tool++;
  }
  if(tool >= numberOfVirtualTools) return;
  
  //if the tool == 0, this is the clipping planes
  //       tool == 1, this is the keyhole planes
  //       tool == 2, this is the orthogonal planes
  //else, custom tool
  switch(tool){
    case 0:
      this->resetClippingPlanes();
      break;
    case 1:
      this->resetKeyholePlanes();
      break;
    case 2:
      this->resetOrthoPlanes();
      break;
    default:
      //TODO:
      break;
  }
}

vtkRenderWindow* virtualToolWidget::GetXPlaneRenderWindow(){
  return windowXPlane;
}

vtkRenderWindow* virtualToolWidget::GetYPlaneRenderWindow(){
  return windowYPlane;
}

vtkRenderWindow* virtualToolWidget::GetZPlaneRenderWindow(){
  return windowZPlane;
}


// ---------------------------------------------------------------------------------------
//Code that interacts with the model

void virtualToolWidget::selectImage(vtkImageData* image){

  //create extraction widget
  vtkImageExtractComponents* extractor = vtkImageExtractComponents::New();
  extractor->SetInput(image);
  if(image->GetNumberOfScalarComponents()==1){
    extractor->SetComponents(0);
  }else if(image->GetNumberOfScalarComponents()==2){
    extractor->SetComponents(0,1,1);
  }else{
    extractor->SetComponents(parent->GetRComponent(),
                 parent->GetGComponent(),
                 parent->GetBComponent() );
  }
  extractor->Modified();
  extractor->Update();

  //prepare the orthogonal planes for use
  int oldIndex[3];
  oldIndex[0] = xPlaneReslice->GetSliceIndex();
  oldIndex[1] = yPlaneReslice->GetSliceIndex();
  oldIndex[2] = zPlaneReslice->GetSliceIndex();
  xPlaneReslice->SetSliceIndex( oldIndex[0] );
  yPlaneReslice->SetSliceIndex( oldIndex[1] );
  zPlaneReslice->SetSliceIndex( oldIndex[2] );

  //adjust colour map
  if(image->GetNumberOfScalarComponents() > 2){
    double minMax[6];
    minMax[0] = parent->GetRMin();
    minMax[1] = parent->GetRMax();
    minMax[2] = parent->GetGMin();
    minMax[3] = parent->GetGMax();
    minMax[4] = parent->GetBMin();
    minMax[5] = parent->GetBMax();
    xPlaneReslice->SetInput( extractor->GetOutput(), minMax );
    yPlaneReslice->SetInput( extractor->GetOutput(), minMax );
    zPlaneReslice->SetInput( extractor->GetOutput(), minMax );
  }else{
    xPlaneReslice->SetInput( extractor->GetOutput() );
    yPlaneReslice->SetInput( extractor->GetOutput() );
    zPlaneReslice->SetInput( extractor->GetOutput() );
  }

  //prepare the clipping planes for use
  clippingPlanes->SetInput( image );
  clippingPlanes->PlaceWidget();
  clippingPlanes->EnabledOn();
  
  //prepare the keyhole planes for use
  keyholePlanes->SetInput( image );
  keyholePlanes->PlaceWidget();
  keyholePlanes->EnabledOn();

  //clean-up
  extractor->Delete();
}


bool virtualToolWidget::addVTKFile(std::string filename){
  
  //copy the polydata in
  vtkPolyDataReader* vrInput = vtkPolyDataReader::New();
  vrInput->SetFileName(filename.c_str());
  vrInput->Update();
  virtualToolReaders.push_back(vrInput);

  //create a mapper
  vtkPolyDataMapper* vrMapper = vtkPolyDataMapper::New();
  vrMapper->SetInput(vrInput->GetOutput());
  vrMapper->SetScalarVisibility( 0 );
  virtualToolMappers.push_back(vrMapper);

  //create a property
  vtkProperty* vrProp = vtkProperty::New();
  vrProp->SetColor(1,1,1);
  vrProp->SetOpacity(1);
  vrProp->Modified();
  virtualToolProperties.push_back( vrProp );

  //create an actor
  vtkActor* vrActor = vtkActor::New();
  vrActor->SetMapper(vrMapper);
  vrActor->SetProperty( vrProp );
  virtualToolActors.push_back(vrActor);

  //add the actors to the renderers
  renderer->AddActor(vrActor);

  return false;

}


bool virtualToolWidget::setCustomVirtualToolVisibility(int tool, bool b){

  //find the actor for the tool we are trying to modify
  std::vector<vtkActor*>::iterator it = virtualToolActors.begin();
  while(it != virtualToolActors.end() && tool > 0){
    it++;
    tool--;
  }

  //if we can't find it, return an error
  if(it == virtualToolActors.end()) return true;

  //else, toggle whether it's attached to the renderer
  if(b){
    renderer->AddActor(*it);
  }else{
    renderer->RemoveActor(*it);
  }

  //update the render window
  window->Render();

  //return without error
  return false;

}

void virtualToolWidget::changeCustomVirtualToolColour( int tool, float r,
                                  float g, float b ){
  //find the actor for the tool we are trying to modify
  std::vector<vtkActor*>::iterator it = virtualToolActors.begin();
  std::vector<vtkProperty*>::iterator itProp = virtualToolProperties.begin();
  while(it != virtualToolActors.end() && tool > 0){
    it++;
    itProp++;
    tool--;
  }

  //if we can't find it, return without doing anything
  if(it == virtualToolActors.end()) return;

  //give the thing a new property with the new colour
  (*itProp)->SetColor(r,g,b);
  (*itProp)->Modified();
  
  //update the render window
  window->Render();
}

void virtualToolWidget::resetClippingPlanes(){
  vtkTransform* resetTransform = vtkTransform::New();
  resetTransform->Identity();
  clippingPlanes->SetTransform(resetTransform);
  resetTransform->Delete();
}

void virtualToolWidget::resetKeyholePlanes(){
  vtkTransform* resetTransform = vtkTransform::New();
  resetTransform->Identity();
  keyholePlanes->SetTransform(resetTransform);
  resetTransform->Delete();
}

void virtualToolWidget::resetOrthoPlanes(){
  vtkTransform* resetTransform = vtkTransform::New();
  resetTransform->Identity();
  xPlaneReslice->SetSliceIndex(0);
  yPlaneReslice->SetSliceIndex(0);
  zPlaneReslice->SetSliceIndex(0);
  resetTransform->Delete();
}

void virtualToolWidget::setClippingPlanesVisibility(bool b){
  if(b){
    clippingPlanes->SetInteractor( window->GetInteractor() );
    clippingPlanes->On();
  }else{
    clippingPlanes->Off();
  }
  window->Render();
}

void virtualToolWidget::setKeyholePlanesVisibility(bool b){
  if(b){
    keyholePlanes->SetInteractor( window->GetInteractor() );
    keyholePlanes->On();
  }else{
    keyholePlanes->Off();
  }
  window->Render();
}

void virtualToolWidget::setOrthoPlanesVisibility(bool b){
  if(b){
    xPlaneReslice->SetInteractor( window->GetInteractor() );
    yPlaneReslice->SetInteractor( window->GetInteractor() );
    zPlaneReslice->SetInteractor( window->GetInteractor() );
    xPlaneReslice->On();
    yPlaneReslice->On();
    zPlaneReslice->On();
  }else{
    xPlaneReslice->Off();
    yPlaneReslice->Off();
    zPlaneReslice->Off();
  }
  window->Render();
}