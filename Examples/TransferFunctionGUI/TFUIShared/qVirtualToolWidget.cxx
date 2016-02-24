#include "qTransferFunctionWindowWidgetInterface.h"
#include "qVirtualToolWidget.h"
#include "vtkActor.h"
#include "vtkActor2D.h"
#include "vtkBoxWidget.h"
#include "vtkColorTransferFunction.h"
#include "vtkCommand.h"
#include "vtkCudaVolumeMapper.h"
#include "vtkImageData.h"
#include "vtkImageExtractComponents.h"
#include "vtkImageHackedPlaneWidget.h"
#include "vtkImageMapToColors.h"
#include "vtkImageMapper.h"
#include "vtkPlanes.h"
#include "vtkPolyDataMapper.h"
#include "vtkPolyDataReader.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkTransform.h"
#include <QColorDialog>
#include <QFileDialog>
#include <QInputDialog>
#include <QMenu>
#include <QAction>
#include <vtkVersion.h>

// ---------------------------------------------------------------------------------------
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

//---------------------------------------------------------------------------------
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
    if (this->Mapper)
    {
      vtkPlanes *planes = vtkPlanes::New();
      widget->GetPlanes(planes);
      this->Mapper->SetClippingPlanes(planes);
      planes->Delete();
    }
  }
  void SetMapper(vtkCudaVolumeMapper* m)
  {
    this->Mapper = m;
  }

protected:
  vtkClippingBoxWidgetCallback()
  {
    this->Mapper = 0;
  }

  vtkCudaVolumeMapper *Mapper;
};

//---------------------------------------------------------------------------------
class vtkKeyholeBoxWidgetCallback : public vtkCommand
{
public:
  static vtkKeyholeBoxWidgetCallback *New()
  {
    return new vtkKeyholeBoxWidgetCallback;
  }
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
  {
    this->Mapper = m;
  }

protected:
  vtkKeyholeBoxWidgetCallback()
  {
    this->Mapper = 0;
  }

  vtkCudaVolumeMapper *Mapper;
};

// ---------------------------------------------------------------------------------------
qVirtualToolWidget::qVirtualToolWidget( qTransferFunctionWindowWidgetInterface* p ) :
  QWidget(p)
{
  Parent = p;
  Window = 0;
  Renderer = 0;
  ClippingPlanes = 0;
  KeyholePlanes = 0;
  xPlaneReslice = 0;
  yPlaneReslice = 0;
  zPlaneReslice = 0;

  WidgetMenu = new QMenu("&Virtual Tools",this);
  AvailableWidgetsMenu = new QMenu("&Current Tools",this);
  setupMenu();
}

//---------------------------------------------------------------------------------
qVirtualToolWidget::~qVirtualToolWidget()
{

  //remove any added actions
  for(std::vector<QAction*>::iterator it = AvailableWidgetsVisibility.begin(); it != AvailableWidgetsVisibility.end(); it++)
  {
    delete *it;
  }
  for(std::vector<QAction*>::iterator it = AvailableWidgetsReset.begin(); it != AvailableWidgetsReset.end(); it++)
  {
    delete *it;
  }
  for(std::vector<QMenu*>::iterator it = AvailableWidgetMenus.begin(); it != AvailableWidgetMenus.end(); it++)
  {
    delete *it;
  }
  AvailableWidgetMenus.clear();
  AvailableWidgetsVisibility.clear();
  AvailableWidgetsReset.clear();
  AvailableWidgetStatus.clear();

  //delete plane pipeline elements
  WindowXPlane->Delete();
  WindowYPlane->Delete();
  WindowZPlane->Delete();
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
  ClippingPlanes->Delete();
  KeyholePlanes->Delete();

  //clear the nonstandard virtual tools
  for(std::vector<vtkActor*>::iterator it = VirtualToolActors.begin();
      it != VirtualToolActors.end(); it++)
  {
    (*it)->Delete();
  }
  VirtualToolActors.clear();
  for(std::vector<vtkPolyDataMapper*>::iterator it = VirtualToolMappers.begin();
      it != VirtualToolMappers.end(); it++)
  {
    (*it)->Delete();
  }
  VirtualToolMappers.clear();
  for(std::vector<vtkPolyDataReader*>::iterator it = VirtualToolReaders.begin();
      it != VirtualToolReaders.end(); it++)
  {
    (*it)->Delete();
  }
  VirtualToolReaders.clear();
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCudaVolumeMapper* c )
{
  //load up the shared bits of the pipeline
  this->Window = w;
  this->Renderer = r;
  this->Mapper = c;

  //declare plane pipeline elements
  WindowXPlane = vtkRenderWindow::New();
  WindowYPlane = vtkRenderWindow::New();
  WindowZPlane = vtkRenderWindow::New();
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
#if (VTK_MAJOR_VERSION < 6)
  xPlaneMapper->SetInput(xPlaneReslice->GetResliceOutput());
  yPlaneMapper->SetInput(yPlaneReslice->GetResliceOutput());
  zPlaneMapper->SetInput(zPlaneReslice->GetResliceOutput());
#else
  xPlaneMapper->SetInputConnection(xPlaneReslice->GetResliceOutputPort());
  yPlaneMapper->SetInputConnection(yPlaneReslice->GetResliceOutputPort());
  zPlaneMapper->SetInputConnection(zPlaneReslice->GetResliceOutputPort());
#endif
  xPlaneActor->SetMapper(xPlaneMapper);
  yPlaneActor->SetMapper(yPlaneMapper);
  zPlaneActor->SetMapper(zPlaneMapper);
  xPlaneRenderer->AddActor(xPlaneActor);
  yPlaneRenderer->AddActor(yPlaneActor);
  zPlaneRenderer->AddActor(zPlaneActor);
  WindowXPlane->AddRenderer(xPlaneRenderer);
  WindowYPlane->AddRenderer(yPlaneRenderer);
  WindowZPlane->AddRenderer(zPlaneRenderer);
  xPlaneReslice->SetInteractor( Window->GetInteractor() );
  yPlaneReslice->SetInteractor( Window->GetInteractor() );
  zPlaneReslice->SetInteractor( Window->GetInteractor() );
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
  xPlaneCommand->SetWindow(WindowXPlane);
  vtkPlaneWidgetCallback* yPlaneCommand = vtkPlaneWidgetCallback::New();
  yPlaneCommand->SetWindow(WindowYPlane);
  vtkPlaneWidgetCallback* zPlaneCommand = vtkPlaneWidgetCallback::New();
  zPlaneCommand->SetWindow(WindowZPlane);
  xPlaneReslice->AddObserver(vtkCommand::InteractionEvent, xPlaneCommand);
  yPlaneReslice->AddObserver(vtkCommand::InteractionEvent, yPlaneCommand);
  zPlaneReslice->AddObserver(vtkCommand::InteractionEvent, zPlaneCommand);
  xPlaneCommand->Delete();
  yPlaneCommand->Delete();
  zPlaneCommand->Delete();

  //set up clipping planes
  ClippingPlanes = vtkBoxWidget::New();
  ClippingPlanes->SetInteractor(Window->GetInteractor());
  ClippingPlanes->SetPlaceFactor(1.01);
  ClippingPlanes->SetDefaultRenderer(Renderer);
  ClippingPlanes->InsideOutOn();
  vtkClippingBoxWidgetCallback *clippingCallback = vtkClippingBoxWidgetCallback::New();
  clippingCallback->SetMapper(Mapper);
  ClippingPlanes->AddObserver(vtkCommand::InteractionEvent, clippingCallback);
  clippingCallback->Delete();
  ClippingPlanes->GetSelectedFaceProperty()->SetOpacity(0.05);
  ClippingPlanes->Off();

  //set up the keyhole planes
  KeyholePlanes = vtkBoxWidget::New();
  KeyholePlanes->SetInteractor(Window->GetInteractor());
  KeyholePlanes->SetPlaceFactor(1.01);
  KeyholePlanes->SetDefaultRenderer(Renderer);
  KeyholePlanes->InsideOutOn();
  vtkKeyholeBoxWidgetCallback *keyholeCallback = vtkKeyholeBoxWidgetCallback::New();
  keyholeCallback->SetMapper(Mapper);
  KeyholePlanes->AddObserver(vtkCommand::InteractionEvent, keyholeCallback);
  keyholeCallback->Delete();
  KeyholePlanes->GetSelectedFaceProperty()->SetOpacity(0.05);
  KeyholePlanes->Off();
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::setupMenu()
{
  QAction* newVTKFileMenuOption = new QAction("Add VTK Virtual Tool",this);
  connect(newVTKFileMenuOption, SIGNAL(triggered()), this, SLOT(addVTKFile()) );

  //add the first two main widgets (clipping box, keyhole box and ortho planes) and connect them
  this->NumberOfVirtualTools = 3;

  //clipping planes
  QMenu* clippingPlanesMenuOption = new QMenu("Clipping Planes",this);
  QAction* clippingPlanesToggleMenuOption = new QAction("Toggle Visibility",this);
  connect(clippingPlanesToggleMenuOption, SIGNAL(triggered()), this, SLOT(toggleVirtualTool()) );
  QAction* clippingPlanesResetMenuOption = new QAction("Reset Position",this);
  connect(clippingPlanesResetMenuOption, SIGNAL(triggered()), this, SLOT(resetVirtualTool()) );
  clippingPlanesMenuOption->addAction(clippingPlanesToggleMenuOption);
  clippingPlanesMenuOption->addAction(clippingPlanesResetMenuOption);
  AvailableWidgetStatus.push_back(0);
  AvailableWidgetsMenu->addMenu(clippingPlanesMenuOption);
  AvailableWidgetMenus.push_back(clippingPlanesMenuOption);
  AvailableWidgetsVisibility.push_back(clippingPlanesToggleMenuOption);
  AvailableWidgetsReset.push_back(clippingPlanesResetMenuOption);

  //keyhole planes
  QMenu* keyholePlanesMenuOption = new QMenu("Keyhole Planes",this);
  QAction* keyholePlanesToggleMenuOption = new QAction("Toggle Visibility",this);
  connect(keyholePlanesToggleMenuOption, SIGNAL(triggered()), this, SLOT(toggleVirtualTool()) );
  QAction* keyholePlanesResetMenuOption = new QAction("Reset Position",this);
  connect(keyholePlanesResetMenuOption, SIGNAL(triggered()), this, SLOT(resetVirtualTool()) );
  keyholePlanesMenuOption->addAction(keyholePlanesToggleMenuOption);
  keyholePlanesMenuOption->addAction(keyholePlanesResetMenuOption);
  AvailableWidgetStatus.push_back(0);
  AvailableWidgetsMenu->addMenu(keyholePlanesMenuOption);
  AvailableWidgetMenus.push_back(keyholePlanesMenuOption);
  AvailableWidgetsVisibility.push_back(keyholePlanesToggleMenuOption);
  AvailableWidgetsReset.push_back(keyholePlanesResetMenuOption);

  //orthoplanes
  QMenu* orthoPlanesMenuOption = new QMenu("Orthogonal Planes",this);
  QAction* orthoPlanesToggleMenuOption = new QAction("Toggle Visibility",this);
  connect(orthoPlanesToggleMenuOption, SIGNAL(triggered()), this, SLOT(toggleVirtualTool()) );
  QAction* orthoPlanesResetMenuOption = new QAction("Reset Position",this);
  connect(orthoPlanesResetMenuOption, SIGNAL(triggered()), this, SLOT(resetVirtualTool()) );
  AvailableWidgetMenus.push_back(orthoPlanesMenuOption);
  AvailableWidgetsVisibility.push_back(orthoPlanesToggleMenuOption);
  AvailableWidgetsReset.push_back(orthoPlanesResetMenuOption);
  AvailableWidgetStatus.push_back(0);
  orthoPlanesMenuOption->addAction(orthoPlanesToggleMenuOption);
  orthoPlanesMenuOption->addAction(orthoPlanesResetMenuOption);
  AvailableWidgetsMenu->addMenu(orthoPlanesMenuOption);
  AvailableWidgetsMenu->addSeparator();

  WidgetMenu->addAction(newVTKFileMenuOption);
  WidgetMenu->addSeparator();
  WidgetMenu->addMenu(AvailableWidgetsMenu);
}

//---------------------------------------------------------------------------------
QMenu* qVirtualToolWidget::getMenuOptions()
{
  return WidgetMenu;
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::changeVirtualToolColour()
{
  //find the appropriate sender
  QAction* sender = dynamic_cast<QAction*>(QObject::sender());
  unsigned int tool = 0;
  for(std::vector<QAction*>::iterator it = AvailableWidgetsColour.begin(); it != AvailableWidgetsColour.end(); it++)
  {
    if( (*it) == sender )
    {
      break;
    }
    tool++;
  }
  if(tool >= NumberOfVirtualTools - 3)
  {
    return;
  }

  //request a colour
  QColor org;
  org.setHsl(0,255,255);
  Parent->releaseKeyboard();
  QColor colour = QColorDialog::getColor(org,this,"TF Colour",QColorDialog::ShowAlphaChannel);
  Parent->grabKeyboard();

  //apply the value to the virtual tool
  if(!colour.isValid())
  {
    return;
  }
  this->changeCustomVirtualToolColour( tool, colour.redF(), colour.greenF(), colour.blueF() );
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::addVTKFile()
{
  //find the requisite filename
  QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath(),"VTK Polydata Files (*.vtk)" );

  //if we cancel the window
  if(filename.isNull())
  {
    return;
  }

  //add image to the model
  bool result = this->addVTKFile(filename.toStdString());

  //if an error occured, print message and do not continue
  if(result)
  {
    std::cerr << "Could not load tool " << filename.toStdString() << "." << std::endl;
    return;
  }

  //query for a tool name (defaulting to "Tool #")
  QString toolname = QString::Null();
  Parent->releaseKeyboard();
  while(toolname.isNull())
  {
    toolname = QInputDialog::getText(this,"Add Virtual Tools","Enter tool name.",QLineEdit::Normal,"New Tool");
  }
  Parent->grabKeyboard();

  //add the tool to the menu
  NumberOfVirtualTools++;

  QMenu* newToolMenu = new QMenu(toolname,this);
  QAction* newToolMenuVisibilityOption = new QAction("Toggle Visibility", this);
  connect(newToolMenuVisibilityOption, SIGNAL(triggered()), this, SLOT(toggleVirtualTool()) );
  QAction* newToolMenuChangeColourOption = new QAction("Change Colour", this);
  connect(newToolMenuChangeColourOption, SIGNAL(triggered()), this, SLOT(changeVirtualToolColour()) );

  AvailableWidgetsColour.push_back( newToolMenuChangeColourOption );
  AvailableWidgetsVisibility.push_back(newToolMenuVisibilityOption);
  AvailableWidgetStatus.push_back(1);

  newToolMenu->addAction( newToolMenuVisibilityOption );
  newToolMenu->addAction( newToolMenuChangeColourOption );
  AvailableWidgetsMenu->addMenu(newToolMenu);
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::toggleVirtualTool()
{

  //find the appropriate sender
  QAction* sender = dynamic_cast<QAction*>(QObject::sender());
  unsigned int tool = 0;
  for(std::vector<QAction*>::iterator it = AvailableWidgetsVisibility.begin(); it != AvailableWidgetsVisibility.end(); it++)
  {
    if( (*it) == sender )
    {
      break;
    }
    tool++;
  }
  if(tool >= NumberOfVirtualTools)
  {
    return;
  }

  //toggle the tool status
  unsigned int status = AvailableWidgetStatus[tool];
  AvailableWidgetStatus[tool] = (status==0) ? 1 : 0;

  //if the tool == 0, this is the clipping planes
  //       tool == 1, this is the keyhole planes
  //       tool == 2, this is the orthogonal planes
  //else, custom tool
  switch(tool)
  {
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

//---------------------------------------------------------------------------------
void qVirtualToolWidget::resetVirtualTool()
{
  //find the appropriate sender
  QAction* sender = dynamic_cast<QAction*>(QObject::sender());
  unsigned int tool = 0;
  for(std::vector<QAction*>::iterator it = AvailableWidgetsReset.begin(); it != AvailableWidgetsReset.end(); it++)
  {
    if( (*it) == sender )
    {
      break;
    }
    tool++;
  }
  if(tool >= NumberOfVirtualTools)
  {
    return;
  }

  //if the tool == 0, this is the clipping planes
  //       tool == 1, this is the keyhole planes
  //       tool == 2, this is the orthogonal planes
  //else, custom tool
  switch(tool)
  {
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

//---------------------------------------------------------------------------------
vtkRenderWindow* qVirtualToolWidget::GetXPlaneRenderWindow()
{
  return WindowXPlane;
}

//---------------------------------------------------------------------------------
vtkRenderWindow* qVirtualToolWidget::GetYPlaneRenderWindow()
{
  return WindowYPlane;
}

//---------------------------------------------------------------------------------
vtkRenderWindow* qVirtualToolWidget::GetZPlaneRenderWindow()
{
  return WindowZPlane;
}

// ---------------------------------------------------------------------------------------
void qVirtualToolWidget::selectImage(vtkImageData* image)
{
  //create extraction widget
  vtkImageExtractComponents* extractor = vtkImageExtractComponents::New();
#if (VTK_MAJOR_VERSION < 6)
  extractor->SetInput(image);
#else
  extractor->SetInputData(image);
#endif
  if(image->GetNumberOfScalarComponents()==1)
  {
    extractor->SetComponents(0);
  }
  else if(image->GetNumberOfScalarComponents()==2)
  {
    extractor->SetComponents(0,1,1);
  }
  else
  {
    extractor->SetComponents(Parent->GetRComponent(),
                             Parent->GetGComponent(),
                             Parent->GetBComponent() );
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
  if(image->GetNumberOfScalarComponents() > 2)
  {
    double minMax[6];
    minMax[0] = Parent->GetRMin();
    minMax[1] = Parent->GetRMax();
    minMax[2] = Parent->GetGMin();
    minMax[3] = Parent->GetGMax();
    minMax[4] = Parent->GetBMin();
    minMax[5] = Parent->GetBMax();
    xPlaneReslice->SetInput( extractor->GetOutput(), minMax );
    yPlaneReslice->SetInput( extractor->GetOutput(), minMax );
    zPlaneReslice->SetInput( extractor->GetOutput(), minMax );
  }
  else
  {
    xPlaneReslice->SetInput( extractor->GetOutput() );
    yPlaneReslice->SetInput( extractor->GetOutput() );
    zPlaneReslice->SetInput( extractor->GetOutput() );
  }

  //prepare the clipping planes for use
#if (VTK_MAJOR_VERSION < 6)
  ClippingPlanes->SetInput( image );
#else
  ClippingPlanes->SetInputData( image );
#endif
  ClippingPlanes->PlaceWidget();
  ClippingPlanes->EnabledOn();

  //prepare the keyhole planes for use
#if (VTK_MAJOR_VERSION < 6)
  KeyholePlanes->SetInput( image );
#else
  KeyholePlanes->SetInputData( image );
#endif
  KeyholePlanes->PlaceWidget();
  KeyholePlanes->EnabledOn();

  //clean-up
  extractor->Delete();
}

//---------------------------------------------------------------------------------
bool qVirtualToolWidget::addVTKFile(std::string filename)
{
  //copy the polydata in
  vtkPolyDataReader* vrInput = vtkPolyDataReader::New();
  vrInput->SetFileName(filename.c_str());
  vrInput->Update();
  VirtualToolReaders.push_back(vrInput);

  //create a mapper
  vtkPolyDataMapper* vrMapper = vtkPolyDataMapper::New();
#if (VTK_MAJOR_VERSION < 6)
  vrMapper->SetInput(vrInput->GetOutput());
#else
  vrMapper->SetInputConnection(vrInput->GetOutputPort());
#endif
  vrMapper->SetScalarVisibility( 0 );
  VirtualToolMappers.push_back(vrMapper);

  //create a property
  vtkProperty* vrProp = vtkProperty::New();
  vrProp->SetColor(1,1,1);
  vrProp->SetOpacity(1);
  vrProp->Modified();
  VirtualToolProperties.push_back( vrProp );

  //create an actor
  vtkActor* vrActor = vtkActor::New();
  vrActor->SetMapper(vrMapper);
  vrActor->SetProperty( vrProp );
  VirtualToolActors.push_back(vrActor);

  //add the actors to the renderers
  Renderer->AddActor(vrActor);

  return false;
}

//---------------------------------------------------------------------------------
bool qVirtualToolWidget::setCustomVirtualToolVisibility(int tool, bool b)
{
  //find the actor for the tool we are trying to modify
  std::vector<vtkActor*>::iterator it = VirtualToolActors.begin();
  while(it != VirtualToolActors.end() && tool > 0)
  {
    it++;
    tool--;
  }

  //if we can't find it, return an error
  if(it == VirtualToolActors.end())
  {
    return true;
  }

  //else, toggle whether it's attached to the renderer
  if(b)
  {
    Renderer->AddActor(*it);
  }
  else
  {
    Renderer->RemoveActor(*it);
  }

  //update the render window
  Window->Render();

  //return without error
  return false;
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::changeCustomVirtualToolColour( int tool, float r,
    float g, float b )
{
  //find the actor for the tool we are trying to modify
  std::vector<vtkActor*>::iterator it = VirtualToolActors.begin();
  std::vector<vtkProperty*>::iterator itProp = VirtualToolProperties.begin();
  while(it != VirtualToolActors.end() && tool > 0)
  {
    it++;
    itProp++;
    tool--;
  }

  //if we can't find it, return without doing anything
  if(it == VirtualToolActors.end())
  {
    return;
  }

  //give the thing a new property with the new colour
  (*itProp)->SetColor(r,g,b);
  (*itProp)->Modified();

  //update the render window
  Window->Render();
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::resetClippingPlanes()
{
  vtkTransform* resetTransform = vtkTransform::New();
  resetTransform->Identity();
  ClippingPlanes->SetTransform(resetTransform);
  resetTransform->Delete();
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::resetKeyholePlanes()
{
  vtkTransform* resetTransform = vtkTransform::New();
  resetTransform->Identity();
  KeyholePlanes->SetTransform(resetTransform);
  resetTransform->Delete();
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::resetOrthoPlanes()
{
  vtkTransform* resetTransform = vtkTransform::New();
  resetTransform->Identity();
  xPlaneReslice->SetSliceIndex(0);
  yPlaneReslice->SetSliceIndex(0);
  zPlaneReslice->SetSliceIndex(0);
  resetTransform->Delete();
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::setClippingPlanesVisibility(bool b)
{
  if(b)
  {
    ClippingPlanes->SetInteractor( Window->GetInteractor() );
    ClippingPlanes->On();
  }
  else
  {
    ClippingPlanes->Off();
  }
  Window->Render();
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::setKeyholePlanesVisibility(bool b)
{
  if(b)
  {
    KeyholePlanes->SetInteractor( Window->GetInteractor() );
    KeyholePlanes->On();
  }
  else
  {
    KeyholePlanes->Off();
  }
  Window->Render();
}

//---------------------------------------------------------------------------------
void qVirtualToolWidget::setOrthoPlanesVisibility(bool b)
{
  if(b)
  {
    xPlaneReslice->SetInteractor( Window->GetInteractor() );
    yPlaneReslice->SetInteractor( Window->GetInteractor() );
    zPlaneReslice->SetInteractor( Window->GetInteractor() );
    xPlaneReslice->On();
    yPlaneReslice->On();
    zPlaneReslice->On();
  }
  else
  {
    xPlaneReslice->Off();
    yPlaneReslice->Off();
    zPlaneReslice->Off();
  }
  Window->Render();
}