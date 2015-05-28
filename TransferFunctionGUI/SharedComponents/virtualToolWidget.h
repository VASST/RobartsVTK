#ifndef VIRTUALTOOLWIDGET
#define VIRTUALTOOLWIDGET

#include <QObject>
#include <QWidget>
#include <QMenu>
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkBoxWidget.h"
#include "vtkImageHackedPlaneWidget.h"
#include "vtkPolyDataReader.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkCudaVolumeMapper.h"
#include "vtkImageMapper.h"
#include "vtkImageData.h"

#include "transferFunctionWindowWidgetInterface.h"
class transferFunctionWindowWidgetInterface;

class virtualToolWidget : public QWidget
{
  Q_OBJECT
public:

  virtualToolWidget( transferFunctionWindowWidgetInterface* parent );
  ~virtualToolWidget();
  QMenu* getMenuOptions( );
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );

  void selectImage(vtkImageData*);
  
  vtkRenderWindow* GetXPlaneRenderWindow();
  vtkRenderWindow* GetYPlaneRenderWindow();
  vtkRenderWindow* GetZPlaneRenderWindow();

private slots:
  
  void addVTKFile();
  void toggleVirtualTool();
  void resetVirtualTool();
  void changeVirtualToolColour();

private:
  
  void setupMenu();

  transferFunctionWindowWidgetInterface* parent;

  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCudaVolumeMapper* mapper;

  unsigned int numberOfVirtualTools;

  //standard widgets
  vtkBoxWidget*  clippingPlanes;
  vtkBoxWidget*  keyholePlanes;
  vtkRenderWindow* windowXPlane;
  vtkRenderWindow* windowYPlane;
  vtkRenderWindow* windowZPlane;
  vtkImageHackedPlaneWidget* xPlaneReslice;
  vtkImageHackedPlaneWidget* yPlaneReslice;
  vtkImageHackedPlaneWidget* zPlaneReslice;
  vtkImageMapper* xPlaneMapper;
  vtkImageMapper* yPlaneMapper;
  vtkImageMapper* zPlaneMapper;
  vtkActor2D* xPlaneActor;
  vtkActor2D* yPlaneActor;
  vtkActor2D* zPlaneActor;
  vtkRenderer* xPlaneRenderer;
  vtkRenderer* yPlaneRenderer;
  vtkRenderer* zPlaneRenderer;
  
  //push commands to model (widgets)
  bool addVTKFile(std::string filename);
  bool setCustomVirtualToolVisibility(int, bool);
  void changeCustomVirtualToolColour( int tool, float r, float g, float b );

  void setClippingPlanesVisibility(bool);
  void setKeyholePlanesVisibility(bool);
  void setOrthoPlanesVisibility(bool);
  void resetClippingPlanes();
  void resetKeyholePlanes();
  void resetOrthoPlanes();
  
  //interface connections for virtual tools
  QMenu* widgetMenu;
  QMenu* availableWidgetsMenu;
  std::vector<QMenu*> availableWidgetMenus;
  std::vector<QAction*> availableWidgetsVisibility;
  std::vector<QAction*> availableWidgetsReset;
  std::vector<QAction*> availableWidgetsColour;
  std::vector<unsigned int> availableWidgetStatus;

  //pipeline pieces for virtual tools
  std::vector<vtkPolyDataReader*> virtualToolReaders;
  std::vector<vtkPolyDataMapper*> virtualToolMappers;
  std::vector<vtkProperty*> virtualToolProperties;
  std::vector<vtkActor*> virtualToolActors;

};

#endif