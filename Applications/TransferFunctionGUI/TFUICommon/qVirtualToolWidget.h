/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef VIRTUALTOOLWIDGET
#define VIRTUALTOOLWIDGET

#include "TFUICommonModule.h"

#include <QWidget>

class QAction;
class QMenu;
class qTransferFunctionWindowWidgetInterface;
class vtkActor2D;
class vtkActor;
class vtkBoxWidget;
class vtkCudaVolumeMapper;
class vtkImageData;
class vtkImageHackedPlaneWidget;
class vtkImageMapper;
class vtkPolyDataMapper;
class vtkPolyDataReader;
class vtkProperty;
class vtkRenderWindow;
class vtkRenderer;

class TFUICOMMON_EXPORT qVirtualToolWidget : public QWidget
{
  Q_OBJECT

public:
  qVirtualToolWidget( qTransferFunctionWindowWidgetInterface* parent );
  ~qVirtualToolWidget();
  QMenu* getMenuOptions( );
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );

  void selectImage(vtkImageData*);

  vtkRenderWindow* GetXPlaneRenderWindow();
  vtkRenderWindow* GetYPlaneRenderWindow();
  vtkRenderWindow* GetZPlaneRenderWindow();

protected slots:
  void addVTKFile();
  void toggleVirtualTool();
  void resetVirtualTool();
  void changeVirtualToolColour();

protected:
  void setupMenu();

  qTransferFunctionWindowWidgetInterface* Parent;

  vtkRenderWindow* Window;
  vtkRenderer* Renderer;
  vtkCudaVolumeMapper* Mapper;

  unsigned int NumberOfVirtualTools;

  //standard widgets
  vtkBoxWidget*  ClippingPlanes;
  vtkBoxWidget*  KeyholePlanes;
  vtkRenderWindow* WindowXPlane;
  vtkRenderWindow* WindowYPlane;
  vtkRenderWindow* WindowZPlane;
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
  QMenu* WidgetMenu;
  QMenu* AvailableWidgetsMenu;
  std::vector<QMenu*> AvailableWidgetMenus;
  std::vector<QAction*> AvailableWidgetsVisibility;
  std::vector<QAction*> AvailableWidgetsReset;
  std::vector<QAction*> AvailableWidgetsColour;
  std::vector<unsigned int> AvailableWidgetStatus;

  //pipeline pieces for virtual tools
  std::vector<vtkPolyDataReader*> VirtualToolReaders;
  std::vector<vtkPolyDataMapper*> VirtualToolMappers;
  std::vector<vtkProperty*> VirtualToolProperties;
  std::vector<vtkActor*> VirtualToolActors;
};

#endif