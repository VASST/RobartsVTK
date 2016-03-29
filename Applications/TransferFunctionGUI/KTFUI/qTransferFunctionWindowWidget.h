/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef TRANSFERFUNCTIONWINDOWWIDGET
#define TRANSFERFUNCTIONWINDOWWIDGET

#include "TFUIKohonenModule.h"
#include "qTransferFunctionWindowWidgetInterface.h"

class QHBoxLayout;
class QMenu;
class QMenuBar;
class QTabWidget;
class QVBoxLayout;
class QVTKWidget;
class qDeviceManagementWidget;
class qFileManagementWidget;
class qSegmentationWidget;
class qShadingWidget;
class qStereoRenderingWidget;
class qTransferFunctionDefinitionWidget;
class qVirtualToolWidget;
class vtkCudaDualImageVolumeMapper;
class vtkInteractorStyleTrackballActor;
class vtkInteractorStyleTrackballCamera;
class vtkRenderWindow;
class vtkRenderWindowInteractor;
class vtkRenderer;

class TFUIKOHONEN_EXPORT qTransferFunctionWindowWidget : public qTransferFunctionWindowWidgetInterface
{
  Q_OBJECT

public:
  qTransferFunctionWindowWidget(QWidget* parent = 0);
  ~qTransferFunctionWindowWidget();

  //keyboard options
  void keyPressEvent(QKeyEvent* e);
  void keyReleaseEvent(QKeyEvent* e);
  
  void LoadedImageData();
  void UpdateScreen();
  
  int GetRComponent();
  int GetGComponent();
  int GetBComponent();
  double GetRMax();
  double GetGMax();
  double GetBMax();
  double GetRMin();
  double GetGMin();
  double GetBMin();

private slots:

  //tab changing slot
  void changeTab();

  void changeColouring();

private:

  //shared and communally modified pipeline pieces
  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCudaDualImageVolumeMapper* mapper;

  //tab bar to manage multiple property options
  QMenuBar* menubar;
  QTabWidget* tabbar;

  //additional widgets and their menus
  qShadingWidget* shWidget;
  qFileManagementWidget* fmWidget;
  QMenu* fileMenu;
  qVirtualToolWidget* vtWidget;
  QMenu* widgetMenu;
  qStereoRenderingWidget* srWidget;
  QMenu* stereoRenderingMenu;
  qTransferFunctionDefinitionWidget* tfWidget;
  QMenu* transferFunctionMenu;
  qTransferFunctionDefinitionWidget* ktfWidget;
  QMenu* keyholetransferFunctionMenu;
  qSegmentationWidget* segWidget;
  QMenu*segmentationMenu;
  qDeviceManagementWidget* devWidget;
  
  //screens for viewing
  QVTKWidget* screen;
  QVTKWidget* planeXScreen;
  QVTKWidget* planeYScreen;
  QVTKWidget* planeZScreen;
  QVBoxLayout* screenMainLayout;
  QHBoxLayout* screenPlanesLayout;
  vtkInteractorStyleTrackballActor* actorManipulationStyle;
  vtkInteractorStyleTrackballCamera* cameraManipulationStyle;
  bool usingCamera;

  //layout managers
  QVBoxLayout* menuLayout;
  QHBoxLayout* mainLayout;
};

#endif