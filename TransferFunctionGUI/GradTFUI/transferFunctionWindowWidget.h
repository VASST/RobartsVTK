#ifndef TRANSFERFUNCTIONWINDOWWIDGET
#define TRANSFERFUNCTIONWINDOWWIDGET

//include outward definiton
#include "transferFunctionWindowWidgetInterface.h"

//Include sub-interfaces
#include "virtualToolWidget.h"
class virtualToolWidget;
#include "fileManagementWidget.h"
class fileManagementWidget;
#include "shadingWidget.h"
class shadingWidget;
#include "stereoRenderingWidget.h"
class stereoRenderingWidget;
#include "transferFunctionDefinitionWidget.h"
class transferFunctionDefinitionWidget;
#include "segmentationWidget.h"
class segmentationWidget;
#include "deviceManagementWidget.h"
class deviceManagementWidget;

#include <QObject>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include "QVTKWidget.h"
#include <QMenu>
#include <QMenuBar>
#include <QAction>
#include <QTabWidget>
#include <QCheckBox>

#include <QThread>
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"

#include "vtkInteractorStyleTrackballActor.h"
#include "vtkInteractorStyleTrackballCamera.h"

class transferFunctionWindowWidget : public transferFunctionWindowWidgetInterface
{
  Q_OBJECT

public:
  transferFunctionWindowWidget(QWidget *parent = 0);
  ~transferFunctionWindowWidget();

  //keyboard options
  void keyPressEvent(QKeyEvent* e);
  void keyReleaseEvent(QKeyEvent* e);
  
  void LoadedImageData();
  void UpdateScreen();

private slots:

  //tab changing slot
  void changeTab();

private:

  //shared and communally modified pipeline pieces
  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCuda2DVolumeMapper* mapper;

  //tab bar to manage multiple property options
  QMenuBar* menubar;
  QTabWidget* tabbar;

  //additional widgets and their menus
  shadingWidget* shWidget;
  fileManagementWidget* fmWidget;
  QMenu* fileMenu;
  virtualToolWidget* vtWidget;
  QMenu* widgetMenu;
  stereoRenderingWidget* srWidget;
  QMenu* stereoRenderingMenu;
  transferFunctionDefinitionWidget* tfWidget;
  QMenu* transferFunctionMenu;
  transferFunctionDefinitionWidget* ktfWidget;
  QMenu* ktransferFunctionMenu;
  segmentationWidget* segWidget;
  QMenu*segmentationMenu;
  deviceManagementWidget* devWidget;
  
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