#ifndef TRANSFERFUNCTIONWINDOWWIDGET
#define TRANSFERFUNCTIONWINDOWWIDGET

//include outward definition
#include "qTransferFunctionWindowWidgetInterface.h"

//Include sub-interfaces
class qDeviceManagementWidget;
class qFileManagementWidget;
class qSegmentationWidget;
class qShadingWidget;
class qStereoRenderingWidget;
class qTransferFunctionDefinitionWidget;
class qVirtualToolWidget;
class vtkCuda2DVolumeMapper;

#include "QVTKWidget.h"
#include "vtkInteractorStyleTrackballActor.h"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include <QAction>
#include <QCheckBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QMenu>
#include <QMenuBar>
#include <QObject>
#include <QTabWidget>
#include <QThread>
#include <QVBoxLayout>
#include <QWidget>

class qTransferFunctionWindowWidget : public qTransferFunctionWindowWidgetInterface
{
  Q_OBJECT

public:
  qTransferFunctionWindowWidget(QWidget *parent = 0);
  ~qTransferFunctionWindowWidget();

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
  QMenu* ktransferFunctionMenu;
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