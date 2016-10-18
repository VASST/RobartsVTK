/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef STEREORENDERINGWIDGET
#define STEREORENDERINGWIDGET

#include "TFUICommonExport.h"

#include <QWidget>

class QLabel;
class QMenu;
class QSlider;
class qTransferFunctionWindowWidgetInterface;
class vtkCudaVolumeMapper;
class vtkRenderWindow;
class vtkRenderer;

class TFUICommonExport qStereoRenderingWidget : public QWidget
{
  Q_OBJECT

public:
  qStereoRenderingWidget( qTransferFunctionWindowWidgetInterface* parent );
  ~qStereoRenderingWidget();
  QMenu* getMenuOptions();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );

protected slots:

  //stereo related slots
  void setStereoOn();
  void setStereoOff();
  void toggleEyes();

protected:
  void setupMenu();
  void setStereoEnabled(bool);
  void toggleStereoEyes();

  qTransferFunctionWindowWidgetInterface* parent;

  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCudaVolumeMapper* mapper;

  //stereo menu variables
  QMenu* stereoMenu;
  QAction* stereoOnMenuOption;
  QAction* stereoOffMenuOption;
  QAction* stereoEyeToggleMenuOption;

  unsigned int old;
};

#endif