/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef qSegmentationWidget_H
#define qSegmentationWidget_H

#include "qTransferFunctionWindowWidget.h"
#include <QWidget>

class QAction;
class QMenu;
class qTransferFunctionWindowWidget;
class vtkCudaDualImageVolumeMapper;
class vtkRenderWindow;
class vtkRenderer;

class qSegmentationWidget : public QWidget
{
  Q_OBJECT

public:
  qSegmentationWidget( qTransferFunctionWindowWidget* parent );
  ~qSegmentationWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaDualImageVolumeMapper* caster );
  QMenu* getMenuOptions();

protected slots:
  //shading related slots
  void segment();

protected:
  void setupMenu();

protected:
  QAction* segmentNowOption;
  QMenu* segmentationMenu;
  qTransferFunctionWindowWidget* parent;
  vtkCudaDualImageVolumeMapper* mapper;
  vtkRenderWindow* window;
  vtkRenderer* renderer;
};

#endif