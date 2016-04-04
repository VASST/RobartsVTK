/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef SEGMENTATIONWIDGET
#define SEGMENTATIONWIDGET

#include <QWidget>

class QMenu;
class QPushButton;
class qTransferFunctionWindowWidget;
class vtkCuda2DVolumeMapper;
class vtkRenderWindow;
class vtkRenderer;

class qSegmentationWidget : public QWidget
{
  Q_OBJECT

public:
  qSegmentationWidget( qTransferFunctionWindowWidget* parent );
  ~qSegmentationWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCuda2DVolumeMapper* caster );
  QMenu* getMenuOptions();

private slots:
  //shading related slots
  void segment();

private:
  void setupMenu();
  QMenu* segmentationMenu;
  QAction* segmentNowOption;

  qTransferFunctionWindowWidget* parent;

  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCuda2DVolumeMapper* mapper;
};

#endif