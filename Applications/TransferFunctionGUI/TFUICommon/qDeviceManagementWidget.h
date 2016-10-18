/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef DEVICEMANAGEMENTWIDGET
#define DEVICEMANAGEMENTWIDGET

#include "TFUICommonExport.h"

#include <QWidget>

class QSlider;
class qTransferFunctionWindowWidgetInterface;
class vtkCudaVolumeMapper;
class vtkRenderWindow;
class vtkRenderer;

class TFUICommonExport qDeviceManagementWidget : public QWidget
{
  Q_OBJECT

public:
  qDeviceManagementWidget( qTransferFunctionWindowWidgetInterface* parent );
  ~qDeviceManagementWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );

protected slots:
  //shading related slots
  void changeDevice(int d);

protected:
  qTransferFunctionWindowWidgetInterface* parent;

  vtkCudaVolumeMapper* caster;
  QSlider* device;
};

#endif