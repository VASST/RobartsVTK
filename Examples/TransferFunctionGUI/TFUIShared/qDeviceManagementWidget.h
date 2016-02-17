#ifndef DEVICEMANAGEMENTWIDGET
#define DEVICEMANAGEMENTWIDGET

#include <QWidget>
#include <QSlider>
#include "qTransferFunctionWindowWidgetInterface.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkCudaVolumeMapper.h"

class qDeviceManagementWidget : public QWidget
{
  Q_OBJECT

public:
  qDeviceManagementWidget( qTransferFunctionWindowWidgetInterface* parent );
  ~qDeviceManagementWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );
  
private slots:
  //shading related slots
  void changeDevice(int d);

private:
  qTransferFunctionWindowWidgetInterface* parent;

  vtkCudaVolumeMapper* caster;
  QSlider* device;
};

#endif