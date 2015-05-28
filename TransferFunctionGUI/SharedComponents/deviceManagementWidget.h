#ifndef DEVICEMANAGEMENTWIDGET
#define DEVICEMANAGEMENTWIDGET

#include <QWidget>
#include <QSlider>
#include "transferFunctionWindowWidgetInterface.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkCudaVolumeMapper.h"

class deviceManagementWidget : public QWidget
{
  Q_OBJECT

public:
  deviceManagementWidget( transferFunctionWindowWidgetInterface* parent );
  ~deviceManagementWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );
  
private slots:
  
  //shading related slots
  void changeDevice(int d);

private:

  transferFunctionWindowWidgetInterface* parent;

  vtkCudaVolumeMapper* caster;
  QSlider* device;
};

#endif