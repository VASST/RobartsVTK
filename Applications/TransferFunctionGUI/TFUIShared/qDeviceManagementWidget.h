#ifndef DEVICEMANAGEMENTWIDGET
#define DEVICEMANAGEMENTWIDGET

#include "TFUICommonModule.h"

#include <QWidget>

class QSlider;
class qTransferFunctionWindowWidgetInterface;
class vtkCudaVolumeMapper;
class vtkRenderWindow;
class vtkRenderer;

class TFUICOMMON_EXPORT qDeviceManagementWidget : public QWidget
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