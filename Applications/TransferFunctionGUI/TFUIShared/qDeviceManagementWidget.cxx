#include "qDeviceManagementWidget.h"
#include "qTransferFunctionWindowWidgetInterface.h"
#include "vtkCudaDeviceManager.h"
#include "vtkCudaVolumeMapper.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include <QSlider>

qDeviceManagementWidget::qDeviceManagementWidget( qTransferFunctionWindowWidgetInterface* parent ) :
  QWidget(parent)
{
  this->parent = parent;
  this->device = new QSlider(Qt::Orientation::Horizontal, this);
  this->device->setMinimum(0);
  this->device->setMaximum( vtkCudaDeviceManager::Singleton()->GetNumberOfDevices() - 1 );
  connect(this->device, SIGNAL(valueChanged(int)), this, SLOT(changeDevice(int)));
  device->show();
}

qDeviceManagementWidget::~qDeviceManagementWidget()
{

}

void qDeviceManagementWidget::setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster )
{
  this->caster = caster;
}

void qDeviceManagementWidget::changeDevice(int d)
{
  this->caster->SetDevice( d, 1 );
}