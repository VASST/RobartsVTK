#include "deviceManagementWidget.h"
#include "vtkCudaDeviceManager.h"

deviceManagementWidget::deviceManagementWidget( transferFunctionWindowWidgetInterface* parent ) :
	QWidget(parent)
{
	this->parent = parent;
	this->device = new QSlider(Qt::Orientation::Horizontal, this);
	this->device->setMinimum(0);
	this->device->setMaximum( vtkCudaDeviceManager::Singleton()->GetNumberOfDevices() - 1 );
	connect(this->device, SIGNAL(valueChanged(int)), this, SLOT(changeDevice(int)));
	device->show();
}

deviceManagementWidget::~deviceManagementWidget(){

}

void deviceManagementWidget::setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster ){
	this->caster = caster;
}

void deviceManagementWidget::changeDevice(int d){
	this->caster->SetDevice( d, 1 );
}