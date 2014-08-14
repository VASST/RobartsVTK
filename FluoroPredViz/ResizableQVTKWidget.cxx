#include "ResizableQVTKWidget.h"
#include "vtkRenderWindow.h"

ResizableQVTKWidget::ResizableQVTKWidget(QWidget* p):
QVTKWidget(p), ready(false) {}


void ResizableQVTKWidget::resizeEvent(QResizeEvent * e ){
	if( ready && this->GetRenderWindow() ) this->GetRenderWindow()->Render();
	this->QVTKWidget::resizeEvent(e);
}

void ResizableQVTKWidget::changeEvent ( QEvent * e ){
	if( ready && this->GetRenderWindow() ) this->GetRenderWindow()->Render();
	this->QVTKWidget::changeEvent(e);
}

void ResizableQVTKWidget::paintEvent ( QPaintEvent * e ){
	if( ready && this->GetRenderWindow() ) this->GetRenderWindow()->Render();
	this->QVTKWidget::paintEvent(e);
}