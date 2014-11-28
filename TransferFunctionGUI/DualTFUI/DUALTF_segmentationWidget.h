#ifndef DUALTF_segmentationWidget_H
#define DUALTF_segmentationWidget_H

#include <QObject>
#include <QWidget>
#include <QMenu>
#include <QPushButton>

#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkCudaDualImageVolumeMapper.h"

#include "DUALTF_transferFunctionWindowWidget.h"
class DUALTF_transferFunctionWindowWidget;

class DUALTF_segmentationWidget : public QWidget
{
	Q_OBJECT
public:

	DUALTF_segmentationWidget( DUALTF_transferFunctionWindowWidget* parent );
	~DUALTF_segmentationWidget();
	void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaDualImageVolumeMapper* caster );
	QMenu* getMenuOptions();

private slots:
	
	//shading related slots
	void segment();

private:
	
	void setupMenu();
	QMenu* segmentationMenu;
	QAction* segmentNowOption;

	DUALTF_transferFunctionWindowWidget* parent;
	
	vtkRenderWindow* window;
	vtkRenderer* renderer;
	vtkCudaDualImageVolumeMapper* mapper;


};

#endif