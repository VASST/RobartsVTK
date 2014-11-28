#ifndef STEREORENDERINGWIDGET
#define STEREORENDERINGWIDGET

#include <QObject>
#include <QWidget>
#include <QMenu>
#include <QSlider>
#include <QLabel>

#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkCudaVolumeMapper.h"

#include "transferFunctionWindowWidgetInterface.h"
class transferFunctionWindowWidgetInterface;

class stereoRenderingWidget : public QWidget
{
	Q_OBJECT
public:

	stereoRenderingWidget( transferFunctionWindowWidgetInterface* parent );
	~stereoRenderingWidget();
	QMenu* getMenuOptions();
	void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );

private slots:
	
	//stereo related slots
	void setStereoOn();
	void setStereoOff();
	void toggleEyes();

private:
	
	void setupMenu();
	void setStereoEnabled(bool);
	void toggleStereoEyes();

	transferFunctionWindowWidgetInterface* parent;
	
	vtkRenderWindow* window;
	vtkRenderer* renderer;
	vtkCudaVolumeMapper* mapper;
	
	//stereo menu variables
	QMenu* stereoMenu;
	QAction* stereoOnMenuOption;
	QAction* stereoOffMenuOption;
	QAction* stereoEyeToggleMenuOption;

	unsigned int old;

};

#endif