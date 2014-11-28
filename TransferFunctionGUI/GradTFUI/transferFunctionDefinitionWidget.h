#ifndef TRANSFERFUNCTIONDEFINITIONWIDGET
#define TRANSFERFUNCTIONDEFINITIONWIDGET

#include <QObject>
#include <QWidget>
#include <QMenu>
#include <QSlider>

#include <list>

//include files for the transfer function objects
#include "vtkCudaFunctionPolygon.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkCuda2DVolumeMapper.h"
#include "vtkImageData.h"

class transferFunctionWindowWidget;
#include "transferFunctionWindowWidget.h"
class HistogramHolderDefault;
#include "HistogramHolderDefault.h"

class transferFunctionDefinitionWidget : public QWidget
{
	Q_OBJECT
public:
	
	static const unsigned int HISTOSIZE = 184;

	transferFunctionDefinitionWidget( transferFunctionWindowWidget* parent, vtkCuda2DTransferFunction* f );
	~transferFunctionDefinitionWidget();
	void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCuda2DVolumeMapper* caster );

	QMenu* getMenuOptions();
	unsigned int getHistoSize();
	vtkCuda2DTransferFunction* getTransferFunction();
	
	void selectImage(vtkImageData*);
	void repaintHistograms();
	void keyPressEvent(QKeyEvent* e); 
	void keyReleaseEvent(QKeyEvent* e); 

public slots:
	void computeHistogram();
	void updateFunction();

private slots:
	
	//histogram related slots
	void computeZoomHistogram();
	void selectZoomRegion();
	void viewAllObjects();
	void viewOneObject();

	//transfer function related slots
	void updateFunctionShading();
	void selectFunctionObject();
	void addFunctionObject();
	void removeFunctionObject();
	void setObjectProperties();
	void saveTransferFunction();
	void loadTransferFunction();

private:
	
	void addFunctionObject(vtkCudaFunctionPolygon* object);
	void removeFunctionObject(vtkCudaFunctionPolygon* object);
	char* getHistogram(vtkImageData* data, float& retIntensityLow, float& retIntensityHigh, float& retGradientLow, float& retGradientHigh, bool setSize);

	void setupMenu();

	//parent window useful for pushing updates
	transferFunctionWindowWidget* parent;
	
	//pipeline pieces
	vtkRenderWindow* window;
	vtkRenderer* renderer;
	vtkCuda2DVolumeMapper* mapper;
	vtkImageData* data;
	vtkCuda2DTransferFunction* function;
	std::list<vtkCudaFunctionPolygon*> functionObjects;
	vtkCudaFunctionPolygon* currObject;
	
	//histogram variables
	float maxGradient;
	float minGradient;
	float maxIntensity;
	float minIntensity;
	HistogramHolderDefault* histogramHolder;
	HistogramHolderDefault* zoomHistogramHolder;
	QImage* histogram;
	QImage* zoomHistogram;
	QSlider* zoomLeft;
	QSlider* zoomRight;
	QSlider* zoomUp;
	QSlider* zoomDown;

	//shading variables
	QSlider* opacityShader;
	QSlider* ambientShader;
	QSlider* diffuseShader;
	QSlider* specularShader;
	QSlider* specularPowerShader;

	//function object variables
	short maxClassifier;
	void recalculateMaxClassifier();
	QListWidget* objectList;

	//transfer function menu variables
	QMenu* transferFunctionMenu;

};

#endif