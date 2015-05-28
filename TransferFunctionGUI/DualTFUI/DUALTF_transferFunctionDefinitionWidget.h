#ifndef DUALTF_transferFunctionDefinitionWidget_H
#define DUALTF_transferFunctionDefinitionWidget_H

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
#include "vtkCudaDualImageVolumeMapper.h"
#include "vtkImageData.h"

class DUALTF_transferFunctionWindowWidget;
#include "DUALTF_transferFunctionWindowWidget.h"
class DUALTF_HistogramHolderDefault;
#include "DUALTF_HistogramHolderDefault.h"

#define HISTOSIZE 200

class DUALTF_transferFunctionDefinitionWidget : public QWidget
{
  Q_OBJECT
public:

  DUALTF_transferFunctionDefinitionWidget( DUALTF_transferFunctionWindowWidget* parent, vtkCuda2DTransferFunction* function );
  ~DUALTF_transferFunctionDefinitionWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaDualImageVolumeMapper* caster);

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
  DUALTF_transferFunctionWindowWidget* parent;
  
  //pipeline pieces
  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCudaDualImageVolumeMapper* mapper;
  vtkImageData* data;
  vtkCuda2DTransferFunction* function;
  std::list<vtkCudaFunctionPolygon*> functionObjects;
  vtkCudaFunctionPolygon* currObject;
  
  //histogram variables
  float maxIntensity1;
  float minIntensity1;
  float maxIntensity2;
  float minIntensity2;
  DUALTF_HistogramHolderDefault* histogramHolder;
  DUALTF_HistogramHolderDefault* zoomHistogramHolder;
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