#ifndef TRANSFERFUNCTIONDEFINITIONWIDGET
#define TRANSFERFUNCTIONDEFINITIONWIDGET

#include "vtkCuda2DTransferFunction.h"
#include "vtkCuda2DVolumeMapper.h"
#include "vtkCudaFunctionPolygon.h"
#include "vtkImageData.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include <QMenu>
#include <QObject>
#include <QSlider>
#include <QWidget>
#include <list>

class qTransferFunctionWindowWidget;
class QListWidget;
class qHistogramHolderLabel;

class qTransferFunctionDefinitionWidget : public QWidget
{
  Q_OBJECT

public:
  static const unsigned int HISTOSIZE = 184;

  qTransferFunctionDefinitionWidget( qTransferFunctionWindowWidget* parent, vtkCuda2DTransferFunction* f );
  ~qTransferFunctionDefinitionWidget();
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
  qTransferFunctionWindowWidget* parent;

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
  qHistogramHolderLabel* histogramHolder;
  qHistogramHolderLabel* zoomHistogramHolder;
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