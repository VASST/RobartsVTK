#ifndef qTransferFunctionDefinitionWidget_H
#define qTransferFunctionDefinitionWidget_H

#include <list>
#include <QWidget>

class qHistogramHolderLabel;
class QListWidget;
class QMenu;
class QSlider;
class qTransferFunctionWindowWidget;
class vtkCuda2DTransferFunction;
class vtkCudaDualImageVolumeMapper;
class vtkCudaFunctionPolygon;
class vtkImageData;
class vtkRenderWindow;
class vtkRenderer;

#define HISTOSIZE 200

class qTransferFunctionDefinitionWidget : public QWidget
{
  Q_OBJECT
public:

  qTransferFunctionDefinitionWidget( qTransferFunctionWindowWidget* parent, vtkCuda2DTransferFunction* function );
  ~qTransferFunctionDefinitionWidget();
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
  qTransferFunctionWindowWidget* parent;

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