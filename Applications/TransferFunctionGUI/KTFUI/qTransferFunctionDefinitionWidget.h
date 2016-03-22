#ifndef TRANSFERFUNCTIONDEFINITIONWIDGET
#define TRANSFERFUNCTIONDEFINITIONWIDGET

#include <QWidget>
#include <list>
#include <vector>

class qHistogramHolderLabel;
class QComboBox;
class QMenu;
class QSlider;
class QListWidget;
class qTransferFunctionWindowWidget;
class vtkCuda2DTransferFunction;
class vtkCudaDualImageVolumeMapper;
class vtkCudaFunctionPolygon;
class vtkImageData;
class vtkRenderWindow;
class vtkRenderer;

class qTransferFunctionDefinitionWidget : public QWidget
{
  Q_OBJECT

public:
  qTransferFunctionDefinitionWidget( qTransferFunctionWindowWidget* parent, vtkCuda2DTransferFunction* function, int s1, int s2 );
  ~qTransferFunctionDefinitionWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaDualImageVolumeMapper* caster);

  QMenu* getMenuOptions();
  unsigned int getHistoSize(int i);
  void setHistoSize(int s1, int s2);
  vtkCuda2DTransferFunction* getTransferFunction();
  
  void selectImage(vtkImageData*);
  void repaintHistograms();
  void keyPressEvent(QKeyEvent* e); 
  void keyReleaseEvent(QKeyEvent* e); 
  
  int GetRed(){return this->redChosen;}
  int GetGreen(){return this->greenChosen;}
  int GetBlue(){return this->blueChosen;}

  double GetRMax(){return this->Max[this->GetRed()];}
  double GetGMax(){return this->Max[this->GetGreen()];}
  double GetBMax(){return this->Max[this->GetBlue()];}
  double GetRMin(){return this->Min[this->GetRed()];}
  double GetGMin(){return this->Min[this->GetGreen()];}
  double GetBMin(){return this->Min[this->GetBlue()];}
  
  void SetMax(double val, int scalar);
  void SetMin(double val, int scalar);

signals:
  void SignalRed(int index);
  void SignalGreen(int index);
  void SignalBlue(int index);

public slots:
  void computeHistogram();
  void updateFunction();
  
  void setRed(int index);
  void setGreen(int index);
  void setBlue(int index);

private slots:
  // histogram related slots
  void computeZoomHistogram();
  void selectZoomRegion();
  void viewAllObjects();
  void viewOneObject();

  // transfer function related slots
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
  char* getHistogram(vtkImageData* data, float retIntensityLow, float retIntensityHigh, float retGradientLow, float retGradientHigh, bool setSize);

  void setupMenu();

  // parent window useful for pushing updates
  qTransferFunctionWindowWidget* parent;
  
  // pipeline pieces
  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCudaDualImageVolumeMapper* mapper;
  vtkImageData* data;
  vtkCuda2DTransferFunction* function;
  std::list<vtkCudaFunctionPolygon*> functionObjects;
  vtkCudaFunctionPolygon* currObject;
  
  // histogram variables
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

  // Coloring variables
  QComboBox* redVariable;
  QComboBox* greenVariable;
  QComboBox* blueVariable;
  int redChosen;
  int blueChosen;
  int greenChosen;
  int scalars;
  
  float* Max;
  float* Min;

  // shading variables
  QSlider* opacityShader;
  QSlider* ambientShader;
  QSlider* diffuseShader;
  QSlider* specularShader;
  QSlider* specularPowerShader;

  // function object variables
  short maxClassifier;
  void recalculateMaxClassifier();
  QListWidget* objectList;

  // transfer function menu variables
  QMenu* transferFunctionMenu;
  
  int HISTOSIZE1;
  int HISTOSIZE2;

};

#endif