/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef TRANSFERFUNCTIONDEFINITIONWIDGET
#define TRANSFERFUNCTIONDEFINITIONWIDGET

#include "TFUIGradientModule.h"
#include <QWidget>
#include <list>

class QListWidget;
class QMenu;
class QSlider;
class qHistogramHolderLabel;
class qTransferFunctionWindowWidget;
class vtkCuda2DTransferFunction;
class vtkCuda2DVolumeMapper;
class vtkCudaFunctionPolygon;
class vtkImageData;
class vtkRenderWindow;
class vtkRenderer;

class TFUIGRADIENT_EXPORT qTransferFunctionDefinitionWidget : public QWidget
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

protected slots:
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

protected:
  void addFunctionObject(vtkCudaFunctionPolygon* object);
  void removeFunctionObject(vtkCudaFunctionPolygon* object);
  char* getHistogram(vtkImageData* data, float& retIntensityLow, float& retIntensityHigh, float& retGradientLow, float& retGradientHigh, bool setSize);
  void setupMenu();

protected:
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