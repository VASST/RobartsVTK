#ifndef SHADINGWIDGET
#define SHADINGWIDGET

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

class shadingWidget : public QWidget
{
  Q_OBJECT
public:

  shadingWidget( transferFunctionWindowWidgetInterface* parent );
  ~shadingWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );

private slots:
  
  //shading related slots
  void changeShading();

private:
  
  //push commands to model (shading parameters)
  void setCelShadingConstants(float d, float a, float b);
  void setDistanceShadingConstants(float d, float a, float b);

  transferFunctionWindowWidgetInterface* parent;
  
  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCudaVolumeMapper* mapper;
  
  //for cel-shading
  QSlider* CelDarknessSlider;
  QLabel* CelDarknessSliderLabel;
  QLabel* CelDarknessSliderValue;
  QSlider* CelASlider;
  QLabel* CelASliderLabel;
  QLabel* CelASliderValue;
  QSlider* CelBSlider;
  QLabel* CelBSliderLabel;
  QLabel* CelBSliderValue;

  //for distance-shading
  QSlider* DistDarknessSlider;
  QLabel* DistDarknessSliderLabel;
  QLabel* DistDarknessSliderValue;
  QSlider* DistASlider;
  QLabel* DistASliderLabel;
  QLabel* DistASliderValue;
  QSlider* DistBSlider;
  QLabel* DistBSliderLabel;
  QLabel* DistBSliderValue;

};

#endif