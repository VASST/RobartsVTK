/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef SHADINGWIDGET
#define SHADINGWIDGET

#include "TFUICommonExport.h"

#include <QWidget>

class QLabel;
class QMenu;
class QSlider;
class qTransferFunctionWindowWidgetInterface;
class vtkCudaVolumeMapper;
class vtkRenderWindow;
class vtkRenderer;

class TFUICommonExport qShadingWidget : public QWidget
{
  Q_OBJECT
public:

  qShadingWidget( qTransferFunctionWindowWidgetInterface* parent );
  ~qShadingWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );

protected slots:
  //shading related slots
  void changeShading();

protected:
  //push commands to model (shading parameters)
  void setCelShadingConstants(float d, float a, float b);
  void setDistanceShadingConstants(float d, float a, float b);

  qTransferFunctionWindowWidgetInterface* parent;
  
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