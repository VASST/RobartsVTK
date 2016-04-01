/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef qHistogramHolderLabel_H
#define qHistogramHolderLabel_H

#include "TFUIGradientModule.h"
#include <QLabel>

class qTransferFunctionDefinitionWidget;
class qTransferFunctionWindowWidget;
class vtkCuda2DTransferFunction;
class vtkCudaFunctionPolygon;

class TFUIGRADIENT_EXPORT qHistogramHolderLabel : public QLabel
{
  Q_OBJECT

public:
  qHistogramHolderLabel(qTransferFunctionDefinitionWidget* parent, vtkCuda2DTransferFunction* f);
  ~qHistogramHolderLabel();
  void paintEvent( QPaintEvent * );
  void setObject(vtkCudaFunctionPolygon* object);
  void setSize(unsigned int size);

  void giveHistogramDimensions(float maxG,float minG, float maxI, float minI);

  void mouseMoveEvent(QMouseEvent*);
  void mousePressEvent(QMouseEvent*);
  void mouseReleaseEvent(QMouseEvent*);

  void keyPressEvent(QKeyEvent* e);
  void keyReleaseEvent(QKeyEvent* e);

  void setZoomSquare(float maxG,float minG, float maxI, float minI);
  void setAutoUpdate(bool au);

  void visualizeAllObjects(bool b);

private:
  qTransferFunctionDefinitionWidget* manager;
  vtkCuda2DTransferFunction* func;
  vtkCudaFunctionPolygon* object;

  float maxGradient;
  float minGradient;
  float minIntensity;
  float maxIntensity;
  double size;
  bool histogram;
  bool visAll;

  float zoomMaxGradient;
  float zoomMinGradient;
  float zoomMinIntensity;
  float zoomMaxIntensity;
  bool hasZoomSquare;

  //original click locations
  float origClickI;
  float origClickG;
  float centroidI;
  float centroidG;
  float rotationCentreI;
  float rotationCentreG;
  float closenessRadius;

  //booleans to interpret click by
  bool shiftHeld;
  bool ctrlHeld;
  bool translating;
  bool scaling;
  bool vertexDragging;
  bool autoUpdate;
  size_t vertexInUse;
};

#endif