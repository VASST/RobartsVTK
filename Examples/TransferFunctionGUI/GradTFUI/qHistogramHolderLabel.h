#ifndef qHistogramHolderLabel_H
#define qHistogramHolderLabel_H

#include <QObject>
#include <QWidget>
#include <QLabel>
#include <QImage>
#include <QPaintEvent>

//include files for the transfer function objects
class qTransferFunctionDefinitionWidget;
#include "qTransferFunctionDefinitionWidget.h"
#include "vtkCudaFunctionPolygon.h"

#include "qTransferFunctionWindowWidget.h"
class qTransferFunctionWindowWidget;

class qHistogramHolderLabel : public QLabel
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

private slots:

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
  unsigned int vertexInUse;

};

#endif