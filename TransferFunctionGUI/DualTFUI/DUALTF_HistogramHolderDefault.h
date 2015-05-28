#ifndef DUALTF_HistogramHolderDefault_H
#define DUALTF_HistogramHolderDefault_H

#include <QObject>
#include <QWidget>
#include <QLabel>
#include <QImage>
#include <QPaintEvent>

//include files for the transfer function objects
class DUALTF_transferFunctionDefinitionWidget;
#include "DUALTF_transferFunctionDefinitionWidget.h"
#include "vtkCudaFunctionPolygon.h"

#include "DUALTF_transferFunctionWindowWidget.h"
class DUALTF_transferFunctionWindowWidget;

class DUALTF_HistogramHolderDefault : public QLabel
{
  Q_OBJECT

public:
  DUALTF_HistogramHolderDefault(DUALTF_transferFunctionDefinitionWidget* parent, vtkCuda2DTransferFunction* f);
  ~DUALTF_HistogramHolderDefault();
  void paintEvent( QPaintEvent * );
  void setObject(vtkCudaFunctionPolygon* object);
  void setSize(unsigned int size);

  void giveHistogramDimensions(float maxI1,float minI1, float maxI2, float minI2);

  void mouseMoveEvent(QMouseEvent*);
  void mousePressEvent(QMouseEvent*);
  void mouseReleaseEvent(QMouseEvent*);
  
  void keyPressEvent(QKeyEvent* e);
  void keyReleaseEvent(QKeyEvent* e);

  void setZoomSquare(float maxI1,float minI1, float maxI2, float minI2);
  void setAutoUpdate(bool au);

  void visualizeAllObjects(bool b);

private slots:

private:

  DUALTF_transferFunctionDefinitionWidget* manager;
  vtkCuda2DTransferFunction* func;
  vtkCudaFunctionPolygon* object;
  
  float minIntensity1;
  float maxIntensity1;
  float minIntensity2;
  float maxIntensity2;
  double size;
  bool histogram;
  bool visAll;

  float zoomMinIntensity1;
  float zoomMaxIntensity1;
  float zoomMinIntensity2;
  float zoomMaxIntensity2;
  bool hasZoomSquare;

  //original click locations
  float origClickI1;
  float origClickI2;
  float centroidI1;
  float centroidI2;
  float rotationCentreI1;
  float rotationCentreI2;
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