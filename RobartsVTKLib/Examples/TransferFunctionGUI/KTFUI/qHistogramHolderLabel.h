#ifndef qHistogramHolderLabel_H
#define qHistogramHolderLabel_H

#include <QLabel>

class qTransferFunctionDefinitionWidget;
class qTransferFunctionDefinitionWidget;
class qTransferFunctionWindowWidget;
class qTransferFunctionWindowWidget;
class vtkCuda2DTransferFunction;
class vtkCudaFunctionPolygon;
class vtkCudaFunctionPolygon;

class qHistogramHolderLabel : public QLabel
{
  Q_OBJECT

public:
  qHistogramHolderLabel(qTransferFunctionDefinitionWidget* parent, vtkCuda2DTransferFunction* f);
  ~qHistogramHolderLabel();
  void paintEvent( QPaintEvent * );
  void setObject(vtkCudaFunctionPolygon* object);
  void setSize(unsigned int size1, unsigned int size2);

  void giveHistogramDimensions(float maxI1,float minI1, float maxI2, float minI2);

  void mouseMoveEvent(QMouseEvent*);
  void mousePressEvent(QMouseEvent*);
  void mouseReleaseEvent(QMouseEvent*);
  
  void keyPressEvent(QKeyEvent* e);
  void keyReleaseEvent(QKeyEvent* e);

  void setZoomSquare(float maxI1,float minI1, float maxI2, float minI2);
  void setAutoUpdate(bool au);

  void visualizeAllObjects(bool b);

private:
  qTransferFunctionDefinitionWidget* manager;
  vtkCuda2DTransferFunction* func;
  vtkCudaFunctionPolygon* object;
  
  float minIntensity1;
  float maxIntensity1;
  float minIntensity2;
  float maxIntensity2;
  double size1;
  double size2;
  bool histogram;
  bool visAll;

  float zoomMinIntensity1;
  float zoomMaxIntensity1;
  float zoomMinIntensity2;
  float zoomMaxIntensity2;
  bool hasZoomSquare;

  // original click locations
  float origClickI1;
  float origClickI2;
  float centroidI1;
  float centroidI2;
  float rotationCentreI1;
  float rotationCentreI2;
  float closenessRadius;

  // booleans to interpret click by
  bool shiftHeld;
  bool ctrlHeld;
  bool translating;
  bool scaling;
  bool vertexDragging;
  bool autoUpdate;
  unsigned int vertexInUse;
};

#endif