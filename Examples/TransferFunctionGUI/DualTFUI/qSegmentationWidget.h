#ifndef qSegmentationWidget_H
#define qSegmentationWidget_H

#include <QObject>
#include <QWidget>
#include <QMenu>
#include <QPushButton>

#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkCudaDualImageVolumeMapper.h"

#include "qTransferFunctionWindowWidget.h"
class qTransferFunctionWindowWidget;

class qSegmentationWidget : public QWidget
{
  Q_OBJECT
public:

  qSegmentationWidget( qTransferFunctionWindowWidget* parent );
  ~qSegmentationWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaDualImageVolumeMapper* caster );
  QMenu* getMenuOptions();

private slots:
  
  //shading related slots
  void segment();

private:
  
  void setupMenu();
  QMenu* segmentationMenu;
  QAction* segmentNowOption;

  qTransferFunctionWindowWidget* parent;
  
  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCudaDualImageVolumeMapper* mapper;


};

#endif