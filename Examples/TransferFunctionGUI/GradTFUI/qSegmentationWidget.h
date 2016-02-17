#ifndef SEGMENTATIONWIDGET
#define SEGMENTATIONWIDGET

#include <QObject>
#include <QWidget>
#include <QMenu>
#include <QPushButton>

#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkCuda2DVolumeMapper.h"

#include "qTransferFunctionWindowWidget.h"
class qTransferFunctionWindowWidget;

class qSegmentationWidget : public QWidget
{
  Q_OBJECT
public:

  qSegmentationWidget( qTransferFunctionWindowWidget* parent );
  ~qSegmentationWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCuda2DVolumeMapper* caster );
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
  vtkCuda2DVolumeMapper* mapper;


};

#endif