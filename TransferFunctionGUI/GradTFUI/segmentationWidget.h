#ifndef SEGMENTATIONWIDGET
#define SEGMENTATIONWIDGET

#include <QObject>
#include <QWidget>
#include <QMenu>
#include <QPushButton>

#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkCuda2DVolumeMapper.h"

#include "transferFunctionWindowWidget.h"
class transferFunctionWindowWidget;

class segmentationWidget : public QWidget
{
  Q_OBJECT
public:

  segmentationWidget( transferFunctionWindowWidget* parent );
  ~segmentationWidget();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCuda2DVolumeMapper* caster );
  QMenu* getMenuOptions();

private slots:
  
  //shading related slots
  void segment();

private:
  
  void setupMenu();
  QMenu* segmentationMenu;
  QAction* segmentNowOption;

  transferFunctionWindowWidget* parent;
  
  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCuda2DVolumeMapper* mapper;


};

#endif