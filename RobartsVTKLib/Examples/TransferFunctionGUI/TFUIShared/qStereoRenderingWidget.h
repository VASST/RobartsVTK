#ifndef STEREORENDERINGWIDGET
#define STEREORENDERINGWIDGET

#include <QObject>
#include <QWidget>
#include <QMenu>
#include <QSlider>
#include <QLabel>

#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkCudaVolumeMapper.h"

#include "qTransferFunctionWindowWidgetInterface.h"
class qTransferFunctionWindowWidgetInterface;

class qStereoRenderingWidget : public QWidget
{
  Q_OBJECT
public:

  qStereoRenderingWidget( qTransferFunctionWindowWidgetInterface* parent );
  ~qStereoRenderingWidget();
  QMenu* getMenuOptions();
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );

private slots:

  //stereo related slots
  void setStereoOn();
  void setStereoOff();
  void toggleEyes();

private:

  void setupMenu();
  void setStereoEnabled(bool);
  void toggleStereoEyes();

  qTransferFunctionWindowWidgetInterface* parent;

  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCudaVolumeMapper* mapper;

  //stereo menu variables
  QMenu* stereoMenu;
  QAction* stereoOnMenuOption;
  QAction* stereoOffMenuOption;
  QAction* stereoEyeToggleMenuOption;

  unsigned int old;

};

#endif