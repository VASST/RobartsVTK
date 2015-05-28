#ifndef KFILEMANAGEMENTWIDGET
#define KFILEMANAGEMENTWIDGET

#include <QObject>
#include <QWidget>
#include <QMenu>
#include <QTimer>
#include <QListWidgetItem>
#include <QPushButton>

#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkVolume.h"
#include "vtkCudaVolumeMapper.h"
#include "vtkImageData.h"
#include "vtkImageReader2.h"
#include "vtkVolume.h"

#include "transferFunctionWindowWidgetInterface.h"
class transferFunctionWindowWidgetInterface;

class kFileManagementWidget : public QWidget
{
  Q_OBJECT
public:
  kFileManagementWidget( transferFunctionWindowWidgetInterface* parent );
  ~kFileManagementWidget();
  QMenu* getMenuOptions( );
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );
  vtkImageData* getCurrentImage();
  unsigned int getNumFrames();

  void SetMap(vtkImageData* m);
  vtkImageData* GetMap(){ return this->Map; }

private slots:
  
  //file related slots
  void addMNCFile();
  void addMHDFile();
  void addDICOMFile();
  void selectFrame();
  void toggleMode();
  void nextFrame();

private:

  transferFunctionWindowWidgetInterface* parent;
  
  bool addMetaImage(std::string filename);
  bool addMincImage(std::string filename);
  bool addDICOMImage(std::string dirname);
  bool addImageToMapper(vtkImageData* data);
  bool selectFrame(std::string dirname);
  bool removeImage(std::string filename);
  
  vtkRenderWindow* window;
  vtkRenderer* renderer;
  vtkCudaVolumeMapper* mapper;
  vtkVolume* volume;
  
  void setupMenu();
  QMenu* fileMenu;
  
  //file related viewing
  QListWidget* files;
  QPushButton* toggleModeButton;

  //control variables
  bool isStatic;
  QTimer* timer;

  //image management variables
  std::vector<std::string> nameVector;
  std::vector<vtkImageReader2*> readers;
  unsigned int maxframes;
  unsigned int numFrames;
  unsigned int currFrame;
  
  //KSOM Components
  vtkImageData* Map;
  int NumberOfComponents;
  
  std::vector<vtkImageData*>* ColourImages;
  std::vector<vtkImageData*> ProjectedImages;
};

#endif