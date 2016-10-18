/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef FILEMANAGEMENTWIDGET
#define FILEMANAGEMENTWIDGET

#include "TFUICommonExport.h"

#include <QWidget>

class QListWidget;
class QListWidgetItem;
class QMenu;
class QPushButton;
class QTimer;
class vtkCudaVolumeMapper;
class vtkImageData;
class vtkImageReader2;
class vtkRenderWindow;
class vtkRenderer;
class vtkVolume;

class qTransferFunctionWindowWidgetInterface;

class TFUICommonExport qFileManagementWidget : public QWidget
{
  Q_OBJECT

public:
  qFileManagementWidget( qTransferFunctionWindowWidgetInterface* parent );
  ~qFileManagementWidget();
  QMenu* getMenuOptions( );
  void setStandardWidgets( vtkRenderWindow* window, vtkRenderer* renderer, vtkCudaVolumeMapper* caster );
  vtkImageData* getCurrentImage();
  unsigned int getNumFrames();

  void SetMap(vtkImageData* aImageData);
  vtkImageData* GetMap();

public slots:
  //file related slots
  void addMNCFile();
  void addMHDFile();
  void addDICOMFile();
  void selectFrame();
  void toggleMode();
  void nextFrame();

private:
  qTransferFunctionWindowWidgetInterface* parent;

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

  //KSOM Components
  vtkImageData* Map;
  int NumberOfComponents;
  std::vector<vtkImageData*>* ColourImages;
  std::vector<vtkImageData*> ProjectedImages;

  //image management variables
  std::vector<std::string> nameVector;
  std::vector<vtkImageReader2*> readers;
  unsigned int maxframes;
  unsigned int numFrames;
  unsigned int currFrame;
};

#endif