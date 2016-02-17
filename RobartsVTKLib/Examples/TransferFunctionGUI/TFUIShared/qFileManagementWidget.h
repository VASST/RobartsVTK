#ifndef FILEMANAGEMENTWIDGET
#define FILEMANAGEMENTWIDGET

#include <QWidget>

class vtkImageReader2;
class vtkRenderWindow;
class vtkRenderer;
class vtkCudaVolumeMapper;
class vtkVolume;
class vtkImageData;
class QTimer;
class QMenu;
class QListWidget;
class QListWidgetItem;
class QPushButton;

class qTransferFunctionWindowWidgetInterface;

class qFileManagementWidget : public QWidget
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