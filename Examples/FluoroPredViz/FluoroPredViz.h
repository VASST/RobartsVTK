#ifndef FLUOROPREDVIZ_H
#define FLUOROPREDVIZ_H

#include <QWidget>
#include <QObject>
#include <QSlider>
#include <QBoxLayout>
#include <QSplitter>
#include <QLineEdit>

class QVTKWidget;
class ResizableQVTKWidget;
class vtkImageReader2;
class vtkSphereSource;
class vtkConeSource;
class vtkImagePlaneWidget;
class vtkActor;
class vtkCamera;
class vtkTransform;
class vtkVolume;
class vtkImageExtractComponents;
class vtkCuda2DTransferFunction;
class vtkCudaDualImageVolumeMapper;
class vtkCudaDRRImageVolumeMapper;
class vtkBoxWidget;
class vtkCommand;

class FluoroPredViz : public QWidget
{
  Q_OBJECT

public:
  FluoroPredViz( QWidget* parent = 0 );
  ~FluoroPredViz();

  bool GetSuccessInit();

public slots:
  void UpdateViz();
  void UpdateImageRegistration();

  //fluoro params slots
  void SetFocusX(double);
  void SetFocusY(double);
  void SetPrincipleX(double);
  void SetPrincipleY(double);
  void SetDetectorDistance(double);
  void SetWidth(double);
  void SetAngle(double);
  void SetCrAngle(double);

  //object param slots
  void SetTranslationX(double);
  void SetTranslationY(double);
  void SetTranslationZ(double);
  void SetOrientationX(double);
  void SetOrientationY(double);
  void SetOrientationZ(double);

  //image file management
  int RequestImage();

private slots:
  //fluoro params slots
  void MimicView();
  void SetFocusX(int);
  void SetFocusY(int);
  void SetPrincipleX(int);
  void SetPrincipleY(int);
  void SetDetectorDistance(int);
  void SetWidth(int);
  void SetAngle(int);
  void SetCrAngle(int);

  //object params slots
  void SetTranslationX(int);
  void SetTranslationY(int);
  void SetTranslationZ(int);
  void SetOrientationX(int);
  void SetOrientationY(int);
  void SetOrientationZ(int);

  //TF slots
  void SetTFName();

private:
  //fluoro params
  class MimicViewCallback;
  friend class MimicViewCallback;
  void SetupFluoroParams(QBoxLayout*);
  QSlider* FocusX;
  double      FocusXVal;
  QLineEdit*    FocusXValBox;
  QSlider* FocusY;
  double      FocusYVal;
  QLineEdit*    FocusYValBox;
  QSlider* PrincipleX;
  double      PrincipleXVal;
  QLineEdit*    PrincipleXValBox;
  QSlider* PrincipleY;
  double      PrincipleYVal;
  QLineEdit*    PrincipleYValBox;
  QSlider* DetectorDistance;
  double      DetectorDistanceVal;
  QLineEdit*    DetectorDistanceValBox;
  QSlider* Angle;
  double      AngleVal;
  QLineEdit*    AngleValBox;
  QSlider* CrAngle;
  double      CrAngleVal;
  QLineEdit*    CrAngleValBox;
  QSlider* Width;
  double      WidthVal;
  QLineEdit*    WidthValBox;

  //object params
  void SetupObjectParams(QBoxLayout*);
  QSlider* TranslationX;
  double      TranslationXVal;
  QLineEdit*    TranslationXValBox;
  QSlider* TranslationY;
  double      TranslationYVal;
  QLineEdit*    TranslationYValBox;
  QSlider* TranslationZ;
  double      TranslationZVal;
  QLineEdit*    TranslationZValBox;
  QSlider* OrientationX;
  double      OrientationXVal;
  QLineEdit*    OrientationXValBox;
  QSlider* OrientationY;
  double      OrientationYVal;
  QLineEdit*    OrientationYValBox;
  QSlider* OrientationZ;
  double      OrientationZVal;
  QLineEdit*    OrientationZValBox;
  vtkTransform* ObjectParams;

  //file management
  QString RequestFilename();
  bool SetUpReader(QString);
  vtkImageReader2* Reader;
  vtkImageExtractComponents* Extractor;

  //clipping planes
  vtkBoxWidget* ClippingPlanes;
  vtkTransform* ClippingPlanesPosition;
  vtkCommand* ClippingCallback;

  //screens
  void ConnectUpPipeline();
  void SetupDRRScreen(QSplitter*);
  void SetupDVRScreen(QSplitter*);
  void SetupSchematicScreen(QSplitter*);
  QVTKWidget*    DVRScreen;
  ResizableQVTKWidget*    DRRScreen;
  ResizableQVTKWidget*    SchematicScreen[3];
  vtkVolume*    ImageVolumeDVR;
  vtkCudaDualImageVolumeMapper* DVRMapper;
  vtkVolume*    ImageVolumeDRR;
  vtkCudaDRRImageVolumeMapper* DRRMapper;
  QLineEdit*    TFName;
  vtkCuda2DTransferFunction* TransferFunction;

  //schematic
  void UpdateDegreeMarkers();
  void UpdateXrayMarker();
  vtkSphereSource** DegreeMarkers;
  int NumMarkers;
  vtkConeSource* XrayMarker;
  vtkCamera* XraySource;
  vtkCamera* DVRSource;
  vtkActor* ImageBoundsMarkerActor;
  bool SuccessInit;
  bool Paused;
};

#endif //FLUOROPREDVIZ_H
