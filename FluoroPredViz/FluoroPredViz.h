#ifndef FLUOROPREDVIZ_H
#define FLUOROPREDVIZ_H

#include <QWidget>
#include <QObject>
#include <QSlider>
#include <QBoxLayout>
#include <QSplitter>

class ResizableQVTKWidget;
class vtkImageReader2;
class vtkSphereSource;
class vtkConeSource;
class vtkImagePlaneWidget;
class vtkActor;
class vtkCamera;
class vtkTransform;
class vtkVolume;

class FluoroPredViz : public QWidget {
	Q_OBJECT

public:
	FluoroPredViz( QWidget* parent = 0 );
	~FluoroPredViz();
	
	int GetSuccessInit();

public slots:
	void UpdateViz();

	//fluoro params slots
	void SetFocusX(double);
	void SetFocusY(double);
	void SetPrincipleX(double);
	void SetPrincipleY(double);
	void SetRadius(double);
	void SetWidth(double);
	void SetAngle(double);

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
	void SetFocusX(int);
	void SetFocusY(int);
	void SetPrincipleX(int);
	void SetPrincipleY(int);
	void SetRadius(int);
	void SetWidth(int);
	void SetAngle(int);

	//object params slots
	void SetTranslationX(int);
	void SetTranslationY(int);
	void SetTranslationZ(int);
	void SetOrientationX(int);
	void SetOrientationY(int);
	void SetOrientationZ(int);

private:
	
	int SuccessInit;

	//fluoro params
	void SetupFluoroParams(QBoxLayout*);
	QSlider* FocusX;		double FocusXVal;
	QSlider* FocusY;		double FocusYVal;
	QSlider* PrincipleX;	double PrincipleXVal;
	QSlider* PrincipleY;	double PrincipleYVal;
	QSlider* Radius;		double RadiusVal;
	QSlider* Angle;			double AngleVal;
	QSlider* Width;			double WidthVal;

	//object params
	void SetupObjectParams(QBoxLayout*);
	QSlider* TranslationX;	double TranslationXVal;
	QSlider* TranslationY;	double TranslationYVal;
	QSlider* TranslationZ;	double TranslationZVal;
	QSlider* OrientationX;	double OrientationXVal;
	QSlider* OrientationY;	double OrientationYVal;
	QSlider* OrientationZ;	double OrientationZVal;
	vtkTransform* ObjectParams;

	//file management
	QString RequestFilename();
	int SetUpReader(QString);
	vtkImageReader2* Reader;

	//screens
	void ConnectUpPipeline();
	void SetupDRRScreen(QSplitter*);
	void SetupSchematicScreen(QSplitter*);
	ResizableQVTKWidget*		DRRScreen;
	ResizableQVTKWidget*		SchematicScreen[3];
	vtkVolume*		ImageVolume;

	//schematic
	void UpdateDegreeMarkers();
	void UpdateXrayMarker();
	void UpdateImageBoundsMarker();
	vtkSphereSource** DegreeMarkers;
	int NumMarkers;
	vtkConeSource* XrayMarker;
	vtkCamera* XraySource;
	vtkActor* ImageBoundsMarkerActor;

};

#endif //FLUOROPREDVIZ_H