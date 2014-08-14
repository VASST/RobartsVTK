#include "FluoroPredViz.h"
#include "ResizableQVTKWidget.h"

#include "qboxlayout.h"
#include "qgridlayout.h"
#include "qsplitter.h"
#include "qlabel.h"
#include "qframe.h"
#include "qgroupbox.h"
#include "qfiledialog.h"

#include "vtkMetaImageReader.h"
#include "vtkMINCImageReader.h"
#include "vtkDICOMImageReader.h"
#include "vtkTransform.h"

FluoroPredViz::FluoroPredViz( QWidget* parent ) :
QWidget(0), SuccessInit(0)
{

	//create main layout
	QHBoxLayout* MainLayout = new QHBoxLayout();
	this->setLayout(MainLayout);
	QWidget* Params = new QWidget(0);
	QVBoxLayout* ParamsLayout = new QVBoxLayout();
	Params->setLayout(ParamsLayout);
	QSplitter* WindowSplitter = new QSplitter(Qt::Orientation::Vertical);
	MainLayout->addWidget(Params);
	MainLayout->addWidget(WindowSplitter);
	WindowSplitter->setSizePolicy(QSizePolicy::Policy::Expanding,QSizePolicy::Policy::Expanding);
	
	//set up parameters bar
	ParamsLayout->addStrut(20);
	SetupFluoroParams(ParamsLayout);
	ParamsLayout->addSpacerItem(new QSpacerItem(40, 20, QSizePolicy::Fixed, QSizePolicy::Fixed) );
	SetupObjectParams(ParamsLayout);
	ParamsLayout->addSpacerItem(new QSpacerItem(40, 200, QSizePolicy::Fixed, QSizePolicy::Expanding) );
	ParamsLayout->setSizeConstraint(QLayout::SetFixedSize);

	//set up screens
	SetupDRRScreen(WindowSplitter);
	SetupSchematicScreen(WindowSplitter);

	//get initial image
	Reader = 0;
	SuccessInit = SetUpReader(RequestFilename());
	if(SuccessInit != 0) return;
	ConnectUpPipeline();
	UpdateDegreeMarkers();
	UpdateXrayMarker();
	UpdateImageBoundsMarker();

	//final prep on the screens
	DRRScreen->ready = true;
	for(int i = 0; i < 3; i++)
		SchematicScreen[i]->ready = true;
	UpdateViz();

}

int FluoroPredViz::GetSuccessInit(){
	return SuccessInit;
}

FluoroPredViz::~FluoroPredViz(){
	
	//fluoro params sliders
	delete FocusX;
	delete FocusY;
	delete PrincipleX;
	delete PrincipleY;
	delete Radius;
	delete Angle;

	//object params sliders
	delete TranslationX;
	delete TranslationY;
	delete TranslationZ;
	delete OrientationX;
	delete OrientationY;
	delete OrientationZ;

	//screens/pipelines
	for(int i = 0; i < 3; i++)
		delete SchematicScreen[i];
	delete DRRScreen;
	if(Reader) Reader->Delete();

}



//-------------------------------------------------------------------------------//
// Manage fluoro parameters
//-------------------------------------------------------------------------------//


void FluoroPredViz::SetupFluoroParams(QBoxLayout* ParamsLayout){
	
	//fluoro params tab labels
	QGroupBox* FluoroTab = new QGroupBox("Fluoro Parameters",this);
	QGridLayout* FluoroTabLayout = new QGridLayout();
	FluoroTabLayout->addWidget(new QLabel("Focus (x)"),0,0);
	FluoroTabLayout->addWidget(new QLabel("Focus (y)"),1,0);
	FluoroTabLayout->addWidget(new QLabel("Radius (d)"),2,0);
	FluoroTabLayout->addWidget(new QLabel("Principle Point (x)"),0,3);
	FluoroTabLayout->addWidget(new QLabel("Principle Point (y)"),1,3);
	FluoroTabLayout->addWidget(new QLabel("Width (w)"),2,3);
	QString ThetaName = "Angle (";
	ThetaName.append( QChar(0x98, 0x03) );
	ThetaName.append( ")" );
	FluoroTabLayout->addWidget(new QLabel( ThetaName ),3,0);

	//fluoro params sliders
	WidthVal = 200.0;
	Width = new QSlider(Qt::Orientation::Horizontal);
	Width->setMaximum(2000);		Width->setValue(200);
	FocusXVal = 1000.0;
	FocusX = new QSlider(Qt::Orientation::Horizontal);
	FocusX->setMaximum(2000);		FocusX->setValue(1000);
	FocusYVal = 1000.0;
	FocusY = new QSlider(Qt::Orientation::Horizontal);
	FocusY->setMaximum(2000);		FocusY->setValue(1000);
	PrincipleXVal = 1000.0;
	PrincipleX = new QSlider(Qt::Orientation::Horizontal);
	PrincipleX->setMaximum(2000);	PrincipleX->setValue(1000);
	PrincipleYVal = 0.0;
	PrincipleY = new QSlider(Qt::Orientation::Horizontal);
	PrincipleY->setMaximum(2000);	PrincipleY->setValue(1000);
	RadiusVal = 1000.0;
	Radius = new QSlider(Qt::Orientation::Horizontal);
	Radius->setMaximum(2000);		Radius->setValue(1000);
	Angle = new QSlider(Qt::Orientation::Horizontal);
	Angle->setMaximum(720);			Angle->setValue(360);
	AngleVal = 0.0;
	FluoroTabLayout->addWidget(FocusX,0,1);
	FluoroTabLayout->addWidget(FocusY,1,1);
	FluoroTabLayout->addWidget(Radius,2,1);
	FluoroTabLayout->addWidget(PrincipleX,0,4);
	FluoroTabLayout->addWidget(PrincipleY,1,4);
	FluoroTabLayout->addWidget(Width,2,4);
	FluoroTabLayout->addWidget(Angle,3,1,1,4);

	//fluoro slider slots
	connect( FocusX, SIGNAL(valueChanged(int)), this, SLOT(SetFocusX(int)) );
	connect( FocusY, SIGNAL(valueChanged(int)), this, SLOT(SetFocusY(int)) );
	connect( PrincipleX, SIGNAL(valueChanged(int)), this, SLOT(SetPrincipleX(int)) );
	connect( PrincipleY, SIGNAL(valueChanged(int)), this, SLOT(SetPrincipleY(int)) );
	connect( Radius, SIGNAL(valueChanged(int)), this, SLOT(SetRadius(int)) );
	connect( Width, SIGNAL(valueChanged(int)), this, SLOT(SetWidth(int)) );
	connect( Angle, SIGNAL(valueChanged(int)), this, SLOT(SetAngle(int)) );

	//add fluoro params to main screen
	FluoroTab->setLayout(FluoroTabLayout);
	FluoroTab->setMaximumWidth(500);
	FluoroTab->setMinimumWidth(300);
	ParamsLayout->addWidget(FluoroTab);
}


void FluoroPredViz::SetFocusX(int v){
	double Range = 2000.0; double Offset = 0.0;
	double aV = Offset + Range*((double) v / (double) (FocusX->maximum() - FocusX->minimum()));
	SetFocusX(aV);
}

void FluoroPredViz::SetFocusY(int v){
	double Range = 2000.0; double Offset = 0.0;
	double aV = Offset + Range*((double) v / (double) (FocusY->maximum() - FocusY->minimum()));
	SetFocusY(aV);
}

void FluoroPredViz::SetPrincipleX(int v){
	double Range = 2000.0; double Offset = -1000.0;
	double aV = Offset + Range*((double) v / (double) (PrincipleX->maximum() - PrincipleX->minimum()));
	SetPrincipleX(aV);
}

void FluoroPredViz::SetPrincipleY(int v){
	double Range = 2000.0; double Offset = -1000.0;
	double aV = Offset + Range*((double) v / (double) (PrincipleY->maximum() - PrincipleY->minimum()));
	SetPrincipleY(aV);
}

void FluoroPredViz::SetRadius(int v){
	double Range = 2000.0; double Offset = 0.0;
	double aV = Offset + Range*((double) v / (double) (Radius->maximum() - Radius->minimum()));
	SetRadius(aV);
}

void FluoroPredViz::SetAngle(int v){
	double Range = 180; double Offset = -90.0;
	double aV = Offset + Range*((double) v / (double) (Angle->maximum() - Angle->minimum()));
	SetAngle(aV);
}

void FluoroPredViz::SetWidth(int v){
	double Range = 2000; double Offset = 0.0;
	double aV = Offset + Range*((double) v / (double) (Width->maximum() - Width->minimum()));
	SetWidth(aV);
}

void FluoroPredViz::SetFocusX(double v){
	FocusXVal = v;
	UpdateXrayMarker();
	UpdateViz();
}

void FluoroPredViz::SetFocusY(double v){
	FocusYVal = v;
	UpdateXrayMarker();
	UpdateViz();
}

void FluoroPredViz::SetPrincipleX(double v){
	PrincipleYVal = v;
	UpdateViz();
}

void FluoroPredViz::SetPrincipleY(double v){
	PrincipleXVal = v;
	UpdateViz();
}

void FluoroPredViz::SetRadius(double v){
	RadiusVal = v;
	UpdateDegreeMarkers();
	UpdateXrayMarker();
	UpdateViz();
}

void FluoroPredViz::SetAngle(double v){
	AngleVal = v;
	UpdateXrayMarker();
	UpdateViz();
}

void FluoroPredViz::SetWidth(double v){
	WidthVal = v;
	UpdateXrayMarker();
	UpdateViz();
}

//-------------------------------------------------------------------------------//
// Manage object parameters
//-------------------------------------------------------------------------------//

void FluoroPredViz::SetupObjectParams(QBoxLayout* ParamsLayout){
	
	//clear transform
	ObjectParams = vtkTransform::New();
	ObjectParams->Identity();
	ObjectParams->PostMultiply();

	//Object params tab labels
	QGroupBox* ObjectTab = new QGroupBox("Object Parameters",this);
	QGridLayout* ObjectTabLayout = new QGridLayout();
	ObjectTabLayout->addWidget(new QLabel("Position (x)"),0,0);
	ObjectTabLayout->addWidget(new QLabel("Position (y)"),1,0);
	ObjectTabLayout->addWidget(new QLabel("Position (z)"),2,0);
	ObjectTabLayout->addWidget(new QLabel("Orientation (x)"),0,3);
	ObjectTabLayout->addWidget(new QLabel("Orientation (y)"),1,3);
	ObjectTabLayout->addWidget(new QLabel("Orientation (z)"),2,3);

	//Object params sliders
	TranslationX = new QSlider(Qt::Orientation::Horizontal);
	TranslationX->setMaximum(2000);		TranslationX->setValue(1000);
	TranslationXVal = 0.0;
	TranslationY = new QSlider(Qt::Orientation::Horizontal);
	TranslationY->setMaximum(2000);		TranslationY->setValue(1000);
	TranslationYVal = 0.0;
	TranslationZ = new QSlider(Qt::Orientation::Horizontal);
	TranslationZ->setMaximum(2000);		TranslationZ->setValue(1000);
	TranslationZVal = 0.0;
	OrientationX = new QSlider(Qt::Orientation::Horizontal);
	OrientationX->setMaximum(720);		OrientationX->setValue(360);
	OrientationXVal = 0.0;
	OrientationY = new QSlider(Qt::Orientation::Horizontal);
	OrientationY->setMaximum(720);		OrientationY->setValue(360);
	OrientationYVal = 0.0;
	OrientationZ = new QSlider(Qt::Orientation::Horizontal);
	OrientationZ->setMaximum(720);		OrientationZ->setValue(360);
	OrientationZVal = 0.0;
	ObjectTabLayout->addWidget(TranslationX,0,1);
	ObjectTabLayout->addWidget(TranslationY,1,1);
	ObjectTabLayout->addWidget(TranslationZ,2,1);
	ObjectTabLayout->addWidget(OrientationX,0,4);
	ObjectTabLayout->addWidget(OrientationY,1,4);
	ObjectTabLayout->addWidget(OrientationZ,2,4);

	//Object slider slots
	connect( TranslationX, SIGNAL(valueChanged(int)), this, SLOT(SetTranslationX(int)) );
	connect( TranslationY, SIGNAL(valueChanged(int)), this, SLOT(SetTranslationY(int)) );
	connect( TranslationZ, SIGNAL(valueChanged(int)), this, SLOT(SetTranslationZ(int)) );
	connect( OrientationX, SIGNAL(valueChanged(int)), this, SLOT(SetOrientationX(int)) );
	connect( OrientationY, SIGNAL(valueChanged(int)), this, SLOT(SetOrientationY(int)) );
	connect( OrientationZ, SIGNAL(valueChanged(int)), this, SLOT(SetOrientationZ(int)) );

	//add Object params to main screen
	ObjectTab->setLayout(ObjectTabLayout);
	ObjectTab->setMaximumWidth(500);
	ObjectTab->setMinimumWidth(300);
	ParamsLayout->addWidget(ObjectTab);
	
}

void FluoroPredViz::SetTranslationX(int v){
	double Range = 2000.0; double Offset = -1000.0;
	double aV = Offset + Range*((double) v / (double) (TranslationX->maximum() - TranslationX->minimum()));
	SetTranslationX(aV);
}

void FluoroPredViz::SetTranslationY(int v){
	double Range = 2000.0; double Offset = -1000.0;
	double aV = Offset + Range*((double) v / (double) (TranslationY->maximum() - TranslationY->minimum()));
	SetTranslationY(aV);
}

void FluoroPredViz::SetTranslationZ(int v){
	double Range = 2000.0; double Offset = -1000.0;
	double aV = Offset + Range*((double) v / (double) (TranslationZ->maximum() - TranslationZ->minimum()));
	SetTranslationZ(aV);
}

void FluoroPredViz::SetOrientationX(int v){
	double Range = 360.0; double Offset = -180.0;
	double aV = Offset + Range*((double) v / (double) (OrientationX->maximum() - OrientationX->minimum()));
	SetOrientationX(aV);
}

void FluoroPredViz::SetOrientationY(int v){
	double Range = 360.0; double Offset = -180.0;
	double aV = Offset + Range*((double) v / (double) (OrientationY->maximum() - OrientationY->minimum()));
	SetOrientationY(aV);
}

void FluoroPredViz::SetOrientationZ(int v){
	double Range = 360.0; double Offset = -180.0;
	double aV = Offset + Range*((double) v / (double) (OrientationZ->maximum() - OrientationZ->minimum()));
	SetOrientationZ(aV);
}

void FluoroPredViz::SetTranslationX(double v){
	TranslationXVal = v;
	UpdateImageBoundsMarker();
	UpdateViz();
}

void FluoroPredViz::SetTranslationY(double v){
	TranslationYVal = v;
	UpdateImageBoundsMarker();
	UpdateViz();
}

void FluoroPredViz::SetTranslationZ(double v){
	TranslationZVal = v;
	UpdateImageBoundsMarker();
	UpdateViz();
}

void FluoroPredViz::SetOrientationX(double v){
	OrientationXVal = v;
	UpdateImageBoundsMarker();
	UpdateViz();
}

void FluoroPredViz::SetOrientationY(double v){
	OrientationYVal = v;
	UpdateImageBoundsMarker();
	UpdateViz();
}

void FluoroPredViz::SetOrientationZ(double v){
	OrientationZVal = v;
	UpdateImageBoundsMarker();
	UpdateViz();
}




//-------------------------------------------------------------------------------//
// Manage Image
//-------------------------------------------------------------------------------//

QString FluoroPredViz::RequestFilename(){

	//get file name
	QString filename = QFileDialog::getOpenFileName(this,"Open Image","", tr("Meta (*.mhd *.mha);; MINC (*.mnc *.minc)") );
	return filename;

}

int FluoroPredViz::SetUpReader(QString filename){

	//return if there is no filename
	if( filename.isNull() || filename.isEmpty() ) return-1;
	
	//create reader based on file name extension
	if( filename.endsWith(".mhd",Qt::CaseInsensitive) || filename.endsWith(".mha",Qt::CaseInsensitive) ){
		if( this->Reader ) this->Reader->Delete();
		this->Reader = vtkMetaImageReader::New();
	}else if( filename.endsWith(".mnc",Qt::CaseInsensitive) || filename.endsWith(".minc",Qt::CaseInsensitive) ){
		if( this->Reader ) this->Reader->Delete();
		this->Reader = vtkMINCImageReader::New();
	}else{
		return -1;
	}

	//check validity of reader and load file in
	if( !this->Reader->CanReadFile( filename.toStdString().c_str() ) ){
		return -1;
	}
	this->Reader->SetFileName( filename.toStdString().c_str() );
	this->Reader->Update();

	return 0;

}

int FluoroPredViz::RequestImage(){
	
	//get file name and set up reader
	int failed = SetUpReader(RequestFilename());
	if( failed ) return failed;

	//connect up remainder of the pipeline
	ConnectUpPipeline();

	//update viz
	UpdateViz();

	return 0;

}

//-------------------------------------------------------------------------------//
// Update Visualization
//-------------------------------------------------------------------------------//
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"

#include "vtkImageData.h"

#include "vtkActor.h"
#include "vtkPolyDataMapper.h"
#include "vtkArrowSource.h"
#include "vtkSphereSource.h"
#include "vtkConeSource.h"
#include "vtkCubeSource.h"
#include "vtkCamera.h"
#include "vtkProperty.h"
#include "vtkImagePlaneWidget.h"
#include "vtkTransform.h"

#include "vtkCudaDRRImageVolumeMapper.h"
#include "vtkVolume.h"

class vtkPlaneWidgetCallback : public vtkCommand {
public:
	static vtkPlaneWidgetCallback *New()
		{ return new vtkPlaneWidgetCallback; }
	virtual void Execute(vtkObject *caller, unsigned long, void*){
		if(window) window->Render();
	}
	void SetWindow(vtkRenderWindow* w) 
		{ this->window = w; }
private:

	vtkRenderWindow* window;
};


void FluoroPredViz::UpdateViz(){
	DRRScreen->GetRenderWindow()->Render();
	for(int i = 0; i < 3; i++)
		SchematicScreen[i]->GetRenderWindow()->Render();
}

void FluoroPredViz::ConnectUpPipeline(){
	
	//build remaining DRR pipeline
	vtkCudaDRRImageVolumeMapper* Mapper = vtkCudaDRRImageVolumeMapper::New();
	Mapper->SetInput(Reader->GetOutput());
	ImageVolume = vtkVolume::New();
	ImageVolume->SetMapper(Mapper);
	vtkRenderer* DRR_Renderer = vtkRenderer::New();
	DRR_Renderer->SetBackground(1,1,1);
	DRR_Renderer->AddVolume(ImageVolume);
	Mapper->Delete();
	DRRScreen->GetRenderWindow()->AddRenderer(DRR_Renderer);
	XraySource = DRR_Renderer->GetActiveCamera();
	DRR_Renderer->Delete();
	DRRScreen->GetInteractor()->Disable();

	//create schematic renderer
	vtkRenderer* Schematic_Renderer[3];
	for(int i = 0; i< 3; i++){
		Schematic_Renderer[i] = vtkRenderer::New();
		Schematic_Renderer[i]->SetBackground(0,0,0);
		SchematicScreen[i]->GetRenderWindow()->AddRenderer(Schematic_Renderer[i]);
		SchematicScreen[i]->GetInteractor()->Disable();
	}

	//build angular trajectory markers
	double DegreeIncrements = 10.0;
	NumMarkers = (int)(180.0/DegreeIncrements + 1.5);
	DegreeMarkers = new vtkSphereSource*[NumMarkers];
	for(int i = 0; i < NumMarkers; i++){
		DegreeMarkers[i] = vtkSphereSource::New();
		DegreeMarkers[i]->SetRadius(10);
		vtkPolyDataMapper* Mapper = vtkPolyDataMapper::New();
		Mapper->SetInputConnection(DegreeMarkers[i]->GetOutputPort());
		vtkActor* Actor = vtkActor::New();
		Actor->SetMapper(Mapper);
		for(int i = 0; i < 3; i++)
			Schematic_Renderer[i]->AddActor(Actor);
		Mapper->Delete();
		Actor->Delete();

	}
	UpdateDegreeMarkers();

	//build center point
	vtkSphereSource* CenterPoint = vtkSphereSource::New();
	CenterPoint->SetCenter(0,0,0);
	CenterPoint->SetRadius(20);
	vtkPolyDataMapper* CenterPointMapper = vtkPolyDataMapper::New();
	CenterPointMapper->SetInputConnection(CenterPoint->GetOutputPort());
	CenterPoint->Delete();
	vtkActor* CenterPointActor = vtkActor::New();
	CenterPointActor->SetMapper(CenterPointMapper);
	CenterPointActor->GetProperty()->SetColor(0,1,1);
	CenterPointMapper->Delete();
	for(int i = 0; i < 3; i++)
		Schematic_Renderer[i]->AddActor(CenterPointActor);
	CenterPointActor->Delete();

	//build axis
	vtkArrowSource* AxisXSource = vtkArrowSource::New();
	vtkPolyDataMapper* AxisXMapper = vtkPolyDataMapper::New();
	AxisXMapper->SetInputConnection(AxisXSource->GetOutputPort());
	AxisXSource->Delete();
	vtkActor* AxisXActor = vtkActor::New();
	AxisXActor->SetMapper(AxisXMapper);
	AxisXMapper->Delete();
	AxisXActor->SetScale(500);
	AxisXActor->GetProperty()->SetColor(1,0,0);
	for(int i = 0; i < 3; i++)
		if(i != 2) Schematic_Renderer[i]->AddActor(AxisXActor);
	AxisXActor->Delete();
	vtkArrowSource* AxisYSource = vtkArrowSource::New();
	vtkPolyDataMapper* AxisYMapper = vtkPolyDataMapper::New();
	AxisYMapper->SetInputConnection(AxisYSource->GetOutputPort());
	AxisYSource->Delete();
	vtkActor* AxisYActor = vtkActor::New();
	AxisYActor->SetMapper(AxisYMapper);
	AxisYMapper->Delete();
	AxisYActor->SetScale(500);
	AxisYActor->SetOrientation(0,0,90);
	AxisYActor->GetProperty()->SetColor(0,1,0);
	for(int i = 0; i < 3; i++)
		if(i != 1) Schematic_Renderer[i]->AddActor(AxisYActor);
	AxisYActor->Delete();
	vtkArrowSource* AxisZSource = vtkArrowSource::New();
	vtkPolyDataMapper* AxisZMapper = vtkPolyDataMapper::New();
	AxisZMapper->SetInputConnection(AxisZSource->GetOutputPort());
	AxisZSource->Delete();
	vtkActor* AxisZActor = vtkActor::New();
	AxisZActor->SetMapper(AxisZMapper);
	AxisZMapper->Delete();
	AxisZActor->SetScale(500);
	AxisZActor->GetProperty()->SetColor(0,0,1);
	AxisZActor->SetOrientation(0,-90,0);
	for(int i = 0; i < 3; i++)
		if(i != 0) Schematic_Renderer[i]->AddActor(AxisZActor);
	AxisZActor->Delete();

	//build fluoro location
	XrayMarker = vtkConeSource::New();
	XrayMarker->SetResolution(100);
	vtkPolyDataMapper* XrayMarkerMapper = vtkPolyDataMapper::New();
	XrayMarkerMapper->SetInputConnection(XrayMarker->GetOutputPort());
	vtkActor* XrayMarkerActor = vtkActor::New();
	XrayMarkerActor->SetMapper(XrayMarkerMapper);
	XrayMarkerActor->GetProperty()->SetColor(1,0.75,0);
	XrayMarkerActor->GetProperty()->SetOpacity(0.5);
	XrayMarkerMapper->Delete();
	for(int i = 0; i < 3; i++)
		Schematic_Renderer[i]->AddActor(XrayMarkerActor);
	XrayMarkerActor->Delete();
	UpdateXrayMarker();

	//build object location
	int Extent[6]; double Spacing[3]; double Origin[3];
	Reader->GetOutput()->GetExtent(Extent);
	Reader->GetOutput()->GetSpacing(Spacing);
	Reader->GetOutput()->GetOrigin(Origin);
	vtkCubeSource* ImageBoundsMarker = vtkCubeSource::New();
	ImageBoundsMarker->SetXLength( abs( (double)(Extent[1]-Extent[0]+1) * Spacing[0]) );
	ImageBoundsMarker->SetYLength( abs( (double)(Extent[3]-Extent[2]+1) * Spacing[1]) );
	ImageBoundsMarker->SetZLength( abs( (double)(Extent[5]-Extent[4]+1) * Spacing[2]) );
	ImageBoundsMarker->SetCenter(	Origin[0] + (double)(Extent[1]+Extent[0]) * Spacing[0] / 2.0,
									Origin[1] + (double)(Extent[3]+Extent[2]) * Spacing[1] / 2.0,
									Origin[2] + (double)(Extent[5]+Extent[4]) * Spacing[2] / 2.0 );
	vtkPolyDataMapper* ImageBoundsMarkerMapper = vtkPolyDataMapper::New();
	ImageBoundsMarkerMapper->SetInputConnection(ImageBoundsMarker->GetOutputPort());
	ImageBoundsMarker->Delete();
	ImageBoundsMarkerActor = vtkActor::New();
	ImageBoundsMarkerActor->SetMapper(ImageBoundsMarkerMapper);
	ImageBoundsMarkerMapper->Delete();
	for(int i = 0; i < 3; i++)
		Schematic_Renderer[i]->AddActor(ImageBoundsMarkerActor);
	ImageBoundsMarkerActor->GetProperty()->SetColor(1,1,1);
	ImageBoundsMarkerActor->GetProperty()->SetAmbient(1);
	ImageBoundsMarkerActor->GetProperty()->SetDiffuse(0);
	ImageBoundsMarkerActor->GetProperty()->SetRepresentationToWireframe();
	
	//turn on the camera
	for(int i = 0; i < 3; i++){
		vtkCamera* schemCamera = Schematic_Renderer[i]->GetActiveCamera();
		double pos[3] = {0,RadiusVal/2,0};
		if(i==0)
			pos[2] = 5*RadiusVal;
		if(i==1)
			pos[1] = -5*RadiusVal;
		if(i==2)
			pos[0] = 5*RadiusVal;
		schemCamera->SetPosition(pos);
		schemCamera->SetFocalPoint(0,0,0);
		if(i==0)
			schemCamera->SetViewUp(0,0,-1);
		if(i==1)
			schemCamera->SetViewUp(0,0,1);
		if(i==2)
			schemCamera->SetViewUp(0,0,1);
		schemCamera->SetClippingRange(RadiusVal,8*RadiusVal);
		Schematic_Renderer[i]->Delete();
		SchematicScreen[i]->GetRenderWindow()->Render();
	}

}


void FluoroPredViz::UpdateDegreeMarkers(){
	//set location of markers based on angle and C-arm radius
	for(int i = 0; i < NumMarkers; i++){
		double DegreeLocation = -0.5*3.1415926 + 3.1415926 * (double)i / (double) (NumMarkers-1);
		double LX = RadiusVal*std::sin(DegreeLocation);
		double LY = RadiusVal*std::cos(DegreeLocation);
		DegreeMarkers[i]->SetCenter(LX,LY,0);
		DegreeMarkers[i]->Update();
	}

}

void FluoroPredViz::UpdateXrayMarker(){
	//set location and orientation of x-ray schematic marker
	double focus = 0.5*FocusXVal + 0.5*FocusYVal;
	double DegreeLocation = 3.141592 * AngleVal / 180;
	double LX = RadiusVal*std::sin(DegreeLocation);
	double LY = RadiusVal*std::cos(DegreeLocation);
	XrayMarker->SetCenter(0.5*LX,0.5*LY,0);
	XrayMarker->SetDirection(LX,LY,0);
	XrayMarker->SetHeight(RadiusVal);
	XrayMarker->SetRadius(WidthVal*RadiusVal/focus);

	//set location and orientation of x-ray source
	double aspect = (double) this->DRRScreen->width() /  (double) this->DRRScreen->height();
	XraySource->SetPosition(LX,LY,0);
	XraySource->SetFocalPoint(0,0,0);
	XraySource->SetViewUp(0,0,-1);
	XraySource->SetClippingRange(RadiusVal/2,RadiusVal*2);
	double angle = 360 * atan((WidthVal*RadiusVal/focus)/(2*RadiusVal)) / 3.1415926;
	XraySource->SetViewAngle(angle*aspect);
}

void FluoroPredViz::UpdateImageBoundsMarker(){
	
	//reset transform
	ObjectParams->Identity();
	ObjectParams->RotateZ(OrientationZVal);
	ObjectParams->RotateX(OrientationXVal);
	ObjectParams->RotateY(OrientationYVal);
	ObjectParams->Translate(TranslationXVal,TranslationYVal,TranslationZVal);
	ObjectParams->Update();

	//update image bounding box
	ImageBoundsMarkerActor->SetUserTransform(ObjectParams);
	ImageVolume->SetUserTransform(ObjectParams);

}

void FluoroPredViz::SetupDRRScreen(QSplitter* WindowsLayout){
	DRRScreen = new ResizableQVTKWidget(0);
	DRRScreen->setMinimumHeight(500);
	DRRScreen->setSizePolicy( QSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum) );
	WindowsLayout->addWidget(DRRScreen);
}

void FluoroPredViz::SetupSchematicScreen(QSplitter* WindowsLayout){

	QHBoxLayout* layout = new QHBoxLayout();
	QWidget* widget = new QWidget();
	widget->setLayout(layout);
	widget->setSizePolicy( QSizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding) );
	for(int i = 0; i < 3; i++){
		SchematicScreen[i] = new ResizableQVTKWidget(0);
		SchematicScreen[i]->setMinimumWidth(200);
		SchematicScreen[i]->setMinimumHeight(200);
		SchematicScreen[i]->setSizePolicy( QSizePolicy( QSizePolicy::Expanding, QSizePolicy::Expanding) );
		layout->addWidget(SchematicScreen[i]);
	}

	WindowsLayout->addWidget(widget);

}