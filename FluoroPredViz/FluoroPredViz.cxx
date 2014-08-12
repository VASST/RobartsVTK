#include "FluoroPredViz.h"
#include "QVTKWidget.h"

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

FluoroPredViz::FluoroPredViz( QWidget* parent ) :
QWidget(0), SuccessInit(0)
{

	//create main layout
	QSplitter* MainLayout = new QSplitter(Qt::Orientation::Horizontal);
	MainLayout->setStretchFactor(1,0);
	MainLayout->setStretchFactor(2,1);
	QList<int> list; list.append(400); list.append(400);
	MainLayout->setSizes(list);
	this->setLayout(new QHBoxLayout());
	this->layout()->addWidget(MainLayout);
	QWidget* Params = new QWidget(0);
	QVBoxLayout* ParamsLayout = new QVBoxLayout();
	Params->setLayout(ParamsLayout);
	QSplitter* WindowSplitter = new QSplitter(Qt::Orientation::Vertical);
	MainLayout->addWidget(Params);
	MainLayout->addWidget(WindowSplitter);
	
	//set up parameters bar
	SetupFluoroParams(ParamsLayout);
	ParamsLayout->addSpacerItem(new QSpacerItem(40, 20, QSizePolicy::Minimum, QSizePolicy::Fixed) );
	SetupObjectParams(ParamsLayout);
	ParamsLayout->addStretch(1);

	//set up screens
	SetupDRRScreen(WindowSplitter);
	SetupSchematicScreen(WindowSplitter);

	//get initial image
	Reader = 0;
	SuccessInit = SetUpReader(RequestFilename());
	if(SuccessInit == 0) ConnectUpPipeline();
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
	delete SchematicScreen;
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
	WidthVal = 500.0;
	Width = new QSlider(Qt::Orientation::Horizontal);
	Width->setMaximum(200);		Width->setValue(50);
	FocusXVal = 1000.0;
	FocusX = new QSlider(Qt::Orientation::Horizontal);
	FocusX->setMaximum(200);		FocusX->setValue(100);
	FocusYVal = 1000.0;
	FocusY = new QSlider(Qt::Orientation::Horizontal);
	FocusY->setMaximum(200);		FocusY->setValue(100);
	PrincipleXVal = 1000.0;
	PrincipleX = new QSlider(Qt::Orientation::Horizontal);
	PrincipleX->setMaximum(200);	PrincipleX->setValue(100);
	PrincipleYVal = 0.0;
	PrincipleY = new QSlider(Qt::Orientation::Horizontal);
	PrincipleY->setMaximum(200);	PrincipleY->setValue(100);
	RadiusVal = 1000.0;
	Radius = new QSlider(Qt::Orientation::Horizontal);
	Radius->setMaximum(200);		Radius->setValue(100);
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
	TranslationX->setMaximum(200);		TranslationX->setValue(100);
	TranslationXVal = 0.0;
	TranslationY = new QSlider(Qt::Orientation::Horizontal);
	TranslationY->setMaximum(200);		TranslationY->setValue(100);
	TranslationYVal = 0.0;
	TranslationZ = new QSlider(Qt::Orientation::Horizontal);
	TranslationZ->setMaximum(200);		TranslationZ->setValue(100);
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
	double Range = 2000.0; double Offset = 0.0;
	double aV = Offset + Range*((double) v / (double) (TranslationX->maximum() - TranslationX->minimum()));
	SetTranslationX(aV);
}

void FluoroPredViz::SetTranslationY(int v){
	double Range = 2000.0; double Offset = 0.0;
	double aV = Offset + Range*((double) v / (double) (TranslationY->maximum() - TranslationY->minimum()));
	SetTranslationY(aV);
}

void FluoroPredViz::SetTranslationZ(int v){
	double Range = 2000.0; double Offset = 0.0;
	double aV = Offset + Range*((double) v / (double) (TranslationZ->maximum() - TranslationZ->minimum()));
	SetTranslationZ(aV);
}

void FluoroPredViz::SetOrientationX(int v){
	double Range = 360.0; double Offset = -180.0;
	double aV = Offset + Range*((double) v / (double) (OrientationX->maximum() - OrientationX->minimum()));
	SetOrientationX(aV);
}

void FluoroPredViz::SetOrientationY(int v){
	double Range = 360.0; double Offset = 0.0;
	double aV = Offset + Range*((double) v / (double) (OrientationY->maximum() - OrientationY->minimum()));
	SetOrientationY(aV);
}

void FluoroPredViz::SetOrientationZ(int v){
	double Range = 360.0; double Offset = 0.0;
	double aV = Offset + Range*((double) v / (double) (OrientationZ->maximum() - OrientationZ->minimum()));
	SetOrientationZ(aV);
}

void FluoroPredViz::SetTranslationX(double v){
	TranslationXVal = v;
	UpdateViz();
}

void FluoroPredViz::SetTranslationY(double v){
	TranslationYVal = v;
	UpdateViz();
}

void FluoroPredViz::SetTranslationZ(double v){
	TranslationZVal = v;
	UpdateViz();
}

void FluoroPredViz::SetOrientationX(double v){
	OrientationXVal = v;
	UpdateViz();
}

void FluoroPredViz::SetOrientationY(double v){
	OrientationYVal = v;
	UpdateViz();
}

void FluoroPredViz::SetOrientationZ(double v){
	OrientationZVal = v;
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

#include "vtkActor.h"
#include "vtkPolyDataMapper.h"
#include "vtkSphereSource.h"
#include "vtkConeSource.h"
#include "vtkCamera.h"
#include "vtkProperty.h"

void FluoroPredViz::UpdateViz(){
	DRRScreen->repaint();
	SchematicScreen->repaint();
}

void FluoroPredViz::ConnectUpPipeline(){



	//build remaining DRR pipeline
	vtkRenderer* DRR_Renderer = vtkRenderer::New();
	vtkRenderWindow* DRR_RenderWindow = vtkRenderWindow::New();
	DRR_RenderWindow->AddRenderer(DRR_Renderer);
	DRR_Renderer->Delete();
	vtkRenderWindowInteractor* DRR_Interactor = vtkRenderWindowInteractor::New();
	DRR_Interactor->Disable();
	DRR_RenderWindow->SetInteractor(DRR_Interactor);
	DRR_Interactor->Delete();
	DRRScreen->SetRenderWindow(DRR_RenderWindow);
	DRR_RenderWindow->Delete();
	DRRScreen->repaint();


	//build angular trajectory markers
	vtkRenderer* Schematic_Renderer = vtkRenderer::New();
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
		Schematic_Renderer->AddActor(Actor);
		Mapper->Delete();
		Actor->Delete();

	}
	UpdateDegreeMarkers();
	Schematic_Renderer->SetBackground(0,0,0);

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
	Schematic_Renderer->AddActor(CenterPointActor);
	CenterPointActor->Delete();

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
	Schematic_Renderer->AddActor(XrayMarkerActor);
	XrayMarkerActor->Delete();
	UpdateXrayMarker();

	//build object location


	
	SchematicScreen->GetRenderWindow()->AddRenderer(Schematic_Renderer);
	Schematic_Renderer->ResetCamera();
	Schematic_Renderer->Delete();
	SchematicScreen->repaint();

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
}

void FluoroPredViz::SetupDRRScreen(QSplitter* WindowsLayout){
	DRRScreen = new QVTKWidget(0);
	DRRScreen->setMinimumWidth(400);
	DRRScreen->setMaximumWidth(2<<16);
	DRRScreen->setSizePolicy( QSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum) );
	WindowsLayout->addWidget(DRRScreen);
}

void FluoroPredViz::SetupSchematicScreen(QSplitter* WindowsLayout){
	SchematicScreen = new QVTKWidget(0);
	SchematicScreen->setMinimumWidth(400);
	SchematicScreen->setMaximumWidth(2<<16);
	SchematicScreen->setSizePolicy( QSizePolicy( QSizePolicy::Maximum, QSizePolicy::Maximum) );
	WindowsLayout->addWidget(SchematicScreen);

}