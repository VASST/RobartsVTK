#include "QLabellingWidget.h"

#include "qgridlayout.h"
#include "qpixmap.h"
#include "qlabel.h"
#include "qpushbutton.h"

#include "vtkLookupTable.h"
#include "vtkProperty.h"

#include "vtkMarchingCubes.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkImageThreshold.h"
#include "vtkImageConstantPad.h"

#include <iostream>

QLabellingWidget::QLabellingWidget(vtkImageReader2* fileReader, vtkLookupTable* CT, QWidget* parent)
	: QWidget(parent), CurrentLabel(0), Reader(fileReader), LabelOpacity(1.0), CutOpacity(0.5)
{

	this->BinaryLookupTable = vtkLookupTable::New();
	this->BinaryLookupTable->SetNumberOfColors(1);
	this->BinaryLookupTable->SetRange(0.0, 0.0);
	this->BinaryLookupTable->SetTableValue(0,0.0,0.0,0.0,0.0);
	this->BinaryLookupTable->Build();
	
	this->CutLookupTable = vtkLookupTable::New();
	this->CutLookupTable->SetNumberOfColors(1);
	this->CutLookupTable->SetRange(0.0, 0.0);
	this->CutLookupTable->SetTableValue(0,0.0,0.0,0.0,0.0);
	this->CutLookupTable->Build();

	QGridLayout* MainLayout = new QGridLayout();
	this->setLayout(MainLayout);

	ImagePlane[0] = new SliceViewer(0);
	ImagePlane[1] = new SliceViewer(0);
	ImagePlane[2] = new SliceViewer(0);
	
	fileReader->Update();
	mInput = fileReader->GetOutput();
	mInput->Register(0);
	ImagePlane[0]->SetVolume(mInput);
	ImagePlane[1]->SetVolume(mInput);
	ImagePlane[2]->SetVolume(mInput);

	double spacing[3];
	int ext[6];
	int dim[3];
	double orig[3];
	fileReader->GetOutput()->GetExtent(ext);
	fileReader->GetOutput()->GetSpacing(spacing);
	fileReader->GetOutput()->GetDimensions(dim);
	fileReader->GetOutput()->GetOrigin(orig);
	int volumeSize = dim[0]*dim[1]*dim[2];

	mBin=vtkImageData::New();
	mBin->SetSpacing(spacing);
	mBin->SetExtent(ext);
	mBin->SetDimensions(dim);
	mBin->SetOrigin(orig);
	mBin->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
	std::fill_n((unsigned char*)mBin->GetScalarPointer(),sizeof(unsigned char)*dim[0]*dim[1]*dim[2], static_cast<unsigned char>(0));
	
	ImagePlane[0]->SetBin(mBin);
	ImagePlane[1]->SetBin(mBin);
	ImagePlane[2]->SetBin(mBin);

	mCut=vtkImageData::New();
	mCut->SetSpacing(spacing);
	mCut->SetExtent(ext);
	mCut->SetDimensions(dim);
	mCut->SetOrigin(orig);
	mCut->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
	std::fill_n((unsigned char*)mCut->GetScalarPointer(),sizeof(unsigned char)*dim[0]*dim[1]*dim[2], static_cast<unsigned char>(0));
	
	ImagePlane[0]->SetCut(mCut);
	ImagePlane[1]->SetCut(mCut);
	ImagePlane[2]->SetCut(mCut);

	ImagePlane[0]->SetOrientation(0);
	ImagePlane[1]->SetOrientation(1);
	ImagePlane[2]->SetOrientation(2);
	
	double range[2];
	fileReader->GetOutput()->GetScalarRange(range);
	CT->SetRange(range);
	ImagePlane[0]->SetLookupTable(CT);
	ImagePlane[1]->SetLookupTable(CT);
	ImagePlane[2]->SetLookupTable(CT);
	ImagePlane[0]->SetBinaryColorTable(this->BinaryLookupTable);
	ImagePlane[1]->SetBinaryColorTable(this->BinaryLookupTable);
	ImagePlane[2]->SetBinaryColorTable(this->BinaryLookupTable);
	ImagePlane[0]->SetCutColorTable(this->CutLookupTable);
	ImagePlane[1]->SetCutColorTable(this->CutLookupTable);
	ImagePlane[2]->SetCutColorTable(this->CutLookupTable);

	MainLayout->addWidget(ImagePlane[2],0,0);
	MainLayout->addWidget(ImagePlane[1],0,1);
	MainLayout->addWidget(ImagePlane[0],1,1);

	ImagePlane[0]->show();
	ImagePlane[1]->show();
	ImagePlane[2]->show();
	ImagePlane[0]->Create();
	ImagePlane[1]->Create();
	ImagePlane[2]->Create();

	QPushButton* UpdateMeshButton = new QPushButton("Update Mesh");
	QObject::connect( UpdateMeshButton, SIGNAL(pressed()), this, SLOT(Update3DRendering()));

	QVBoxLayout* meshLayout = new QVBoxLayout();
	this->MeshedView = new QVTKWidget();
	this->MeshedRenderer = vtkRenderer::New();
	this->MeshedView->GetRenderWindow()->AddRenderer(this->MeshedRenderer);
	meshLayout->addWidget(this->MeshedView);
	meshLayout->addWidget(UpdateMeshButton);
	MainLayout->addLayout(meshLayout,1,0);
	this->MeshedView->show();

	QObject::connect( ImagePlane[0], SIGNAL(crossSelected(int, int, int)), this, SLOT(MoveToPlanes(int,int,int)));
	QObject::connect( ImagePlane[1], SIGNAL(crossSelected(int, int, int)), this, SLOT(MoveToPlanes(int,int,int)));
	QObject::connect( ImagePlane[2], SIGNAL(crossSelected(int, int, int)), this, SLOT(MoveToPlanes(int,int,int)));

	this->show();

}

QLabellingWidget::~QLabellingWidget(){
	delete ImagePlane[0];
	delete ImagePlane[1];
	delete ImagePlane[2];
	BinaryLookupTable->Delete();
	CutLookupTable->Delete();
	mBin->Delete();
	mCut->Delete();
	mInput->Delete();

	delete MeshedView;
	this->MeshedRenderer->Delete();
}

void QLabellingWidget::ChangeToLabel(int NewLabel){
	this->CurrentLabel = NewLabel;
	this->ImagePlane[0]->labelSlot(NewLabel);
	this->ImagePlane[1]->labelSlot(NewLabel);
	this->ImagePlane[2]->labelSlot(NewLabel);
	//std::cout << "QLabellingWidget::ChangeToLabel( " << NewLabel << " ) called." << std::endl;
}

void QLabellingWidget::ClearSamplePoints(int Label){
	//std::cout << "QLabellingWidget::ClearSamplePoints( " << Label << " ) called." << std::endl;
	
	int Extent[6];
	this->mBin->GetExtent(Extent);
	unsigned char* ptr = (unsigned char*) this->mBin->GetScalarPointer();
	int VolumeSize = (Extent[1]-Extent[0]+1)*(Extent[3]-Extent[2]+1)*(Extent[5]-Extent[4]+1);
	for(int x = 0; x < VolumeSize; x++, ptr++)
		if( *ptr == Label ) *ptr = 0;

	this->Update();

}

void QLabellingWidget::UpdateColourTable(vtkLookupTable* CT){
	double range[2];
	Reader->GetOutput()->GetScalarRange(range);
	CT->SetRange(range);
	this->ImagePlane[0]->SetLookupTable(CT);
	this->ImagePlane[1]->SetLookupTable(CT);
	this->ImagePlane[2]->SetLookupTable(CT);
	this->Update();
}

void QLabellingWidget::Update(){
	this->ImagePlane[0]->Render();
	this->ImagePlane[1]->Render();
	this->ImagePlane[2]->Render();
}


void QLabellingWidget::SetColour(int Label, int r, int g, int b){
	if( this->BinaryLookupTable->GetNumberOfColors() <= Label ){
		this->BinaryLookupTable->SetNumberOfColors(Label+1);
		this->BinaryLookupTable->SetTableRange( 0.0, (double) Label );
	}
	this->BinaryLookupTable->SetTableValue(Label,(double)r/255.0,(double)g/255.0,(double)b/255.0,this->LabelOpacity);

	if( this->CutLookupTable->GetNumberOfColors() <= Label ){
		this->CutLookupTable->SetNumberOfColors(Label+1);
		this->CutLookupTable->SetTableRange( 0.0, (double) Label );
	}
	this->CutLookupTable->SetTableValue(Label,(double)r/255.0,(double)g/255.0,(double)b/255.0,this->CutOpacity);

	this->Update();
}


vtkImageData* QLabellingWidget::GetSeeding(){
	return this->mBin;
}

void QLabellingWidget::SetCut(vtkImageData* newCut ){
	this->mCut->DeepCopy(newCut);
}

vtkImageData* QLabellingWidget::GetCut(){
	return this->mCut;
}

void QLabellingWidget::SetLabelOpacity(int percentage){

	double opacity = (double) percentage / 100.0;
	this->LabelOpacity = opacity;

	for(int i = 1; i < this->BinaryLookupTable->GetNumberOfTableValues(); i++){
		double r = this->BinaryLookupTable->GetTableValue(i)[0];
		double g = this->BinaryLookupTable->GetTableValue(i)[1];
		double b = this->BinaryLookupTable->GetTableValue(i)[2];
		this->BinaryLookupTable->SetTableValue(i,r,g,b,opacity);
	}

	this->Update();

}

void QLabellingWidget::SetCutOpacity(int percentage){

	double opacity = (double) percentage / 100.0;
	this->CutOpacity = opacity;

	for(int i = 1; i < this->CutLookupTable->GetNumberOfTableValues(); i++){
		double r = this->CutLookupTable->GetTableValue(i)[0];
		double g = this->CutLookupTable->GetTableValue(i)[1];
		double b = this->CutLookupTable->GetTableValue(i)[2];
		this->CutLookupTable->SetTableValue(i,r,g,b,0.5*opacity);
	}

	this->Update();

}

void QLabellingWidget::Update3DRendering(){

	this->MeshedRenderer->RemoveAllViewProps();

	for(int i = 1; i < this->BinaryLookupTable->GetNumberOfTableValues(); i++){

		vtkImageThreshold* thresholder = vtkImageThreshold::New();
		thresholder->SetOutputScalarTypeToChar();
		thresholder->SetReplaceIn(1);
		thresholder->SetReplaceOut(1);
		thresholder->SetInValue(255);
		thresholder->SetOutValue(0);
		thresholder->ThresholdBetween( (double)i, (double)i);
		thresholder->SetInputDataObject(this->mCut);
		thresholder->Update();

		bool flag = true;
		for(int x = 0; x < mCut->GetDimensions()[0]*mCut->GetDimensions()[1]*mCut->GetDimensions()[2]; x++){
			if( ((unsigned char*)mCut->GetScalarPointer())[x] == i ){
				flag = false;
				break;
			}
		}
		if( flag ) break;

		vtkImageConstantPad* padder = vtkImageConstantPad::New();
		padder->SetConstant(0);
		padder->SetInputDataObject(thresholder->GetOutput());
		padder->SetOutputWholeExtent( mCut->GetExtent()[0]-1, mCut->GetExtent()[1]+1,
										mCut->GetExtent()[2]-1, mCut->GetExtent()[3]+1,
										mCut->GetExtent()[4]-1, mCut->GetExtent()[5]+1 );
		padder->Update();
		thresholder->Delete();

		vtkMarchingCubes* polyCube = vtkMarchingCubes::New();
		polyCube->SetInputDataObject(padder->GetOutput());
		polyCube->SetValue(0,127);
		polyCube->ComputeNormalsOn();
		polyCube->Update();
		padder->Delete();
		vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
		mapper->SetInputDataObject(polyCube->GetOutput());
		mapper->SetScalarVisibility(false);
		mapper->Update();
		polyCube->Delete();

		vtkProperty* actorProperty = vtkProperty::New();
		double rgb[4];
		this->BinaryLookupTable->GetTableValue(i,rgb);
		actorProperty->SetColor(rgb[0],rgb[1],rgb[2]);
		actorProperty->SetOpacity(0.5);

		vtkActor* actor = vtkActor::New();
		actor->SetMapper(mapper);
		actor->SetProperty(actorProperty);
		this->MeshedRenderer->AddActor(actor);
		actor->Delete();
		mapper->Delete();
		actorProperty->Delete();

	}

	this->MeshedRenderer->ResetCamera();
	this->MeshedView->repaint();

}

void QLabellingWidget::SetBrushSize(int pixelRadius){
	this->ImagePlane[0]->radiusSlot(pixelRadius);
	this->ImagePlane[1]->radiusSlot(pixelRadius);
	this->ImagePlane[2]->radiusSlot(pixelRadius);
}


void QLabellingWidget::SwapLabels(int Node1, int Node2){
	
	int Extent[6];
	this->mBin->GetExtent(Extent);
	unsigned char* ptr = (unsigned char*) this->mBin->GetScalarPointer();
	int VolumeSize = (Extent[1]-Extent[0]+1)*(Extent[3]-Extent[2]+1)*(Extent[5]-Extent[4]+1);
	for(int x = 0; x < VolumeSize; x++, ptr++){
		if( *ptr == Node1 )			*ptr = Node2;
		else if( *ptr == Node2 )	*ptr = Node1;
	}

	this->Update();
}

void QLabellingWidget::MoveToPlanes(int x, int y, int z){
	this->ImagePlane[0]->sliceSlot(x);
	this->ImagePlane[1]->sliceSlot(y);
	this->ImagePlane[2]->sliceSlot(z);
	this->ImagePlane[0]->SetSliceNumber(x);
	this->ImagePlane[1]->SetSliceNumber(y);
	this->ImagePlane[2]->SetSliceNumber(z);
	this->Update();
}


void QLabellingWidget::BrushSize(int s){
	this->ImagePlane[0]->radiusSlot(s);
	this->ImagePlane[1]->radiusSlot(s);
	this->ImagePlane[2]->radiusSlot(s);
}