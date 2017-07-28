#include "SEGUEMainWindow.h"

#include "qpushbutton.h"
#include "qdockwidget.h"
#include "qgridlayout.h"
#include "qmenubar.h"

#include "qshortcut.h"
#include "qkeysequence.h"
#include "qstring.h"
#include "qfileinfo.h"

#include "vtkMetaImageReader.h"
#include "vtkMINCImageReader.h"
#include "vtkMetaImageWriter.h"
#include "vtkMINCImageWriter.h"

#include "vtkTreeWriter.h"
#include "vtkTreeReader.h"

#include <iostream>
#include <stdlib.h>
#include <sstream>

#include "qsizepolicy.h"
#include "qfiledialog.h"

#include "vtkImageThreshold.h"
#include "vtkImageVote.h"
#include "vtkImageMathematics.h"

#include "vtkTreeDFSIterator.h"
#include "vtkCudaHierarchicalMaxFlowSegmentation2.h"
#include "vtkImageEntropyPlaneSelection.h"
#include "vtkHierarchicalMaxFlowSegmentation.h"
#include "vtkCudaImageAtlasLabelProbability.h"
#include "vtkCudaImageLogLikelihood.h"
#include "vtkImageMathematics.h"
#include "vtkCudaImageVote.h"
#include "vtkType.h"
#include "vtkTrivialProducer.h"

#include "QNumberAction.h"

#include "vtkMetaImageWriter.h"

SEGUEMainWindow::SEGUEMainWindow():
lWidget(0), Reader(0)
{

	//set to delete on closure
	this->setAttribute(Qt::WA_DeleteOnClose);

	//set up menus
	QMenu* fileMenu = new QMenu("File",0);
	QMenu* editMenu = new QMenu("Edit",0);
	QMenu* gpusMenu = new QMenu("GPUs",0);
	QMenu* viewMenu = new QMenu("View",0);
	QMenu* helpMenu = new QMenu("Help",0);
	this->menuBar()->addMenu(fileMenu);
	this->menuBar()->addMenu(editMenu);
	this->menuBar()->addMenu(gpusMenu);
	this->menuBar()->addMenu(viewMenu);
	this->menuBar()->addMenu(helpMenu);

	//set up file menu
	QAction* file_updateSegAction = new QAction("Update Segmentation",0);
	file_updateSegAction->setShortcut(QKeySequence("Ctrl+U"));
	QObject::connect(file_updateSegAction,SIGNAL(triggered()),this,SLOT(UpdateSegmentation()));
	fileMenu->addAction(file_updateSegAction);
	QAction* file_suggestPlanesAction = new QAction("Suggest Planes",0);
	file_suggestPlanesAction->setShortcut(QKeySequence("Ctrl+P"));
	QObject::connect(file_suggestPlanesAction,SIGNAL(triggered()),this,SLOT(SuggestPlanes()));
	fileMenu->addAction(file_suggestPlanesAction);
	fileMenu->addSeparator();
	QAction* file_openImageAction = new QAction("Open Image",0);
	file_openImageAction->setShortcut(QKeySequence("Ctrl+O"));
	QObject::connect(file_openImageAction,SIGNAL(triggered()),this,SLOT(OpenImage()));
	fileMenu->addAction(file_openImageAction);
	QAction* file_seedImageAction = new QAction("Load Seeding",0);
	file_seedImageAction->setShortcut(QKeySequence("Ctrl+L"));
	QObject::connect(file_seedImageAction,SIGNAL(triggered()),this,SLOT(LoadSeeding()));
	fileMenu->addAction(file_seedImageAction);
	QAction* file_loadTreeAction = new QAction("Load Tree",0);
	QObject::connect(file_loadTreeAction,SIGNAL(triggered()),this,SLOT(LoadTree()));
	fileMenu->addAction(file_loadTreeAction);
	fileMenu->addSeparator();
	QAction* file_saveSegAction = new QAction("Export Segmentation",0);
	file_saveSegAction->setShortcut(QKeySequence("Ctrl+S"));
	QObject::connect(file_saveSegAction,SIGNAL(triggered()),this,SLOT(SaveSegmentation()));
	fileMenu->addAction(file_saveSegAction);
	QAction* file_saveSeedAction = new QAction("Export Seeds",0);
	file_saveSeedAction->setShortcut(QKeySequence("Ctrl+Shift+S"));
	QObject::connect(file_saveSeedAction,SIGNAL(triggered()),this,SLOT(SaveSeeding()));
	fileMenu->addAction(file_saveSeedAction);
	QAction* file_saveTreeAction = new QAction("Export Tree",0);
	QObject::connect(file_saveTreeAction,SIGNAL(triggered()),this,SLOT(SaveTree()));
	fileMenu->addAction(file_saveTreeAction);

	//set up which GPUs to use
	NumGPUs = vtkCudaDeviceManager::Singleton()->GetNumberOfDevices();
	GPUUsed = new bool[NumGPUs];
	for(int i = 0; i < NumGPUs; i++){
		std::stringstream ss;
		ss << "GPU " << i;
		std::string name = ss.str();
		QNumberAction* gpuToggleAction = new QNumberAction(name.c_str(),0);
		gpuToggleAction->value = i; 
		gpuToggleAction->setCheckable(true);
		GPUUsed[i] = (i == 0);
		gpuToggleAction->setChecked( GPUUsed[i] );
		QObject::connect(gpuToggleAction,SIGNAL(triggered(int)),this,SLOT(ToggleGPU(int)));
		gpusMenu->addAction(gpuToggleAction);
	}

	//set up brush definition dock
	QDockWidget* brushDockHolder = new QDockWidget("Brush Definition",this);
	brushDockHolder->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
	this->addDockWidget(Qt::DockWidgetArea::LeftDockWidgetArea,brushDockHolder);

	//set up brush/update region
	QWidget* brushCollection = new QWidget(brushDockHolder);
	brushCollection->setLayout(new QVBoxLayout());
	QPushButton* UpdateButton = new QPushButton("Update Segmentation",0);
	QObject::connect(UpdateButton,SIGNAL(pressed()),this,SLOT(UpdateSegmentation()));
	brushCollection->layout()->addWidget(UpdateButton);
	QSlider* brushSize = new QSlider(Qt::Orientation::Horizontal);
	brushSize->setMinimum(1);
	brushSize->setMaximum(20);
	brushSize->setValue(2);
	QObject::connect(brushSize,SIGNAL(valueChanged(int)),this,SLOT(BrushSize(int)));
	brushCollection->layout()->addWidget(brushSize);
	brushDockHolder->setWidget(brushCollection);
	brushDockHolder->setSizePolicy(QSizePolicy::Policy::Minimum,QSizePolicy::Policy::Minimum);

	//set up colour/opacity table dock
	cutAlphaSlider = new QSlider(Qt::Orientation::Horizontal);
	cutAlphaSlider->setMinimum(0);
	cutAlphaSlider->setMaximum(100);
	cutAlphaSlider->setValue(50);
	labAlphaSlider = new QSlider(Qt::Orientation::Horizontal);
	labAlphaSlider->setMinimum(0);
	labAlphaSlider->setMaximum(100);
	labAlphaSlider->setValue(100);
	QDockWidget* ctWidgetHolder = new QDockWidget("Image Colour/Opacity",this);
	ctWidgetHolder->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
	ctWidgetHolder->setSizePolicy(QSizePolicy::Policy::Minimum,QSizePolicy::Policy::Minimum);
	this->addDockWidget(Qt::DockWidgetArea::LeftDockWidgetArea,ctWidgetHolder);
	ctWidget = new QColorTableWidget();
	QWidget* temp = new QWidget();
	temp->setLayout(new QVBoxLayout());
	temp->layout()->addWidget(ctWidget);
	temp->layout()->addWidget(new QLabel("Label Opacity"));
	temp->layout()->addWidget(labAlphaSlider);
	temp->layout()->addWidget(new QLabel("Cut Opacity"));
	temp->layout()->addWidget(cutAlphaSlider);
	ctWidgetHolder->setWidget(temp);

	//set up smoothness definition dock
	QDockWidget* sWidgetHolder = new QDockWidget("Smoothness Parameters",this);
	sWidgetHolder->setSizePolicy(QSizePolicy::Policy::Minimum,QSizePolicy::Policy::Minimum);
	sWidgetHolder->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
	this->addDockWidget(Qt::DockWidgetArea::LeftDockWidgetArea,sWidgetHolder);
	smWidget = new QSmoothnessScalarWidget();
	sWidgetHolder->setWidget(smWidget);
	//sWidgetHolder->setSizePolicy(QSizePolicy::Policy::Minimum,QSizePolicy::Policy::Minimum);

	//set up hierarchy definition dock
	QDockWidget* hWidgetHolder = new QDockWidget("Hierarchy",this);
	hWidgetHolder->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
	this->addDockWidget(Qt::DockWidgetArea::LeftDockWidgetArea,hWidgetHolder);
	
	//set up hierarchy widget
	this->hWidget = new QHierarchyWidget(hWidgetHolder);
	hWidgetHolder->setWidget(this->hWidget);
	//hWidgetHolder->setSizePolicy(QSizePolicy::Policy::Minimum,QSizePolicy::Policy::Minimum);
	//this->hWidget->setSizePolicy(QSizePolicy::Policy::Minimum,QSizePolicy::Policy::Maximum);
	QObject::connect(this->hWidget,SIGNAL(AddLabel(int)),this,SLOT(LabelAdded(int)));
	QObject::connect(this->hWidget,SIGNAL(RemoveLabel(int)),this,SLOT(LabelRemoved(int)));
	QObject::connect(this->hWidget,SIGNAL(SelectLabel(int)),this,SLOT(LabelSelected(int)));
	
	this->hWidget->Initialize();

	this->OpenImage();
	this->showMaximized();
	
	SuggestedPlanes[0] = SuggestedPlanes[1] = SuggestedPlanes[2] = 0;
	
	ctWidgetHolder->resize(sWidgetHolder->minimumSize());
	sWidgetHolder->resize(sWidgetHolder->minimumSize());
	brushDockHolder->resize(sWidgetHolder->minimumSize());
}

SEGUEMainWindow::~SEGUEMainWindow(){
	delete this->hWidget;
	if( this->lWidget ) delete this->lWidget;
	if( this->Reader ) this->Reader->Delete();
	delete this->ctWidget;
	delete this->GPUUsed;
}

void SEGUEMainWindow::OpenImage(){

	//get file name
	QString filename = QFileDialog::getOpenFileName(this,"Open Image","", tr("Meta (*.mhd *.mha);; MINC (*.mnc *.minc)") );
	if( filename.isNull() || filename.isEmpty() ) return;
	
	//if any image is already open, remove it
	if( this->lWidget ){
		this->setCentralWidget(0);
		delete this->lWidget;
		this->lWidget = 0;
	}

	//create reader and check validity
	if( this->Reader ) this->Reader->Delete();
	if( filename.endsWith(".mhd",Qt::CaseInsensitive) || filename.endsWith(".mha",Qt::CaseInsensitive) )
		this->Reader = vtkMetaImageReader::New();
	else if( filename.endsWith(".mnc",Qt::CaseInsensitive) || filename.endsWith(".minc",Qt::CaseInsensitive) )
		this->Reader = vtkMINCImageReader::New();
	if( !this->Reader->CanReadFile( filename.toStdString().c_str() ) ){
		this->Reader->Delete();
		this->Reader = 0;
	}
	this->Reader->SetFileName( filename.toStdString().c_str() );
	this->Reader->Update();

	//add the lWidget in
	this->lWidget = new QLabellingWidget(this->Reader, this->ctWidget->GetLookupTable(), 0);
	this->setCentralWidget(this->lWidget);
	QObject::connect(this->ctWidget,SIGNAL(LookupTableChange(vtkLookupTable*)),this->lWidget,SLOT(UpdateColourTable(vtkLookupTable*)));
	QObject::connect(this->lWidget, SIGNAL(destroyed(QObject*)),this,SLOT(ClosingLabellingWindow()));
	QObject::connect(this->hWidget,SIGNAL(ClearLabel(int)),this->lWidget,SLOT(ClearSamplePoints(int)));
	QObject::connect(this->hWidget,SIGNAL(RemoveLabel(int)),this->lWidget,SLOT(ClearSamplePoints(int)));
	QObject::connect(this->hWidget,SIGNAL(SwapLabels(int,int)),this->lWidget,SLOT(SwapLabels(int,int)));
	QObject::connect(this->hWidget,SIGNAL(SelectLabel(int)),this->lWidget,SLOT(ChangeToLabel(int)));
	QObject::connect(this->hWidget,SIGNAL(RecolourLabel(int,int,int,int)),this->lWidget,SLOT(SetColour(int,int,int,int)));
	QObject::connect(this->cutAlphaSlider,SIGNAL(valueChanged(int)),this->lWidget,SLOT(SetCutOpacity(int)));
	QObject::connect(this->labAlphaSlider,SIGNAL(valueChanged(int)),this->lWidget,SLOT(SetLabelOpacity(int)));

	this->hWidget->ForceColourReiterate();
	this->lWidget->ChangeToLabel( this->hWidget->GetCurrentItem() );
	
	SuggestedPlanes[0] = SuggestedPlanes[1] = SuggestedPlanes[2] = 0;

}

void SEGUEMainWindow::ClosingLabellingWindow(){
	//remove the reader
	if(this->Reader) this->Reader->Delete();
	this->Reader = 0;

	//remove any connections
	this->lWidget = 0;
}

vtkImageData* SEGUEMainWindow::LeafSegmentation(int leaf, vtkImageData* cut){
	vtkImageThreshold* thresholder = vtkImageThreshold::New();
	thresholder->SetInputDataObject(cut);
	thresholder->SetInValue(1.0);
	thresholder->SetOutValue(0.0);
	thresholder->ReplaceInOn();
	thresholder->ReplaceOutOn();
	thresholder->ThresholdBetween( (double) leaf, (double) leaf);
	thresholder->SetOutputScalarTypeToChar();
	thresholder->Modified();
	thresholder->Update();
	vtkImageData* retData = vtkImageData::New();
	retData->DeepCopy( thresholder->GetOutput() );
	thresholder->Delete();
	return retData;
}

vtkImageData* SEGUEMainWindow::BranchSegmentation(int branch, vtkImageData* cut){

	std::vector<int>* children = this->hWidget->GetChildren(branch);
	std::vector<vtkObject*> filters;
	
	vtkImageAlgorithm* finalFilter = 0;
	vtkImageMathematics* accumulator = 0;

	for(std::vector<int>::iterator it = children->begin(); it != children->end(); it++){
		vtkImageThreshold* thresholder = vtkImageThreshold::New();
		thresholder->SetInputDataObject(cut);
		thresholder->SetInValue(1.0);
		thresholder->SetOutValue(0.0);
		thresholder->ReplaceInOn();
		thresholder->ReplaceOutOn();
		thresholder->ThresholdBetween( (double) *it, (double) *it);
		thresholder->SetOutputScalarTypeToChar();
		thresholder->Modified();
		thresholder->Update();
		filters.push_back(thresholder);
		
		vtkImageMathematics* newAccumulator = vtkImageMathematics::New();
		filters.push_back(newAccumulator);
		newAccumulator->SetOperationToAdd();
		if( accumulator ){
			accumulator->SetInput2Data( thresholder->GetOutput() );
			newAccumulator->SetInput1Data(accumulator->GetOutput());
			finalFilter = accumulator;
			accumulator->Modified();
			accumulator->Update();
		}else{
			newAccumulator->SetInput1Data(thresholder->GetOutput());
			finalFilter = thresholder;
		}
		accumulator = newAccumulator;

	}
	
	vtkImageData* retVal = vtkImageData::New();
	retVal->DeepCopy(finalFilter->GetOutput());

	//clean pipeline
	for(std::vector<vtkObject*>::iterator it = filters.begin(); it != filters.end(); it++)
		(*it)->Delete();
	delete children;

	return retVal;
}

void SEGUEMainWindow::SaveSegmentation(){

	
	int LabelToSave = this->hWidget->GetCurrentItem();
	std::string LabelName = "Save ";
	LabelName.append(this->hWidget->GetName(LabelToSave));
	LabelName.append(" Label");

	//find the file to save over
	QString saveFilename = QFileDialog::getSaveFileName(this,
		tr(LabelName.c_str()), tr(this->hWidget->GetName(LabelToSave).c_str()),
		tr("Meta (*.mhd *.mhd);;MINC (*.mnc *.minc)"));
	if( saveFilename.isEmpty() || saveFilename.isNull() )
		return;
	QFileInfo saveFile = QFileInfo(saveFilename);

	//get the label to save
	vtkImageData* cutLabel = this->lWidget->GetCut();
	vtkImageData* saveLabel = 0;
	if( this->hWidget->IsBranch(LabelToSave) )
		saveLabel = this->BranchSegmentation(LabelToSave,cutLabel);
	else
		saveLabel = this->LeafSegmentation(LabelToSave,cutLabel);

	//save it
	if( saveFilename.endsWith( QString(".mhd"),Qt::CaseInsensitive) ||
		saveFilename.endsWith( QString(".mha"),Qt::CaseInsensitive) ){
		QString rawFilename = saveFile.baseName();
		rawFilename.append(".raw");
		vtkMetaImageWriter* writer = vtkMetaImageWriter::New();
		writer->SetInputDataObject(saveLabel);
		writer->SetFileName( saveFilename.toStdString().c_str() );
		writer->SetRAWFileName( rawFilename.toStdString().c_str() );
		writer->Write();
		writer->Delete();
	}else if( saveFilename.endsWith( QString(".mnc"),Qt::CaseInsensitive) ||
		saveFilename.endsWith( QString(".minc"),Qt::CaseInsensitive) ){
		vtkMINCImageWriter* writer = vtkMINCImageWriter::New();
		writer->SetInputDataObject(saveLabel);
		writer->SetFileName( saveFilename.toStdString().c_str() );
		writer->Write();
		writer->Delete();
	}

	//clean up
	saveLabel->Delete();

}

void SEGUEMainWindow::SaveSeeding(){

	int LabelToSave = this->hWidget->GetCurrentItem();
	std::string LabelName = "Save ";
	LabelName.append(this->hWidget->GetName(LabelToSave));
	LabelName.append(" Seeds");

	//find the file to save over
	QString saveFilename = QFileDialog::getSaveFileName(this,
		tr(LabelName.c_str()), tr(this->hWidget->GetName(LabelToSave).c_str()),
		tr("Meta (*.mhd *.mhd);;MINC (*.mnc *.minc)"));
	if( saveFilename.isEmpty() || saveFilename.isNull() )
		return;
	QFileInfo saveFile = QFileInfo(saveFilename);

	//get the label to save
	vtkImageData* seedLabel = this->lWidget->GetSeeding();
	vtkImageData* saveLabel = 0;
	if( this->hWidget->IsBranch(LabelToSave) )
		saveLabel = this->BranchSegmentation(LabelToSave,seedLabel);
	else
		saveLabel = this->LeafSegmentation(LabelToSave,seedLabel);

	//save it
	if( saveFilename.endsWith( QString(".mhd"),Qt::CaseInsensitive) ||
		saveFilename.endsWith( QString(".mha"),Qt::CaseInsensitive) ){
		QString rawFilename = saveFile.baseName();
		rawFilename.append(".raw");
		vtkMetaImageWriter* writer = vtkMetaImageWriter::New();
		writer->SetInputDataObject(saveLabel);
		writer->SetFileName( saveFilename.toStdString().c_str() );
		writer->SetRAWFileName( rawFilename.toStdString().c_str() );
		writer->Write();
		writer->Delete();
	}else if( saveFilename.endsWith( QString(".mnc"),Qt::CaseInsensitive) ||
		saveFilename.endsWith( QString(".minc"),Qt::CaseInsensitive) ){
		vtkMINCImageWriter* writer = vtkMINCImageWriter::New();
		writer->SetInputDataObject(saveLabel);
		writer->SetFileName( saveFilename.toStdString().c_str() );
		writer->Write();
		writer->Delete();
	}

	//clean up
	saveLabel->Delete();

}

void SEGUEMainWindow::BrushTypeChanged(QString& NewBrushType){

}

void SEGUEMainWindow::LabelRemoved(int RemovedLabel){
	//std::cout << "Label removed:\t" << RemovedLabel << " : " << this->hWidget->GetName(RemovedLabel) << std::endl;

	//remove any autoselecting shortcuts
	if( RemovedLabel < 10 ){
		QShortcut *shortcut = this->IdentifierToShortcut[RemovedLabel];
		QObject::disconnect(shortcut, SIGNAL(activated()), this, SLOT(SelectLabelFromKey()));
		this->ShortcutToIdentifier.erase(this->ShortcutToIdentifier.find(shortcut));
		this->IdentifierToShortcut.erase(this->IdentifierToShortcut.find(RemovedLabel));
		delete shortcut;
	}

	this->smWidget->RemoveLabel( RemovedLabel );
}

void SEGUEMainWindow::LabelAdded(int AddedLabel){

	//std::cout << "Label added:\t" << AddedLabel << " : " << this->hWidget->GetName(AddedLabel) << std::endl;

	//add autoselecting shortcuts
	if( AddedLabel < 10 ){
		char NumberSymbol = '0' + AddedLabel;
		QString NewShortCut = "Ctrl+";
		NewShortCut.append(QChar(NumberSymbol));
		QShortcut *shortcut = new QShortcut(QKeySequence(NewShortCut), this);
		QObject::connect(shortcut, SIGNAL(activated()), this, SLOT(SelectLabelFromKey()));
		this->ShortcutToIdentifier[shortcut] = AddedLabel;
		this->IdentifierToShortcut[AddedLabel] = shortcut;
	}

	QColor NewLabelColour = this->hWidget->GetColour(AddedLabel);
	if( this->lWidget ) this->lWidget->SetColour(AddedLabel,NewLabelColour.red(),NewLabelColour.green(),NewLabelColour.blue());

	this->smWidget->AddLabel(AddedLabel);
}

void SEGUEMainWindow::SelectLabelFromKey(){
	QShortcut* Sender = (QShortcut*) QObject::sender();
	if( this->ShortcutToIdentifier.find(Sender) == this->ShortcutToIdentifier.end() ) return;
	int LabelReference = this->ShortcutToIdentifier[Sender];
	this->hWidget->ForceSelectLabel( LabelReference );
}

void SEGUEMainWindow::LabelSelected(int SelectedLabel){
	this->smWidget->SelectLabel( SelectedLabel );
}

void SEGUEMainWindow::UpdateSegmentation(){

	//check if a device is usable
	int i = 0;
	for(; i < NumGPUs; i++)
		if( GPUUsed[i] )
			break;
	if( i == NumGPUs)
		return;

	//get input from file reader
	if( !this->Reader ) return;
	vtkImageData* inData = this->Reader->GetOutput();
	
	//get the hierarchy
	std::map<vtkIdType,int> NodeToInt;
	std::map<int,vtkIdType> IntToNode;
	vtkTree* Hierarchy = vtkTree::New();
	this->hWidget->GetHierarchy( Hierarchy, &NodeToInt, &IntToNode);

	//create segmenter
	vtkCudaHierarchicalMaxFlowSegmentation2* segmenter = vtkCudaHierarchicalMaxFlowSegmentation2::New();
	segmenter->SetStructure( Hierarchy );

	
	vtkImageThreshold* zeroThresholder = vtkImageThreshold::New();
	zeroThresholder->SetInputDataObject(this->lWidget->GetSeeding());
	zeroThresholder->SetInValue( -(double) Hierarchy->GetNumberOfVertices() * 4.0);
	zeroThresholder->SetOutValue(0.0);
	zeroThresholder->ReplaceInOn();
	zeroThresholder->ReplaceOutOn();
	zeroThresholder->ThresholdBetween( 0.0, 0.0 );
	zeroThresholder->SetOutputScalarTypeToFloat();
	zeroThresholder->Modified();
	zeroThresholder->Update();

	//put in information into the segmenter
	vtkTreeDFSIterator* Iterator = vtkTreeDFSIterator::New();
	Iterator->SetTree(Hierarchy);
	Iterator->SetStartVertex(Hierarchy->GetRoot());
	Iterator->Next();
	while( Iterator->HasNext() ){
		vtkIdType CurrentNode = Iterator->Next();

		//get the smoothness terms
		segmenter->AddSmoothnessScalar(CurrentNode, this->smWidget->GetSmoothness(NodeToInt[CurrentNode]));

		//get the data terms
		if( Hierarchy->IsLeaf(CurrentNode) ){

			vtkImageThreshold* thresholder = vtkImageThreshold::New();
			thresholder->SetInputDataObject(this->lWidget->GetSeeding());
			thresholder->SetInValue(0.0);
			thresholder->SetOutValue((double) Hierarchy->GetNumberOfVertices() * 4.0);
			thresholder->ReplaceInOn();
			thresholder->ReplaceOutOn();
			thresholder->ThresholdBetween( (double) NodeToInt[CurrentNode], (double) NodeToInt[CurrentNode]);
			thresholder->SetOutputScalarTypeToFloat();
			thresholder->Modified();
			thresholder->Update();

			vtkImageMathematics* addition1 = vtkImageMathematics::New();
			addition1->SetInput1Data(thresholder->GetOutput());
			addition1->SetInput2Data(zeroThresholder->GetOutput());
			addition1->SetOperationToAdd();
			addition1->Update();

			vtkCudaImageLogLikelihood* likelihood = vtkCudaImageLogLikelihood::New();
			vtkTrivialProducer* trivProd2 = vtkTrivialProducer::New();
			trivProd2->SetOutput(inData);
			likelihood->SetInputImageConnection(trivProd2->GetOutputPort());
			trivProd2->Delete();
			vtkTrivialProducer* trivProd = vtkTrivialProducer::New();
			trivProd->SetOutput(this->lWidget->GetSeeding());
			likelihood->SetInputLabelMap(trivProd->GetOutputPort(),0);
			trivProd->Delete();
			likelihood->SetLabelID(NodeToInt[CurrentNode]);
			likelihood->SetNormalizeDataTermOn();
			//likelihood->SetHistogramResolution(0.01);
			likelihood->Update();
			
			vtkImageMathematics* addition2 = vtkImageMathematics::New();
			addition2->SetInput1Data(addition1->GetOutput());
			addition2->SetInput2Data(likelihood->GetOutput());
			addition2->SetOperationToAdd();
			addition2->Update();

			segmenter->SetDataInputDataObject(CurrentNode,addition2->GetOutput());

			likelihood->Delete();
			thresholder->Delete();
			addition1->Delete();
			addition2->Delete();
		}
	}
	Iterator->Delete();

	//set GPUs in use
	segmenter->ClearDevices();
	for(int i = 0; i < NumGPUs; i++)
		if( GPUUsed[i] )
			segmenter->AddDevice(i);

	//run segmentation
	segmenter->SetNumberOfIterations(200);
	segmenter->SetMaxGPUUsage(0.90);
	segmenter->Update();
	vtkImageVote* voter = vtkImageVote::New();
	voter->SetNumberOfThreads(8);
	Iterator = vtkTreeDFSIterator::New();
	Iterator->SetTree(Hierarchy);
	Iterator->SetStartVertex(Hierarchy->GetRoot());
	vtkImageEntropyPlaneSelection* planeSelection = vtkImageEntropyPlaneSelection::New();
	while( Iterator->HasNext() ){
		vtkIdType CurrentNode = Iterator->Next();
		if( Hierarchy->IsLeaf(CurrentNode) ){
			planeSelection->SetInput(NodeToInt[CurrentNode],segmenter->GetOutputDataObject(CurrentNode));
			voter->SetInput(NodeToInt[CurrentNode],segmenter->GetOutputDataObject(CurrentNode));
			voter->SetOutputDataType( VTK_UNSIGNED_CHAR );
		}
	}
	voter->Update();
	planeSelection->Update();
	Iterator->Delete();
	SuggestedPlanes[0] = planeSelection->GetSliceInX();
	SuggestedPlanes[1] = planeSelection->GetSliceInX();
	SuggestedPlanes[2] = planeSelection->GetSliceInZ();

	this->lWidget->SetCut( voter->GetOutput() );
	this->lWidget->Update();

	voter->Delete();
	planeSelection->Delete();
	segmenter->Delete();

}

#include "vtkDoubleArray.h"
#include "vtkDataSetAttributes.h"

void SEGUEMainWindow::SaveTree(){

	QString saveFilename = QFileDialog::getSaveFileName(this,
		tr("Save Tree"), tr("tree.vtk"),
		tr("VTK (*.vtk)"));
	if( saveFilename.isEmpty() || saveFilename.isNull() )
		return;
	
	std::map<vtkIdType,int> NodeToInt;
	std::map<int,vtkIdType> IntToNode;
	vtkTree* Hierarchy = vtkTree::New();
	this->hWidget->GetHierarchy( Hierarchy, &NodeToInt, &IntToNode);

	vtkDoubleArray* smoothnessTerms = vtkDoubleArray::New();
	smoothnessTerms->SetName("Smoothness");
	for( std::map<int,vtkIdType>::iterator it = IntToNode.begin(); it != IntToNode.end(); it++ )
		smoothnessTerms->InsertTuple1( it->second, this->smWidget->GetSmoothness(it->first) );
	Hierarchy->GetVertexData()->AddArray(smoothnessTerms);

	vtkTreeWriter* writer = vtkTreeWriter::New();
	writer->SetFileName(saveFilename.toStdString().c_str());
	writer->SetInputDataObject(Hierarchy);
	writer->Write();
	writer->Delete();

	smoothnessTerms->Delete();

}

void SEGUEMainWindow::LoadSeeding(){
	
	int LabelToLoad = this->hWidget->GetCurrentItem();
	std::string LabelName = "Load ";
	LabelName.append(this->hWidget->GetName(LabelToLoad));
	LabelName.append(" Seeds");

	if( this->hWidget->IsBranch(LabelToLoad) ) return;

	//find the file to load from
	QString loadFilename = QFileDialog::getOpenFileName(this,
		tr(LabelName.c_str()), tr(this->hWidget->GetName(LabelToLoad).c_str()),
		tr("Meta (*.mhd *.mhd);;MINC (*.mnc *.minc)"));
	if( loadFilename.isEmpty() || loadFilename.isNull() )
		return;
	QFileInfo loadFile = QFileInfo(loadFilename);

	//load file
	vtkImageReader2* reader;
	if( loadFilename.endsWith( QString(".mhd"),Qt::CaseInsensitive) ||
		loadFilename.endsWith( QString(".mha"),Qt::CaseInsensitive) )
		reader = vtkMetaImageReader::New();
	else if( loadFilename.endsWith( QString(".mnc"),Qt::CaseInsensitive) ||
		loadFilename.endsWith( QString(".minc"),Qt::CaseInsensitive) )
		reader = vtkMINCImageReader::New();
	reader->SetFileName(loadFilename.toStdString().c_str());
	reader->Update();
	vtkImageData* loadedFile = reader->GetOutput();
	
	//apply over the seeding
	vtkImageData* seeding = this->lWidget->GetSeeding();

	//check file
	if( loadedFile->GetScalarType() != VTK_CHAR &&
		loadedFile->GetScalarType() != VTK_SIGNED_CHAR &&
		loadedFile->GetScalarType() != VTK_UNSIGNED_CHAR ){
		reader->Delete();
		return;
	}
	if( loadedFile->GetDimensions()[0] != seeding->GetDimensions()[0] ||
		loadedFile->GetDimensions()[1] != seeding->GetDimensions()[1] ||
		loadedFile->GetDimensions()[2] != seeding->GetDimensions()[2] ){
		reader->Delete();
		return;
	}

	//apply label
	int VolumeSize = loadedFile->GetDimensions()[0] * loadedFile->GetDimensions()[1] * loadedFile->GetDimensions()[2];
	for(int x = 0; x < VolumeSize; x++){
		if( ((char*)loadedFile->GetScalarPointer())[x] )
			((char*)seeding->GetScalarPointer())[x] = LabelToLoad;
	}

	reader->Delete();

	this->lWidget->Update();

}

void SEGUEMainWindow::LoadTree(){

	QString loadFilename = QFileDialog::getOpenFileName(this,
		tr("Save Tree"), tr("tree.vtk"),
		tr("VTK (*.vtk)"));
	if( loadFilename.isEmpty() || loadFilename.isNull() )
		return;
	
	vtkTreeReader* reader = vtkTreeReader::New();
	reader->SetFileName(loadFilename.toStdString().c_str());
	reader->Update();
	
	std::map<vtkIdType,int> NodeToInt;
	std::map<int,vtkIdType> IntToNode;
	this->hWidget->SetHierarchy( reader->GetOutput(), &NodeToInt, &IntToNode );
	
	for( std::map<int,vtkIdType>::iterator it = IntToNode.begin(); it != IntToNode.end(); it++ )
		this->smWidget->SetSmoothness(it->first, reader->GetOutput()->GetVertexData()->GetArray("Smoothness")->GetTuple1(it->second));
	this->smWidget->update();

	reader->Delete();

}

void SEGUEMainWindow::ToggleGPU(int device){
	this->GPUUsed[device] = !this->GPUUsed[device];
}

void SEGUEMainWindow::SuggestPlanes(){
	this->lWidget->MoveToPlanes(this->SuggestedPlanes[0],this->SuggestedPlanes[1],this->SuggestedPlanes[2]);
}

void SEGUEMainWindow::BrushSize(int s){
	if( this->lWidget ) this->lWidget->BrushSize(s);
}