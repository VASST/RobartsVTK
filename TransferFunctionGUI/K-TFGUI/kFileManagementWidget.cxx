#include "kFileManagementWidget.h"

#include <QVBoxLayout>
#include <QDir>
#include <QFileDialog>

//image file readers
#include "vtkMINCImageReader.h"
#include "vtkMetaImageReader.h"
#include "vtkDICOMImageReader.h"

#include "vtkCudaKohonenApplication.h"

kFileManagementWidget::kFileManagementWidget( transferFunctionWindowWidgetInterface* p ) :
	QWidget( p ), isStatic(true), Map(0)
{
	//keep track of the parent
	parent = p;

	//set the other variables
	maxframes = 30;
	numFrames = 0;
	currFrame = 0;
	nameVector.clear();
	readers.clear();

	//set up dynamic control timer for 20fps
	timer = new QTimer( this );
	timer->stop();
	timer->setSingleShot(false);
	timer->setInterval( 1000/20 );
	connect(timer, SIGNAL(timeout()), this, SLOT(nextFrame()) );
	
	//file layout manages the file list and related buttons
	QVBoxLayout* fileLayout = new QVBoxLayout();
	this->setLayout(fileLayout);

	//set up the file list tab
	files = new QListWidget(this);
	files->setMinimumWidth(100);
	files->setMaximumWidth(250);
	fileLayout->addWidget(files);
	connect(files, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(selectFrame()) );
	files->show();

	//set up the button for toggling between static and dynamic
	toggleModeButton = new QPushButton(this);
	toggleModeButton->setText("Dynamic");
	connect(toggleModeButton, SIGNAL(clicked()), this, SLOT(toggleMode()) );
	fileLayout->addWidget(toggleModeButton);
	toggleModeButton->show();

	setupMenu();
}
	
kFileManagementWidget::~kFileManagementWidget( ) {
	mapper->ClearInput();

	timer->stop();
	delete timer;
	
	//clear up pipeline
	volume->Delete();
	
	//clear the readers
	for(std::vector<vtkImageReader2*>::iterator it = readers.begin();
		it != readers.end(); it++){
		(*it)->Delete();
	}
	readers.clear();
	nameVector.clear();
	
	//clean up Kohonen components
	if(this->Map) this->Map->Delete();
	for(int i = 0; i < this->NumberOfComponents; i++)
		for(std::vector<vtkImageData*>::iterator it = this->ColourImages[i].begin();
			it != this->ColourImages[i].end(); it++)
			(*it)->Delete();
	delete[] this->ColourImages;
	for(std::vector<vtkImageData*>::iterator it = this->ProjectedImages.begin();
		it != this->ProjectedImages.end(); it++)
		(*it)->Delete();
}

void kFileManagementWidget::SetMap(vtkImageData* m){
	if( !this->Map ) this->Map = vtkImageData::New();
	this->Map->DeepCopy(m);
	this->NumberOfComponents = (this->Map->GetNumberOfScalarComponents() -1) / 2;
	this->ColourImages = new std::vector<vtkImageData*> [this->NumberOfComponents];
}

QMenu* kFileManagementWidget::getMenuOptions( ){
	return this->fileMenu;
}

unsigned int kFileManagementWidget::getNumFrames(){
	return this->numFrames;
}

void kFileManagementWidget::setupMenu(){
	
	fileMenu = new QMenu("&File",this);

	//create menu options
	QAction* newMHDFileMenuOption = new QAction("Add Meta Image",fileMenu);
	connect(newMHDFileMenuOption, SIGNAL(triggered()), this, SLOT(addMHDFile()) );
	QAction* newMNCFileMenuOption = new QAction("Add MINC Image",fileMenu);
	connect(newMNCFileMenuOption, SIGNAL(triggered()), this, SLOT(addMNCFile()) );
	QAction* newDICOMFileMenuOption = new QAction("Add DICOM Image",fileMenu);
	connect(newDICOMFileMenuOption, SIGNAL(triggered()), this, SLOT(addDICOMFile()) );
	QAction*exitProgramMenuOption = new QAction("Exit",fileMenu);
	connect(exitProgramMenuOption, SIGNAL(triggered()), parent, SLOT(close()) );

	//add them to the menu
	fileMenu->addAction(newMHDFileMenuOption);
	fileMenu->addAction(newMNCFileMenuOption);
	fileMenu->addAction(newDICOMFileMenuOption);
	fileMenu->addSeparator();
	fileMenu->addAction(exitProgramMenuOption);
	
}

void kFileManagementWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCudaVolumeMapper* c ){
	
	//copy over the pointers to the shared pipeline
	window = w;
	renderer = r;
	mapper = c;

	//set up the remainder of the permanent pipeline
	volume = vtkVolume::New();
	volume->SetMapper(mapper);
	renderer->AddVolume(volume);


}


// ---------------------------------------------------------------------------------------
// Construction and destruction code

void kFileManagementWidget::addMHDFile(){

	//find the requisite filename
	QStringList filenameList = QFileDialog::getOpenFileNames(this, tr("Open File"), QDir::currentPath(),"Meta Image Files (*.mhd)" );

	//if we cancel the window
	if(filenameList.size() == 0) return;
	filenameList.sort();

	for(int i = 0; i < filenameList.size(); i++){
		//add image to the model
		bool result = addMetaImage(filenameList[i].toStdString());

		//if an error occured, print message and do not continue
		if(result){
			std::cerr << "Could not load image " << filenameList[i].toStdString() << "." << std::endl;
			continue;
		}
		// add the image to the list of images
		files->addItem(filenameList[i]);

		//inform parent if successful
		parent->LoadedImageData();
	}


}

void kFileManagementWidget::addMNCFile(){

	//find the requisite filename
	QStringList filenameList = QFileDialog::getOpenFileNames(this, tr("Open File"), QDir::currentPath(), "MINC Image Files (*.mnc)");

	//if we cancel the window
	if(filenameList.size() == 0) return;
	filenameList.sort();

	for(int i = 0; i < filenameList.size(); i++){
		//add image to the model
		bool result = addMincImage(filenameList[i].toStdString());

		//if an error occured, print message and do not continue
		if(result){
			std::cerr << "Could not load image " << filenameList[i].toStdString() << "." << std::endl;
			return;
		}
		
		// add the image to the list of images
		files->addItem(filenameList[i]);
	
		//enable the menu bars if successful
		parent->LoadedImageData();
	}

}

void kFileManagementWidget::addDICOMFile(){

	//find the requisite filename
	QString filename = QFileDialog::getExistingDirectory(this, tr("Open File"), QDir::currentPath() );

	//if we cancel the window
	if(filename.size() == 0) return;

	//add image to the model
	bool result = addDICOMImage(filename.toStdString());

	//if an error occured, print message and do not continue
	if(result){
		std::cerr << "Could not load image " << filename.toStdString() << "." << std::endl;
		return;
	}
	
	// add the image to the list of images
	files->addItem(filename);
	
	//enable the menu bars if successful
	parent->LoadedImageData();

}

void kFileManagementWidget::selectFrame(){

	//grab the current item selected
	QListWidgetItem* curr = files->currentItem();

	//if no item is selected, do nothing
	if(!curr) return;

	//if an item has been selected, grab its filename and move along
	QString filename = curr->text();

	//tell the manager to display this image
	selectFrame(filename.toStdString());
	window->Render();
}

void kFileManagementWidget::toggleMode(){
	//if we are currently static, then we must be toggling to dynamic
	if(isStatic){
		isStatic = false;
		timer->start();
		toggleModeButton->setText("Static");
	}else{
		isStatic = true;
		timer->stop();
		toggleModeButton->setText("Dynamic");
	}
}


vtkImageData* kFileManagementWidget::getCurrentImage(){
	if( numFrames == 0 ) return 0;
	return readers[currFrame]->GetOutput();
}

void kFileManagementWidget::nextFrame(){
	mapper->AdvanceFrame();
	
	//display the screen
	parent->UpdateScreen();
}

// ---------------------------------------------------------------------------------------
// Intraction with model

#include "vtkMetaImageWriter.h"

bool kFileManagementWidget::addImageToMapper(vtkImageData* data){

	//convert frame to projected version
	vtkCudaKohonenApplication* applier = vtkCudaKohonenApplication::New();
	applier->SetMapInput(this->Map);
	applier->SetDataInput(data);
	applier->Update();
	vtkImageData* projected = vtkImageData::New();
	projected->DeepCopy(applier->GetOutput());

	//add frame to map
	numFrames++;
	mapper->SetNumberOfFrames(numFrames);
	if( numFrames == mapper->GetNumberOfFrames() ){
		mapper->SetInput(projected,numFrames-1);
		mapper->ChangeFrame(numFrames-1);
		mapper->Update();
		applier->Delete();
		window->Render();
		projected->Delete();

		return true;
	}
	numFrames--;
	applier->Delete();
	projected->Delete();




	return false;
}

bool kFileManagementWidget::addMetaImage(std::string filename){

	//if we have too many images, don't let us add another
	if(numFrames > maxframes - 1) return true;

	//copy the image in
	vtkImageReader2* metareader = vtkMetaImageReader::New();
	if(!metareader->CanReadFile(filename.c_str())) return true;
	metareader->SetFileName(filename.c_str());
	metareader->Update();
	readers.push_back(metareader);
	vtkImageData* data = metareader->GetOutput();
	data->Update();

	//load image into CUDA
	this->addImageToMapper(data);

	//load image into image container
	nameVector.push_back(filename);
	volume->Update();

	//select the new image
	selectFrame(filename);

	//completed without error
	return false;
}

bool kFileManagementWidget::addMincImage(std::string filename){

	//if we have too many images, don't let us add another
	if(numFrames > maxframes - 1) return true;

	//copy the image in
	vtkImageReader2* mincreader = vtkMINCImageReader::New();
	if(!mincreader->CanReadFile(filename.c_str())) return true;
	mincreader->SetFileName(filename.c_str());
	readers.push_back(mincreader);
	vtkImageData* data = mincreader->GetOutput();
	data->Update();

	//load image into CUDA
	this->addImageToMapper(data);

	//load image into image container
	nameVector.push_back(filename);
	volume->Update();

	//select the new image
	selectFrame(filename);

	//completed without error
	return false;
}

bool kFileManagementWidget::addDICOMImage(std::string dirname){

	//if we have too many images, don't let us add another
	if(numFrames > maxframes - 1) return true;

	//copy the image in
	vtkDICOMImageReader* dicomreader = vtkDICOMImageReader::New();
	if(!dicomreader->CanReadFile(dirname.c_str())) return true;
	dicomreader->SetDirectoryName(dirname.c_str());
	dicomreader->Update();
	readers.push_back(dicomreader);
	vtkImageData* data = dicomreader->GetOutput();
	data->Update();

	//load image into CUDA
	this->addImageToMapper(data);

	//load image into image container
	nameVector.push_back(dirname);
	volume->Update();

	//select the new image
	selectFrame(dirname);

	//completed without error
	return false;
}

bool kFileManagementWidget::removeImage(std::string filename){
	return true;
}

bool kFileManagementWidget::selectFrame(std::string filename){

	//iterate through the list to get the file used
	unsigned int temp = 0;
	std::vector<vtkImageReader2*>::iterator Readit = readers.begin();
	for(std::vector<std::string>::iterator it = nameVector.begin(); it != nameVector.end(); it++){
		if( it->compare(filename) == 0 ) break;
		Readit++;
		temp++;
	}
	currFrame = temp;

	//if we don't have the string, return an error
	if( nameVector[currFrame].compare(filename) != 0 ) return 0;

	mapper->SetNumberOfFrames(numFrames);
	mapper->ChangeFrame(temp);

	return (*Readit)->GetOutput();
}