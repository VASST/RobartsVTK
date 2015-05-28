#include "fileManagementWidget.h"

#include <QVBoxLayout>
#include <QDir>
#include <QFileDialog>

//image file readers
#include "vtkMINCImageReader.h"
#include "vtkMetaImageReader.h"
#include "vtkDICOMImageReader.h"

fileManagementWidget::fileManagementWidget( transferFunctionWindowWidgetInterface* p ) :
  QWidget( p ), isStatic(true)
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
  
fileManagementWidget::~fileManagementWidget( ) {
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

}

QMenu* fileManagementWidget::getMenuOptions( ){
  return this->fileMenu;
}

unsigned int fileManagementWidget::getNumFrames(){
  return this->numFrames;
}

void fileManagementWidget::setupMenu(){
  
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

void fileManagementWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCudaVolumeMapper* c ){
  
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

void fileManagementWidget::addMHDFile(){

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

void fileManagementWidget::addMNCFile(){

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

void fileManagementWidget::addDICOMFile(){

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

void fileManagementWidget::selectFrame(){

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

void fileManagementWidget::toggleMode(){
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


vtkImageData* fileManagementWidget::getCurrentImage(){
  if( numFrames == 0 ) return 0;
  return readers[currFrame]->GetOutput();
}

void fileManagementWidget::nextFrame(){
  mapper->AdvanceFrame();
  
  //display the screen
  parent->UpdateScreen();
}

// ---------------------------------------------------------------------------------------
// Intraction with model

bool fileManagementWidget::addImageToMapper(vtkImageData* data){
  numFrames++;
  mapper->SetNumberOfFrames(numFrames);
  std::cout <<  mapper->GetNumberOfFrames() << std::endl;
  int numFramesInMapper = mapper->GetNumberOfFrames();
  if( numFrames == numFrames ){
    mapper->SetInput(data,numFrames-1);
    mapper->ChangeFrame(numFrames-1);
    mapper->Update();
    return true;
  }
  numFrames--;
  return false;
}

bool fileManagementWidget::addMetaImage(std::string filename){

  //if we have too many images, don't let us add another
  if(numFrames > maxframes - 1) return true;

  //copy the image in
  vtkImageReader2* metareader = vtkMetaImageReader::New();
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

bool fileManagementWidget::addMincImage(std::string filename){

  //if we have too many images, don't let us add another
  if(numFrames > maxframes - 1) return true;

  //copy the image in
  vtkImageReader2* mincreader = vtkMINCImageReader::New();
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

bool fileManagementWidget::addDICOMImage(std::string dirname){

  //if we have too many images, don't let us add another
  if(numFrames > maxframes - 1) return true;

  //copy the image in
  vtkDICOMImageReader* dicomreader = vtkDICOMImageReader::New();
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

bool fileManagementWidget::removeImage(std::string filename){
  return true;
}

bool fileManagementWidget::selectFrame(std::string filename){

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