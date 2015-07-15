#include "DUALTF_transferFunctionDefinitionWidget.h"

#include "vtkCudaFunctionPolygonReader.h"
#include "vtkCudaFunctionPolygonWriter.h"

#include <QFileDialog>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QColorDialog>

#include "vtkDataArray.h"

// ---------------------------------------------------------------------------------------
// Construction and destruction code
DUALTF_transferFunctionDefinitionWidget::DUALTF_transferFunctionDefinitionWidget( DUALTF_transferFunctionWindowWidget* p, vtkCuda2DTransferFunction* f ) :
  QWidget(p), function(f)
{
  parent = p;
  function->Register(0);

  //prepare the histogram holder
  QVBoxLayout* histogramLayout = new QVBoxLayout();
  QGridLayout* DUALTF_HistogramHolderDefaultLayout = new QGridLayout();
  this->setLayout(histogramLayout);
  histogramHolder = new DUALTF_HistogramHolderDefault(this, function);
  zoomHistogramHolder = new DUALTF_HistogramHolderDefault(this, function);
  histogramHolder->setSize(HISTOSIZE);
  zoomHistogramHolder->setSize(HISTOSIZE);
  histogramLayout->addLayout(DUALTF_HistogramHolderDefaultLayout);
  DUALTF_HistogramHolderDefaultLayout->addWidget(histogramHolder,0,2);
  DUALTF_HistogramHolderDefaultLayout->addWidget(zoomHistogramHolder,3,2);
  objectList = new QListWidget(this);
  objectList->setMinimumHeight(50);
  connect(objectList, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(selectFunctionObject()) );
  connect(objectList, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(setObjectProperties()) );
  maxClassifier = 1;
  histogram = 0;
  zoomHistogram = 0;
  functionObjects.clear();
  currObject = 0;

  //prepare the histogram sliders
  zoomLeft = new QSlider(Qt::Orientation::Horizontal,this);
  zoomRight = new QSlider(Qt::Orientation::Horizontal,this);
  zoomUp = new QSlider(Qt::Orientation::Vertical,this);
  zoomDown = new QSlider(Qt::Orientation::Vertical,this);
  zoomLeft->setRange(0,HISTOSIZE);
  zoomRight->setRange(0,HISTOSIZE);
  zoomUp->setRange(0,HISTOSIZE);
  zoomDown->setMaximumHeight( HISTOSIZE );
  zoomDown->setRange(0,HISTOSIZE);
  zoomUp->setMaximumHeight( HISTOSIZE );
  DUALTF_HistogramHolderDefaultLayout->addWidget(zoomUp,0,0);
  DUALTF_HistogramHolderDefaultLayout->addWidget(zoomDown,0,1);
  DUALTF_HistogramHolderDefaultLayout->addWidget(zoomLeft,1,2);
  DUALTF_HistogramHolderDefaultLayout->addWidget(zoomRight,2,2);
  connect(zoomLeft, SIGNAL(valueChanged(int)), this, SLOT(selectZoomRegion()) );
  connect(zoomRight, SIGNAL(valueChanged(int)), this, SLOT(selectZoomRegion()) );
  connect(zoomUp, SIGNAL(valueChanged(int)), this, SLOT(selectZoomRegion()) );
  connect(zoomDown, SIGNAL(valueChanged(int)), this, SLOT(selectZoomRegion()) );

  //prepare the shading sliders
  opacityShader= new QSlider(Qt::Orientation::Horizontal,this);
  ambientShader= new QSlider(Qt::Orientation::Horizontal,this);
  diffuseShader= new QSlider(Qt::Orientation::Horizontal,this);
  specularShader= new QSlider(Qt::Orientation::Horizontal,this);
  specularPowerShader= new QSlider(Qt::Orientation::Horizontal,this);
  opacityShader->setRange(0,1000);
  opacityShader->setValue(1000);
  ambientShader->setRange(0,1000);
  ambientShader->setValue(1000);
  diffuseShader->setRange(0,1000);
  diffuseShader->setValue(0);
  specularShader->setRange(0,1000);
  specularShader->setValue(0);
  specularPowerShader->setRange(0,1000);
  specularPowerShader->setValue(100);
  connect(opacityShader, SIGNAL(valueChanged(int)), this, SLOT(updateFunctionShading()) );
  connect(ambientShader, SIGNAL(valueChanged(int)), this, SLOT(updateFunctionShading()) );
  connect(diffuseShader, SIGNAL(valueChanged(int)), this, SLOT(updateFunctionShading()) );
  connect(specularShader, SIGNAL(valueChanged(int)), this, SLOT(updateFunctionShading()) );
  connect(specularPowerShader, SIGNAL(valueChanged(int)), this, SLOT(updateFunctionShading()) );

  histogramLayout->addWidget( new QLabel("Opacity",this) );
  histogramLayout->addWidget( opacityShader );
  histogramLayout->addWidget( new QLabel("Ambient",this) );
  histogramLayout->addWidget( ambientShader );
  histogramLayout->addWidget( new QLabel("Diffuse",this) );
  histogramLayout->addWidget( diffuseShader );
  histogramLayout->addWidget( new QLabel("Specular",this) );
  histogramLayout->addWidget( specularShader );
  histogramLayout->addWidget( new QLabel("Specular Power",this) );
  histogramLayout->addWidget( specularPowerShader );
  histogramLayout->addWidget( objectList );
  setupMenu();
}

DUALTF_transferFunctionDefinitionWidget::~DUALTF_transferFunctionDefinitionWidget(){
  
  //delete the histogram pictures
  delete histogramHolder;
  delete zoomHistogramHolder;

  //delete the zooming sliders
  delete zoomUp;
  delete zoomDown;
  delete zoomLeft;
  delete zoomRight;

  delete transferFunctionMenu;
  
  function->Delete();
  for( std::list<vtkCudaFunctionPolygon*>::iterator it = functionObjects.begin(); it != functionObjects.end(); it++){
    (*it)->Delete();
  }
  functionObjects.clear();

}

void DUALTF_transferFunctionDefinitionWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCudaDualImageVolumeMapper* c){
  window = w;
  renderer = r;
  mapper = c;
}

vtkCuda2DTransferFunction* DUALTF_transferFunctionDefinitionWidget::getTransferFunction(){
  return function;
}

void DUALTF_transferFunctionDefinitionWidget::setupMenu(){

  transferFunctionMenu = new QMenu("Transfer Function", this);
  
  //save and load transfer functions
  QAction* loadTFObjectMenuOption = new QAction("Load Transfer Function",this);
  connect(loadTFObjectMenuOption, SIGNAL(triggered()), this, SLOT(loadTransferFunction()) );  
  transferFunctionMenu->addAction( loadTFObjectMenuOption );
  QAction* saveTFObjectMenuOption= new QAction("Save Transfer Function",this);
  connect(saveTFObjectMenuOption, SIGNAL(triggered()), this, SLOT(saveTransferFunction()) );
  transferFunctionMenu->addAction( saveTFObjectMenuOption );
  
  //modify current transfer function
  transferFunctionMenu->addSeparator();
  QAction* newTFObjectMenuOption = new QAction("New TF Object",this);
  connect(newTFObjectMenuOption, SIGNAL(triggered()), this, SLOT(addFunctionObject()) );
  transferFunctionMenu->addAction( newTFObjectMenuOption );
  QAction* removeTFObjectMenuOption = new QAction("Remove TF Object",this);
  connect(removeTFObjectMenuOption, SIGNAL(triggered()), this, SLOT(removeFunctionObject()) );
  transferFunctionMenu->addAction( removeTFObjectMenuOption );
  QAction* settingsTFObjectMenuOption = new QAction("Oject Settings",this);
  connect(settingsTFObjectMenuOption, SIGNAL(triggered()), this, SLOT(setObjectProperties()) );
  transferFunctionMenu->addAction( settingsTFObjectMenuOption );
  
  //object viewing functions
  transferFunctionMenu->addSeparator();
  QAction* viewAllTFObjectMenuOption = new QAction("View all objects", this);
  connect(viewAllTFObjectMenuOption,SIGNAL(triggered()),this,SLOT(viewAllObjects()));
  viewAllTFObjectMenuOption->setEnabled(false);
  transferFunctionMenu->addAction( viewAllTFObjectMenuOption );
  QAction* viewOneTFObjectMenuOption = new QAction("View one object at a time", this);
  connect(viewOneTFObjectMenuOption,SIGNAL(triggered()),this,SLOT(viewOneObject()));
  viewAllTFObjectMenuOption->setEnabled(true);
  transferFunctionMenu->addAction( viewOneTFObjectMenuOption );
  
  //histogram updating actions
  transferFunctionMenu->addSeparator();
  QAction* computeZoomHistogramMenuOption = new QAction("Re-compute zoomed-in histogram",this);
  connect(computeZoomHistogramMenuOption, SIGNAL(triggered()), this, SLOT(computeZoomHistogram()) );
  transferFunctionMenu->addAction( computeZoomHistogramMenuOption );
  QAction* computeHistogramMenuOption = new QAction("Re-compute whole histogram",this);
  connect(computeHistogramMenuOption, SIGNAL(triggered()), this, SLOT(computeHistogram()) );
  transferFunctionMenu->addAction( computeHistogramMenuOption );

}

QMenu* DUALTF_transferFunctionDefinitionWidget::getMenuOptions(){
  return transferFunctionMenu;
}

unsigned int DUALTF_transferFunctionDefinitionWidget::getHistoSize(){
  return HISTOSIZE;
}

// ---------------------------------------------------------------------------------------
// Interface with the slots and the interface

void DUALTF_transferFunctionDefinitionWidget::keyPressEvent(QKeyEvent* e){
  histogramHolder->keyPressEvent(e);
  zoomHistogramHolder->keyPressEvent(e);
}

void DUALTF_transferFunctionDefinitionWidget::keyReleaseEvent(QKeyEvent* e){
  histogramHolder->keyReleaseEvent(e);
  zoomHistogramHolder->keyReleaseEvent(e);
}

void DUALTF_transferFunctionDefinitionWidget::selectImage(vtkImageData*d){
  data = d;
}

void DUALTF_transferFunctionDefinitionWidget::repaintHistograms(){
  this->histogramHolder->repaint();
  this->zoomHistogramHolder->repaint();
}

void DUALTF_transferFunctionDefinitionWidget::computeHistogram(){

  //get the histogram from the manager
  uchar* histoPtr = (uchar*) getHistogram(data, minIntensity1,maxIntensity1,minIntensity2,maxIntensity2,false);
  if(!histoPtr) return;

  //if the histogram picture is still in use, free it up
  if(histogram){
    histogramHolder->setPixmap(0);
    delete histogram;
  }

  //create a new histogram image and attach it to the viewing area
  histogram = new QImage(histoPtr,HISTOSIZE,HISTOSIZE,QImage::Format_RGB888);
  histogramHolder->giveHistogramDimensions(maxIntensity1, minIntensity1, maxIntensity2, minIntensity2);
  histogramHolder->setPixmap(QPixmap::fromImage(*histogram));

  //free the redundant image memory
  delete histoPtr;
  
}

void DUALTF_transferFunctionDefinitionWidget::computeZoomHistogram(){

  //if we don't have a histogram to zoom in on, don't bother
  if(!histogram) return;

  //grab slider values
  //get up and down values, making sure up is less than down
  unsigned int upVal = HISTOSIZE-zoomUp->value();
  unsigned int downVal = HISTOSIZE-zoomDown->value();
  if(upVal > downVal){
    unsigned int temp = upVal;
    upVal = downVal;
    downVal = temp;
  }

  //get the left and right values, making sure left is less than right
  unsigned int leftVal = zoomLeft->value();
  unsigned int rightVal = zoomRight->value();
  if(leftVal > rightVal){
    unsigned int temp = leftVal;
    leftVal = rightVal;
    rightVal = temp;
  }

  //return if we get an invalid range
  if( leftVal > rightVal - 5 ||
    upVal > downVal - 5 ) return;

  //calculate the required gradient and intensity values from the UDLR values
  float lowIntensity1 = (double) leftVal *(maxIntensity1 - minIntensity1) / (double) HISTOSIZE + minIntensity1;
  float highIntensity1 = (double) rightVal *(maxIntensity1 - minIntensity1) / (double) HISTOSIZE + minIntensity1;
  float lowIntensity2 = (double) upVal *(maxIntensity2 - minIntensity2) / (double) HISTOSIZE + minIntensity2;
  float highIntensity2 = (double) downVal *(maxIntensity2 - minIntensity2) / (double) HISTOSIZE + minIntensity2;

  //get the histogram from the manager
  uchar* histoPtr = (uchar*) getHistogram(data, lowIntensity1,highIntensity1,lowIntensity2,highIntensity2,true);

  //if the histogram picture is still in use, free it up
  if(zoomHistogram){
    zoomHistogramHolder->setPixmap(0);
    delete zoomHistogram;
  }

  //create a new histogram image and attach it to the viewing area
  zoomHistogram = new QImage(histoPtr,HISTOSIZE,HISTOSIZE,QImage::Format_RGB888);
  zoomHistogramHolder->giveHistogramDimensions(highIntensity1, lowIntensity1, highIntensity2, lowIntensity2);
  zoomHistogramHolder->setPixmap(QPixmap::fromImage(*zoomHistogram));
  
  //free the redundant image memory
  delete histoPtr;
  
}

void DUALTF_transferFunctionDefinitionWidget::selectFunctionObject(){
  
  if(!histogram) return;

  //get the selected objects identifier
  QListWidgetItem* curr = objectList->currentItem();
  if(!curr) return;
  short currClassifier = curr->text().toInt();

  //grab the object from the list, if it exists, else, set it to NULL
  this->currObject = 0;
  vtkCudaFunctionPolygon* tempObject = 0;
  for(std::list<vtkCudaFunctionPolygon*>::iterator it = functionObjects.begin(); it != functionObjects.end(); it++){
    if( (*it)->GetIdentifier() == currClassifier ){
      tempObject = *it;
      break;
    }
  }

  if(!tempObject) return;
  
  //if we have a valid object, start setting the sliders
  this->opacityShader->setValue( tempObject->GetOpacity() *  1000.0f );
  this->ambientShader->setValue( tempObject->GetAmbient() *  1000.0f );
  this->diffuseShader->setValue( tempObject->GetDiffuse() *  1000.0f );
  this->specularShader->setValue( tempObject->GetSpecular() *  1000.0f );
  this->specularPowerShader->setValue( tempObject->GetSpecularPower() *  250.0f );
  this->currObject = tempObject;

  //if we have a valid object, start drawing it on the diagram
  histogramHolder->setObject(currObject);
  zoomHistogramHolder->setObject(currObject);
  repaintHistograms();

}

void DUALTF_transferFunctionDefinitionWidget::setObjectProperties(){

  //if we don't have an object to set the properties of, return
  if(!currObject) return;

  //calculate the original colour
  QColor org;
  org.setAlphaF(currObject->GetOpacity());
  org.setRedF(currObject->GetRedColourValue());
  org.setGreenF(currObject->GetGreenColourValue());
  org.setBlueF(currObject->GetBlueColourValue());

  //open a colour dialog and fetch a floating point rgba value
  parent->releaseKeyboard();
  QColor colour = QColorDialog::getColor(org,this,"TF Colour",QColorDialog::ShowAlphaChannel);
  parent->grabKeyboard();

  //apply the value to the transfer function object
  if(!colour.isValid()) return;
  currObject->SetColour(colour.redF(),colour.greenF(),colour.blueF());
  //currObject->SetOpacity(colour.alphaF());

  //update the transfer function
  function->Modified();

}

void DUALTF_transferFunctionDefinitionWidget::updateFunctionShading(){
  if(currObject){
    currObject->SetOpacity( (float) this->opacityShader->value() / 1000.0f );
    currObject->SetAmbient( (float) this->ambientShader->value() / 1000.0f );
    currObject->SetDiffuse( (float) this->diffuseShader->value() / 1000.0f );
    currObject->SetSpecular( (float) this->specularShader->value() / 1000.0f );
    currObject->SetSpecularPower( (float) this->specularPowerShader->value() / 250.0f );
    function->Modified();
    parent->UpdateScreen();
  }
}

void DUALTF_transferFunctionDefinitionWidget::updateFunction(){

  function->Modified();
  repaintHistograms();
  parent->UpdateScreen();
}

void DUALTF_transferFunctionDefinitionWidget::addFunctionObject(){
  
  if(!histogram) return;

  //get a new classifier
  short currClassifier = maxClassifier;
  QString label = QString::number(maxClassifier);
  maxClassifier++;

  //create a new object
  vtkCudaFunctionPolygon* polygon = vtkCudaFunctionPolygon::New();
  polygon->SetIdentifier(currClassifier);
  
  //update the transfer function
  function->AddFunctionObject(polygon);
  updateFunction();

  //add object to the lists
  objectList->addItem(label);
  functionObjects.push_back(polygon);

  //get a new object
  polygon->SetColour(1.0f, 0.0f, 0.0f);
  polygon->SetOpacity(1.0f);
  polygon->AddVertex((this->maxIntensity1-this->minIntensity1) / 4.0 + this->minIntensity1, (this->maxIntensity2-this->minIntensity2) / 4.0 + this->minIntensity2);
  polygon->AddVertex((this->maxIntensity1-this->minIntensity1) / 4.0 + this->minIntensity1, 3.0*(this->maxIntensity2-this->minIntensity2) / 4.0 + this->minIntensity2);
  polygon->AddVertex(3.0*(this->maxIntensity1-this->minIntensity1) / 4.0 + this->minIntensity1, 3.0*(this->maxIntensity2-this->minIntensity2) / 4.0 + this->minIntensity2);
  polygon->AddVertex(3.0*(this->maxIntensity1-this->minIntensity1) / 4.0 + this->minIntensity1, (this->maxIntensity2-this->minIntensity2) / 4.0 + this->minIntensity2);

  //set this object as the current object
  currObject = polygon;
  histogramHolder->setObject(polygon);
  zoomHistogramHolder->setObject(polygon);
  repaintHistograms();

}

void DUALTF_transferFunctionDefinitionWidget::recalculateMaxClassifier(){

  //try to find the maximum classifier value
  short maxEstimate = 0;
  for(unsigned int i = 0; i < objectList->count(); i++){
    QListWidgetItem *item = objectList->item(i);
    if( item->text().toInt() > maxEstimate )
      maxEstimate = item->text().toInt();
  }
  maxClassifier = maxEstimate+1;

}

void DUALTF_transferFunctionDefinitionWidget::removeFunctionObject(){

  if(!histogram) return;

  //get the selected objects identifier and remove it from the list
  QListWidgetItem* curr = objectList->currentItem();
  if(!curr) return;
  short currClassifier = curr->text().toInt();
  delete curr;

  //recalculate the max classifier
  recalculateMaxClassifier();

  //grab the object from the list, if it exists, else, set it to NULL
  vtkCudaFunctionPolygon* chosenObject = 0;
  for(std::list<vtkCudaFunctionPolygon*>::iterator it = functionObjects.begin(); it != functionObjects.end(); it++){
    if( (*it)->GetIdentifier() == currClassifier ){
      chosenObject = *it;
      break;
    }
  }

  //if we have no object, do nothing
  if(!chosenObject) return;

  removeFunctionObject(chosenObject);
  
  if(currObject == chosenObject){
    histogramHolder->setObject(0);
    zoomHistogramHolder->setObject(0);
    currObject = 0;
    repaintHistograms();
  }

  functionObjects.remove(chosenObject);
  chosenObject->Delete();

}

void DUALTF_transferFunctionDefinitionWidget::saveTransferFunction(){

  //only allow this if we have a histogram
  if(!histogram) return;

  //get a file name
  parent->releaseKeyboard();
  QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), QDir::currentPath(),"2D Transfer Function File (*.2tf)" );
  parent->grabKeyboard();

  //if filename is valid, open a writer
  if(filename.isNull()) return;
  
  vtkCudaFunctionPolygonWriter* writer = vtkCudaFunctionPolygonWriter::New();
  writer->SetFileName( filename.toStdString() );
  for( std::list<vtkCudaFunctionPolygon*>::iterator it = this->functionObjects.begin(); it != this->functionObjects.end(); it++){
    writer->AddInput( *it );
  }
  writer->Write();
  writer->Delete();


}

void DUALTF_transferFunctionDefinitionWidget::loadTransferFunction(){

  //only allow this if we have a histogram
  if(!histogram) return;

  //get a file name
  parent->releaseKeyboard();
  QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath(),"2D Transfer Function File (*.2tf)" );
  parent->grabKeyboard();

  //if filename is valid, open a writer
  if(filename.isNull()) return;
  

  vtkCudaFunctionPolygonReader* reader = vtkCudaFunctionPolygonReader::New();
  reader->SetFileName( filename.toStdString() );
  reader->Read();
  for( int n = 0; n < reader->GetNumberOfOutputs(); n++ ){
    
    //get a new classifier
    short currClassifier = maxClassifier;
    QString label = QString::number(maxClassifier);
    vtkCudaFunctionPolygon* polygon = reader->GetOutput(n);
    polygon->SetIdentifier( maxClassifier );

    //add object to the lists
    this->objectList->addItem(label);
    this->functionObjects.push_back(polygon);
    this->function->AddFunctionObject( polygon );
    polygon->Register( 0 );

    maxClassifier++;
  }
  reader->Delete();

  //repaint the histogram to show changes
  repaintHistograms();
  parent->UpdateScreen();

}

void DUALTF_transferFunctionDefinitionWidget::selectZoomRegion(){
  if(!histogram) return;

  //get up and down values, making sure up is less than down
  unsigned int upVal = HISTOSIZE - zoomUp->value();
  unsigned int downVal = HISTOSIZE - zoomDown->value();
  if(upVal > downVal){
    unsigned int temp = upVal;
    upVal = downVal;
    downVal = temp;
  }

  //get the left and right values, making sure left is less than right
  unsigned int leftVal = zoomLeft->value();
  unsigned int rightVal = zoomRight->value();
  if(leftVal > rightVal){
    unsigned int temp = leftVal;
    leftVal = rightVal;
    rightVal = temp;
  }

  //calculate the required gradient and intensity values for the UDLR values
  float lowIntensity1 = (double) leftVal *(maxIntensity1 - minIntensity1) / (double) HISTOSIZE + minIntensity1;
  float highIntensity1 = (double) rightVal *(maxIntensity1 - minIntensity1) / (double) HISTOSIZE + minIntensity1;
  float lowIntensity2 = (double) upVal *(maxIntensity2 - minIntensity2) / (double) HISTOSIZE + minIntensity2;
  float highIntensity2 = (double) downVal *(maxIntensity2 - minIntensity2) / (double) HISTOSIZE + minIntensity2;

  //pass this information to the histogram holder to draw
  histogramHolder->setZoomSquare(highIntensity1,lowIntensity1,highIntensity2,lowIntensity2);
  histogramHolder->repaint();

}

void DUALTF_transferFunctionDefinitionWidget::viewAllObjects(){
  histogramHolder->visualizeAllObjects(true);
  zoomHistogramHolder->visualizeAllObjects(true);
  repaintHistograms();
}

void DUALTF_transferFunctionDefinitionWidget::viewOneObject(){
  histogramHolder->visualizeAllObjects(false);
  zoomHistogramHolder->visualizeAllObjects(false);
  repaintHistograms();
}

// ---------------------------------------------------------------------------------------
// Interface with the model

void DUALTF_transferFunctionDefinitionWidget::addFunctionObject(vtkCudaFunctionPolygon* object){
  function->AddFunctionObject(object);
  functionObjects.push_back(object);
}

void DUALTF_transferFunctionDefinitionWidget::removeFunctionObject(vtkCudaFunctionPolygon* object){
  function->RemoveFunctionObject(object);

  bool erase = false;
  std::list<vtkCudaFunctionPolygon*>::iterator it;
  for( it = functionObjects.begin(); it != functionObjects.end(); it++){
    if( *it == object ){
      erase = true;
      break;
    }
  }
  if(erase){
    functionObjects.erase(it);
  }
}

// ---------------------------------------------------------------------------------------
// Histogram computation
//------- warning: thar be concurrent programming afoot -------//
#include <QThread>

#include "vtkMutexLock.h"
#include "vtkImageGradientMagnitude.h"
#include "vtkPointData.h"

vtkMutexLock** histoMutex;

template < typename T >
class histoThread : public QThread
{
public:
  unsigned int dimX;
  unsigned int dimY;
  unsigned int z;
  float minIntensity1;
  float maxIntensity1;
  float minIntensity2;
  float maxIntensity2;
  T* image;
  unsigned int* histogram2d;
  
  void run(){
    unsigned int index = 0;
    for(int y = 0; y < dimY; y++){
      for(int x = 0; x < dimX; x++,index++){

        //calculate the intensity index
        T currVal = image[2*index];
        unsigned int intensity1Index = (double) HISTOSIZE * (currVal - minIntensity1) / (maxIntensity1 - minIntensity1);
        currVal = image[2*index+1];
        unsigned int intensity2Index = (double) HISTOSIZE * (currVal - minIntensity2) / (maxIntensity2 - minIntensity2);

        //increment that portion of the histogram
        if(intensity1Index >= 0 && intensity1Index < HISTOSIZE && intensity2Index >= 0 && intensity2Index < HISTOSIZE){
          histoMutex[intensity2Index*HISTOSIZE+intensity1Index]->Lock();
          histogram2d[intensity2Index*HISTOSIZE+intensity1Index]++;
          histoMutex[intensity2Index*HISTOSIZE+intensity1Index]->Unlock();
        }

      }
    }

  }

};

char* DUALTF_transferFunctionDefinitionWidget::getHistogram(vtkImageData* image, float& retIntensity1Low, float& retIntensity1High, float& retIntensity2Low, float& retIntensity2High, bool setSize){
  
  //if we have no images to make histograms of, then leave
  if(image == 0 || image->GetNumberOfScalarComponents() != 2) return 0;

  //create the mutex locks
  histoMutex = new vtkMutexLock*[HISTOSIZE*HISTOSIZE];
  for(int i = 0; i < HISTOSIZE*HISTOSIZE; i++)
    histoMutex[i] = vtkMutexLock::New();

  //get image spacing and dimensions
  int dims[3];
  image->GetDimensions(dims);
  double spacing[3];
  image->GetSpacing(spacing);

  //if we don't provide the histogram parameters, find them from the image data
  if(!setSize){
    //get intensity range
    double rangeInt[2];
    image->GetPointData()->GetScalars()->GetRange(rangeInt,0);
    retIntensity1High = rangeInt[1];
    retIntensity1Low = rangeInt[0];
    image->GetPointData()->GetScalars()->GetRange(rangeInt,1);
    retIntensity2High = rangeInt[1];
    retIntensity2Low = rangeInt[0];
  }

  //create a clear histogram
  unsigned int* tempHisto = new unsigned int[HISTOSIZE*HISTOSIZE];
  for(int i = 0; i < HISTOSIZE*HISTOSIZE; i++){
    tempHisto[i] = 0;
  }

  //populate the temporary histogram buckets using threads (one for each slice)
  if(image->GetScalarType() == VTK_SHORT){
    histoThread<short>** threads = new histoThread<short>*[dims[2]];
    for(int z = 0; z < dims[2]; z++){
      threads[z] = new histoThread<short>();
      threads[z]->z = z;
      threads[z]->dimX = dims[0];
      threads[z]->dimY = dims[1];
      threads[z]->minIntensity1 = retIntensity1Low;
      threads[z]->maxIntensity1 = retIntensity1High;
      threads[z]->minIntensity2 = retIntensity2Low;
      threads[z]->maxIntensity2 = retIntensity2High;
      threads[z]->image = (short*) image->GetScalarPointer(0,0,z);
      threads[z]->histogram2d = tempHisto;
      threads[z]->start();
    }
    for(int z = 0; z < dims[2]; z++){
      (threads[z])->wait();
      delete threads[z];
    }
    delete threads;
  }else if(image->GetScalarType() == VTK_UNSIGNED_SHORT){
    histoThread<unsigned short>** threads = new histoThread<unsigned short>*[dims[2]];
    for(int z = 0; z < dims[2]; z++){
      threads[z] = new histoThread<unsigned short>();
      threads[z]->z = z;
      threads[z]->dimX = dims[0];
      threads[z]->dimY = dims[1];
      threads[z]->minIntensity1 = retIntensity1Low;
      threads[z]->maxIntensity1 = retIntensity1High;
      threads[z]->minIntensity2 = retIntensity2Low;
      threads[z]->maxIntensity2 = retIntensity2High;
      threads[z]->image = (unsigned short*) image->GetScalarPointer(0,0,z);
      threads[z]->histogram2d = tempHisto;
      threads[z]->start();
    }
    for(int z = 0; z < dims[2]; z++){
      (threads[z])->wait();
      delete threads[z];
    }
    delete threads;
  }else if(image->GetScalarType() == VTK_CHAR){
    histoThread<char>** threads = new histoThread<char>*[dims[2]];
    for(int z = 0; z < dims[2]; z++){
      threads[z] = new histoThread<char>();
      threads[z]->z = z;
      threads[z]->dimX = dims[0];
      threads[z]->dimY = dims[1];
      threads[z]->minIntensity1 = retIntensity1Low;
      threads[z]->maxIntensity1 = retIntensity1High;
      threads[z]->minIntensity2 = retIntensity2Low;
      threads[z]->maxIntensity2 = retIntensity2High;
      threads[z]->image = (char*) image->GetScalarPointer(0,0,z);
      threads[z]->histogram2d = tempHisto;
      threads[z]->start();
    }
    for(int z = 0; z < dims[2]; z++){
      (threads[z])->wait();
      delete threads[z];
    }
    delete threads;
  }else if(image->GetScalarType() == VTK_UNSIGNED_CHAR){
    histoThread<unsigned char>** threads = new histoThread<unsigned char>*[dims[2]];
    for(int z = 0; z < dims[2]; z++){
      threads[z] = new histoThread<unsigned char>();
      threads[z]->z = z;
      threads[z]->dimX = dims[0];
      threads[z]->dimY = dims[1];
      threads[z]->minIntensity1 = retIntensity1Low;
      threads[z]->maxIntensity1 = retIntensity1High;
      threads[z]->minIntensity2 = retIntensity2Low;
      threads[z]->maxIntensity2 = retIntensity2High;
      threads[z]->image = (unsigned char*) image->GetScalarPointer(0,0,z);
      threads[z]->histogram2d = tempHisto;
      threads[z]->start();
    }
    for(int z = 0; z < dims[2]; z++){
      (threads[z])->wait();
      delete threads[z];
    }
    delete threads;
  }else if(image->GetScalarType() == VTK_INT){
    histoThread<int>** threads = new histoThread<int>*[dims[2]];
    for(int z = 0; z < dims[2]; z++){
      threads[z] = new histoThread<int>();
      threads[z]->z = z;
      threads[z]->dimX = dims[0];
      threads[z]->dimY = dims[1];
      threads[z]->minIntensity1 = retIntensity1Low;
      threads[z]->maxIntensity1 = retIntensity1High;
      threads[z]->minIntensity2 = retIntensity2Low;
      threads[z]->maxIntensity2 = retIntensity2High;
      threads[z]->image = (int*) image->GetScalarPointer(0,0,z);
      threads[z]->histogram2d = tempHisto;
      threads[z]->start();
    }
    for(int z = 0; z < dims[2]; z++){
      (threads[z])->wait();
      delete threads[z];
    }
    delete threads;
  }else if(image->GetScalarType() == VTK_UNSIGNED_INT){
    histoThread<unsigned int>** threads = new histoThread<unsigned int>*[dims[2]];
    for(int z = 0; z < dims[2]; z++){
      threads[z] = new histoThread<unsigned int>();
      threads[z]->z = z;
      threads[z]->dimX = dims[0];
      threads[z]->dimY = dims[1];
      threads[z]->minIntensity1 = retIntensity1Low;
      threads[z]->maxIntensity1 = retIntensity1High;
      threads[z]->minIntensity2 = retIntensity2Low;
      threads[z]->maxIntensity2 = retIntensity2High;
      threads[z]->image = (unsigned int*) image->GetScalarPointer(0,0,z);
      threads[z]->histogram2d = tempHisto;
      threads[z]->start();
    }
    for(int z = 0; z < dims[2]; z++){
      (threads[z])->wait();
      delete threads[z];
    }
    delete threads;
  }else if(image->GetScalarType() == VTK_FLOAT){
    histoThread<float>** threads = new histoThread<float>*[dims[2]];
    for(int z = 0; z < dims[2]; z++){
      threads[z] = new histoThread<float>();
      threads[z]->z = z;
      threads[z]->dimX = dims[0];
      threads[z]->dimY = dims[1];
      threads[z]->minIntensity1 = retIntensity1Low;
      threads[z]->maxIntensity1 = retIntensity1High;
      threads[z]->minIntensity2 = retIntensity2Low;
      threads[z]->maxIntensity2 = retIntensity2High;
      threads[z]->image = (float*) image->GetScalarPointer(0,0,z);
      threads[z]->histogram2d = tempHisto;
      threads[z]->start();
    }
    for(int z = 0; z < dims[2]; z++){
      (threads[z])->wait();
      delete threads[z];
    }
    delete threads;
  }else{
  }

  //destroy the mutex locks and gradient calculator
  for(int i = 0; i < HISTOSIZE*HISTOSIZE; i++)
    histoMutex[i]->Delete();
  delete histoMutex;

  //find the first and second most full box
  unsigned int maxBox1 = 0;
  unsigned int maxBox2 = 0;
  for(int g = 0; g < HISTOSIZE; g++){
    for(int i = 0; i < HISTOSIZE; i++){
      if(tempHisto[g*HISTOSIZE+i] > maxBox2)
        if(tempHisto[g*HISTOSIZE+i] > maxBox1){
          maxBox2 = maxBox1;
          maxBox1 = tempHisto[g*HISTOSIZE+i];
        }else{
          maxBox2 = tempHisto[g*HISTOSIZE+i];
        }
    }
  }

  //change the temporary histogram into an outputtable one, normalizing it to 256-RGB
  unsigned char* retHisto = new unsigned char[3*HISTOSIZE*HISTOSIZE];
  const double amplifySize = 1.0;
  const double logAmplify = log((double) amplifySize);
  for(int g = 0; g < HISTOSIZE; g++){
    for(int i = 0; i < HISTOSIZE; i++){
      int index = g * HISTOSIZE + i;

      double value = 256.0 - 256.0 * exp( -1.0 * (log((double)(tempHisto[index]+amplifySize)) - logAmplify) / (log((double)(maxBox2+amplifySize)) - logAmplify) );
      if( value < 0.0 ) value = 0.0;
      if( value > 255.0 ) value = 255.0;

      retHisto[3*index] = value;
      retHisto[3*index+1] = value;
      retHisto[3*index+2] = value;

    }
  }

  //return image
  delete tempHisto;
  return (char*) retHisto;
}