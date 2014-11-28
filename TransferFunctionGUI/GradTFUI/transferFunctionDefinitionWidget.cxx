#include "transferFunctionDefinitionWidget.h"

#include "vtkCudaFunctionPolygonReader.h"
#include "vtkCudaFunctionPolygonWriter.h"

#include <QFileDialog>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QColorDialog>

// ---------------------------------------------------------------------------------------
// Construction and destruction code
transferFunctionDefinitionWidget::transferFunctionDefinitionWidget( transferFunctionWindowWidget* p,vtkCuda2DTransferFunction* f ) :
	QWidget(p), function(f)
{
	parent = p;
	function->Register(0);

	//prepare the histogram holder
	QVBoxLayout* histogramLayout = new QVBoxLayout();
	QGridLayout* HistogramHolderDefaultLayout = new QGridLayout();
	this->setLayout(histogramLayout);
	histogramHolder = new HistogramHolderDefault(this, function);
	zoomHistogramHolder = new HistogramHolderDefault(this, function);
	histogramHolder->setSize(HISTOSIZE);
	zoomHistogramHolder->setSize(HISTOSIZE);
	histogramLayout->addLayout(HistogramHolderDefaultLayout);
	HistogramHolderDefaultLayout->addWidget(histogramHolder,0,2);
	HistogramHolderDefaultLayout->addWidget(zoomHistogramHolder,3,2);
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
	HistogramHolderDefaultLayout->addWidget(zoomUp,0,0);
	HistogramHolderDefaultLayout->addWidget(zoomDown,0,1);
	HistogramHolderDefaultLayout->addWidget(zoomLeft,1,2);
	HistogramHolderDefaultLayout->addWidget(zoomRight,2,2);
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

transferFunctionDefinitionWidget::~transferFunctionDefinitionWidget(){
	
	//delete the histogram pictures
	delete histogramHolder;
	delete zoomHistogramHolder;

	//delete the zooming sliders
	delete zoomUp;
	delete zoomDown;
	delete zoomLeft;
	delete zoomRight;

	delete transferFunctionMenu;
	
	std::list<vtkCudaFunctionPolygon*>::iterator it;
	for( it = functionObjects.begin(); it != functionObjects.end(); it++){
		(*it)->Delete();
	}
	functionObjects.clear();
	function->Delete();

}

void transferFunctionDefinitionWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCuda2DVolumeMapper* c ){
	window = w;
	renderer = r;
	mapper = c;
}

vtkCuda2DTransferFunction* transferFunctionDefinitionWidget::getTransferFunction(){
	return function;
}

void transferFunctionDefinitionWidget::setupMenu(){

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

QMenu* transferFunctionDefinitionWidget::getMenuOptions(){
	return transferFunctionMenu;
}

unsigned int transferFunctionDefinitionWidget::getHistoSize(){
	return HISTOSIZE;
}

// ---------------------------------------------------------------------------------------
// Interface with the slots and the interface

void transferFunctionDefinitionWidget::keyPressEvent(QKeyEvent* e){
	histogramHolder->keyPressEvent(e);
	zoomHistogramHolder->keyPressEvent(e);
}

void transferFunctionDefinitionWidget::keyReleaseEvent(QKeyEvent* e){
	histogramHolder->keyReleaseEvent(e);
	zoomHistogramHolder->keyReleaseEvent(e);
}

void transferFunctionDefinitionWidget::selectImage(vtkImageData*d){
	data = d;
}

void transferFunctionDefinitionWidget::repaintHistograms(){
	this->histogramHolder->repaint();
	this->zoomHistogramHolder->repaint();
}

void transferFunctionDefinitionWidget::computeHistogram(){

	//get the histogram from the manager
	uchar* histoPtr = (uchar*) getHistogram(data, minIntensity,maxIntensity,minGradient,maxGradient,false);
	if(!histoPtr) return;

	//if the histogram picture is still in use, free it up
	if(histogram){
		histogramHolder->setPixmap(0);
		delete histogram;
	}

	//create a new histogram image and attach it to the viewing area
	histogram = new QImage(histoPtr,HISTOSIZE,HISTOSIZE,QImage::Format_RGB888);
	histogramHolder->giveHistogramDimensions(maxGradient, minGradient, maxIntensity, minIntensity);
	histogramHolder->setPixmap(QPixmap::fromImage(*histogram));

	//free the redundant image memory
	delete histoPtr;
	
}

void transferFunctionDefinitionWidget::computeZoomHistogram(){

	//if we don't have a histogram to zoom in on, don't bother
	if(!histogram) return;

	//grab slider values
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

	//return if we get an invalid range
	if( leftVal > rightVal - 10 ||
		upVal > downVal - 10 ) return;

	//calculate the required gradient and intensity values from the UDLR values
	float lowIntensity = (double) leftVal *(maxIntensity - minIntensity) / (double) HISTOSIZE + minIntensity;
	float highIntensity = (double) rightVal *(maxIntensity - minIntensity) / (double) HISTOSIZE + minIntensity;
	float lowGradient = (double) upVal *(maxGradient - minGradient) / (double) HISTOSIZE + minGradient;
	float highGradient = (double) downVal *(maxGradient - minGradient) / (double) HISTOSIZE + minGradient;

	//get the histogram from the manager
	uchar* histoPtr = (uchar*) getHistogram(data, lowIntensity,highIntensity,lowGradient,highGradient,true);

	//if the histogram picture is still in use, free it up
	if(zoomHistogram){
		zoomHistogramHolder->setPixmap(0);
		delete zoomHistogram;
	}

	//create a new histogram image and attach it to the viewing area
	zoomHistogram = new QImage(histoPtr,HISTOSIZE,HISTOSIZE,QImage::Format_RGB888);
	zoomHistogramHolder->giveHistogramDimensions(highGradient, lowGradient, highIntensity, lowIntensity);
	zoomHistogramHolder->setPixmap(QPixmap::fromImage(*zoomHistogram));
	
	//free the redundant image memory
	delete histoPtr;
	
}

void transferFunctionDefinitionWidget::selectFunctionObject(){
	
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

void transferFunctionDefinitionWidget::setObjectProperties(){

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

void transferFunctionDefinitionWidget::updateFunctionShading(){
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

void transferFunctionDefinitionWidget::updateFunction(){

	function->Modified();
	repaintHistograms();
	parent->UpdateScreen();
}

void transferFunctionDefinitionWidget::addFunctionObject(){
	
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

	//set object properties
	polygon->SetColour(1.0f, 0.0f, 0.0f);
	polygon->SetOpacity(1.0f);
	polygon->AddVertex((this->maxIntensity-this->minIntensity) / 4.0 + this->minIntensity, (this->maxGradient-this->minGradient) / 4.0 + this->minGradient);
	polygon->AddVertex((this->maxIntensity-this->minIntensity) / 4.0 + this->minIntensity, 3.0*(this->maxGradient-this->minGradient) / 4.0 + this->minGradient);
	polygon->AddVertex(3.0*(this->maxIntensity-this->minIntensity) / 4.0 + this->minIntensity, 3.0*(this->maxGradient-this->minGradient) / 4.0 + this->minGradient);
	polygon->AddVertex(3.0*(this->maxIntensity-this->minIntensity) / 4.0 + this->minIntensity, (this->maxGradient-this->minGradient) / 4.0 + this->minGradient);

	//set this object as the current object
	currObject = polygon;
	histogramHolder->setObject(polygon);
	zoomHistogramHolder->setObject(polygon);
	repaintHistograms();

}

void transferFunctionDefinitionWidget::recalculateMaxClassifier(){

	//try to find the maximum classifier value
	short maxEstimate = 0;
	for(unsigned int i = 0; i < objectList->count(); i++){
		QListWidgetItem *item = objectList->item(i);
		if( item->text().toInt() > maxEstimate )
			maxEstimate = item->text().toInt();
	}
	maxClassifier = maxEstimate+1;

}

void transferFunctionDefinitionWidget::removeFunctionObject(){

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

void transferFunctionDefinitionWidget::saveTransferFunction(){

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

void transferFunctionDefinitionWidget::loadTransferFunction(){

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
		polygon->Register( 0 );

		//add object to the lists
		this->objectList->addItem(label);
		this->functionObjects.push_back(polygon);
		this->function->AddFunctionObject( polygon );

		maxClassifier++;
	}
	reader->Delete();

	//repaint the histogram to show changes
	repaintHistograms();
	parent->UpdateScreen();

}

void transferFunctionDefinitionWidget::selectZoomRegion(){
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
	float lowIntensity = (double) leftVal *(maxIntensity - minIntensity) / (double) HISTOSIZE + minIntensity;
	float highIntensity = (double) rightVal *(maxIntensity - minIntensity) / (double) HISTOSIZE + minIntensity;
	float lowGradient = (double) upVal *(maxGradient - minGradient) / (double) HISTOSIZE + minGradient;
	float highGradient = (double) downVal *(maxGradient - minGradient) / (double) HISTOSIZE + minGradient;

	//pass this information to the histogram holder to draw
	histogramHolder->setZoomSquare(highGradient,lowGradient,highIntensity,lowIntensity);
	histogramHolder->repaint();

}

void transferFunctionDefinitionWidget::viewAllObjects(){
	histogramHolder->visualizeAllObjects(true);
	zoomHistogramHolder->visualizeAllObjects(true);
	repaintHistograms();
}

void transferFunctionDefinitionWidget::viewOneObject(){
	histogramHolder->visualizeAllObjects(false);
	zoomHistogramHolder->visualizeAllObjects(false);
	repaintHistograms();
}

// ---------------------------------------------------------------------------------------
// Interface with the model

void transferFunctionDefinitionWidget::addFunctionObject(vtkCudaFunctionPolygon* object){
	function->AddFunctionObject(object);
	functionObjects.push_back(object);
}

void transferFunctionDefinitionWidget::removeFunctionObject(vtkCudaFunctionPolygon* object){
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
	unsigned int histoSize;
	float minIntensity;
	float maxIntensity;
	float minGradient;
	float maxGradient;
	T* image;
	T* gradientImage;
	unsigned int* histogram2d;
	
	void run(){
		unsigned int index = 0;
		for(int y = 0; y < dimY; y++){
			for(int x = 0; x < dimX; x++,index++){

				//calculate the intensity index
				T currVal = image[index];
				unsigned int intensityIndex = (double) histoSize * (currVal - minIntensity) / (maxIntensity - minIntensity);

				//calculate the gradient index
				double gradMag = gradientImage[index];
				unsigned int gradientIndex = (double) histoSize * (gradMag-minGradient) / (maxGradient-minGradient);

				//increment that portion of the histogram
				if(intensityIndex >= 0 && intensityIndex < histoSize && gradientIndex >= 0 && gradientIndex < histoSize){
					histoMutex[gradientIndex*histoSize+intensityIndex]->Lock();
					histogram2d[gradientIndex*histoSize+intensityIndex]++;
					histoMutex[gradientIndex*histoSize+intensityIndex]->Unlock();
				}

			}
		}

	}

};

char* transferFunctionDefinitionWidget::getHistogram(vtkImageData* image, float& retIntensityLow, float& retIntensityHigh, float& retGradientLow, float& retGradientHigh, bool setSize){
	
	//if we have no images to make histograms of, then leave
	if(image == 0) return 0;

	//create the mutex locks
	histoMutex = new vtkMutexLock*[HISTOSIZE*HISTOSIZE];
	for(int i = 0; i < HISTOSIZE*HISTOSIZE; i++)
		histoMutex[i] = vtkMutexLock::New();

	//grab the image you intend to work with
	vtkImageGradientMagnitude* gradientCalculator = vtkImageGradientMagnitude::New();
	gradientCalculator->SetInput(image);
	gradientCalculator->SetDimensionality(3);
	gradientCalculator->Update();
	vtkImageData* gradient = gradientCalculator->GetOutput();

	//get image spacing and dimensions
	int dims[3];
	image->GetDimensions(dims);
	double spacing[3];
	image->GetSpacing(spacing);

	//if we don't provide the histogram parameters, find them from the image data
	float maxVal = retIntensityHigh;
	float minVal = retIntensityLow;
	float maxGrad = retGradientHigh;
	float minGrad = retGradientLow;
	if(!setSize){
		//get intensity range
		double rangeInt[2];
		image->GetPointData()->GetScalars()->GetRange(rangeInt);
		maxVal = rangeInt[1];
		minVal = rangeInt[0];

		//get gradient range
		double rangeGrad[2];
		gradient->GetPointData()->GetScalars()->GetRange(rangeGrad);
		maxGrad = rangeGrad[1];
		minGrad = rangeGrad[0];

		//store returned boundary values
		retIntensityLow = minVal;
		retIntensityHigh = maxVal;
		retGradientHigh = maxGrad;
		retGradientLow = minGrad;
	}

	//create a clear histogram
	unsigned int* tempHisto = new unsigned int[HISTOSIZE*HISTOSIZE];
	for(int i = 0; i < HISTOSIZE*HISTOSIZE; i++){
		tempHisto[i] = 0;
	}

	//populate the temporary histogram buckets using threads (one for each slice)
	unsigned int gradType = gradient->GetScalarType();
	if(image->GetScalarType() != gradType){
		std::cout << "Error - incompatible image and gradient types for histogram" << std::endl;
		exit(1);
	}
	if(image->GetScalarType() == VTK_SHORT){
		histoThread<short>** threads = new histoThread<short>*[dims[2]];
		for(int z = 0; z < dims[2]; z++){
			threads[z] = new histoThread<short>();
			threads[z]->z = z;
			threads[z]->dimX = dims[0];
			threads[z]->dimY = dims[1];
			threads[z]->histoSize = HISTOSIZE;
			threads[z]->minIntensity = minVal;
			threads[z]->maxIntensity = maxVal;
			threads[z]->minGradient = minGrad;
			threads[z]->maxGradient = maxGrad;
			threads[z]->image = (short*) image->GetScalarPointer(0,0,z);
			threads[z]->gradientImage = (short*) gradient->GetScalarPointer(0,0,z);
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
			threads[z]->histoSize = HISTOSIZE;
			threads[z]->minIntensity = minVal;
			threads[z]->maxIntensity = maxVal;
			threads[z]->minGradient = minGrad;
			threads[z]->maxGradient = maxGrad;
			threads[z]->image = (unsigned short*) image->GetScalarPointer(0,0,z);
			threads[z]->gradientImage = (unsigned short*) gradient->GetScalarPointer(0,0,z);
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
			threads[z]->histoSize = HISTOSIZE;
			threads[z]->minIntensity = minVal;
			threads[z]->maxIntensity = maxVal;
			threads[z]->minGradient = minGrad;
			threads[z]->maxGradient = maxGrad;
			threads[z]->image = (char*) image->GetScalarPointer(0,0,z);
			threads[z]->gradientImage = (char*) gradient->GetScalarPointer(0,0,z);
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
			threads[z]->histoSize = HISTOSIZE;
			threads[z]->minIntensity = minVal;
			threads[z]->maxIntensity = maxVal;
			threads[z]->minGradient = minGrad;
			threads[z]->maxGradient = maxGrad;
			threads[z]->image = (unsigned char*) image->GetScalarPointer(0,0,z);
			threads[z]->gradientImage = (unsigned char*) gradient->GetScalarPointer(0,0,z);
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
			threads[z]->histoSize = HISTOSIZE;
			threads[z]->minIntensity = minVal;
			threads[z]->maxIntensity = maxVal;
			threads[z]->minGradient = minGrad;
			threads[z]->maxGradient = maxGrad;
			threads[z]->image = (int*) image->GetScalarPointer(0,0,z);
			threads[z]->gradientImage = (int*) gradient->GetScalarPointer(0,0,z);
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
			threads[z]->histoSize = HISTOSIZE;
			threads[z]->minIntensity = minVal;
			threads[z]->maxIntensity = maxVal;
			threads[z]->minGradient = minGrad;
			threads[z]->maxGradient = maxGrad;
			threads[z]->image = (unsigned int*) image->GetScalarPointer(0,0,z);
			threads[z]->gradientImage = (unsigned int*) gradient->GetScalarPointer(0,0,z);
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
			threads[z]->histoSize = HISTOSIZE;
			threads[z]->minIntensity = minVal;
			threads[z]->maxIntensity = maxVal;
			threads[z]->minGradient = minGrad;
			threads[z]->maxGradient = maxGrad;
			threads[z]->image = (float*) image->GetScalarPointer(0,0,z);
			threads[z]->gradientImage = (float*) gradient->GetScalarPointer(0,0,z);
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
	gradientCalculator->Delete();

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