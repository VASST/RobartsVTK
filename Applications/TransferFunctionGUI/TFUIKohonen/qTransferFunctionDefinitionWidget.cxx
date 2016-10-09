/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "qHistogramHolderLabel.h"
#include "qTransferFunctionDefinitionWidget.h"
#include "qTransferFunctionWindowWidget.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaDualImageVolumeMapper.h"
#include "vtkCudaFunctionPolygon.h"
#include "vtkCudaFunctionPolygonReader.h"
#include "vtkCudaFunctionPolygonWriter.h"
#include "vtkImageData.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include <QColorDialog>
#include <QComboBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QListWidget>
#include <QListWidgetItem>
#include <QMenu>
#include <QObject>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>
#include <float.h>

// ---------------------------------------------------------------------------------------
// Construction and destruction code
qTransferFunctionDefinitionWidget::qTransferFunctionDefinitionWidget( qTransferFunctionWindowWidget* p, vtkCuda2DTransferFunction* f, int s1, int s2 ) :
  QWidget(p), function(f), HISTOSIZE1(s1), HISTOSIZE2(s2), Max(0), Min(0)
{
  parent = p;
  function->Register(0);

  //get the widget's overall layout
  QVBoxLayout* histogramLayout = new QVBoxLayout();

  //prepare the colour variables
  QHBoxLayout* colourVariableLayout = new QHBoxLayout();
  redVariable = new QComboBox(this);
  greenVariable = new QComboBox(this);
  blueVariable = new QComboBox(this);
  colourVariableLayout->addWidget( new QLabel("R",this) );
  colourVariableLayout->addWidget( redVariable );
  colourVariableLayout->addWidget( new QLabel("G",this) );
  colourVariableLayout->addWidget( greenVariable );
  colourVariableLayout->addWidget( new QLabel("B",this) );
  colourVariableLayout->addWidget( blueVariable );
  histogramLayout->addLayout(colourVariableLayout);
  redChosen = greenChosen = blueChosen = 0;
  connect(redVariable, SIGNAL(currentIndexChanged (int)), this, SLOT(setRed(int)) );
  connect(greenVariable, SIGNAL(currentIndexChanged (int)), this, SLOT(setGreen(int)) );
  connect(blueVariable, SIGNAL(currentIndexChanged (int)), this, SLOT(setBlue(int)) );

  //prepare the histogram holder
  QGridLayout* qHistogramHolderLabelLayout = new QGridLayout();
  this->setLayout(histogramLayout);
  histogramHolder = new qHistogramHolderLabel(this, function);
  zoomHistogramHolder = new qHistogramHolderLabel(this, function);
  histogramHolder->setSize(HISTOSIZE1,HISTOSIZE2);
  zoomHistogramHolder->setSize(HISTOSIZE1,HISTOSIZE2);
  histogramLayout->addLayout(qHistogramHolderLabelLayout);
  qHistogramHolderLabelLayout->addWidget(histogramHolder,0,2);
  qHistogramHolderLabelLayout->addWidget(zoomHistogramHolder,3,2);
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
  zoomLeft->setRange(0,HISTOSIZE1);
  zoomRight->setRange(0,HISTOSIZE1);
  zoomUp->setRange(0,HISTOSIZE2);
  zoomDown->setMaximumHeight( HISTOSIZE2 );
  zoomDown->setRange(0,HISTOSIZE2);
  zoomUp->setMaximumHeight( HISTOSIZE2 );
  qHistogramHolderLabelLayout->addWidget(zoomUp,0,0);
  qHistogramHolderLabelLayout->addWidget(zoomDown,0,1);
  qHistogramHolderLabelLayout->addWidget(zoomLeft,1,2);
  qHistogramHolderLabelLayout->addWidget(zoomRight,2,2);
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

qTransferFunctionDefinitionWidget::~qTransferFunctionDefinitionWidget()
{

  //delete the histogram pictures
  delete histogramHolder;
  delete zoomHistogramHolder;

  //delete the zooming sliders
  delete zoomUp;
  delete zoomDown;
  delete zoomLeft;
  delete zoomRight;

  //delete the colour variables
  delete redVariable;
  delete greenVariable;
  delete blueVariable;

  delete transferFunctionMenu;

  function->Delete();
  for( std::list<vtkCudaFunctionPolygon*>::iterator it = functionObjects.begin(); it != functionObjects.end(); it++)
  {
    (*it)->Delete();
  }
  functionObjects.clear();

}

void qTransferFunctionDefinitionWidget::setStandardWidgets( vtkRenderWindow* w, vtkRenderer* r, vtkCudaDualImageVolumeMapper* c)
{
  window = w;
  renderer = r;
  mapper = c;
}

void qTransferFunctionDefinitionWidget::setHistoSize(int s1, int s2)
{
  this->HISTOSIZE1 = s1;
  this->HISTOSIZE2 = s2;
}

vtkCuda2DTransferFunction* qTransferFunctionDefinitionWidget::getTransferFunction()
{
  return function;
}

void qTransferFunctionDefinitionWidget::setupMenu()
{

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

QMenu* qTransferFunctionDefinitionWidget::getMenuOptions()
{
  return transferFunctionMenu;
}

unsigned int qTransferFunctionDefinitionWidget::getHistoSize(int i)
{
  if(i == 1)
  {
    return HISTOSIZE1;
  }
  if(i == 2)
  {
    return HISTOSIZE2;
  }
  return -1;
}

// ---------------------------------------------------------------------------------------
// Interface with the slots and the interface

void qTransferFunctionDefinitionWidget::keyPressEvent(QKeyEvent* e)
{
  histogramHolder->keyPressEvent(e);
  zoomHistogramHolder->keyPressEvent(e);
}

void qTransferFunctionDefinitionWidget::keyReleaseEvent(QKeyEvent* e)
{
  histogramHolder->keyReleaseEvent(e);
  zoomHistogramHolder->keyReleaseEvent(e);
}

//----------------------------------------------------------------------------
double qTransferFunctionDefinitionWidget::GetRMax()
{
  return this->Max[this->GetRed()];
}

//----------------------------------------------------------------------------
double qTransferFunctionDefinitionWidget::GetGMax()
{
  return this->Max[this->GetGreen()];
}

//----------------------------------------------------------------------------
double qTransferFunctionDefinitionWidget::GetBMax()
{
  return this->Max[this->GetBlue()];
}

//----------------------------------------------------------------------------
double qTransferFunctionDefinitionWidget::GetRMin()
{
  return this->Min[this->GetRed()];
}

//----------------------------------------------------------------------------
double qTransferFunctionDefinitionWidget::GetGMin()
{
  return this->Min[this->GetGreen()];
}

//----------------------------------------------------------------------------
double qTransferFunctionDefinitionWidget::GetBMin()
{
  return this->Min[this->GetBlue()];
}

void qTransferFunctionDefinitionWidget::SetMax(double val, int scalar)
{
  if(scalar >= 0 && scalar < scalars)
  {
    Max[scalar] = (Max[scalar] < val) ? val: Max[scalar];
  }
}

void qTransferFunctionDefinitionWidget::SetMin(double val, int scalar)
{
  if(scalar >= 0 && scalar < scalars)
  {
    Min[scalar] = (Min[scalar] > val) ? val: Min[scalar];
  }
}

void qTransferFunctionDefinitionWidget::selectImage(vtkImageData*d)
{
  data = d;

  //update max-min
  float* weights = (float*) data->GetScalarPointer();
  int sJump = data->GetNumberOfScalarComponents();
  scalars = (sJump-1)/2;
  if(this->Max)
  {
    delete this->Max;
  }
  if(this->Min)
  {
    delete this->Min;
  }
  this->Max = new float[scalars];
  this->Min = new float[scalars];
  for(int i = 0; i < scalars; i++)
  {
    this->Max[i] = -FLT_MAX;
    this->Min[i] = FLT_MAX;
  }
  for(int j = 0; j < HISTOSIZE1*HISTOSIZE2; j++)
  {
    weights++;
    for(int i = 0; i < scalars; i++)
    {
      this->Max[i] = (Max[i] > *weights) ? Max[i]: *weights;
      weights++;
      this->Min[i] = (Min[i] < *weights) ? Min[i]: *weights;
      weights++;
    }
  }

  //update combo boxes
  redVariable->clear();
  greenVariable->clear();
  blueVariable->clear();
  for(int s = 0; s < scalars; s++)
  {
    redVariable->addItem(QString::number(s));
    greenVariable->addItem(QString::number(s));
    blueVariable->addItem(QString::number(s));
  }

}

void qTransferFunctionDefinitionWidget::repaintHistograms()
{
  this->histogramHolder->repaint();
  this->zoomHistogramHolder->repaint();
}

void qTransferFunctionDefinitionWidget::computeHistogram()
{
  //get the histogram from the manager
  uchar* histoPtr = (uchar*) getHistogram(data,0.0,(double)HISTOSIZE1-1,0.0,(double)HISTOSIZE2-1,false);
  if(!histoPtr)
  {
    return;
  }

  //if the histogram picture is still in use, free it up
  if(histogram)
  {
    delete histogram;
  }

  //create a new histogram image and attach it to the viewing area
  histogram = new QImage(histoPtr,HISTOSIZE1,HISTOSIZE2,QImage::Format_RGB888);
  histogramHolder->giveHistogramDimensions(HISTOSIZE1, 0, HISTOSIZE2, 0);
  histogramHolder->setPixmap(QPixmap::fromImage(*histogram));

  //free the redundant image memory
  delete histoPtr;
}

void qTransferFunctionDefinitionWidget::computeZoomHistogram()
{
  //if we don't have a histogram to zoom in on, don't bother
  if(!histogram)
  {
    return;
  }

  //grab slider values
  //get up and down values, making sure up is less than down
  unsigned int upVal = HISTOSIZE2-zoomUp->value();
  unsigned int downVal = HISTOSIZE2-zoomDown->value();
  if(upVal > downVal)
  {
    unsigned int temp = upVal;
    upVal = downVal;
    downVal = temp;
  }

  //get the left and right values, making sure left is less than right
  unsigned int leftVal = zoomLeft->value();
  unsigned int rightVal = zoomRight->value();
  if(leftVal > rightVal)
  {
    unsigned int temp = leftVal;
    leftVal = rightVal;
    rightVal = temp;
  }

  //return if we get an invalid range
  if( leftVal > rightVal - 5 ||
      upVal > downVal - 5 )
  {
    return;
  }

  //calculate the required gradient and intensity values from the UDLR values
  float lowIntensity1 = (double) leftVal *(HISTOSIZE1 - 0) / (double) HISTOSIZE1 + 0;
  float highIntensity1 = (double) rightVal *(HISTOSIZE1 - 0) / (double) HISTOSIZE1 + 0;
  float lowIntensity2 = (double) upVal *(HISTOSIZE2 - 0) / (double) HISTOSIZE2 + 0;
  float highIntensity2 = (double) downVal *(HISTOSIZE2 - 0) / (double) HISTOSIZE2 + 0;

  //get the histogram from the manager
  uchar* histoPtr = (uchar*) getHistogram(data, lowIntensity1,highIntensity1,lowIntensity2,highIntensity2,true);

  //if the histogram picture is still in use, free it up
  if(zoomHistogram)
  {
    delete zoomHistogram;
  }

  //create a new histogram image and attach it to the viewing area
  zoomHistogram = new QImage(histoPtr,HISTOSIZE1,HISTOSIZE2,QImage::Format_RGB888);
  zoomHistogramHolder->giveHistogramDimensions(highIntensity1, lowIntensity1, highIntensity2, lowIntensity2);
  zoomHistogramHolder->setPixmap(QPixmap::fromImage(*zoomHistogram));

  //free the redundant image memory
  delete histoPtr;
}

void qTransferFunctionDefinitionWidget::selectFunctionObject()
{
  if(!histogram)
  {
    return;
  }

  //get the selected objects identifier
  QListWidgetItem* curr = objectList->currentItem();
  if(!curr)
  {
    return;
  }
  short currClassifier = curr->text().toInt();

  //grab the object from the list, if it exists, else, set it to NULL
  this->currObject = 0;
  vtkCudaFunctionPolygon* tempObject = 0;
  for(std::list<vtkCudaFunctionPolygon*>::iterator it = functionObjects.begin(); it != functionObjects.end(); it++)
  {
    if( (*it)->GetIdentifier() == currClassifier )
    {
      tempObject = *it;
      break;
    }
  }

  if(!tempObject)
  {
    return;
  }

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

void qTransferFunctionDefinitionWidget::setObjectProperties()
{
  //if we don't have an object to set the properties of, return
  if(!currObject)
  {
    return;
  }

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
  if(!colour.isValid())
  {
    return;
  }
  currObject->SetColour(colour.redF(),colour.greenF(),colour.blueF());
  //currObject->SetOpacity(colour.alphaF());

  //update the transfer function
  function->Modified();
}

void qTransferFunctionDefinitionWidget::updateFunctionShading()
{
  if(currObject)
  {
    currObject->SetOpacity( (float) this->opacityShader->value() / 1000.0f );
    currObject->SetAmbient( (float) this->ambientShader->value() / 1000.0f );
    currObject->SetDiffuse( (float) this->diffuseShader->value() / 1000.0f );
    currObject->SetSpecular( (float) this->specularShader->value() / 1000.0f );
    currObject->SetSpecularPower( (float) this->specularPowerShader->value() / 250.0f );
    function->Modified();
    parent->UpdateScreen();
  }
}

void qTransferFunctionDefinitionWidget::updateFunction()
{
  function->Modified();
  repaintHistograms();
  parent->UpdateScreen();
}

void qTransferFunctionDefinitionWidget::addFunctionObject()
{
  if(!histogram)
  {
    return;
  }

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
  polygon->AddVertex((double)(this->HISTOSIZE1-1) / 4.0, (double)(this->HISTOSIZE2-1) / 4.0);
  polygon->AddVertex((this->HISTOSIZE1-1) / 4.0, 3.0*(double)(this->HISTOSIZE2-1) / 4.0);
  polygon->AddVertex(3.0*(double)(this->HISTOSIZE1-1) / 4.0, 3.0*(double)(this->HISTOSIZE2-1) / 4.0);
  polygon->AddVertex(3.0*(double)(this->HISTOSIZE1-1) / 4.0, (double)(this->HISTOSIZE2-1) / 4.0);

  //set this object as the current object
  currObject = polygon;
  histogramHolder->setObject(polygon);
  zoomHistogramHolder->setObject(polygon);
  repaintHistograms();
}

void qTransferFunctionDefinitionWidget::recalculateMaxClassifier()
{
  //try to find the maximum classifier value
  short maxEstimate = 0;
  for(int i = 0; i < objectList->count(); i++)
  {
    QListWidgetItem *item = objectList->item(i);
    if( item->text().toInt() > maxEstimate )
    {
      maxEstimate = item->text().toInt();
    }
  }
  maxClassifier = maxEstimate+1;
}

void qTransferFunctionDefinitionWidget::removeFunctionObject()
{
  if(!histogram)
  {
    return;
  }

  //get the selected objects identifier and remove it from the list
  QListWidgetItem* curr = objectList->currentItem();
  if(!curr)
  {
    return;
  }
  short currClassifier = curr->text().toInt();
  delete curr;

  //recalculate the max classifier
  recalculateMaxClassifier();

  //grab the object from the list, if it exists, else, set it to NULL
  vtkCudaFunctionPolygon* chosenObject = 0;
  for(std::list<vtkCudaFunctionPolygon*>::iterator it = functionObjects.begin(); it != functionObjects.end(); it++)
  {
    if( (*it)->GetIdentifier() == currClassifier )
    {
      chosenObject = *it;
      break;
    }
  }

  //if we have no object, do nothing
  if(!chosenObject)
  {
    return;
  }

  removeFunctionObject(chosenObject);

  if(currObject == chosenObject)
  {
    histogramHolder->setObject(0);
    zoomHistogramHolder->setObject(0);
    currObject = 0;
    repaintHistograms();
  }

  functionObjects.remove(chosenObject);
  chosenObject->Delete();
}

void qTransferFunctionDefinitionWidget::saveTransferFunction()
{
  //only allow this if we have a histogram
  if(!histogram)
  {
    return;
  }

  //get a file name
  parent->releaseKeyboard();
  QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), QDir::currentPath(),"2D Transfer Function File (*.2tf)" );
  parent->grabKeyboard();

  //if filename is valid, open a writer
  if(filename.isNull())
  {
    return;
  }

  vtkCudaFunctionPolygonWriter* writer = vtkCudaFunctionPolygonWriter::New();
  writer->SetFileName( std::string(filename.toLatin1().data()) );
  for( std::list<vtkCudaFunctionPolygon*>::iterator it = this->functionObjects.begin(); it != this->functionObjects.end(); it++)
  {
    writer->AddInput( *it );
  }
  writer->Write();
  writer->Delete();
}

void qTransferFunctionDefinitionWidget::loadTransferFunction()
{
  //only allow this if we have a histogram
  if(!histogram)
  {
    return;
  }

  //get a file name
  parent->releaseKeyboard();
  QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::currentPath(),"2D Transfer Function File (*.2tf)" );
  parent->grabKeyboard();

  //if filename is valid, open a writer
  if(filename.isNull())
  {
    return;
  }

  vtkCudaFunctionPolygonReader* reader = vtkCudaFunctionPolygonReader::New();
  reader->SetFileName( std::string(filename.toLatin1().data()) );
  reader->Read();
  for( int n = 0; n < reader->GetNumberOfOutputs(); n++ )
  {

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

void qTransferFunctionDefinitionWidget::selectZoomRegion()
{
  if(!histogram)
  {
    return;
  }

  //get up and down values, making sure up is less than down
  unsigned int upVal = HISTOSIZE2 - zoomUp->value();
  unsigned int downVal = HISTOSIZE2 - zoomDown->value();
  if(upVal > downVal)
  {
    unsigned int temp = upVal;
    upVal = downVal;
    downVal = temp;
  }

  //get the left and right values, making sure left is less than right
  unsigned int leftVal = zoomLeft->value();
  unsigned int rightVal = zoomRight->value();
  if(leftVal > rightVal)
  {
    unsigned int temp = leftVal;
    leftVal = rightVal;
    rightVal = temp;
  }

  //calculate the required gradient and intensity values for the UDLR values
  float lowIntensity1 = (double) leftVal *(HISTOSIZE1 - 0) / (double) HISTOSIZE1 + 0;
  float highIntensity1 = (double) rightVal *(HISTOSIZE1 - 0) / (double) HISTOSIZE1 + 0;
  float lowIntensity2 = (double) upVal *(HISTOSIZE2 - 0) / (double) HISTOSIZE2 + 0;
  float highIntensity2 = (double) downVal *(HISTOSIZE2 - 0) / (double) HISTOSIZE2 + 0;

  //pass this information to the histogram holder to draw
  histogramHolder->setZoomSquare(highIntensity1,lowIntensity1,highIntensity2,lowIntensity2);
  histogramHolder->repaint();
}

void qTransferFunctionDefinitionWidget::viewAllObjects()
{
  histogramHolder->visualizeAllObjects(true);
  zoomHistogramHolder->visualizeAllObjects(true);
  repaintHistograms();
}

void qTransferFunctionDefinitionWidget::viewOneObject()
{
  histogramHolder->visualizeAllObjects(false);
  zoomHistogramHolder->visualizeAllObjects(false);
  repaintHistograms();
}

// ---------------------------------------------------------------------------------------
// Interface with the model

void qTransferFunctionDefinitionWidget::addFunctionObject(vtkCudaFunctionPolygon* object)
{
  function->AddFunctionObject(object);
  functionObjects.push_back(object);
}

void qTransferFunctionDefinitionWidget::removeFunctionObject(vtkCudaFunctionPolygon* object)
{
  function->RemoveFunctionObject(object);

  bool erase = false;
  std::list<vtkCudaFunctionPolygon*>::iterator it;
  for( it = functionObjects.begin(); it != functionObjects.end(); it++)
  {
    if( *it == object )
    {
      erase = true;
      break;
    }
  }
  if(erase)
  {
    functionObjects.erase(it);
  }
}

// ---------------------------------------------------------------------------------------
// Get picture for histogram

void qTransferFunctionDefinitionWidget::setRed(int index)
{
  redChosen = index;
  computeHistogram();
  SignalRed(index);
}

void qTransferFunctionDefinitionWidget::setGreen(int index)
{
  greenChosen = index;
  computeHistogram();
  SignalGreen(index);
}

void qTransferFunctionDefinitionWidget::setBlue(int index)
{
  blueChosen = index;
  computeHistogram();
  SignalBlue(index);
}

char* qTransferFunctionDefinitionWidget::getHistogram(vtkImageData* map, float retIntensity1Low, float retIntensity1High, float retIntensity2Low, float retIntensity2High, bool setSize)
{
  float* weights = (float*) map->GetScalarPointer();
  int jump = map->GetNumberOfScalarComponents();

  int redVar = 2*redChosen+1;
  int greenVar = 2*greenChosen+1;
  int blueVar = 2*blueChosen+1;

  //find the maximal element
  float maxR = this->GetRMax();
  float minR = this->GetRMin();
  float maxG = this->GetGMax();
  float minG = this->GetGMin();
  float maxB = this->GetBMax();
  float minB = this->GetBMin();

  //change the temporary histogram into an outputtable one, normalizing it to 256-RGB
  unsigned char* retHisto = new unsigned char[3*HISTOSIZE1*HISTOSIZE2];
  for(int i = 0; i < HISTOSIZE1*HISTOSIZE2; i++)
  {
    retHisto[3*i]  = 254.5*(weights[i*jump+redVar  ]-minR)/(maxR-minR) + 0.5;
    retHisto[3*i+1] = 254.5*(weights[i*jump+greenVar]-minG)/(maxG-minG) + 0.5;;
    retHisto[3*i+2] = 254.5*(weights[i*jump+blueVar ]-minB)/(maxB-minB) + 0.5;;
  }

  //return image
  return (char*) retHisto;
}
