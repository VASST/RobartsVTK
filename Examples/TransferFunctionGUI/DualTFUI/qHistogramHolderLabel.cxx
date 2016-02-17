#include "qHistogramHolderLabel.h"
#include "qTransferFunctionDefinitionWidget.h"
#include "qTransferFunctionWindowWidget.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaFunctionPolygon.h"
#include <QImage>
#include <QPaintEvent>
#include <QPainter>
#include <iostream>

qHistogramHolderLabel::qHistogramHolderLabel(qTransferFunctionDefinitionWidget* parent, vtkCuda2DTransferFunction* f)
  : QLabel(parent)
  , func(f)
{
  manager = parent;

  histogram = false;
  hasZoomSquare = false;
  maxIntensity1 = 0.0f;
  minIntensity1 = 0.0f;
  zoomMinIntensity1 = 0.0f;
  zoomMaxIntensity1 = 0.0f;
  maxIntensity2 = 0.0f;
  minIntensity2 = 0.0f;
  zoomMinIntensity2 = 0.0f;
  zoomMaxIntensity2 = 0.0f;
  object = 0;
  visAll = true;

  setMouseTracking(false);

  closenessRadius = 100.0f;
  translating = false;
  scaling = false;
  vertexDragging = false;
  shiftHeld = false;
  ctrlHeld = false;
  autoUpdate = true;
  origClickI1 = 0.0f;
  origClickI2 = 0.0f;
  centroidI1 = 0.0f;
  centroidI2 = 0.0f;
  rotationCentreI1 = 0.0f;
  rotationCentreI2 = 0.0f;
}

qHistogramHolderLabel::~qHistogramHolderLabel()
{
}

void qHistogramHolderLabel::setAutoUpdate(bool au)
{
  autoUpdate = au;
}

void qHistogramHolderLabel::giveHistogramDimensions(float maxI1,float minI1, float maxI2, float minI2)
{
  histogram = true;
  maxIntensity1 = maxI1;
  minIntensity1 = minI1;
  maxIntensity2 = maxI2;
  minIntensity2 = minI2;
}

void qHistogramHolderLabel::setZoomSquare(float maxI1,float minI1, float maxI2, float minI2)
{
  hasZoomSquare = true;
  zoomMaxIntensity1 = maxI1;
  zoomMinIntensity1 = minI1;
  zoomMaxIntensity2 = maxI2;
  zoomMinIntensity2 = minI2;
}

void qHistogramHolderLabel::setObject(vtkCudaFunctionPolygon* object)
{
  this->object = object;
}

void qHistogramHolderLabel::setSize(unsigned int s)
{
  QWidget::setMaximumSize(s,s);
  QWidget::setMinimumSize(s,s);
  size = s;
}


void qHistogramHolderLabel::keyPressEvent(QKeyEvent* e)
{
  if(e->key() == Qt::Key::Key_Shift)
  {
    shiftHeld = true;
  }
  if(e->key() == Qt::Key::Key_Control)
  {
    ctrlHeld = true;
  }
}

void qHistogramHolderLabel::keyReleaseEvent(QKeyEvent* e)
{
  if(e->key() == Qt::Key::Key_Shift)
  {
    shiftHeld = false;
  }
  if(e->key() == Qt::Key::Key_Control)
  {
    ctrlHeld = false;
  }
}

void qHistogramHolderLabel::mousePressEvent(QMouseEvent* e)
{

  //if we do not have a histogram or object, this has no meaning
  if(!histogram || !object)
  {
    return;
  }

  //grab mouse position and translate to a gradient and intensity amount
  float intensity1 = (float) e->x() / size * (maxIntensity1-minIntensity1) + minIntensity1;
  float intensity2 = (float) e->y() / size * (maxIntensity2-minIntensity2) + minIntensity2;

  //grab original click location
  origClickI1 = intensity1;
  origClickI2 = intensity2;

  //if the shift was held, we are translating the entire polygon
  if(shiftHeld)
  {
    translating = true;
    return;
  }

  //if control is held, we are scaling the object
  if(ctrlHeld)
  {
    scaling = true;

    //calculate the centroid
    centroidI1 = 0.0f;
    centroidI2 = 0.0f;
    for(int i = 0; i < object->GetNumVertices(); i++)
    {
      centroidI1 += object->GetVertexIntensity(i);
      centroidI2 += object->GetVertexGradient(i);
    }
    centroidI1 /= object->GetNumVertices();
    centroidI2 /= object->GetNumVertices();

    //record this point as the centre of rotation
    rotationCentreI1 = intensity1;
    rotationCentreI2 = intensity2;

    //finish this method
    return;
  }

  //if the right click was held, we are removing a vertex (if we have enough to do so)
  if(e->buttons() == Qt::RightButton && object->GetNumVertices() > 3)
  {
    //find the closest vertex
    vertexInUse = 0;
    float minDist = (object->GetVertexIntensity(0) - intensity1)*(object->GetVertexIntensity(0) - intensity1) +
                    (object->GetVertexGradient(0) - intensity2)*(object->GetVertexGradient(0) - intensity2);
    for(int i = 1; i < object->GetNumVertices(); i++)
    {
      float vertDist = (object->GetVertexIntensity(i) - intensity1)*(object->GetVertexIntensity(i) - intensity1) +
                       (object->GetVertexGradient(i) - intensity2)*(object->GetVertexGradient(i) - intensity2);
      if(vertDist < minDist)
      {
        minDist = vertDist;
        vertexInUse = i;
      }
    }

    //if the vertex is less than the radius distance away, choose it
    unsigned int intensity1Location = size * (object->GetVertexIntensity(vertexInUse) - minIntensity1) / (maxIntensity1-minIntensity1);
    unsigned int intensity2Location = size * (object->GetVertexGradient(vertexInUse) - minIntensity2) / (maxIntensity2-minIntensity2);
    float pictDistance = (e->x() - intensity1Location)*(e->x() - intensity1Location) + (e->y() - intensity2Location) * (e->y() - intensity2Location);
    if( pictDistance < closenessRadius )
    {
      object->RemoveVertex(vertexInUse);
    }

    //do not continue since this is all that is required for the interaction
    return;
  }

  //else, we are moving a single vertex
  if(e->buttons() == Qt::LeftButton)
  {
    //assume we've missed all vertices
    vertexDragging = false;

    //find the closest vertex
    vertexInUse = 0;
    float minDist = (object->GetVertexIntensity(0) - intensity1)*(object->GetVertexIntensity(0) - intensity1) +
                    (object->GetVertexGradient(0) - intensity2)*(object->GetVertexGradient(0) - intensity2);
    for(int i = 1; i < object->GetNumVertices(); i++)
    {
      float vertDist = (object->GetVertexIntensity(i) - intensity1)*(object->GetVertexIntensity(i) - intensity1) +
                       (object->GetVertexGradient(i) - intensity2)*(object->GetVertexGradient(i) - intensity2);
      if(vertDist < minDist)
      {
        minDist = vertDist;
        vertexInUse = i;
      }
    }

    //if the vertex is less than the radius distance away, choose it
    unsigned int intensity1Location = size * (object->GetVertexIntensity(vertexInUse) - minIntensity1) / (maxIntensity1-minIntensity1);
    unsigned int intensity2Location = size * (object->GetVertexGradient(vertexInUse) - minIntensity2) / (maxIntensity2-minIntensity2);
    float pictDistance = (e->x() - intensity1Location)*(e->x() - intensity1Location) + (e->y() - intensity2Location) * (e->y() - intensity2Location);
    if( pictDistance < closenessRadius )
    {
      vertexDragging = true;
      return;
    }

    //if we didn't get a vertex, we might be trying to create a new one
    //find the closest line segment for(unsigned int i = 1; i < object->GetNumVertices(); i++){
    minDist = closenessRadius;
    float minPointX = 0.0f;
    float minPointY = 0.0f;
    int placeIndex = -1;
    int belowIndex = 0;
    int aboveIndex = object->GetNumVertices() - 1;
    for(int i = 0; i < object->GetNumVertices(); i++)
    {

      //define the line and direct vectors in the intensity gradient scale
      float lineX = object->GetVertexIntensity(belowIndex) - object->GetVertexIntensity(aboveIndex);
      float lineY = object->GetVertexGradient(belowIndex) - object->GetVertexGradient(aboveIndex);
      float directX = intensity1 - object->GetVertexIntensity(aboveIndex);
      float directY = intensity2 - object->GetVertexGradient(aboveIndex);
      float lineMagSquared = lineX*lineX+lineY*lineY;
      float lineMag = sqrt(lineX*lineX+lineY*lineY);
      float directMag = sqrt(directX*directX+directY*directY);

      //get the closest point on the line
      float pointX = (directX*lineX+directY*lineY) * lineX / lineMagSquared + object->GetVertexIntensity(aboveIndex);
      float pointY = (directX*lineX+directY*lineY) * lineY / lineMagSquared + object->GetVertexGradient(aboveIndex);

      //find the distance components in pixels from the point to the line
      float pixDistX = size * (intensity1-pointX) / (maxIntensity1- minIntensity1);
      float pixDistY = size * (intensity2-pointY) / (maxIntensity2- minIntensity2);
      float pixDist = pixDistX*pixDistX+pixDistY*pixDistY;

      //if we are close enough, create a new point here, provided that the point falls in the line segment
      if(pixDist < minDist )
      {

        //verify that the point is in the line segment and not off it
        float distToAbove= (pointX-object->GetVertexIntensity(aboveIndex))*(pointX-object->GetVertexIntensity(aboveIndex))
                           + (pointY-object->GetVertexGradient(aboveIndex))*(pointY-object->GetVertexGradient(aboveIndex));
        float distToBelow = (pointX-object->GetVertexIntensity(belowIndex))*(pointX-object->GetVertexIntensity(belowIndex))
                            + (pointY-object->GetVertexGradient(belowIndex))*(pointY-object->GetVertexGradient(belowIndex));

        if(distToBelow < lineMagSquared && distToAbove < lineMagSquared)
        {
          minDist = pixDist;
          minPointX = pointX;
          minPointY = pointY;
          placeIndex = aboveIndex+1;
        }
      }

      //go to the next line
      belowIndex = i+1;
      aboveIndex = i;
    }

    //if we have a proper point
    if(placeIndex != -1)
    {
      object->AddVertex(minPointX,minPointY,placeIndex);
      vertexInUse = placeIndex;
      vertexDragging = true;
      return;
    }
  }

  //or we might just have missed everything, oh well, let's just go home then
  return;

}


void qHistogramHolderLabel::mouseReleaseEvent(QMouseEvent* e)
{
  //if we do not have a histogram or object, this has no meaning
  if(!histogram)
  {
    return;
  }

  //redraw the image if necessary
  manager->updateFunction();

  //release all variables
  origClickI1 = 0.0f;
  origClickI2 = 0.0f;
  translating = false;
  scaling = false;
  vertexDragging = false;
  vertexInUse = 0;

}

void qHistogramHolderLabel::mouseMoveEvent(QMouseEvent* e)
{

  //if we do not have a histogram, this has no meaning
  if(!histogram || !object)
  {
    return;
  }

  //grab mouse position and translate to a gradient and intensity amount
  double intensity1 = (double) e->x() / size * (maxIntensity1-minIntensity1) + minIntensity1;
  double intensity2 = (double) e->y() / size * (maxIntensity2-minIntensity2) + minIntensity2;
  unsigned int numVertices = object->GetNumVertices();

  //if we are translating, then translate!
  if(translating)
  {
    float intensityDifference = intensity1-origClickI1;
    float gradientDifference = intensity2-origClickI2;
    for(unsigned int i = 0; i < numVertices; i++)
    {
      object->ModifyVertex(object->GetVertexIntensity(i) + intensityDifference, object->GetVertexGradient(i) + gradientDifference, i);
    }
    origClickI1 = intensity1;
    origClickI2 = intensity2;

    //if we are scaling, scale this instance of the polygon
  }
  else if(scaling)
  {

    //calculate the distances from the last click to the centroid and from the current click to the centroid
    float centroidLastDist = 1.0f + sqrt( ((origClickI1 - centroidI1)*(origClickI1 - centroidI1)) / ((maxIntensity1 - minIntensity1) * (maxIntensity1 - minIntensity1))
                                          + ((origClickI2 - centroidI2)*(origClickI2 - centroidI2)) / ((maxIntensity2 - minIntensity2) * (maxIntensity2 - minIntensity2)) );
    float centroidCurrDist = 1.0f + sqrt( ((intensity1 - centroidI1)*(intensity1 - centroidI1)) / ((maxIntensity1 - minIntensity1) * (maxIntensity1 - minIntensity1))
                                          + ((intensity2 - centroidI2)*(intensity2 - centroidI2)) / ((maxIntensity2 - minIntensity2) * (maxIntensity2 - minIntensity2)) );

    //calculate the distances from the last click to the centroid and from the current click to the centroid and the ratio to scale by
    float lastDist = 1.0f + sqrt( ((origClickI1 - rotationCentreI1)*(origClickI1 - rotationCentreI1)) / ((maxIntensity1 - minIntensity1) * (maxIntensity1 - minIntensity1))
                                  + ((origClickI2 - rotationCentreI2)*(origClickI2 - rotationCentreI2)) / ((maxIntensity2 - minIntensity2) * (maxIntensity2 - minIntensity2)) );
    float currDist = 1.0f + sqrt( ((intensity1 - rotationCentreI1)*(intensity1 - rotationCentreI1)) / ((maxIntensity1 - minIntensity1) * (maxIntensity1 - minIntensity1))
                                  + ((intensity2 - rotationCentreI2)*(intensity2 - rotationCentreI2))  / ((maxIntensity2 - minIntensity2) * (maxIntensity2 - minIntensity2)) );
    float ratio = currDist / lastDist;

    //determine whether to scale up or down
    bool scaleUp = (currDist > centroidCurrDist) && (currDist > lastDist);

    //take the reciprocal of the ratio in order to make it congruent with the direction of scaling
    if( ((ratio < 1.0f) || scaleUp) && !((ratio < 1.0f) && scaleUp) )
    {
      ratio = 1.0f / ratio;
    }

    //recalculate the centroid
    centroidI1 = ratio * (centroidI1 - rotationCentreI1) + rotationCentreI1;
    centroidI2 = ratio * (centroidI2 - rotationCentreI2) + rotationCentreI2;

    //update each of the vertices by a factor of the calculated ratio
    for(unsigned int i = 0; i < numVertices; i++)
    {
      float posI = ratio * (object->GetVertexIntensity(i) - rotationCentreI1) + rotationCentreI1;
      float posG = ratio * (object->GetVertexGradient(i)  - rotationCentreI2) + rotationCentreI2;
      object->ModifyVertex(posI, posG, i);
    }

    //set this as the last click
    origClickI1 = intensity1;
    origClickI2 = intensity2;

    //else, we are moving a single vertex
  }
  else if(vertexDragging)
  {
    object->ModifyVertex(intensity1, intensity2, vertexInUse);

    //else, we be out of options
  }
  else
  {
    return;
  }

  //update the transfer function and screen if appropriate, else, just update the histograms
  if(autoUpdate)
  {
    manager->updateFunction();
  }
  else
  {
    manager->repaintHistograms();
  }
}

void qHistogramHolderLabel::paintEvent( QPaintEvent * e)
{
  if(!histogram)
  {
    return;
  }

  //paint histogram
  QLabel::paintEvent(e);

  QPainter painter(this);

  //if we have a function, draw it
  if(!func)
  {
    return;
  }
  unsigned int i = 0;
  while(true)
  {

    //get the next object to be painted
    vtkCudaFunctionPolygon* paintedObject = (vtkCudaFunctionPolygon*) func->GetFunctionObject(i);
    i++;
    if(!paintedObject)
    {
      break;
    }
    if( !visAll && paintedObject != object )
    {
      continue;
    }

    //set the colour to be that of the object being drawn
    QColor colour;
    colour.setAlpha(255);
    colour.setRed(255.0f * paintedObject->GetRedColourValue() + 0.5f);
    colour.setGreen(255.0f * paintedObject->GetGreenColourValue() + 0.5f);
    colour.setBlue(255.0f * paintedObject->GetBlueColourValue() + 0.5f);
    painter.setPen(colour);

    //collect the number of vertices, and if we have less than 3 (not a polyhedron) then return
    unsigned int numVertices = paintedObject->GetNumVertices();
    if(numVertices < 3 )
    {
      return;
    }

    //create a polygon from the vertex information
    int rectangleDim = sqrt(this->closenessRadius)+0.5;
    QPolygon polygon(numVertices);
    polygon.clear();
    for(unsigned int i = 0; i < numVertices; i++)
    {
      unsigned int intensityLocation = size * (paintedObject->GetVertexIntensity(i) - minIntensity1) / (maxIntensity1-minIntensity1);
      unsigned int gradientLocation = size * (paintedObject->GetVertexGradient(i) - minIntensity2) / (maxIntensity2-minIntensity2);
      polygon.putPoints(i,1,intensityLocation,gradientLocation);
      QRect rectangle;
      rectangle.setRect(intensityLocation-rectangleDim/2, gradientLocation-rectangleDim/2, rectangleDim, rectangleDim);
      if(paintedObject == object)
      {
        painter.drawEllipse(rectangle);
      }
    }

    //draw the resulting polygon
    painter.drawConvexPolygon(polygon);

  }

  //draw the zoom square
  if(hasZoomSquare)
  {
    QColor colour;
    colour.setAlpha(255);
    colour.setRed(255);
    colour.setGreen(255);
    colour.setBlue(255);
    painter.setPen(colour);
    painter.drawRect( size * (zoomMinIntensity1 - minIntensity1) / (maxIntensity1-minIntensity1),
                      size * (zoomMinIntensity2 - minIntensity2) / (maxIntensity2-minIntensity2),
                      size * (zoomMaxIntensity1 - zoomMinIntensity1) / (maxIntensity1-minIntensity1),
                      size * (zoomMaxIntensity2 - zoomMinIntensity2) / (maxIntensity2-minIntensity2));
  }
}

void qHistogramHolderLabel::visualizeAllObjects(bool b)
{
  visAll = b;
}