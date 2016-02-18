#include "qTransferFunctionWindowWidgetInterface.h"

#include <algorithm>

qTransferFunctionWindowWidgetInterface::qTransferFunctionWindowWidgetInterface(QWidget *parent /*= 0*/) 
  : QWidget(parent)
{

}

void qTransferFunctionWindowWidgetInterface::AddCapableObject( vtkCudaObject* newObject )
{
  this->capableObjects.push_back( newObject );
}

void qTransferFunctionWindowWidgetInterface::RemoveCapableObject( vtkCudaObject* remObject )
{
  this->capableObjects.erase( std::find(this->capableObjects.begin(), this->capableObjects.end(), remObject) );
}

size_t qTransferFunctionWindowWidgetInterface::GetNumberOfCapableObjects()
{
  return this->capableObjects.size();
}

vtkCudaObject* qTransferFunctionWindowWidgetInterface::GetObject(unsigned int index)
{
  if( index >= 0 && index < this->capableObjects.size() )
  {
    return this->capableObjects[index];
  }
  else
  {
    return 0;
  }
}

int qTransferFunctionWindowWidgetInterface::GetRComponent()
{
  return 0;
}

int qTransferFunctionWindowWidgetInterface::GetGComponent()
{
  return 0;
}

int qTransferFunctionWindowWidgetInterface::GetBComponent()
{
  return 0;
}

double qTransferFunctionWindowWidgetInterface::GetRMax()
{
  return 1.0;
}

double qTransferFunctionWindowWidgetInterface::GetGMax()
{
  return 1.0;
}

double qTransferFunctionWindowWidgetInterface::GetBMax()
{
  return 1.0;
}

double qTransferFunctionWindowWidgetInterface::GetRMin()
{
  return -1.0;
}

double qTransferFunctionWindowWidgetInterface::GetGMin()
{
  return -1.0;
}

double qTransferFunctionWindowWidgetInterface::GetBMin()
{
  return -1.0;
}
