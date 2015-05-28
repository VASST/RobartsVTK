#include "transferFunctionWindowWidgetInterface.h"

#include <algorithm>

void transferFunctionWindowWidgetInterface::AddCapableObject( vtkCudaObject* newObject ){
  this->capableObjects.push_back( newObject );
}

void transferFunctionWindowWidgetInterface::RemoveCapableObject( vtkCudaObject* remObject ){
  this->capableObjects.erase( std::find(this->capableObjects.begin(), this->capableObjects.end(), remObject) );
}

int transferFunctionWindowWidgetInterface::GetNumberOfCapableObjects(){
  return this->capableObjects.size();
}

vtkCudaObject* transferFunctionWindowWidgetInterface::GetObject(int index){
  if( index >= 0 && index < this->capableObjects.size() ){
    return this->capableObjects[index];
  }else{
    return 0;
  }
}

int transferFunctionWindowWidgetInterface::GetRComponent(){
  return 0;
}

int transferFunctionWindowWidgetInterface::GetGComponent(){
  return 0;
}

int transferFunctionWindowWidgetInterface::GetBComponent(){
  return 0;
}

double transferFunctionWindowWidgetInterface::GetRMax(){
  return 1.0;
}

double transferFunctionWindowWidgetInterface::GetGMax(){
  return 1.0;
}

double transferFunctionWindowWidgetInterface::GetBMax(){
  return 1.0;
}

double transferFunctionWindowWidgetInterface::GetRMin(){
  return -1.0;
}

double transferFunctionWindowWidgetInterface::GetGMin(){
  return -1.0;
}

double transferFunctionWindowWidgetInterface::GetBMin(){
  return -1.0;
}
