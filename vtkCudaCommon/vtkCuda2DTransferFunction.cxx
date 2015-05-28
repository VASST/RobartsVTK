#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaFunctionObject.h"
#include "vtkObjectFactory.h"
#include <vector>

vtkStandardNewMacro(vtkCuda2DTransferFunction);

vtkCuda2DTransferFunction::vtkCuda2DTransferFunction(){
  this->components = new std::vector<vtkCudaFunctionObject*>();
}

vtkCuda2DTransferFunction::~vtkCuda2DTransferFunction(){
  for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin();
    it != this->components->end(); it++)
    (*it)->UnRegister(this);
  this->components->clear();
  delete this->components;
}

double vtkCuda2DTransferFunction::getMaxGradient(){

  if(this->components->size() == 0) return 1;

  double max = this->components->front()->getMaxGradient();

  for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
    double localmax = (*it)->getMaxGradient();
    max = (localmax > max) ? localmax : max;
  }

  return max;
}

double vtkCuda2DTransferFunction::getMinGradient(){

  if(this->components->size() == 0) return 0;

  double min = this->components->front()->getMinGradient();

  for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
    double localmin = (*it)->getMinGradient();
    min = (localmin < min) ? localmin : min;
  }

  return min;
}

double vtkCuda2DTransferFunction::getMaxIntensity(){

  if(this->components->size() == 0) return 1;

  double max = this->components->front()->getMaxIntensity();

  for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
    double localmax = (*it)->getMaxIntensity();
    max = (localmax > max) ? localmax : max;
  }

  return max;
}

double vtkCuda2DTransferFunction::getMinIntensity(){

  if(this->components->size() == 0) return 0;

  double min = this->components->front()->getMinIntensity();

  for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
    double localmin = (*it)->getMinIntensity();
    min = (localmin < min) ? localmin : min;
  }

  return min;
}

void vtkCuda2DTransferFunction::GetClassifyTable(  short* outputTable, int sizeI, int sizeG,
                          float lowI, float highI, float offsetI,
                          float lowG, float highG, float offsetG,
                          int logUsed ){
  //return if we don't have a proper table to populate
  if(!outputTable) return;

  //clear the table
  memset( (void*) outputTable, 0, sizeof(short) * sizeI * sizeG);

  //iterate over each object, letting them contribute to the table
  for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
    (*it)->PopulatePortionOfClassifyTable(sizeI, sizeG, lowI, highI, offsetI, lowG, highG, offsetG, outputTable, logUsed);
  }
}

void vtkCuda2DTransferFunction::GetTransferTable(  float* outputRTable, float* outputGTable, float* outputBTable, float* outputATable,
                          int sizeI, int sizeG,
                          float lowI, float highI, float offsetI,
                          float lowG, float highG, float offsetG,
                          int logUsed ){

  //clear the table
  if(outputRTable) memset( (void*) outputRTable, 0.0f, sizeof(float) * sizeI * sizeG);
  if(outputGTable) memset( (void*) outputGTable, 0.0f, sizeof(float) * sizeI * sizeG);
  if(outputBTable) memset( (void*) outputBTable, 0.0f, sizeof(float) * sizeI * sizeG);
  if(outputATable) memset( (void*) outputATable, 0.0f, sizeof(float) * sizeI * sizeG);

  //iterate over each object, letting them contribute to the table
  for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
    (*it)->PopulatePortionOfTransferTable(sizeI, sizeG, lowI, highI, offsetI, lowG, highG, offsetG, outputRTable, outputGTable, outputBTable, outputATable, logUsed);
  }

}

void vtkCuda2DTransferFunction::GetShadingTable(  float* outputATable, float* outputDTable, float* outputSTable, float* outputPTable,
                          int sizeI, int sizeG,
                          float lowI, float highI, float offsetI,
                          float lowG, float highG, float offsetG,
                          int logUsed ){

  //clear the table
  if(outputATable) memset( (void*) outputATable, 0.0f, sizeof(float) * sizeI * sizeG);
  if(outputDTable) memset( (void*) outputDTable, 0.0f, sizeof(float) * sizeI * sizeG);
  if(outputSTable) memset( (void*) outputSTable, 0.0f, sizeof(float) * sizeI * sizeG);
  if(outputPTable) memset( (void*) outputPTable, 0.0f, sizeof(float) * sizeI * sizeG);

  //iterate over each object, letting them contribute to the table
  for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
    (*it)->PopulatePortionOfShadingTable(sizeI, sizeG, lowI, highI, offsetI, lowG, highG, offsetG, outputATable, outputDTable, outputSTable, outputPTable, logUsed);
  }

}

short vtkCuda2DTransferFunction::GetNumberOfClassifications(){
  short max = 0;
  for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
    if( (*it)->GetIdentifier() > max ){
      max = (*it)->GetIdentifier();
    }
  }
  return max;
}

void vtkCuda2DTransferFunction::AddFunctionObject(vtkCudaFunctionObject* object){
  this->components->push_back(object);
  object->Register(this);
  this->Modified();
}

void vtkCuda2DTransferFunction::RemoveFunctionObject(vtkCudaFunctionObject* object){

  bool erase = false;
  std::vector<vtkCudaFunctionObject*>::iterator it;
  for( it = this->components->begin(); it != this->components->end(); it++){
    if( *it == object ){
      erase = true;
      break;
    }
  }
  if(erase){
    (*it)->UnRegister(this);
    this->components->erase(it);
    this->Modified();
  }

}

vtkCudaFunctionObject* vtkCuda2DTransferFunction::GetFunctionObject(unsigned int index){

  if(index >= components->size()) return 0;
  return (*components)[index];

}

int vtkCuda2DTransferFunction::GetNumberOfFunctionObjects(){
  return this->components->size();
}