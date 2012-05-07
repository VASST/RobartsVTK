#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaFunctionObject.h"
#include "vtkObjectFactory.h"
#include <vector>

vtkStandardNewMacro(vtkCuda2DTransferFunction);

vtkCuda2DTransferFunction::vtkCuda2DTransferFunction(){
	this->components = new std::vector<vtkCudaFunctionObject*>();
}

vtkCuda2DTransferFunction::~vtkCuda2DTransferFunction(){
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

void vtkCuda2DTransferFunction::GetClassifyTable(	short* outputTable, int sizeI, int sizeG,
															   float lowI, float highI, float lowG, float highG, float offsetG){

	//clear the table
	for(unsigned int i = 0; i < sizeI*sizeG; i++)
		outputTable[i] = 0;

	//iterate over each object, letting them contribute to the table
	for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
		(*it)->PopulatePortionOfClassifyTable(sizeI, sizeG, lowI, highI, lowG, highG, offsetG, outputTable);
	}
}

void vtkCuda2DTransferFunction::GetTransferTable(	float* outputRTable, float* outputGTable, float* outputBTable, float* outputATable,
															   int sizeI, int sizeG, float lowI, float highI, float lowG, float highG, float offsetG){

	//clear the table
	for(unsigned int i = 0; i < sizeI*sizeG; i++){
		outputRTable[i] = 0.0f;
		outputGTable[i] = 0.0f;
		outputBTable[i] = 0.0f;
		outputATable[i] = 0.0f;
	}

	//iterate over each object, letting them contribute to the table
	for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
		(*it)->PopulatePortionOfTransferTable(sizeI, sizeG, lowI, highI, lowG, highG, offsetG, outputRTable, outputGTable, outputBTable, outputATable);
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
		this->components->erase(it);
		this->Modified();
	}

}

vtkCudaFunctionObject* vtkCuda2DTransferFunction::GetFunctionObject(unsigned int index){

	if(index >= components->size()) return 0;
	return components->at(index);

}