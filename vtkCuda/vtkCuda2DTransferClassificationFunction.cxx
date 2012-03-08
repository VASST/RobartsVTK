#include "vtkCuda2DTransferClassificationFunction.h"
#include "vtkCudaFunctionObject.h"
#include "vtkObjectFactory.h"
#include <vector>

vtkStandardNewMacro(vtkCuda2DTransferClassificationFunction);

vtkCuda2DTransferClassificationFunction::vtkCuda2DTransferClassificationFunction(){
	this->components = new std::vector<vtkCudaFunctionObject*>();
	this->SatisfyUpdate();
}

vtkCuda2DTransferClassificationFunction::~vtkCuda2DTransferClassificationFunction(){
	this->components->clear();
	delete this->components;
}

bool vtkCuda2DTransferClassificationFunction::NeedsUpdate(){
	return this->updateNeeded;
}

void vtkCuda2DTransferClassificationFunction::SatisfyUpdate(){
	this->updateNeeded = false;
}


void vtkCuda2DTransferClassificationFunction::SignalUpdate(){
	this->updateNeeded = true;
}

void vtkCuda2DTransferClassificationFunction::GetClassifyTable(	short* outputTable, int sizeI, int sizeG,
															   float lowI, float highI, float lowG, float highG){
	//iterate over each object, letting them contribute to the table
	for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
		(*it)->PopulatePortionOfClassifyTable(sizeI, sizeG, lowI, highI, lowG, highG, outputTable);
	}

}
void vtkCuda2DTransferClassificationFunction::GetTransferTable(	float* outputRTable, float* outputGTable, float* outputBTable, float* outputATable,
															   int sizeI, int sizeG, float lowI, float highI, float lowG, float highG){
	
	//iterate over each object, letting them contribute to the table
	for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
		(*it)->PopulatePortionOfTransferTable(sizeI, sizeG, lowI, highI, lowG, highG, outputRTable, outputGTable, outputBTable, outputATable);
	}

}

short vtkCuda2DTransferClassificationFunction::GetNumberOfClassifications(){
	short max = 0;
	for( std::vector<vtkCudaFunctionObject*>::iterator it = this->components->begin(); it != this->components->end(); it++){
		if( (*it)->GetIdentifier() > max ){
			max = (*it)->GetIdentifier();
		}
	}
	return max;
}

void vtkCuda2DTransferClassificationFunction::AddFunctionObject(vtkCudaFunctionObject* object){
	this->components->push_back(object);
	this->updateNeeded = true;
}

void vtkCuda2DTransferClassificationFunction::RemoveFunctionObject(vtkCudaFunctionObject* object){

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
		this->updateNeeded = true;
	}

}