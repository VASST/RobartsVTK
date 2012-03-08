#include "vtkCudaFunctionObject.h"

short vtkCudaFunctionObject::GetIdentifier(){
	return 	this->identifier;
}

void vtkCudaFunctionObject::SetIdentifier(short id){
	this->identifier = id;
}
	
float vtkCudaFunctionObject::GetRedColourValue(){
	return this->colourRed;
}

float vtkCudaFunctionObject::GetGreenColourValue(){
	return this->colourGreen;
}

float vtkCudaFunctionObject::GetBlueColourValue(){
	return this->colourBlue;
}

void vtkCudaFunctionObject::SetColour(float R, float G, float B){
	if(R <= 1.0f && R >= 0.0f && G <= 1.0f && G >= 0.0f && B <= 1.0f && B >= 0.0f){
		this->colourRed = R;
		this->colourGreen = G;
		this->colourBlue = B;
	}else{
		//TODO error macro
	}
}

float vtkCudaFunctionObject::GetOpacity(){
	return this->opacity;
}

void vtkCudaFunctionObject::SetOpacity(float alpha){
	this->opacity = alpha;
}
