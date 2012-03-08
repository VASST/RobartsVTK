#include "vtkCudaFunctionSquare.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkCudaFunctionSquare);

vtkCudaFunctionSquare::vtkCudaFunctionSquare(){
	this->identifier = 0;
	this->colourRed = 0.0f;
	this->colourGreen = 0.0f;
	this->colourBlue = 0.0f;
	this->opacity = 0.0f;

	this->intensityHigh = 0.0f;
	this->intensityLow = 0.0f;
	this->gradientHigh = 0.0f;
	this->gradientLow = 0.0f;

}

vtkCudaFunctionSquare::~vtkCudaFunctionSquare(){

}

void vtkCudaFunctionSquare::PopulatePortionOfTransferTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh,
											float GradientLow, float GradientHigh,
											float* rTable, float* gTable, float* bTable, float* aTable){
	
	//find the edges of the square in index co-ordinates
	int lowIntensIndex = 0.5f + (float) IntensitySize * (this->intensityLow - IntensityLow) / (IntensityHigh - IntensityLow);
	int highIntensIndex = 0.5f + (float) IntensitySize * (this->intensityHigh - IntensityLow) / (IntensityHigh - IntensityLow);
	int lowGradIndex = 0.5f + (float) GradientSize * (this->gradientLow - GradientLow) / (GradientHigh - GradientLow);
	int highGradIndex = 0.5f + (float) GradientSize * (this->gradientHigh - GradientLow) / (GradientHigh - GradientLow);

	//make sure the bottom section falls in range
	if(lowIntensIndex < 0) lowIntensIndex = 0;
	if(lowGradIndex < 0) lowGradIndex = 0;

	//populate this section of the tables
	for( int i = lowIntensIndex; i < highIntensIndex && i < IntensitySize; i++){
		for( int g = lowGradIndex; g < highGradIndex && g < GradientSize; g++){
			int tableIndex = i + g * IntensitySize;
			rTable[tableIndex] = this->colourRed;
			gTable[tableIndex] = this->colourGreen;
			bTable[tableIndex] = this->colourBlue;
			aTable[tableIndex] = this->opacity;
		}
	}

}

void vtkCudaFunctionSquare::PopulatePortionOfClassifyTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh,
											float GradientLow, float GradientHigh,
											short* table){

	//find the edges of the square in index co-ordinates
	int lowIntensIndex = 0.5f + (float) IntensitySize * (this->intensityLow - IntensityLow) / (IntensityHigh - IntensityLow);
	int highIntensIndex = 0.5f + (float) IntensitySize * (this->intensityHigh - IntensityLow) / (IntensityHigh - IntensityLow);
	int lowGradIndex = 0.5f + (float) GradientSize * (this->gradientLow - GradientLow) / (GradientHigh - GradientLow);
	int highGradIndex = 0.5f + (float) GradientSize * (this->gradientHigh - GradientLow) / (GradientHigh - GradientLow);

	//make sure the bottom section falls in range
	if(lowIntensIndex < 0) lowIntensIndex = 0;
	if(lowGradIndex < 0) lowGradIndex = 0;

	//populate this section of the table
	for( int i = lowIntensIndex; i < highIntensIndex && i < IntensitySize; i++){
		for( int g = lowGradIndex; g < highGradIndex && g < GradientSize; g++){
			int tableIndex = i + g * IntensitySize;
			table[tableIndex] = this->identifier;
		}
	}

}

float vtkCudaFunctionSquare::GetLowIntensityValue(){
	return this->intensityLow;
}

float vtkCudaFunctionSquare::GetHighIntensityValue(){
	return this->intensityHigh;
}

float vtkCudaFunctionSquare::GetLowGradientValue(){
	return this->gradientLow;
}

float vtkCudaFunctionSquare::GetHighGradientValue(){
	return this->gradientHigh;
}

void vtkCudaFunctionSquare::SetSizeAndPosition( float intensityLow, float intensityHigh, float gradientLow, float gradientHigh ){
	this->intensityLow = intensityLow;
	this->intensityHigh = intensityHigh;
	this->gradientLow = gradientLow;
	this->gradientHigh = gradientHigh;
}