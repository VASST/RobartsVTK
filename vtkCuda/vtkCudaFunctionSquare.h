#ifndef VTKCUDAFUNCTIONSQUARE_H
#define VTKCUDAFUNCTIONSQUARE_H

#include "vtkCudaFunctionObject.h"

class vtkCudaFunctionSquare : public vtkCudaFunctionObject {
public:

	static vtkCudaFunctionSquare*	New();

	//overwritten from vtkCudaFunctionObject
	virtual void	PopulatePortionOfTransferTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh,
											float GradientLow, float GradientHigh,
											float* rTable, float* gTable, float* bTable, float* aTable);
	virtual void	PopulatePortionOfClassifyTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh,
											float GradientLow, float GradientHigh,
											short* table);

	//get the high and low intensity and gradient values which define the square
	float	GetLowIntensityValue();
	float	GetHighIntensityValue();
	float	GetLowGradientValue();
	float	GetHighGradientValue();

	//set the high and low intensity and gradient values which define the square
	void	SetSizeAndPosition( float intensityLow, float intensityHigh, float gradientLow, float gradientHigh );

protected:

	vtkCudaFunctionSquare();
	~vtkCudaFunctionSquare();

	//parameters to describe the position and size of the square
	float	intensityLow;
	float	intensityHigh;
	float	gradientLow;
	float	gradientHigh;


};


#endif