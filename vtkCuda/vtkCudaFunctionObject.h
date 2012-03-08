#ifndef VTKCUDAFUNCTIONOBJECT_H
#define VTKCUDAFUNCTIONOBJECT_H

#include "vtkObject.h"

class vtkCudaFunctionObject : public vtkObject {
public:
	
	short	GetIdentifier();
	void	SetIdentifier(short id);
	
	float	GetRedColourValue();
	float	GetGreenColourValue();
	float	GetBlueColourValue();
	void	SetColour(float R, float G, float B);

	float	GetOpacity();
	void	SetOpacity(float alpha);

	//methods that, given a table to house the transfer/classification function, apply the
	//attributes (RGBA or id) to the parts of the table that are within the object
	virtual void	PopulatePortionOfTransferTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh,
											float GradientLow, float GradientHigh,
											float* rTable, float* gTable, float* bTable, float* aTable) = 0;
	virtual void	PopulatePortionOfClassifyTable(	int IntensitySize, int GradientSize,
											float IntensityLow, float IntensityHigh,
											float GradientLow, float GradientHigh,
											short* table) = 0;

protected:
	//general attributes for objects in a transfer/classification function
	short	identifier;
	float	colourRed;
	float	colourGreen;
	float	colourBlue;
	float	opacity;

};


#endif