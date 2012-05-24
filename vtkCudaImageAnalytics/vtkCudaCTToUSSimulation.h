#ifndef __VTKCUDACTTOUSSIMULATION_H__
#define __VTKCUDACTTOUSSIMULATION_H__

#include "CUDA_cttoussimulation.h"
#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"

#include "vtkCudaObject.h"

class vtkCudaCTToUSSimulation : public vtkAlgorithm, public vtkCudaObject
{
public:
	static vtkCudaCTToUSSimulation *New();

	void SetInput( vtkImageData * );
	void SetTransform( vtkTransform * );
	void Update();
	vtkImageData* GetOutput();
	vtkImageData* GetOutput(int);

	//output parameters
	void SetOutputResolution(int x, int y, int z);
	void SetLogarithmicScaleFactor(float factor);
	void SetTotalReflectionThreshold(float threshold);
	void SetLinearCombinationAlpha(float a); //weighting for the reflection
	void SetLinearCombinationBeta(float b); //weighting for the density
	void SetLinearCombinationBias(float bias); //bias amount
	void SetDensityScaleModel(float scale, float offset);

	//probe geometry
	void SetProbeWidth(float x, float y);
	void SetFanAngle(float xAngle, float yAngle);
	void SetNearClippingDepth(float depth);
	void SetFarClippingDepth(float depth);

protected:
	vtkCudaCTToUSSimulation();
	virtual ~vtkCudaCTToUSSimulation();
	
	void Reinitialize(int withData);
	void Deinitialize(int withData);

private:
	vtkCudaCTToUSSimulation operator=(const vtkCudaCTToUSSimulation&){}
	vtkCudaCTToUSSimulation(const vtkCudaCTToUSSimulation&){}
	
	CT_To_US_Information information;
	vtkTransform* usTransform;

	vtkImageCast* caster;

	vtkImageData* usOutput;
	vtkImageData* densOutput;
	vtkImageData* transOutput;
	vtkImageData* reflOutput;

	float alpha;
	float beta;
	float bias;

};

#endif