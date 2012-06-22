#ifndef __vtkCudaCT2USSimulation_H__
#define __vtkCudaCT2USSimulation_H__

#include "CUDA_cttoussimulation.h"
#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"

#include "vtkCudaObject.h"

class vtkCudaCT2USSimulation : public vtkAlgorithm, public vtkCudaObject
{
public:

	vtkTypeMacro( vtkCudaCT2USSimulation, vtkAlgorithm )

	static vtkCudaCT2USSimulation *New();

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
	vtkCudaCT2USSimulation();
	virtual ~vtkCudaCT2USSimulation();
	
	void Reinitialize(int withData);
	void Deinitialize(int withData);

private:
	vtkCudaCT2USSimulation operator=(const vtkCudaCT2USSimulation&){}
	vtkCudaCT2USSimulation(const vtkCudaCT2USSimulation&){}
	
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