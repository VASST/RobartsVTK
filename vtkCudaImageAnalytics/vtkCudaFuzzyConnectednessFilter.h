#ifndef __vtkCudaFuzzyConnectednessFilter_H__
#define __vtkCudaFuzzyConnectednessFilter_H__

#include "vtkThreadedImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "CUDA_fuzzyconnectednessfilter.h"

struct vtkCudaFuzzyConnectednessFilterInformation;

class vtkCudaFuzzyConnectednessFilter : public vtkImageAlgorithm
{
public:

	vtkTypeMacro( vtkCudaFuzzyConnectednessFilter, vtkThreadedImageAlgorithm )

	static vtkCudaFuzzyConnectednessFilter *New();

	//output parameters
	void SetOutputResolution(int x, int y, int z);
	void SetLogarithmicScaleFactor(double factor);
	void SetTotalReflectionThreshold(double threshold);
	void SetLinearCombinationAlpha(double a); //weighting for the reflection
	void SetLinearCombinationBeta(double b); //weighting for the density
	void SetLinearCombinationBias(double bias); //bias amount
	void SetDensityScaleModel(double scale, double offset);
	
	// Description:
	// Get/Set the t-Norm and s-Norm type
	vtkSetClampMacro( TNorm, int, 0, 2 );
	vtkGetMacro( TNorm, int );
	vtkSetClampMacro( SNorm, int, 0, 2 );
	vtkGetMacro( SNorm, int );
protected:
	
	int RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);

	vtkCudaFuzzyConnectednessFilter();
	virtual ~vtkCudaFuzzyConnectednessFilter();

private:
	vtkCudaFuzzyConnectednessFilter operator=(const vtkCudaFuzzyConnectednessFilter&){}
	vtkCudaFuzzyConnectednessFilter(const vtkCudaFuzzyConnectednessFilter&){}
	
	int TNorm;
	int SNorm;

	Fuzzy_Connectedness_Information* Information;
};

#endif