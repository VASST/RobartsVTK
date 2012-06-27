#ifndef __vtkCudaFuzzyConnectednessFilter_H__
#define __vtkCudaFuzzyConnectednessFilter_H__

#include "vtkThreadedImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "CUDA_fuzzyconnectednessfilter.h"
#include "vtkCudaObject.h"

struct vtkCudaFuzzyConnectednessFilterInformation;

class vtkCudaFuzzyConnectednessFilter : public vtkImageAlgorithm, public vtkCudaObject
{
public:

	vtkTypeMacro( vtkCudaFuzzyConnectednessFilter, vtkImageAlgorithm )

	static vtkCudaFuzzyConnectednessFilter *New();
	
	// Description:
	// Get/Set the t-Norm and s-Norm type
	vtkSetClampMacro( TNorm, int, 0, 2 );
	vtkGetMacro( TNorm, int );
	vtkSetClampMacro( SNorm, int, 0, 2 );
	vtkGetMacro( SNorm, int );

protected:
	
	void Reinitialize(int withData);
	void Deinitialize(int withData);

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