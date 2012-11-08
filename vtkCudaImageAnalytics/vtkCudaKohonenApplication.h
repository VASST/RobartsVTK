#ifndef __VTKCUDAKOHONENAPPLICATION_H__
#define __VTKCUDAKOHONENAPPLICATION_H__

#include "CUDA_kohonenapplication.h"
#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"
#include "vtkCudaObject.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"

class vtkCudaKohonenApplication : public vtkImageAlgorithm, public vtkCudaObject
{
public:
	vtkTypeMacro( vtkCudaKohonenApplication, vtkImageAlgorithm );

	static vtkCudaKohonenApplication *New();
	
	void SetWeight(int index, double weight);
	void SetWeights(const double* weights);
	double GetWeight(int index);
	double* GetWeights();
	void SetWeightNormalization(bool set);
	bool GetWeightNormalization();

	// Description:
	// If the subclass does not define an Execute method, then the task
	// will be broken up, multiple threads will be spawned, and each thread
	// will call this method. It is public so that the thread functions
	// can call this method.
	virtual int RequestData(vtkInformation *request, 
							 vtkInformationVector **inputVector, 
							 vtkInformationVector *outputVector);
	virtual int RequestInformation( vtkInformation* request,
							 vtkInformationVector** inputVector,
							 vtkInformationVector* outputVector);
	virtual int RequestUpdateExtent( vtkInformation* request,
							 vtkInformationVector** inputVector,
							 vtkInformationVector* outputVector);

protected:
	vtkCudaKohonenApplication();
	virtual ~vtkCudaKohonenApplication();
	
	void Reinitialize(int withData);
	void Deinitialize(int withData);

private:
	vtkCudaKohonenApplication operator=(const vtkCudaKohonenApplication&){}
	vtkCudaKohonenApplication(const vtkCudaKohonenApplication&){}
	
	double	UnnormalizedWeights[MAX_DIMENSIONALITY];
	bool	WeightNormalization;

	Kohonen_Application_Information info;
};

#endif