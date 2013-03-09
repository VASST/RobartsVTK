#ifndef __VTKCUDAPAGMMESTIMATOR_H__
#define __VTKCUDAPAGMMESTIMATOR_H__

#include "CUDA_PAGMMestimator.h"
#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"
#include "vtkCudaObject.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"

class vtkCudaPAGMMEstimator : public vtkImageAlgorithm, public vtkCudaObject
{
public:
	vtkTypeMacro( vtkCudaPAGMMEstimator, vtkImageAlgorithm );

	static vtkCudaPAGMMEstimator *New();
	
	void SetWeight(int index, double weight);
	void SetWeights(const double* weights);
	double GetWeight(int index);
	double* GetWeights();
	void SetWeightNormalization(bool set);
	bool GetWeightNormalization();

	void SetConservativeness(double q);
	double GetConservativeness();
	void SetScale(double s);
	double GetScale();

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
	vtkCudaPAGMMEstimator();
	virtual ~vtkCudaPAGMMEstimator();
	
	void Reinitialize(int withData);
	void Deinitialize(int withData);

private:
	vtkCudaPAGMMEstimator operator=(const vtkCudaPAGMMEstimator&){}
	vtkCudaPAGMMEstimator(const vtkCudaPAGMMEstimator&){}
	
	double	Q;
	double	Scale;

	double	UnnormalizedWeights[MAX_DIMENSIONALITY];
	bool	WeightNormalization;

	PAGMM_Information info;
};

#endif