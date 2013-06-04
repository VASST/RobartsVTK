#ifndef __VTKCUDAKSOMLIKELIHOOD_H__
#define __VTKCUDAKSOMLIKELIHOOD_H__

#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"
#include "vtkCudaObject.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"
#include "CUDA_KSOMlikelihood.h"

//INPUT PORT DESCRIPTION
//[0] Input Image - full X by Y by Z image with D float components interlaced (VTK default)
//[1] Input GMM - M by N image with D float components interlaced (VTK default) representing
//                the means of the Gaussian mixture model
//[2] Seed Image - full X by Y by Z image with 1 char component only representing the label
//                 associated with a given location (note that 0 means no label, not background
//                 label. give background a different label altogether. No negative values allowed.)

//OUTPUT PORT DESCRIPTION
//[0] Probability Image - full X by Y by Z image with L float components interlaced (VTK default) which
//                        represents the probability of a point being associated with a label. (can be
//                        transformed into a valid Gibbs energy using -ln() of the image.)
//[1] PAGMM Set - M by N image with L float components interlaced (VTK default) representing the
//                activation of each Gaussian component in the histogram estimate of each label

class vtkCudaKSOMLikelihood : public vtkImageAlgorithm, public vtkCudaObject
{
public:
	vtkTypeMacro( vtkCudaKSOMLikelihood, vtkImageAlgorithm );

	static vtkCudaKSOMLikelihood *New();

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
	vtkCudaKSOMLikelihood();
	virtual ~vtkCudaKSOMLikelihood();
	
	void Reinitialize(int withData);
	void Deinitialize(int withData);

private:
	vtkCudaKSOMLikelihood operator=(const vtkCudaKSOMLikelihood&){}
	vtkCudaKSOMLikelihood(const vtkCudaKSOMLikelihood&){}

	KSOMLL_Information info;
	double	Scale;
};

#endif