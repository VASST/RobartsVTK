
#ifndef _VTKCUDAANISOTROPICDIFFUSIONFILTER_H
#define _VTKCUDAANISOTROPICDIFFUSIONFILTER_H

#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkAlgorithm.h"

class vtkCudaAnisotropicDiffusionFilter : public vtkAlgorithm {
public:
	
	/** @brief VTK compatible constructor method
	 *
	 */
	static vtkCudaAnisotropicDiffusionFilter* New();
	
	/** @brief Sets the 3D image data to be segmented
	 *
	 *  @param image The 3D image data to be segmented
	 */
	void SetInput(vtkImageData* image);
	
	/** @brief Sets the smoothing rate
	 *
	 *  @param lambda The smoothing rate, which should be between 0 and 1
	 */
	void SetLambda(float lambda);
	
	/** @brief Gets the smoothing rate
	 *
	 */
	float GetLambda();

	/** @brief Sets the value on which the Tukey biweight function becomes 0
	 *
	 *  @param sigma The Tukey biweight parameter
	 */
	void SetSigma(float sigma);
	
	/** @brief Gets the smoothing rate
	 *
	 */
	float GetSigma();
	
	/** @brief Sets the number of iterations to run the filter for
	 *
	 *  @param numIterations The number of iterations
	 */
	void SetNumberOfIterations(unsigned int numIterations);
	
	/** @brief Gets the smoothing rate
	 *
	 */
	unsigned int GetNumberOfIterations();
	
	/** @brief Updates the filter, redoing the smoothing process and updating the output image
	 *
	 */
	virtual void Update();
	
	/** @brief Gets the smoothed volumetric image
	 *
	 */
	vtkImageData* GetOutput();

protected:
	vtkCudaAnisotropicDiffusionFilter();
	~vtkCudaAnisotropicDiffusionFilter();

private:
	vtkImageData* output;
	vtkImageCast* caster;
	float lambda;
	float sigma;
	unsigned int numIterations;
};

#endif