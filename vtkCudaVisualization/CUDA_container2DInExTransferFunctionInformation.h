/** @file CUDA_container2DTransferFunctionInformation.h
 *
 *  @brief File for the volume information holding structure used for volume ray casting
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on March 27, 2011
 *
 *  @note This is primarily an internal file used by the vtkCudaVolumeInformationHandler and CUDA_renderAlgo to store and communicate constants
 *
 */

#ifndef __CUDA2DINEXTRANSFERFUNCTIONINFORMATION_H__
#define __CUDA2DINEXTRANSFERFUNCTIONINFORMATION_H__

#include "vector_types.h"

/** @brief A stucture located on the CUDA hardware that holds all the information required about the volume being renderered.
 *
 */
typedef struct __align__(16) {
	// The scale and shift to transform intensity and gradient to indices in the transfer functions
	float			intensityLow;			/**< Minimum intensity of the image */
	float			intensityMultiplier;	/**< Scale factor to normalize intensities to between 0 and 1 */
	float			gradientLow;			/**< The minimum logarithmic gradient index including the offset */
	float			gradientMultiplier;		/**< The maximum logarithmic gradient index including the offset */
	float			gradientOffset;			/**< The offset for the logarithmic scaling of the gradient indexes */
	unsigned int	functionSize;			/**< The size of the lookup table */
	bool			useBlackKeyhole;

	//opague memory back for the transfer function
	cudaArray* alphaTransferArray2D;
	cudaArray* colorRTransferArray2D;
	cudaArray* colorGTransferArray2D;
	cudaArray* colorBTransferArray2D;
	cudaArray* inExLogicTransferArray2D;

} cuda2DInExTransferFunctionInformation;

#endif
