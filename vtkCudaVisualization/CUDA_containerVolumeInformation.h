/** @file cudaVolumeInformation.h
 *
 *  @brief File for the volume information holding structure used for volume ray casting
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on March 27, 2011
 *
 *  @note This is primarily an internal file used by the vtkCudaVolumeInformationHandler and CUDA_renderAlgo to store and communicate constants
 *
 */

#ifndef __CUDAVOLUMEINFORMATION_H__
#define __CUDAVOLUMEINFORMATION_H__

#include "vector_types.h"

/** @brief A stucture located on the CUDA hardware that holds all the information required about the volume being renderered.
 *
 */
typedef struct __align__(16) {
	// The size and spacing of the volume
	int3			VolumeSize;				/**< Size of the volume in X, Y and Z */
	float			Bounds[6];				/**< The bounds of the volume in X, Y and Z */
	float3			SpacingReciprocal;		/**< The reciprocal of the spacing between pixels in the volume in the X, Y and Z directions */
	float3			Spacing;
	float			MinSpacing;

} cudaVolumeInformation;

#endif
