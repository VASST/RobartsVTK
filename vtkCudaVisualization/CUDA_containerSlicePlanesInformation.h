/** @file CUDA_containerSlicePlanesInformation.h
 *
 *  @brief File for the slice planes information for the 
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on May 11, 2012
 *
 *  @note This is primarily an internal file used by the vtkCudaVolumeInformationHandler and CUDA_renderAlgo to store and communicate constants
 *
 */

#ifndef __CUDASLICEPLANESINFORMATION_H__
#define __CUDASLICEPLANESINFORMATION_H__

#include "vector_types.h"

/** @brief A stucture located on the CUDA hardware that holds all the information required about the volume being renderered.
 *
 */
extern "C"
typedef struct __align__(16) {

	int NumberOfSlicingPlanes;		/**< Number of additional user defined slicing planes to a maximum of 6 */
	float SlicingPlanes[24];		/**< Parameters defining each of the additional user defined slicing planes */

} cudaSlicePlanesInformation;

#endif
