/** @file cudaRendererInformation.h
 *
 *  @brief File for renderer information holding structure used for volume ray casting
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on March 27, 2011
 *
 *  @note This is primarily an internal file used by the vtkCudaRendererInformationHandler and CUDA_renderAlgo to store and communicate constants
 *
 */

#ifndef __CUDARENDERERINFORMATION_H__
#define __CUDARENDERERINFORMATION_H__

#include "vector_types.h"

/** @brief A stucture located on the CUDA hardware that holds all the information required about the renderer.
 *
 */
typedef struct __align__(16)
{
	uint2 actualResolution;			/**< The resolution of the rendering screen */

	float ViewToVoxelsMatrix[16];	/**< 4x4 matrix mapping the view space (0 to 1 in each direction, with 0 and 1 in x and y being the borders of the screen, and 0 and 1 in z being the clipping planes) to the volume space */

	int NumberOfClippingPlanes;		/**< Number of additional user defined clipping planes to a maximum of 6 */
	float ClippingPlanes[24];		/**< Parameters defining each of the additional user defined clipping planes */
	
	int NumberOfKeyholePlanes;		/**< Number of additional user defined keyhole planes to a maximum of 6 */
	float KeyholePlanes[24];		/**< Parameters defining each of the additional user defined keyhole planes */

	//Cel shading constants
	float celr;						/**< Multiplicative constant for Cel shading of the volume */
	float cela;						/**< Start */
	float celb;						/**< Stop */
	float celc;						/**< Additive constant for the Cel shading of the volume */
	
	//Distance shading constants
	float disr;						/**< Multiplicative constant for distance shading of the volume */
	float disa;						/**< Start */
	float disb;						/**< Stop */
	float disc;						/**< Additive constant for the distance shading of the volume */

} cudaRendererInformation;

#endif
