#ifndef __KOHONENAPPLICATION_H__
#define __KOHONENAPPLICATION_H__

#include "vector_types.h"

#define MAX_DIMENSIONALITY 16

typedef struct __align__(16)
{
	int VolumeSize[3];				/**< Size of the volume being classified */

	float	TextureSize;
	float	Intensity1Low;			/**< Minimum intensity of the first component of the image */
	float	Intensity1Multiplier;	/**< Scale factor to normalize first intensities to between 0 and 1 */
	float	Intensity2Low;			/**< Minimum intensity of the second component of the image */
	float	Intensity2Multiplier;	/**< Scale factor to normalize second intensities to between 0 and 1 */

	int NumberOfClippingPlanes;		/**< Number of additional user defined clipping planes to a maximum of 6 */
	float ClippingPlanes[24];		/**< Parameters defining each of the additional user defined clipping planes */
	
	int NumberOfKeyholePlanes;		/**< Number of additional user defined keyhole planes to a maximum of 6 */
	float KeyholePlanes[24];		/**< Parameters defining each of the additional user defined keyhole planes */

} Voxel_Classifier_Information;

void CUDAalgo_classifyVoxels( float* inputData, short* inputPrimaryTexture, short* inputKeyholeTexture, int textureSize,
								short* outputData, Voxel_Classifier_Information& information,
								cudaStream_t* stream );

#endif //__KOHONENAPPLICATION_H__