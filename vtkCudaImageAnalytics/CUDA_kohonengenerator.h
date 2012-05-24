#ifndef __KOHONENGENERATOR_H__
#define __KOHONENGENERATOR_H__

#include "vector_types.h"

typedef struct __align__(16)
{

	// The resolution of the rStartering screen.
	int VolumeSize[3];
	float spacing[3];
	short numberOfDimensions;
	short flags;

	float distanceBuffer;

	int OutputResolution[3];

} Kohonen_Generator_Information;

void CUDAsetup_loadNDImage( cudaStream_t* stream );

void CUDAsetup_loadNDImage( float* image, Kohonen_Generator_Information& information, cudaStream_t* stream);

void CUDAalgo_generateKohonenMap( float* outputKohonen, Kohonen_Generator_Information& information, cudaStream_t* stream );

#endif //__KOHONENGENERATOR_H__