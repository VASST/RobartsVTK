#ifndef __CUDA_KSOMLIKELIHOOD_H__
#define __CUDA_KSOMLIKELIHOOD_H__

#include "vector_types.h"

#define MAX_DIMENSIONALITY 16

typedef struct __align__(16)
{
	int VolumeSize[3];
	int GMMSize[3];
	int NumberOfDimensions;
	int NumberOfLabels;

} KSOMLL_Information;

void CUDAalgo_applyKSOMLLModel( float* inputData, float* inputGMM, float* outputGMM,
								char* seededImage, KSOMLL_Information& information, float scale,
								cudaStream_t* stream );

#endif //__KOHONENAPPLICATION_H__