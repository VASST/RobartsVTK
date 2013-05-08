#ifndef __PAGMMESTIMATOR_H__
#define __PAGMMESTIMATOR_H__

#include "vector_types.h"

#define MAX_DIMENSIONALITY 16

typedef struct __align__(16)
{
	int VolumeSize[3];
	int GMMSize[3];
	int NumberOfDimensions;
	int NumberOfLabels;

} PAGMM_Information;

void CUDAalgo_applyPAGMMModel( float* inputData, float* inputGMM, float* outputData, float* outputGMM,
								char* seededImage, PAGMM_Information& information, float P, float Q, float scale,
								cudaStream_t* stream );

#endif //__KOHONENAPPLICATION_H__