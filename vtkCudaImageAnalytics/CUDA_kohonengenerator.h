#ifndef __KOHONENGENERATOR_H__
#define __KOHONENGENERATOR_H__

#include "vector_types.h"

#define MAX_DIMENSIONALITY 16

typedef struct __align__(16)
{
	int KohonenMapSize[3];
	int NumberOfDimensions;
	int flags;

	float Weights[MAX_DIMENSIONALITY];

} Kohonen_Generator_Information;

void CUDAalgo_generateKohonenMap( float** inputData, float* outputKohonen, char** maskData, double* range,
									int* VolumeSize, int NumVolumes,
									Kohonen_Generator_Information& information,
									int MaxEpochs, int BatchSize,
									float alphaVMult, float alphaVShift, float alphaHMult, float alphaHShift,
									float nVMult, float nVShift, float nHMult, float nHShift,
									cudaStream_t* stream );

#endif //__KOHONENGENERATOR_H__