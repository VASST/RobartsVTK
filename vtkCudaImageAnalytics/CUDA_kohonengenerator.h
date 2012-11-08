#ifndef __KOHONENGENERATOR_H__
#define __KOHONENGENERATOR_H__

#include "vector_types.h"

#define MAX_DIMENSIONALITY 16

typedef struct __align__(16)
{
	int VolumeSize[3];
	int KohonenMapSize[3];
	int NumberOfDimensions;
	int flags;

	int MaxEpochs;
	int BatchSize;

	float Weights[MAX_DIMENSIONALITY];

} Kohonen_Generator_Information;

void CUDAalgo_generateKohonenMap( float* inputData, float* outputKohonen, char* maskData, double* range,
									Kohonen_Generator_Information& information,
									float alpha, float alphaDecay,
									float neighbourhood, float nDecay,
									cudaStream_t* stream );

#endif //__KOHONENGENERATOR_H__