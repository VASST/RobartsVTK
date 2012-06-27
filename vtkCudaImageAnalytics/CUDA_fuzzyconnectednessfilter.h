#ifndef __FUZZYCONNECTEDNESSFILTER_H__
#define __FUZZYCONNECTEDNESSFILTER_H__

#include "vector_types.h"

typedef struct __align__(16)
{

	//image parameters
	int3 VolumeSize;
	int NumObjects;
	float3 Spacing;

	//neighbourhood parameters
	char tnorm; // 0 - min, 1 - product, 2 - Hamacher
	char snorm; // 0 - max, 1 - sum,     2 - Einstein

} Fuzzy_Connectedness_Information;

void CUDAalgo_calculateConnectedness( float* connectedness, float* seed, float* affinity, int numIterations,
	Fuzzy_Connectedness_Information& information, cudaStream_t* stream );

#endif