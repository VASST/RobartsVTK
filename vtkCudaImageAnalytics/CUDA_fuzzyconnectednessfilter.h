#ifndef __FUZZYCONNECTEDNESSFILTER_H__
#define __FUZZYCONNECTEDNESSFILTER_H__

#include "vector_types.h"

typedef struct __align__(16)
{

	//image parameters
	int3 VolumeSize;
	float3 Spacing;

	//neighbourhood parameters
	char tnorm; // 0 - min, 1 - product, 2 - Hamacher
	char snorm; // 0 - max, 1 - sum,     2 - Einstein

	//weighting
	float gradientWeight;
	float distanceWeight;

} Fuzzy_Connectedness_Information;

template<class T>
void CUDAalgo_calculateConnectedness( float* connectedness, float* seed, int numIterations, T* image, int numCompo,
	Fuzzy_Connectedness_Information& information, cudaStream_t* stream );

#endif