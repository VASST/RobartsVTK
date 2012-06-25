#ifndef __FUZZYCONNECTEDNESSFILTER_H__
#define __FUZZYCONNECTEDNESSFILTER_H__

#include "vector_types.h"

typedef struct __align__(16)
{

	//image parameters
	uint3 VolumeSize;
	float3 spacing;

	//neiughbourhood parameters
	int connectedness;
	int tnorm; // 0 - min, 1 - product, 2 - Hamacher
	int snorm; // 0 - max, 1 - sum,     2 - Einstein

} Fuzzy_Connectedness_Information;

void CUDAsetup_unloadImage(cudaStream_t* stream);

void CUDAsetup_loadImage( float* CTImage, Fuzzy_Connectedness_Information& information, cudaStream_t* stream);

void CUDAalgo_calculateConnectedness( float* connectedness, int numIterations, vtkType dataType, int numCompo,
	Fuzzy_Connectedness_Information& information, cudaStream_t* stream );

#endif