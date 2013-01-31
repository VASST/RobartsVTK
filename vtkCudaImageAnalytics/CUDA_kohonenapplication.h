#ifndef __KOHONENAPPLICATION_H__
#define __KOHONENAPPLICATION_H__

#include "vector_types.h"

#define MAX_DIMENSIONALITY 16

typedef struct __align__(16)
{
	int VolumeSize[3];
	int KohonenMapSize[3];
	int NumberOfDimensions;

	float Weights[MAX_DIMENSIONALITY];

} Kohonen_Application_Information;

void CUDAalgo_applyKohonenMap( float* inputData, char* inputMask, float* inputKohonen, float* outputData,
									Kohonen_Application_Information& information,
									cudaStream_t* stream );

#endif //__KOHONENAPPLICATION_H__