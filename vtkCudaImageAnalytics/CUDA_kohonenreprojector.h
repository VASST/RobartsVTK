#ifndef __KOHONENREPROJECTOR_H__
#define __KOHONENREPROJECTOR_H__

#include "vector_types.h"

#define MAX_DIMENSIONALITY 16

typedef struct __align__(16)
{
	int VolumeSize[3];
	int KohonenMapSize[3];
	int NumberOfDimensions;

} Kohonen_Reprojection_Information;

void CUDAalgo_reprojectKohonenMap( float* inputData, float* inputKohonen, float* outputData,
									Kohonen_Reprojection_Information& information,
									cudaStream_t* stream );

#endif //__KOHONENAPPLICATION_H__