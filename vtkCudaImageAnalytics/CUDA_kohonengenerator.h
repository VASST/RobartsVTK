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

void CUDAalgo_KSOMInitialize( double* range, Kohonen_Generator_Information& information, int* KMapSize,
								float** device_KohonenMap, float** device_tempSpace,
								float** device_DistanceBuffer, short2** device_IndexBuffer,
								float meansWidth, float varsWidth, cudaStream_t* stream );

void CUDAalgo_KSOMIteration( float** inputData,  char** maskData, int epoch,
								int* KMapSize,
								float** device_KohonenMap, float** device_tempSpace,
								float** device_DistanceBuffer, short2** device_IndexBuffer,
								int* VolumeSize, int NumVolumes,
								Kohonen_Generator_Information& information,
								int BatchSize,
								float meansAlpha, float meansWidth,
								float varsAlpha, float varsWidth,
								cudaStream_t* stream );

void CUDAalgo_KSOMOffLoad( float* outputKohonen, float** device_KohonenMap,
							float** device_tempSpace,
							float** device_DistanceBuffer, short2** device_IndexBuffer,
							Kohonen_Generator_Information& information,
							cudaStream_t* stream );

#endif //__KOHONENGENERATOR_H__