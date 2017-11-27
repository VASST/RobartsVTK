#ifndef __KOHONENGENERATOR_H__
#define __KOHONENGENERATOR_H__

#include "CudaCommon.h"

#define MAX_DIMENSIONALITY 16

typedef struct __align__(16)
{
  int KohonenMapSize[3];
  int NumberOfDimensions;
  int flags;

  float epsilon;

} Kohonen_Generator_Information;

void CUDAalgo_KSOMInitialize( double* Means, double* Covariances, double* Eig1, double* Eig2,
                Kohonen_Generator_Information& information, int* KMapSize,
                float** device_KohonenMap, float** device_tempSpace,
                float** device_DistanceBuffer, short2** device_IndexBuffer, float** device_WeightBuffer, 
                float meansWidth, float varsWidth, float weiWidth,
                cudaStream_t* stream );

void CUDAalgo_KSOMIteration( float** inputData,  char** maskData, int epoch,
                int* KMapSize,
                float** device_KohonenMap, float** device_tempSpace,
                float** device_DistanceBuffer, short2** device_IndexBuffer, float** device_WeightBuffer, 
                int* VolumeSize, int NumVolumes,
                Kohonen_Generator_Information& information,
                int BatchSize,
                float meansAlpha, float meansWidth,
                float varsAlpha, float varsWidth,
                float weiAlpha, float weiWidth,
                cudaStream_t* stream );

void CUDAalgo_KSOMOffLoad( float* outputKohonen, float** device_KohonenMap,
              float** device_tempSpace,
              float** device_DistanceBuffer, short2** device_IndexBuffer, float** device_WeightBuffer, 
              Kohonen_Generator_Information& information,
              cudaStream_t* stream );

#endif //__KOHONENGENERATOR_H__