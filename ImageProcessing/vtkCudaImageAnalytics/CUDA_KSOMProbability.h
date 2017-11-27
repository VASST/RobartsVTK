#ifndef __KOHONENPROBABILITY_H__
#define __KOHONENPROBABILITY_H__

#include "CudaCommon.h"

#define MAX_DIMENSIONALITY 16

typedef struct __align__(16)
{
  int VolumeSize[3];
  int KohonenMapSize[3];
  int NumberOfDimensions;
  int NumberOfLabels;
  int BufferSize;

  float Scale;

} Kohonen_Probability_Information;

void CUDAalgo_applyProbabilityMaps( float* inputData, float* inputKohonen, float** probabilityData,
                  float** outputData, bool useProbData, bool useEntropy,
                  Kohonen_Probability_Information& information, cudaStream_t* stream );

#endif //__KOHONENAPPLICATION_H__