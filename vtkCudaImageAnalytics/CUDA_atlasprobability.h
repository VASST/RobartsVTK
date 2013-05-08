#ifndef __CUDA_ATLASPROBABILITY_H__
#define __CUDA_ATLASPROBABILITY_H__

#include "vector_types.h"

void CUDA_GetRelevantBuffers(short** agreement, float** output, int size, cudaStream_t* stream);
void CUDA_ConvertInformation(short* agreement, float* output, float maxOut, int size, short max, short flags, cudaStream_t* stream);
void CUDA_CopyBackResult(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream);

template<class T>
void CUDA_IncrementInformation(T* labelData, T desiredValue, short* agreement, int size, cudaStream_t* stream);

#endif //__CUDA_ATLASPROBABILITY_H__