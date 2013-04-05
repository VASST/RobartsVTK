#ifndef __CUDA_HIERARCHICALMAXFLOW_H__
#define __CUDA_HIERARCHICALMAXFLOW_H__

#include "vector_types.h"

int CUDA_GetGPUBuffers( int maxNumber, float** buffer, int volSize );
void CUDA_ReturnGPUBuffers(float* buffer);

void CUDA_CopyBufferToCPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream);
void CUDA_CopyBufferToGPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream);

void CUDA_zeroOutBuffer(float* GPUBuffer, int size, cudaStream_t* stream);
void CUDA_divideAndStoreBuffer(float* inBuffer, float* outBuffer, float number, int size, cudaStream_t* stream);

#endif //__CUDA_HIERARCHICALMAXFLOW_H__