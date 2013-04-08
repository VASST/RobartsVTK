#ifndef __CUDA_HIERARCHICALMAXFLOW_H__
#define __CUDA_HIERARCHICALMAXFLOW_H__

#include "vector_types.h"

int CUDA_GetGPUBuffers( int maxNumber, float** buffer, int volSize );
void CUDA_ReturnGPUBuffers(float* buffer);

void CUDA_CopyBufferToCPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream);
void CUDA_CopyBufferToGPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream);

void CUDA_zeroOutBuffer(float* GPUBuffer, int size, cudaStream_t* stream);
void CUDA_SetBufferToValue(float* GPUBuffer, float value, int size, cudaStream_t* stream);
void CUDA_divideAndStoreBuffer(float* inBuffer, float* outBuffer, float number, int size, cudaStream_t* stream);

void CUDA_storeSinkFlowInBuffer(float* workingBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream);
void CUDA_storeSourceFlowInBuffer(float* workingBuffer, float* sinkBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream);

void CUDA_updateLeafSinkFlow(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream );
void CUDA_constrainLeafSinkFlow(float* sinkBuffer, float* capBuffer, int size, cudaStream_t* stream );

void CUDA_updateLabel(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream );

void CUDA_flowGradientStep(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float stepSize, float CC, int size, cudaStream_t* stream );
void CUDA_applyStep(float* divBuffer, float* flowX, float* flowY, float* flowZ, int X, int Y, int Z, int size, cudaStream_t* stream );
void CUDA_computeFlowMag(float* divBuffer, float* flowX, float* flowY, float* flowZ, float* smoothnessTerm, float smoothnessConstant, int X, int Y, int Z, int size, cudaStream_t* stream );
void CUDA_projectOntoSet(float* divBuffer, float* flowX, float* flowY, float* flowZ, int X, int Y, int Z, int size, cudaStream_t* stream );

#endif //__CUDA_HIERARCHICALMAXFLOW_H__