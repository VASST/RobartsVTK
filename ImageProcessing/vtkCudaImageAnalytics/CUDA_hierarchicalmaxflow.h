/** @file CUDA_hierarchicalmaxflow.h
 *
 *  @brief Header file with definitions of GPU kernels used predominantly in GHMF segmentation
 *      These are used only by vtkHierarchicalMaxFlowSegmentation and vtkHierarchicalMaxFlowSegmentation2.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#ifndef __CUDA_HIERARCHICALMAXFLOW_H__
#define __CUDA_HIERARCHICALMAXFLOW_H__

#include "CudaCommon.h"

void CUDA_GetGPUBuffers( int maxNumber, double maxPercent, float** buffer, int pad, int volSize, int* numberAcquired, double* percentAcquired );
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

void CUDA_CopyBuffer(float* dst, float* src, int size, cudaStream_t* stream);
void CUDA_MinBuffer(float* dst, float* src, int size, cudaStream_t* stream);
void CUDA_LblBuffer(float* lbl, float* flo, float* cap, int size, cudaStream_t* stream);
void CUDA_SumBuffer(float* dst, float* src, int size, cudaStream_t* stream);
void CUDA_SumScaledBuffer(float* dst, float* src, float scale, int size, cudaStream_t* stream);
void CUDA_DivBuffer(float* dst, float* src, int size, cudaStream_t* stream);

void CUDA_ShiftBuffer(float* buf, float shift, int size, cudaStream_t* stream);
void CUDA_ResetSinkBuffer(float* sink, float* source, float* div, float* label, float ik, float iCC, int size, cudaStream_t* stream);
void CUDA_PushUpSourceFlows(float* psink, float* sink, float* source, float* div, float* label, float w, float iCC, int size, cudaStream_t* stream);
void CUDA_Copy2Buffers(float* fIn, float* fOut1, float* fOut2, int size, cudaStream_t* stream);

#endif //__CUDA_HIERARCHICALMAXFLOW_H__