/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    CUDA_commonKernels.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file CUDA_commonKernels.h
 *
 *  @brief Header file with definitions of GPU kernels used in several GPU-accelerated classes.
 *      This includes simple utilities like created validly sized grids.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#ifndef __CUDA_COMMON_KERNELS_H__
#define __CUDA_COMMON_KERNELS_H__

#include "vtkCudaCommonStaticExport.h"

//---------------------------------------------------------------------------//
//-------------------------COMMON UNARY OPERATORS----------------------------//
//---------------------------------------------------------------------------//

template<class T> __global__ void ZeroOutBuffer(T* buffer, int size);
template<class T> __global__ void OneOutBuffer(T* buffer, int size);
template<class T> __global__ void SetBufferToConst(T* buffer, T value, int size);
template<class T> __global__ void TranslateBuffer(T* buffer, T scale, T shift, int size);
__global__ void ReplaceNANs(float* buffer, float value, int size);
template<class T, class S> __global__ void IncrementBuffer(T* labelBuffer, T desiredLabel, S* agreement, int size);
__global__ void SetBufferToRandom(float* buffer, float min, float max, int size);
template<class T> __global__ void LogBuffer(T* buffer, int size);
template<class T> __global__ void NegLogBuffer(T* buffer, int size);
template<class T> __global__ void ExpBuffer(T* buffer, int size);
template<class T> __global__ void NegExpBuffer(T* buffer, int size);

//---------------------------------------------------------------------------//
//-------------------------COMMON BINARY OPERATORS---------------------------//
//---------------------------------------------------------------------------//

template<class T> __global__ void SumBuffers(T* outBuffer, T* sumBuffer, int size);
template<class T> __global__ void SumScaledBuffers(T* outBuffer, T* sumBuffer, T scale, int size);
template<class T> __global__ void CopyBuffers(T* outBuffer, T* inBuffer, int size);
template<class T> __global__ void MultiplyBuffers(T* outBuffer, T* multBuffer, T scale, T shift, int size);
template<class T> __global__ void MultiplyBuffers(T* outBuffer, T* multBuffer, int size);
template<class T> __global__ void MinBuffers(T* outBuffer, T* inBuffer, int size);
template<class T> __global__ void DivideBuffers(T* outBuffer, T* denomBuffer, int size);
template<class T> __global__ void MultiplyAndStoreBuffer(T* inBuffer, T* outBuffer, T number, int size);
template<class T> __global__ void MultiplyAndStoreBuffer(T* inBuffer1, T* inBuffer2, T* outBuffer, int size);

//---------------------------------------------------------------------------//
//----------------------------COMMON ACCUMULATORS----------------------------//
//---------------------------------------------------------------------------//

void SumData(int size, int threads, int blocks, float* dataBuffer, cudaStream_t* stream );
template <unsigned int blockSize> __global__ void SumOverSmallBuffer(float *buffer, unsigned int n);
__global__ void SumOverLargeBuffer( float* buffer, int spread, int size );

void LogaritureData(int size, int threads, int blocks, float* dataBuffer, cudaStream_t* stream );
template <unsigned int blockSize> __global__ void LogaritureOverSmallBuffer(float *buffer, unsigned int n);
__global__ void LogaritureOverLargeBuffer( float* buffer, int spread, int size );

#endif