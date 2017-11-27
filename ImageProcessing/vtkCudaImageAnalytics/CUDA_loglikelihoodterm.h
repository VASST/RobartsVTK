/*=========================================================================

  Program:   Visualization Toolkit
  Module:    CUDA_loglikelihoodterm.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file CUDA_loglikelihoodterm.h
 *
 *  @brief Header file with definitions of GPU kernels used predominantly in calculating
 *      log likelihood components of data terms for GHMF.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#ifndef __CUDA_LOGLIKELIHOOD_H__
#define __CUDA_LOGLIKELIHOOD_H__

#include "CudaCommon.h"

void CUDA_ILLT_GetRelevantBuffers(short** agreement, int size, cudaStream_t* stream);
void CUDA_ILLT_CopyBackResult(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream);

template<class T>
void CUDA_ILLT_IncrementInformation(T* labelData, T desiredValue, short* agreement, int size, cudaStream_t* stream);

void CUDA_ILLT_AllocateHistogram(float** histogramGPU, int numBins, cudaStream_t* stream);
void CUDA_ILLT_ReturnBuffer(float* buffer);

template< class T >
void CUDA_ILLT_CalculateHistogramAndTerms(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, T* image, short requiredAgreement, int imageSize, cudaStream_t* stream);

template< class T >
void CUDA_ILLT_CalculateHistogramAndTerms2D(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, T* image, short requiredAgreement, int imageSize, cudaStream_t* stream);


#endif //__CUDA_ATLASPROBABILITY_H__