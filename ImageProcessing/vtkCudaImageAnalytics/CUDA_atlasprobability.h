/*=========================================================================

  Program:   Visualization Toolkit
  Module:    CUDA_atlasprobability.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file CUDA_atlasprobability.h
 *
 *  @brief Header file with definitions of GPU kernels used for the 'atlas probability'
 *      prior.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#ifndef __CUDA_ATLASPROBABILITY_H__
#define __CUDA_ATLASPROBABILITY_H__

#include "CudaCommon.h"

void CUDA_GetRelevantBuffers(short** agreement, float** output, int size, cudaStream_t* stream);
void CUDA_ConvertInformation(short* agreement, float* output, float maxOut, int size, short max, short flags, int gaussWidth[], int imageDims[], cudaStream_t* stream);
void CUDA_CopyBackResult(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream);

template<class T>
void CUDA_IncrementInformation(T* labelData, T desiredValue, short* agreement, int size, cudaStream_t* stream);

#endif //__CUDA_ATLASPROBABILITY_H__