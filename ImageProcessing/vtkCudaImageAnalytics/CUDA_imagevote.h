/*=========================================================================

  Program:   Visualization Toolkit
  Module:    CUDA_imagevote.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file CUDA_imagevote.h
 *
 *  @brief Header file with definitions of GPU kernels used predominantly in performing a voting
 *      operation to merge probabilistic labellings
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#ifndef __CUDA_IMAGEVOTE_H__
#define __CUDA_IMAGEVOTE_H__

#include "CudaCommon.h"

template<typename IT, typename OT>
void CUDA_CIV_COMPUTE( IT** inputBuffers, int inputNum, OT* outputBuffer, OT* map, int size, cudaStream_t* stream);



#endif