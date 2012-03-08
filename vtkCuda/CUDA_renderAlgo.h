/*
 * Volume Rendering on CUDA.
 * Author: Nicholas Herlambang
 * Second Author: Benjamin Grauer
 */

#ifndef CUDA_RENDERALGO_H
#define CUDA_RENDERALGO_H
#include "cudaRendererInformation.h"
#include "cudaVolumeInformation.h"

extern "C"
void CUDArenderAlgo_doRender(const cudaRendererInformation& renderInfo,
                             const cudaVolumeInformation& volumeInfo);

extern "C"
void CUDArenderAlgo_changeFrame(int frame);

extern "C"
void CUDAkernelsetup_initImageArray();

extern "C"
void CUDAkernelsetup_clearImageArray();

extern "C"
void CUDAkernelsetup_loadTextures(const cudaVolumeInformation& volumeInfo, int FunctionSize,
								  float* redTF, float* greenTF, float* blueTF, float* alphaTF);

extern "C"
void CUDAkernelsetup_loadImageInfo(const cudaVolumeInformation& volumeInfo, int index);

extern "C"
void CUDAkernelsetup_loadRandoms(float* randoms);

#endif
