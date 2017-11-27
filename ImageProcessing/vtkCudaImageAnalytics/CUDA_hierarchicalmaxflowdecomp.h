#ifndef __CUDA_HIERARCHICALMAXFLOWDECOMP_H__
#define __CUDA_HIERARCHICALMAXFLOWECOMP_H__

#include "CudaCommon.h"

double CUDA_GHMFD_DataTermForLabel(float* data, float* label, int size, cudaStream_t* stream);
double CUDA_GHMFD_LeafSmoothnessForLabel(float* smoothness, float* label, int x, int y, int z, int size, float* GPUParentLabel, float* devGradientBuffer, cudaStream_t* stream);
double CUDA_GHMFD_LeafNoSmoothnessForLabel( float* label, int x, int y, int z, int size, float* GPUParentLabel, float* devGradientBuffer, cudaStream_t* stream);

double CUDA_GHMFD_BranchSmoothnessForLabel(float* smoothness, float* devLabelBuffer, int x, int y, int z, int size, float* GPUParentLabel, float* devGradientBuffer, cudaStream_t* stream);
double CUDA_GHMFD_BranchNoSmoothnessForLabel( float* devLabelBuffer, int x, int y, int z, int size, float* GPUParentLabel, float* devGradientBuffer, cudaStream_t* stream);

float* CUDA_GHMFD_GetBuffer(int size, cudaStream_t* stream);
void CUDA_GHMFD_ReturnBuffer(float* buffer);

#endif //__CUDA_HIERARCHICALMAXFLOWDECOMP_H__