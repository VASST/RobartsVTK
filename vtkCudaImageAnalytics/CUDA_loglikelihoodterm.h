#ifndef __CUDA_LOGLIKELIHOOD_H__
#define __CUDA_LOGLIKELIHOOD_H__

#include "vector_types.h"

void CUDA_ILLT_GetRelevantBuffers(short** agreement, int size, cudaStream_t* stream);
void CUDA_ILLT_CopyBackResult(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream);

template<class T>
void CUDA_ILLT_IncrementInformation(T* labelData, T desiredValue, short* agreement, int size, cudaStream_t* stream);

void CUDA_ILLT_AllocateHistogram(float** histogramGPU, int numBins, cudaStream_t* stream);
void CUDA_ILLT_ReturnBuffer(float* buffer);

template< class T >
void CUDA_ILLT_CalculateHistogramAndTerms(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, T* image, short requiredAgreement, int imageSize, cudaStream_t* stream);

template< class T >
void CUDA_ILLT_CalculateHistogramAndTerms2D(float* outputBuffer, float* histogramGPU, short* agreement, T* image, short requiredAgreement, int imageSize, cudaStream_t* stream);


#endif //__CUDA_ATLASPROBABILITY_H__