#ifndef __CUDA_COMMON_KERNELS_H__
#define __CUDA_COMMON_KERNELS_H__

//---------------------------------------------------------------------------//
//-------------------------COMMON UNARY OPERATORS----------------------------//
//---------------------------------------------------------------------------//

template<class T> __global__ void ZeroOutBuffer(T* buffer, int size);
template<class T> __global__ void OneOutBuffer(T* buffer, int size);
template<class T> __global__ void SetBufferToConst(T* buffer, T value, int size);
__global__ void ReplaceNANs(float* buffer, float value, int size);
template<class T, class S> __global__ void IncrementBuffer(T* labelBuffer, T desiredLabel, S* agreement, int size);
__global__ void SetBufferToRandom(float* buffer, float min, float max, int size);

//---------------------------------------------------------------------------//
//-------------------------COMMON BINARY OPERATORS---------------------------//
//---------------------------------------------------------------------------//

template<class T> __global__ void SumBuffers(T* outBuffer, T* sumBuffer, int size);
template<class T> __global__ void CopyBuffers(T* outBuffer, T* inBuffer, int size);
template<class T> __global__ void TranslateBuffer(T* buffer, T scale, T shift, int size);
template<class T> __global__ void MultiplyBuffers(T* outBuffer, T* multBuffer, T scale, T shift, int size);
template<class T> __global__ void MultiplyBuffers(T* outBuffer, T* multBuffer, int size);
template<class T> __global__ void MultiplyAndStoreBuffer(T* inBuffer, T* outBuffer, T number, int size);

//---------------------------------------------------------------------------//
//----------------------------COMMON ACCUMULATORS----------------------------//
//---------------------------------------------------------------------------//

void SumData(int size, int threads, int blocks, float* dataBuffer, cudaStream_t* stream );
template <unsigned int blockSize> __global__ void SumOverSmallBuffer(float *buffer, unsigned int n);
__global__ void SumOverLargeBuffer( float* buffer, int spread, int size );

#endif