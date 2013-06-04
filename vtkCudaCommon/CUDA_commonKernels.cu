#include "CUDA_commonKernels.h"
#include <curand_kernel.h>

//---------------------------------------------------------------------------//
//-------------------------COMMON UNARY OPERATORS----------------------------//
//---------------------------------------------------------------------------//

template<class T> 
__global__ void ZeroOutBuffer(T* buffer, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	if(offset < size ) buffer[offset] = 0;
}
template __global__ void ZeroOutBuffer<float> (float* buffer, int size);

template<class T> 
__global__ void OneOutBuffer(T* buffer, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	if(offset < size ) buffer[offset] = 1;
}
template __global__ void OneOutBuffer<float> (float* buffer, int size);

template<class T> 
__global__ void SetBufferToConst(T* buffer, T value, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx < size ) buffer[idx] = value;
}
template __global__ void SetBufferToConst<float> (float* buffer, float value, int size);

__global__ void ReplaceNANs(float* buffer, float value, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	float current = buffer[offset];
	current = isfinite(current) ? current : value;
	if(offset < size ) buffer[offset] = current;
}

template<class T, class S>
__global__ void IncrementBuffer(T* labelBuffer, T desiredLabel, S* agreement, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	S newAgreement = agreement[idx];
	T labelValue = labelBuffer[idx];
	newAgreement += (labelValue == desiredLabel) ? 1 : 0;
	if( idx < size ) agreement[idx] = newAgreement;
}
template __global__ void IncrementBuffer<float, short> (float* labelBuffer, float desiredLabel, short* agreement, int size);

__global__ void SetBufferToRandom(float* buffer, float min, float max, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState;
	curand_init(7+offset, offset, 0, &localState);
	__syncthreads();

	float value = min + (max-min)*curand_uniform(&localState);
	if(offset < size ) buffer[offset] = value;
}

//---------------------------------------------------------------------------//
//-------------------------COMMON BINARY OPERATORS---------------------------//
//---------------------------------------------------------------------------//

template<class T> 
__global__ void SumBuffers(T* outBuffer, T* sumBuffer, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	T value = outBuffer[offset] + sumBuffer[offset];
	if(offset < size ) outBuffer[offset] = value;
}
template __global__ void SumBuffers<float>(float* outBuffer, float* sumBuffer, int size);

template<class T> 
__global__ void CopyBuffers(T* outBuffer, T* inBuffer, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	T value = inBuffer[offset];
	if(offset < size ) outBuffer[offset] = value;
}
template __global__ void CopyBuffers<float>(float* outBuffer, float* inBuffer, int size);

template<class T>
__global__ void TranslateBuffer(T* buffer, T scale, T shift, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	T value = scale * buffer[offset] + shift;
	if(offset < size ) buffer[offset] = value;
}
template __global__ void TranslateBuffer<float>(float* buffer, float scale, float shift, int size);

template<class T>
__global__ void MultiplyBuffers(T* outBuffer, T* multBuffer, T scale, T shift, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	float value = (scale * outBuffer[offset] + shift) * multBuffer[offset];
	if(offset < size ) outBuffer[offset] = value;
}
template __global__ void MultiplyBuffers<float>(float* buffer, float* multBuffer, float scale, float shift, int size);

template<class T>
__global__ void MultiplyBuffers(T* outBuffer, T* multBuffer, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	float value = outBuffer[offset] * multBuffer[offset];
	if(offset < size ) outBuffer[offset] = value;
}
template __global__ void MultiplyBuffers<float>(float* buffer, float* multBuffer, int size);

template<class T>
__global__ void MultiplyAndStoreBuffer(T* inBuffer, T* outBuffer, T number, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	T value = inBuffer[idx] * number;
	if( idx < size ) outBuffer[idx] = value;
}
template __global__ void MultiplyAndStoreBuffer<float>(float* inBuffer, float* outBuffer, float number, int size);


//---------------------------------------------------------------------------//
//----------------------------COMMON ACCUMULATORS----------------------------//
//---------------------------------------------------------------------------//


void SumData(int size, int threads, int blocks, float* dataBuffer, cudaStream_t* stream ){

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * (sizeof(float)+sizeof(short2)) : threads * (sizeof(float)+sizeof(short2));
	switch (threads)
	{
	case 512:
		SumOverSmallBuffer<512><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 256:
		SumOverSmallBuffer<256><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 128:
		SumOverSmallBuffer<128><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 64:
		SumOverSmallBuffer< 64><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 32:
		SumOverSmallBuffer< 32><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 16:
		SumOverSmallBuffer< 16><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 8:
		SumOverSmallBuffer< 8><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 4:
		SumOverSmallBuffer< 4><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 2:
		SumOverSmallBuffer< 2><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 1:
		SumOverSmallBuffer< 1><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	}

}

template <unsigned int blockSize>
__global__ void SumOverSmallBuffer(float *buffer, unsigned int n)
{
	__shared__ float sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0.0f;
	
	while (i < n) {
		sdata[tid] += buffer[i];
		sdata[tid] += buffer[i+blockSize];
		i += gridSize;
		__syncthreads();
	}
	
	if (blockSize >= 512) { if (tid < 256) {
			sdata[tid] += sdata[tid + 256];
	} __syncthreads(); }

	if (blockSize >= 256) { if (tid < 128) {
			sdata[tid] += sdata[tid + 128];
	} __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) {
			sdata[tid] += sdata[tid + 64];
	} __syncthreads(); }
	
	if (tid < 32) {
		if (blockSize >= 64){
			sdata[tid] += sdata[tid + 32];
			__syncthreads();
		}
		if (blockSize >= 32){
			sdata[tid] += sdata[tid + 16];
			__syncthreads();
		}
		if (blockSize >= 16){
			sdata[tid] += sdata[tid + 8];
			__syncthreads();
		}
		if (blockSize >=  8){
			sdata[tid] += sdata[tid + 4];
			__syncthreads();
		}
		if (blockSize >=  4){
			sdata[tid] += sdata[tid + 2];
			__syncthreads();
		}
		if (blockSize >=  2){
			sdata[tid] += sdata[tid + 1];
			__syncthreads();
		}
	}
	if (tid == 0){
		buffer[0] = sdata[0];
	}
}

__global__ void SumOverLargeBuffer( float* buffer, int spread, int size ){
	
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	float value1 = buffer[kOffset];
	float value2 = buffer[kOffset+spread];

	if( kOffset+spread < size )
		buffer[kOffset] = value1+value2;

}