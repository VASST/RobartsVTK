#include "CUDA_commonKernels.h"
#include <curand_kernel.h>
#include <limits.h>
#include <float.h>

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
template __global__ void IncrementBuffer<char, char> (char* labelBuffer, char desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<signed char, char> (signed char* labelBuffer, signed char desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, char> (unsigned char* labelBuffer, unsigned char desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<short, char> (short* labelBuffer, short desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, char> (unsigned short* labelBuffer, unsigned short desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<int, char> (int* labelBuffer, int desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, char> (unsigned int* labelBuffer, unsigned int desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<long, char> (long* labelBuffer, long desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, char> (unsigned long* labelBuffer, unsigned long desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<float, char> (float* labelBuffer, float desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<double, char> (double* labelBuffer, double desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<long long, char> (long long* labelBuffer, long long desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, char> (unsigned long long* labelBuffer, unsigned long long desiredLabel, char* agreement, int size);
template __global__ void IncrementBuffer<char, signed char> (char* labelBuffer, char desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<signed char, signed char> (signed char* labelBuffer, signed char desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, signed char> (unsigned char* labelBuffer, unsigned char desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<short, signed char> (short* labelBuffer, short desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, signed char> (unsigned short* labelBuffer, unsigned short desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<int, signed char> (int* labelBuffer, int desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, signed char> (unsigned int* labelBuffer, unsigned int desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<long, signed char> (long* labelBuffer, long desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, signed char> (unsigned long* labelBuffer, unsigned long desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<float, signed char> (float* labelBuffer, float desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<double, signed char> (double* labelBuffer, double desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<long long, signed char> (long long* labelBuffer, long long desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, signed char> (unsigned long long* labelBuffer, unsigned long long desiredLabel, signed char* agreement, int size);
template __global__ void IncrementBuffer<char, unsigned char> (char* labelBuffer, char desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<signed char, unsigned char> (signed char* labelBuffer, signed char desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, unsigned char> (unsigned char* labelBuffer, unsigned char desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<short, unsigned char> (short* labelBuffer, short desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, unsigned char> (unsigned short* labelBuffer, unsigned short desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<int, unsigned char> (int* labelBuffer, int desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, unsigned char> (unsigned int* labelBuffer, unsigned int desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<long, unsigned char> (long* labelBuffer, long desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, unsigned char> (unsigned long* labelBuffer, unsigned long desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<float, unsigned char> (float* labelBuffer, float desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<double, unsigned char> (double* labelBuffer, double desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<long long, unsigned char> (long long* labelBuffer, long long desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, unsigned char> (unsigned long long* labelBuffer, unsigned long long desiredLabel, unsigned char* agreement, int size);
template __global__ void IncrementBuffer<char, short> (char* labelBuffer, char desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<signed char, short> (signed char* labelBuffer, signed char desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, short> (unsigned char* labelBuffer, unsigned char desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<short, short> (short* labelBuffer, short desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, short> (unsigned short* labelBuffer, unsigned short desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<int, short> (int* labelBuffer, int desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, short> (unsigned int* labelBuffer, unsigned int desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<long, short> (long* labelBuffer, long desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, short> (unsigned long* labelBuffer, unsigned long desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<float, short> (float* labelBuffer, float desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<double, short> (double* labelBuffer, double desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<long long, short> (long long* labelBuffer, long long desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, short> (unsigned long long* labelBuffer, unsigned long long desiredLabel, short* agreement, int size);
template __global__ void IncrementBuffer<char, unsigned short> (char* labelBuffer, char desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<signed char, unsigned short> (signed char* labelBuffer, signed char desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, unsigned short> (unsigned char* labelBuffer, unsigned char desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<short, unsigned short> (short* labelBuffer, short desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, unsigned short> (unsigned short* labelBuffer, unsigned short desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<int, unsigned short> (int* labelBuffer, int desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, unsigned short> (unsigned int* labelBuffer, unsigned int desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<long, unsigned short> (long* labelBuffer, long desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, unsigned short> (unsigned long* labelBuffer, unsigned long desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<float, unsigned short> (float* labelBuffer, float desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<double, unsigned short> (double* labelBuffer, double desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<long long, unsigned short> (long long* labelBuffer, long long desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, unsigned short> (unsigned long long* labelBuffer, unsigned long long desiredLabel, unsigned short* agreement, int size);
template __global__ void IncrementBuffer<char, int> (char* labelBuffer, char desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<signed char, int> (signed char* labelBuffer, signed char desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, int> (unsigned char* labelBuffer, unsigned char desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<short, int> (short* labelBuffer, short desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, int> (unsigned short* labelBuffer, unsigned short desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<int, int> (int* labelBuffer, int desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, int> (unsigned int* labelBuffer, unsigned int desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<long, int> (long* labelBuffer, long desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, int> (unsigned long* labelBuffer, unsigned long desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<float, int> (float* labelBuffer, float desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<double, int> (double* labelBuffer, double desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<long long, int> (long long* labelBuffer, long long desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, int> (unsigned long long* labelBuffer, unsigned long long desiredLabel, int* agreement, int size);
template __global__ void IncrementBuffer<char, unsigned int> (char* labelBuffer, char desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<signed char, unsigned int> (signed char* labelBuffer, signed char desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, unsigned int> (unsigned char* labelBuffer, unsigned char desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<short, unsigned int> (short* labelBuffer, short desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, unsigned int> (unsigned short* labelBuffer, unsigned short desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<int, unsigned int> (int* labelBuffer, int desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, unsigned int> (unsigned int* labelBuffer, unsigned int desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<long, unsigned int> (long* labelBuffer, long desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, unsigned int> (unsigned long* labelBuffer, unsigned long desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<float, unsigned int> (float* labelBuffer, float desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<double, unsigned int> (double* labelBuffer, double desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<long long, unsigned int> (long long* labelBuffer, long long desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, unsigned int> (unsigned long long* labelBuffer, unsigned long long desiredLabel, unsigned int* agreement, int size);
template __global__ void IncrementBuffer<char, long> (char* labelBuffer, char desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<signed char, long> (signed char* labelBuffer, signed char desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, long> (unsigned char* labelBuffer, unsigned char desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<short, long> (short* labelBuffer, short desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, long> (unsigned short* labelBuffer, unsigned short desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<int, long> (int* labelBuffer, int desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, long> (unsigned int* labelBuffer, unsigned int desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<long, long> (long* labelBuffer, long desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, long> (unsigned long* labelBuffer, unsigned long desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<float, long> (float* labelBuffer, float desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<double, long> (double* labelBuffer, double desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<long long, long> (long long* labelBuffer, long long desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, long> (unsigned long long* labelBuffer, unsigned long long desiredLabel, long* agreement, int size);
template __global__ void IncrementBuffer<char, unsigned long> (char* labelBuffer, char desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<signed char, unsigned long> (signed char* labelBuffer, signed char desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, unsigned long> (unsigned char* labelBuffer, unsigned char desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<short, unsigned long> (short* labelBuffer, short desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, unsigned long> (unsigned short* labelBuffer, unsigned short desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<int, unsigned long> (int* labelBuffer, int desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, unsigned long> (unsigned int* labelBuffer, unsigned int desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<long, unsigned long> (long* labelBuffer, long desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, unsigned long> (unsigned long* labelBuffer, unsigned long desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<float, unsigned long> (float* labelBuffer, float desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<double, unsigned long> (double* labelBuffer, double desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<long long, unsigned long> (long long* labelBuffer, long long desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, unsigned long> (unsigned long long* labelBuffer, unsigned long long desiredLabel, unsigned long* agreement, int size);
template __global__ void IncrementBuffer<char, float> (char* labelBuffer, char desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<signed char, float> (signed char* labelBuffer, signed char desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, float> (unsigned char* labelBuffer, unsigned char desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<short, float> (short* labelBuffer, short desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, float> (unsigned short* labelBuffer, unsigned short desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<int, float> (int* labelBuffer, int desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, float> (unsigned int* labelBuffer, unsigned int desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<long, float> (long* labelBuffer, long desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, float> (unsigned long* labelBuffer, unsigned long desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<float, float> (float* labelBuffer, float desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<double, float> (double* labelBuffer, double desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<long long, float> (long long* labelBuffer, long long desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, float> (unsigned long long* labelBuffer, unsigned long long desiredLabel, float* agreement, int size);
template __global__ void IncrementBuffer<char, double> (char* labelBuffer, char desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<signed char, double> (signed char* labelBuffer, signed char desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, double> (unsigned char* labelBuffer, unsigned char desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<short, double> (short* labelBuffer, short desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, double> (unsigned short* labelBuffer, unsigned short desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<int, double> (int* labelBuffer, int desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, double> (unsigned int* labelBuffer, unsigned int desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<long, double> (long* labelBuffer, long desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, double> (unsigned long* labelBuffer, unsigned long desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<float, double> (float* labelBuffer, float desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<double, double> (double* labelBuffer, double desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<long long, double> (long long* labelBuffer, long long desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, double> (unsigned long long* labelBuffer, unsigned long long desiredLabel, double* agreement, int size);
template __global__ void IncrementBuffer<char, long long> (char* labelBuffer, char desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<signed char, long long> (signed char* labelBuffer, signed char desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, long long> (unsigned char* labelBuffer, unsigned char desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<short, long long> (short* labelBuffer, short desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, long long> (unsigned short* labelBuffer, unsigned short desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<int, long long> (int* labelBuffer, int desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, long long> (unsigned int* labelBuffer, unsigned int desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<long, long long> (long* labelBuffer, long desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, long long> (unsigned long* labelBuffer, unsigned long desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<float, long long> (float* labelBuffer, float desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<double, long long> (double* labelBuffer, double desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<long long, long long> (long long* labelBuffer, long long desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, long long> (unsigned long long* labelBuffer, unsigned long long desiredLabel, long long* agreement, int size);
template __global__ void IncrementBuffer<char, unsigned long long> (char* labelBuffer, char desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<signed char, unsigned long long> (signed char* labelBuffer, signed char desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned char, unsigned long long> (unsigned char* labelBuffer, unsigned char desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<short, unsigned long long> (short* labelBuffer, short desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned short, unsigned long long> (unsigned short* labelBuffer, unsigned short desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<int, unsigned long long> (int* labelBuffer, int desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned int, unsigned long long> (unsigned int* labelBuffer, unsigned int desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<long, unsigned long long> (long* labelBuffer, long desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned long, unsigned long long> (unsigned long* labelBuffer, unsigned long desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<float, unsigned long long> (float* labelBuffer, float desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<double, unsigned long long> (double* labelBuffer, double desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<long long, unsigned long long> (long long* labelBuffer, long long desiredLabel, unsigned long long* agreement, int size);
template __global__ void IncrementBuffer<unsigned long long, unsigned long long> (unsigned long long* labelBuffer, unsigned long long desiredLabel, unsigned long long* agreement, int size);

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

#define Logariture(value1, value2)	0.5f * ((isfinite(value1 + log(1.0f + exp(value2-value1)))?value1 + log(1.0f + exp(value2-value1)):value2 + log(1.0f + exp(value1-value2))) + \
											(isfinite(value2 + log(1.0f + exp(value1-value2)))?value2 + log(1.0f + exp(value1-value2)):value1 + log(1.0f + exp(value2-value1))) )

void LogaritureData(int size, int threads, int blocks, float* dataBuffer, cudaStream_t* stream ){

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * (sizeof(float)+sizeof(short2)) : threads * (sizeof(float)+sizeof(short2));
	switch (threads)
	{
	case 512:
		LogaritureOverSmallBuffer<512><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 256:
		LogaritureOverSmallBuffer<256><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 128:
		LogaritureOverSmallBuffer<128><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 64:
		LogaritureOverSmallBuffer< 64><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 32:
		LogaritureOverSmallBuffer< 32><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 16:
		LogaritureOverSmallBuffer< 16><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 8:
		LogaritureOverSmallBuffer< 8><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 4:
		LogaritureOverSmallBuffer< 4><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 2:
		LogaritureOverSmallBuffer< 2><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 1:
		LogaritureOverSmallBuffer< 1><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	}

}

template <unsigned int blockSize>
__global__ void LogaritureOverSmallBuffer(float *buffer, unsigned int n)
{
	__shared__ float sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = -2.0f * FLT_MAX;
	
	while (i < n) {
		sdata[tid] = Logariture(sdata[tid], buffer[i]);
		sdata[tid] = Logariture(sdata[tid], buffer[i+blockSize]);
		i += gridSize;
		__syncthreads();
	}
	
	if (blockSize >= 512) { if (tid < 256) {
			sdata[tid] = Logariture(sdata[tid], sdata[tid + 256]);
	} __syncthreads(); }

	if (blockSize >= 256) { if (tid < 128) {
			sdata[tid] = Logariture(sdata[tid], sdata[tid + 128]);
	} __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) {
			sdata[tid] = Logariture(sdata[tid], sdata[tid + 64]);
	} __syncthreads(); }
	
	if (tid < 32) {
		if (blockSize >= 64){
			sdata[tid] = Logariture(sdata[tid], sdata[tid + 32]);
			__syncthreads();
		}
		if (blockSize >= 32){
			sdata[tid] = Logariture(sdata[tid], sdata[tid + 16]);
			__syncthreads();
		}
		if (blockSize >= 16){
			sdata[tid] = Logariture(sdata[tid], sdata[tid + 8]);
			__syncthreads();
		}
		if (blockSize >=  8){
			sdata[tid] = Logariture(sdata[tid], sdata[tid + 4]);
			__syncthreads();
		}
		if (blockSize >=  4){
			sdata[tid] = Logariture(sdata[tid], sdata[tid + 2]);
			__syncthreads();
		}
		if (blockSize >=  2){
			sdata[tid] = Logariture(sdata[tid], sdata[tid + 1]);
			__syncthreads();
		}
	}
	if (tid == 0){
		buffer[0] = sdata[0];
	}
}

__global__ void LogaritureOverLargeBuffer( float* buffer, int spread, int size ){
	
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	float value1 = buffer[kOffset];
	float value2 = buffer[kOffset+spread];
	
	float result = Logariture(value1, value2);

	if( kOffset+spread < size )
		buffer[kOffset] = result;

}