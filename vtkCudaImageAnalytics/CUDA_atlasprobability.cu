#include "CUDA_atlasprobability.h"
#include "stdio.h"
#include "cuda.h"

#define NUMTHREADS 512

template<class T>
__global__ void kern_IncrementBuffer(T* labelBuffer, T desiredLabel, short* agreement, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	short newAgreement = agreement[idx];
	T labelValue = labelBuffer[idx];
	newAgreement += (labelValue == desiredLabel) ? 1 : 0;
	if( idx < size ) agreement[idx] = newAgreement;
}

template void CUDA_IncrementInformation<float>(float* labelData, float desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<double>(double* labelData, double desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<long>(long* labelData, long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned long>(unsigned long* labelData, unsigned long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<long long>(long long* labelData, long long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned long long>(unsigned long long* labelData, unsigned long long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<int>(int* labelData, int desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned int>(unsigned int* labelData, unsigned int desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<short>(short* labelData, short desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned short>(unsigned short* labelData, unsigned short desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<char>(char* labelData, char desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<signed char>(signed char* labelData, signed char desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned char>(unsigned char* labelData, unsigned char desiredValue, short* agreement, int size, cudaStream_t* stream);



template< class T >
void CUDA_IncrementInformation(T* labelData, T desiredValue, short* agreement, int size, cudaStream_t* stream){
    float* GPUBuffer = 0;
	cudaMalloc((void**) &GPUBuffer, sizeof(T)*size);
	cudaMemcpyAsync( labelData, GPUBuffer, sizeof(T)*size, cudaMemcpyDeviceToHost, *stream );
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_IncrementBuffer<T><<<grid,threads,0,*stream>>>(labelData, desiredValue, agreement, size);
	cudaFree(GPUBuffer);
}

__global__ void kern_ZeroOutBuffer(short* buffer, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx < size ) buffer[idx] = 0;
}

void CUDA_GetRelevantBuffers(short** agreement, float** output, int size, cudaStream_t* stream){
	cudaMalloc((void**) agreement, sizeof(short)*size);
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_ZeroOutBuffer<<<grid,threads,0,*stream>>>(*agreement,size);
	cudaMalloc((void**) output, sizeof(float)*size);
}

void CUDA_CopyBackResult(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream){
	cudaMemcpyAsync( CPUBuffer, GPUBuffer, sizeof(float)*size, cudaMemcpyDeviceToHost, *stream );
	cudaThreadSynchronize();
	cudaFree(GPUBuffer);
}

__global__ void kern_LogBuffer(short* agreement, float* output, float maxOut, int size, short max){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float locAgreement = (float) agreement[idx];
	float logValue = (locAgreement > 0) ? log((float)max)-log(locAgreement): maxOut;
	logValue = (logValue < maxOut) ? logValue: maxOut;
	if( idx < size ) output[idx] = logValue;
}

__global__ void kern_ProbBuffer(short* agreement, float* output, int size, short max){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	short locAgreement = agreement[idx];
	float probValue = (float) locAgreement / (float) max;
	probValue = (probValue < 1.0) ? probValue: 1.0;
	if( idx < size ) output[idx] = probValue;
}

void CUDA_ConvertInformation(short* agreement, float* output, float maxOut, int size, short max, short flags, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	if( flags & 1 )
		kern_LogBuffer<<<grid,threads,0,*stream>>>(agreement, output, maxOut, size, max);
	else
		kern_ProbBuffer<<<grid,threads,0,*stream>>>(agreement, output, size, max);
}
