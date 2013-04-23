#include "CUDA_atlasprobability.h"
#include "stdio.h"
#include "cuda.h"

#define NUMTHREADS 512

//#define DEBUG_VTKCUDA_IALP

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
    T* GPUBuffer = 0;

	cudaMalloc((void**) &GPUBuffer, sizeof(T)*size);
	cudaMemcpyAsync( GPUBuffer, labelData, sizeof(T)*size, cudaMemcpyHostToDevice, *stream );

	#ifdef DEBUG_VTKCUDA_IALP
		cudaThreadSynchronize();
		printf( "CUDA_IncrementInformation: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_IncrementBuffer<T><<<grid,threads,0,*stream>>>(GPUBuffer, desiredValue, agreement, size);
	cudaFree(GPUBuffer);

	#ifdef DEBUG_VTKCUDA_IALP
		cudaThreadSynchronize();
		printf( "CUDA_IncrementInformation: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
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

	#ifdef DEBUG_VTKCUDA_IALP
		cudaThreadSynchronize();
		printf( "CUDA_GetRelevantBuffers: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

void CUDA_CopyBackResult(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream){
	cudaThreadSynchronize();
	cudaMemcpy( CPUBuffer, GPUBuffer, sizeof(float)*size, cudaMemcpyDeviceToHost );
	cudaFree(GPUBuffer);

	#ifdef DEBUG_VTKCUDA_IALP
		cudaThreadSynchronize();
		printf( "CUDA_CopyBackResult: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_LogBuffer(short* agreement, float* output, float maxOut, int size, short max){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float locAgreement = (float) agreement[idx];
	float logValue = (locAgreement > 0.0f) ? log((float)max)-log(locAgreement): maxOut;
	logValue = (logValue > 0.0f) ? logValue : 0.0f;
	logValue = (logValue < maxOut) ? logValue: maxOut;
	if( idx < size ) output[idx] = logValue;
}

__global__ void kern_NormLogBuffer(short* agreement, float* output, float maxOut, int size, short max){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float locAgreement = (float) agreement[idx];
	float logValue = (locAgreement > 0.0f) ? log((float)max)-log(locAgreement): maxOut;
	logValue = (logValue > 0.0f) ? logValue : 0.0f;
	logValue = (logValue < maxOut) ? logValue / maxOut: 1.0f;
	if( idx < size ) output[idx] = logValue;
}

__global__ void kern_ProbBuffer(short* agreement, float* output, int size, short max){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	short locAgreement = agreement[idx];
	float probValue = (float) locAgreement / (float) max;
	probValue = (probValue < 1.0f) ? probValue: 1.0f;
	if( idx < size ) output[idx] = probValue;
}

void CUDA_ConvertInformation(short* agreement, float* output, float maxOut, int size, short max, short flags, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	if( flags & 1 )
		if( flags & 2)
			kern_NormLogBuffer<<<grid,threads,0,*stream>>>(agreement, output, maxOut, size, max);
		else
			kern_LogBuffer<<<grid,threads,0,*stream>>>(agreement, output, maxOut, size, max);
	else
		kern_ProbBuffer<<<grid,threads,0,*stream>>>(agreement, output, size, max);

	#ifdef DEBUG_VTKCUDA_IALP
		cudaThreadSynchronize();
		printf( "CUDA_ConvertInformation: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}
