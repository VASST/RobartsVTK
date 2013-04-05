#include "CUDA_hierarchicalmaxflow.h"
#include "stdio.h"
#include "cuda.h"

#define NUMTHREADS 512

int CUDA_GetGPUBuffers( int maxNumber, float** buffer, int volSize ){

	size_t freeMemory, totalMemory;
	cudaError_t nErr = cudaSuccess;
	cudaMemGetInfo(&freeMemory, &totalMemory);

    printf("===========================================================\n");
    printf("Free/Total(kB): %f/%f\n", (float)freeMemory/1024.0f, (float)totalMemory/1024.0f);

	while( maxNumber > 0 ){
		nErr = cudaMalloc((void**) buffer, sizeof(float)*maxNumber*volSize);
		if( nErr == cudaSuccess ) break;
		maxNumber--; 
	}
	
	cudaMemGetInfo(&freeMemory, &totalMemory);
    printf("===========================================================\n");
    printf("Free/Total(kB): %f/%f\n", (float)freeMemory/1024.0f, (float)totalMemory/1024.0f);

	return maxNumber;

}

void CUDA_ReturnGPUBuffers(float* buffer){
	cudaFree(buffer);
}


void CUDA_CopyBufferToCPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream){
	cudaMemcpyAsync( CPUBuffer, GPUBuffer, sizeof(float)*size, cudaMemcpyDeviceToHost, *stream );
}

void CUDA_CopyBufferToGPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream){
	cudaMemcpyAsync( GPUBuffer, CPUBuffer, sizeof(float)*size, cudaMemcpyHostToDevice, *stream );
}

__global__ void kern_ZeroOutBuffer(float* buffer, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx < size ) buffer[idx] = 0.0f;
}

void CUDA_zeroOutBuffer(float* GPUBuffer, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_ZeroOutBuffer<<<grid,threads,0,*stream>>>(GPUBuffer,size);
}

__global__ void kern_DivideAndStoreBuffer(float* inBuffer, float* outBuffer, float number, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value = inBuffer[idx] * number;
	if( idx < size ) outBuffer[idx] = value;
}

void CUDA_divideAndStoreBuffer(float* inBuffer, float* outBuffer, float number, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_DivideAndStoreBuffer<<<grid,threads,0,*stream>>>(inBuffer,outBuffer,1.0f/number,size);
}