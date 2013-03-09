#include "CUDA_PAGMMestimator.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

__constant__ PAGMM_Information info;
texture<float, 3, cudaReadModeElementType> Mixture_Map;

#define NUMTHREADS 512

__global__ void ZeroOutBuffer(float* buffer, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	if(offset < size ) buffer[offset] = 0.0f;
}

__global__ void OneOutBuffer(float* buffer, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	if(offset < size ) buffer[offset] = 1.0f;
}

__global__ void MultiplyBuffers(float* outBuffer, float* multBuffer, float scale, float shift, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	float value = (scale * outBuffer[offset] + shift) * multBuffer[offset];
	if(offset < size ) outBuffer[offset] = value;
}

__global__ void SumBuffers(float* outBuffer, float* sumBuffer, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	float value = outBuffer[offset] + sumBuffer[offset];
	if(offset < size ) outBuffer[offset] = value;
}

__global__ void NormalizeBuffers(float* bigBuffer, int repeats, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	float denom = 0.0f;
	for( int i = 0; i < repeats; i++ )
		denom = denom + bigBuffer[i*size+offset];
		
	for( int i = 0; i < repeats; i++ ){
		float value = bigBuffer[i*size+offset];
		if(offset < size) bigBuffer[i*size+offset] = value / denom;
	}
}

__global__ void TranslateBuffer(float* buffer, float scale, float shift, int size){
	int offset = blockDim.x * blockIdx.x + threadIdx.x;
	float value = scale * buffer[offset] + shift;
	if(offset < size ) buffer[offset] = value;
}

__global__ void ProcessSample(int samplePointLoc, float* InputData, float* Map, float* OutputBuffer, float scale){

	__shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

	//get sample co-ordinates in buffer
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadIdx.x < MAX_DIMENSIONALITY){
		SamplePointLocal[threadIdx.x] = InputData[info.NumberOfDimensions*samplePointLoc+threadIdx.x];
	}
	__syncthreads();
	
	//calculate the distance
	float distance = 0.0f;
	int bufferSize = info.GMMSize[0]*info.GMMSize[1];
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = info.Weights[i]*(Map[i*bufferSize+kOffset] - SamplePointLocal[i]);
		distance += value*value;
	}

	//output results
	if(kOffset < bufferSize) OutputBuffer[kOffset] = exp( -distance * scale );

}

template <unsigned int blockSize>
__global__ void sumOverSmallBuffer(float *buffer, unsigned int n)
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

__global__ void sumOverLargeBuffer( float* buffer, int spread, int size ){
	
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	float value1 = buffer[kOffset];
	float value2 = buffer[kOffset+spread];

	if( kOffset+spread < size )
		buffer[kOffset] = value1+value2;

}

void sumData(int size, int threads, int blocks, float* dataBuffer, cudaStream_t* stream ){

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * (sizeof(float)+sizeof(short2)) : threads * (sizeof(float)+sizeof(short2));
	switch (threads)
	{
	case 512:
		sumOverSmallBuffer<512><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 256:
		sumOverSmallBuffer<256><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 128:
		sumOverSmallBuffer<128><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 64:
		sumOverSmallBuffer< 64><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 32:
		sumOverSmallBuffer< 32><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 16:
		sumOverSmallBuffer< 16><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 8:
		sumOverSmallBuffer< 8><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 4:
		sumOverSmallBuffer< 4><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 2:
		sumOverSmallBuffer< 2><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	case 1:
		sumOverSmallBuffer< 1><<< dimGrid, dimBlock, smemSize, *stream >>>(dataBuffer, size); break;
	}

}

void CUDAalgo_applyPAGMMModel( float* inputData, float* inputGMM, float* outputData, float* outputGMM,
								char* seededImage, PAGMM_Information& information, float p, float q, float scale,
									cudaStream_t* stream ){

	//useful constants for sizing stuff
	int N = info.GMMSize[0]*info.GMMSize[1];
	int L = info.NumberOfLabels;
	int V = info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2];

	//scale is only used in Gaussians, so, pre-square it for efficiency
	scale = 1.0f / (scale*scale);

	//copy problem information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(PAGMM_Information) );

	//copy input GMM transposed definition to GPU
	float* tempGMM = new float[N*information.NumberOfDimensions];
	for(int i = 0; i < N; i++)
		for( int j = 0; j < information.NumberOfDimensions; j++ )
			tempGMM[j*N+i] = inputGMM[i*information.NumberOfDimensions+j];
	float* dev_GMMMeans = 0;
	cudaMalloc( (void**) &dev_GMMMeans, sizeof(float)*N*information.NumberOfDimensions );
	cudaMemcpyAsync( dev_GMMMeans, tempGMM, sizeof(float)*N*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );
	delete[] tempGMM;

	//copy input image into GPU
	float* dev_inputImage = 0;
	cudaMalloc( (void**) &dev_inputImage, sizeof(float)*N*information.NumberOfDimensions );
	cudaMemcpyAsync( dev_inputImage, inputData, sizeof(float)*V*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );

	//allocate space for estimate co-ordinates
	float* dev_outputGMM = 0;
	cudaMalloc((void**)&dev_outputGMM, sizeof(float)*N*L);

	//preallocate grid and thread sizes for N, and N*L
	dim3 gridN((N-1)/NUMTHREADS+1, 1, 1);
	dim3 gridNL((N*L-1)/NUMTHREADS+1, 1, 1);
	dim3 threadsFull(NUMTHREADS, 1, 1);
	
	//-----------------------------------------------------------------------------------//
	//      Start PAGMM model from the seed points                                       //
	//-----------------------------------------------------------------------------------//

	//allocate space for a working buffer (size N)
	float* dev_workingBuffer = 0;
	cudaMalloc((void**)&dev_workingBuffer, sizeof(float)*N);

	//allocate host space for summation buffer (size V)
	float* hostSummationBuffer = (float*) malloc(V*sizeof(float));

	//keep track of the number of seeds in each label
	int* numSeeds = (int*) malloc(L*sizeof(int));
	for(int i = 0; i < L; i++)
		numSeeds = 0;

	//for each seed sample
	for( int x = 0; x < V; x++){

		//increment number of seeds
		char seedNumber = seededImage[x];
		if( seedNumber < 1 ) continue;
		numSeeds[seedNumber]++;

		//find GMM activation and place in working buffer
		ProcessSample<<<gridN, threadsFull, 0, *stream>>>
			(x, dev_inputImage, dev_GMMMeans, dev_workingBuffer, scale);

		//reduce working buffer by summation
		for(int j = N / 2; j > NUMTHREADS; j = j/2){
			dim3 tempGrid( j>NUMTHREADS ? j/NUMTHREADS : 1, 1, 1);
			sumOverLargeBuffer<<<tempGrid, threadsFull, 0, *stream>>>(dev_workingBuffer,j,N);
		}
		sumData( min(NUMTHREADS,N), min(NUMTHREADS,N), 1, dev_workingBuffer, stream );

		//place summation on host summation buffer
		cudaMemcpyAsync( &(hostSummationBuffer[x]), dev_workingBuffer, sizeof(float), cudaMemcpyHostToDevice, *stream );
		cudaStreamSynchronize(*stream);

	}

	//allocate product buffer (size N*L)
	float* dev_productBuffer = 0;
	cudaMalloc((void**)&dev_productBuffer, sizeof(float)*N*L);

	//zero out the estimate coefficient buffer
	ZeroOutBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, N*L);

	//generate probability values
	float* Prob = (float*) malloc( sizeof(float) * N );
	Prob[0] = p*pow(1.0f-q,N-1);
	for( int k = 2; k <= N; k++ ){
		Prob[k-1] = Prob[k-2] * (q/(1.0f-q)) * ((float)(k-1) / (float)k) * ((float)(N-k+1) / (float)(k-1));
	}

	//estimate coefficients
	for( int k = 1; k <= N; k++ ){
		
		//initialize the product buffer to all 1's
		OneOutBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_productBuffer, N*L);

		for( int x = 0; x < V; x++){

			//find seed number
			char seedNumber = seededImage[x];
			if( seedNumber < 1 ) continue;

			//find GMM activation and place in working buffer
			ProcessSample<<<gridN, threadsFull, 0, *stream>>>
				(x, dev_inputImage, dev_GMMMeans, dev_workingBuffer, scale);

			//combine working buffer with summation buffer and multiply into product buffer
			MultiplyBuffers<<<gridN, threadsFull, 0, *stream>>>
				(dev_productBuffer+seedNumber*N, dev_workingBuffer,
				 (float)(N-k+1) / (float)(N*k), hostSummationBuffer[x] * (float)(k-1) / (float)(N*k), N);

		}

		//multiply product buffer by probability value
		TranslateBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_productBuffer, Prob[k], 0.0f, N*L);

		//sum product buffer into estimate coefficients buffer
		SumBuffers<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, dev_productBuffer, N*L);

	}

	//normalize estimate coefficients
	NormalizeBuffers<<<gridN, threadsFull, 0, *stream>>> (dev_outputGMM, L, N);

	//deallocate product buffer (size N*L) and probabilities
	cudaFree( dev_productBuffer );
	free( Prob );

	//deallocate host summation buffer (size V)
	free(hostSummationBuffer);
	free(numSeeds);

	//copy estimate coefficients to host
	cudaMemcpyAsync( outputGMM, dev_outputGMM, sizeof(float)*L*N, cudaMemcpyDeviceToHost, *stream );

	
	//-----------------------------------------------------------------------------------//
	//      Start creating the probabilistic result                                      //
	//-----------------------------------------------------------------------------------//

	//for each voxel in the image
	for( int x = 0; x < V; x++){

		//find GMM activation and place in working buffer
		ProcessSample<<<gridN, threadsFull, 0, *stream>>>
			(x, dev_inputImage, dev_GMMMeans, dev_workingBuffer, scale);

		//for each label
		for( int curl = 0; curl < L; curl++ ){
			//multiply working buffer with estimate coefficient buffer
			MultiplyBuffers<<<gridN, threadsFull, 0, *stream>>>
				(dev_workingBuffer, dev_outputGMM+curl*N, 1.0f, 0.0f, N);

			//reduce by summation
			for(int j = N / 2; j > NUMTHREADS; j = j/2){
				dim3 tempGrid( j>NUMTHREADS ? j/NUMTHREADS : 1, 1, 1);
				sumOverLargeBuffer<<<tempGrid, threadsFull, 0, *stream>>>(dev_workingBuffer,j,N);
			}
			sumData( min(NUMTHREADS,N), min(NUMTHREADS,N), 1, dev_workingBuffer, stream );

			//copy into host probability buffer
			cudaMemcpyAsync( &(outputData[x]), dev_workingBuffer, sizeof(float), cudaMemcpyHostToDevice, *stream );
			cudaStreamSynchronize(*stream);

		}

	}

	//deallocate working buffer (size N*M)
	cudaFree(dev_workingBuffer);

	//deallocate estimate coefficient buffer (size N*M*L)
	cudaFree(dev_outputGMM);
	
	//deallocate Gaussian mixture (size N*D)
	cudaFree(dev_GMMMeans);

	//deallocate buffer of input data (size V*D)
	cudaFree(dev_inputImage);

	//make sure all memcpy's are done before leaving
	cudaStreamSynchronize(*stream);

}