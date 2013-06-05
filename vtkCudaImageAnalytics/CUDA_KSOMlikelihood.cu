#include "CUDA_KSOMlikelihood.h"
#include "CUDA_commonKernels.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

__constant__ KSOMLL_Information info;

#define NUMTHREADS 512

__global__ void KSOMLL_ProcessSample(int samplePointLoc, float* InputData, float* Map, float* OutputBuffer, float scale){

	__shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

	//get sample co-ordinates in buffer
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadIdx.x < MAX_DIMENSIONALITY)
		SamplePointLocal[threadIdx.x] = InputData[info.NumberOfDimensions*samplePointLoc+threadIdx.x];
	__syncthreads();
	
	//calculate the distance
	float distance = 0.0f;
	float penalty = 1.0f;
	int bufferSize = info.GMMSize[0]*info.GMMSize[1];
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float weight = Map[(2*i+1)*bufferSize+kOffset];
		float value = (Map[(2*i)*bufferSize+kOffset] - SamplePointLocal[i]);
		distance += value * value / weight ;
		penalty *= weight;
	}
	distance += 0.5f * log( penalty );

	//output results
	if(kOffset < bufferSize) OutputBuffer[kOffset] =  - distance * scale;

}

void CUDAalgo_applyKSOMLLModel( float* inputData, float* inputGMM, float* outputGMM,
								char* seededImage, KSOMLL_Information& information, float scale,
								cudaStream_t* stream ){

	//useful constants for sizing stuff
	int N = information.GMMSize[0]*information.GMMSize[1];
	int L = information.NumberOfLabels;
	int V = information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2];

	//scale is only used in Gaussians, so, pre-square it for efficiency
	scale = 1.0f / (scale*scale);

	//copy problem information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(KSOMLL_Information) );
	
	//copy input GMM transposed definition to GPU
	float* tempGMM = new float[2*N*information.NumberOfDimensions];
	for(int i = 0; i < N; i++)
		for( int j = 0; j < 2*information.NumberOfDimensions; j++ )
			tempGMM[j*N+i] = inputGMM[i*2*information.NumberOfDimensions+j];
	float* dev_GMMOrig = 0;
	cudaMalloc( (void**) &dev_GMMOrig, sizeof(float)*2*N*information.NumberOfDimensions );
	cudaMemcpyAsync( dev_GMMOrig, tempGMM, sizeof(float)*2*N*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );
	delete[] tempGMM;

	//copy input image into GPU
	float* dev_inputImage = 0;
	cudaMalloc( (void**) &dev_inputImage, sizeof(float)*V*information.NumberOfDimensions );
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

	//allocate space for two working buffers (size N)
	float* dev_workingBuffer = 0;
	cudaMalloc((void**)&dev_workingBuffer, sizeof(float)*N);
	
	//zero out the estimate coefficient buffer
	ZeroOutBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, N*L);

	//estimate coefficients
	int* NumberOfSeeds = new int[L];
	int TotalNumberOfSeeds = 0;
	for( int i = 0; i < L; i++ ) NumberOfSeeds[i] = 0;
	for( int x = 0; x < V; x++){

		//find seed number
		char seedNumber = seededImage[x] - 1;
		if( seedNumber < 0 ) continue;
		NumberOfSeeds[seedNumber]++;
		TotalNumberOfSeeds++;

		//find GMM activation and place in working buffer
		KSOMLL_ProcessSample<<<gridN, threadsFull, 0, *stream>>>
			(x, dev_inputImage, dev_GMMOrig, dev_workingBuffer, scale);
		SumBuffers<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM+seedNumber*N, dev_workingBuffer, N);

	}

	//adjust coefficients by region size
	for( int i = 0; i < L; i++ )
		if( NumberOfSeeds[i] )
			TranslateBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM+i*N, 1.0f / (float)NumberOfSeeds[i], log((float)TotalNumberOfSeeds) - log((float)NumberOfSeeds[i]), N);
		
	
	//copy estimate coefficients to host
	float* tempPAGMM = new float[N*L];
	cudaMemcpyAsync( tempPAGMM, dev_outputGMM, sizeof(float)*L*N, cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);

	//transpose output PAGMM to right buffer orientation
	for(int i = 0; i < N; i++)
		for( int j = 0; j < L; j++ )
			outputGMM[i*L+j] = tempPAGMM[j*N+i];
	delete[] tempPAGMM;
	delete[] NumberOfSeeds;

	//deallocate working buffer (size N*M)
	cudaFree(dev_workingBuffer);

	//deallocate estimate coefficient buffer (size N*M*L)
	cudaFree(dev_outputGMM);
	
	//deallocate Gaussian mixture (size N*D)
	cudaFree(dev_GMMOrig);

	//deallocate buffer of input data (size V*D)
	cudaFree(dev_inputImage);

	//make sure all memcpy's are done before leaving
	cudaStreamSynchronize(*stream);

}