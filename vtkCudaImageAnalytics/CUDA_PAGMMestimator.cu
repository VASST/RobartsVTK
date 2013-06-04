#include "CUDA_PAGMMestimator.h"
#include "CUDA_commonKernels.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

__constant__ PAGMM_Information info;

#define NUMTHREADS 512

__global__ void ProcessSample(int samplePointLoc, float* InputData, float* Map, float* OutputBuffer, float scale){

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
	if(kOffset < bufferSize) OutputBuffer[kOffset] = exp( -distance * scale );

}

void CUDAalgo_applyPAGMMModel( float* inputData, float* inputGMM, float* outputGMM,
								char* seededImage, PAGMM_Information& information, float p, float q, float scale,
									cudaStream_t* stream ){

	//useful constants for sizing stuff
	int N = information.GMMSize[0]*information.GMMSize[1];
	int L = information.NumberOfLabels;
	int V = information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2];

	//scale is only used in Gaussians, so, pre-square it for efficiency
	scale = 1.0f / (scale*scale);

	//copy problem information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(PAGMM_Information) );
	
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
	float* dev_workingBuffer2 = 0;
	cudaMalloc((void**)&dev_workingBuffer, sizeof(float)*N);
	cudaMalloc((void**)&dev_workingBuffer2, sizeof(float)*N);
	

	//allocate host space for summation buffer (size V)
	float* hostSummationBuffer = (float*) malloc(V*sizeof(float));

	//for each seed sample
	for( int x = 0; x < V; x++){

		//figure out the seed number
		char seedNumber = seededImage[x]-1;
		if( seedNumber < 0 ) continue;
		
		//find GMM activation and place in working buffer
		ProcessSample<<<gridN, threadsFull, 0, *stream>>>
			(x, dev_inputImage, dev_GMMOrig, dev_workingBuffer, scale);

		//reduce working buffer by summation
		for(int j = N / 2; j > NUMTHREADS; j = j/2){
			dim3 tempGrid( j>NUMTHREADS ? j/NUMTHREADS : 1, 1, 1);
			SumOverLargeBuffer<<<tempGrid, threadsFull, 0, *stream>>>(dev_workingBuffer,j,N);
		}
		SumData( min(NUMTHREADS,N), min(NUMTHREADS,N), 1, dev_workingBuffer, stream );

		//place summation on host summation buffer
		hostSummationBuffer[x] = 0.0f;
		cudaMemcpyAsync( &(hostSummationBuffer[x]), dev_workingBuffer, sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaStreamSynchronize(*stream);

	}

	//allocate product buffer (size N*L)
	float* dev_productBuffer = 0;
	cudaMalloc((void**)&dev_productBuffer, sizeof(float)*N*L);
	
	//zero out the estimate coefficient buffer
	ZeroOutBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, N*L);

	//generate probability values
	long double* Prob = (long double*) malloc( sizeof(long double) * N );
	Prob[0] = (long double) N * (long double) p * (long double) N * pow(1.0 - (long double) q,N-1);
	Prob[N-1] = (long double) N * (long double) p *  (long double) N * pow((long double) q,N-1);
	for( int b = 1; b < N-1; b++ ){
		int k = b+1;
		long double upVal = Prob[k-2] * ((long double)(q)/(long double)(1.0-q)) * (long double)(N-k+1) / (long double)(k);
		long double downVal = Prob[k] * ((long double)(1.0-q)/(long double)q) * (long double)(k+1) / (long double)(N-k);
		Prob[k-1] = (upVal > downVal) ? upVal : downVal;
		
		k = N-b;
		upVal = Prob[k-2] * ((long double)(q)/(long double)(1.0-q)) * (long double)(N-k+1) / (long double)(k);
		downVal = Prob[k] * ((long double)(1.0-q)/(long double)q) * (long double)(k+1) / (long double)(N-k);
		Prob[k-1] = (upVal > downVal) ? upVal : downVal;
	}

	//refine the values
	for( int reps = 0 ; reps < (int)(sqrt((double)N)+0.5); reps++ ){
		Prob[0] = (long double) N * (long double) p * (long double) N * pow(1.0 - (long double) q,N-1);
		Prob[N-1] = (long double) N * (long double) p *  (long double) N * pow((long double) q,N-1);
		for( int b = 1; b < N-1; b++ ){
			int k = b+1;
			long double upVal = Prob[k-2] * ((long double)(q)/(long double)(1.0-q)) * (long double)(N-k+1) / (long double)(k);
			long double downVal = Prob[k] * ((long double)(1.0-q)/(long double)q) * (long double)(k+1) / (long double)(N-k);
			Prob[k-1] = (upVal + downVal) / (long double) 2.0;
		
			k = N-b;
			upVal = Prob[k-2] * ((long double)(q)/(long double)(1.0-q)) * (long double)(N-k+1) / (long double)(k);
			downVal = Prob[k] * ((long double)(1.0-q)/(long double)q) * (long double)(k+1) / (long double)(N-k);
			Prob[k-1] = (upVal + downVal) / (long double) 2.0;
		}
		long double probSum = 0.0;
		for( int k = 0; k < N; k++ )
			probSum += Prob[k];
		probSum = probSum;
		for( int b = 0; b < N; b++ )
			Prob[b] *= ((long double)N / probSum);
		probSum = 0.0;
		for( int k = 0; k < N; k++ )
			probSum += Prob[k];
		probSum = probSum;
	}

	//estimate coefficients
	for( int k = 1; k <= N; k++ ){
		
		bool applicable = ((float) Prob[k-1] > 0.0f);
		if( !applicable ) continue;

		//initialize the product buffer to all 1's
		OneOutBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_productBuffer, N*L);

		for( int x = 0; x < V; x++){

			//find seed number
			char seedNumber = seededImage[x] - 1;
			if( seedNumber < 0 ) continue;

			//find GMM activation and place in working buffer
			ProcessSample<<<gridN, threadsFull, 0, *stream>>>
				(x, dev_inputImage, dev_GMMOrig, dev_workingBuffer, scale);

			//combine working buffer with summation buffer and multiply into product buffer
			MultiplyBuffers<<<gridN, threadsFull, 0, *stream>>>
				(dev_productBuffer+seedNumber*N, dev_workingBuffer,
				 (float)(N-k+1) / (float)(N*k), hostSummationBuffer[x] * (float)(k-1) / (float)(N*k), N);

		}

		//multiply product buffer by probability value
		TranslateBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_productBuffer, (float) Prob[k], 0.0f, N*L);

		//sum product buffer into estimate coefficients buffer
		SumBuffers<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, dev_productBuffer, N*L);

	}

	//replace all NaN values with zeros
	ReplaceNANs<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, 1.0f/(float)N, N*L);

	//normalize estimate coefficients
	for( int curl = 0; curl < L; curl++ ){
		CopyBuffers<<<gridN, threadsFull, 0, *stream>>>(dev_workingBuffer, dev_outputGMM+curl*N, N);
		for(int j = N / 2; j > NUMTHREADS; j = j/2){
			dim3 tempGrid( j>NUMTHREADS ? j/NUMTHREADS : 1, 1, 1);
			SumOverLargeBuffer<<<tempGrid, threadsFull, 0, *stream>>>(dev_workingBuffer,j,N);
		}
		SumData( min(NUMTHREADS,N), min(NUMTHREADS,N), 1, dev_workingBuffer, stream );
		float sum = 0.0f;
		cudaMemcpyAsync( &sum, dev_workingBuffer, sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaStreamSynchronize(*stream);
		//TranslateBuffer<<<gridN, threadsFull, 0, *stream>>>(dev_outputGMM+curl*N, 1.0f/sum, 0.0f, N);
	}

	//replace all NaN values with zeros
	ReplaceNANs<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, 1.0f/(float)N, N*L);

	//deallocate product buffer (size N*L) and probabilities
	cudaFree( dev_productBuffer );
	free( Prob );

	//deallocate host summation buffer (size V)
	free(hostSummationBuffer);
	
	//copy estimate coefficients to host
	float* tempPAGMM = new float[N*L];
	cudaMemcpyAsync( tempPAGMM, dev_outputGMM, sizeof(float)*L*N, cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);

	//transpose output PAGMM to right buffer orientation
	for(int i = 0; i < N; i++)
		for( int j = 0; j < L; j++ )
			outputGMM[i*L+j] = tempPAGMM[j*N+i];
	delete[] tempPAGMM;

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