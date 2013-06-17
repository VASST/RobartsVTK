#include "CUDA_KSOMlikelihood.h"
#include "CUDA_commonKernels.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

__constant__ KSOMLL_Information info;

__global__ void KSOMLL_ProcessSample(int samplePointLoc, float* InputData, float* Map, float* OutputBuffer, float scale){

	__shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

	//get sample co-ordinates in buffer
	int kOffset = CUDASTDOFFSET;
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
		penalty += log(weight);
	}
	distance += 0.5f * penalty;

	//output results
	if(kOffset < bufferSize) OutputBuffer[kOffset] =  exp( - distance * scale );

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
	dim3 gridN = GetGrid(N);
	dim3 gridNL = GetGrid(N*L);
	dim3 threadsFull(NUMTHREADS, 1, 1);

	
	//-----------------------------------------------------------------------------------//
	//      Grab denominator information                                                 //
	//-----------------------------------------------------------------------------------//

	//allocate host space for summation buffer (size V)
	float* hostSummationBuffer = (float*) malloc(V*sizeof(float));
	
	//allocate space for a working buffer (size N)
	float* dev_workingBuffer = 0;
	float* dev_workingBuffer2 = 0;
	cudaMalloc((void**)&dev_workingBuffer, sizeof(float)*N);
	cudaMalloc((void**)&dev_workingBuffer2, sizeof(float)*N);

	//create container for seed samples
	int* NumberOfSeeds = new int[L];
	for( int i = 0; i < L; i++ ) NumberOfSeeds[i] = 0;
	int TotalNumberOfSeeds = 0;
	double Denominator = 0.0;

	//for each seed sample
	for( int x = 0; x < V; x++){

		//figure out the seed number
		char seedNumber = seededImage[x]-1;
		if( seedNumber < 0 ) continue;
		NumberOfSeeds[seedNumber]++;
		TotalNumberOfSeeds++;
		
		//find GMM activation and place in working buffer
		KSOMLL_ProcessSample<<<gridN, threadsFull, 0, *stream>>>
			(x, dev_inputImage, dev_GMMOrig, dev_workingBuffer, scale);

		//reduce working buffer by summation
		hostSummationBuffer[x] = 0.0f;
		cudaMemcpyAsync( &(hostSummationBuffer[x]), dev_workingBuffer, sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaStreamSynchronize(*stream);
		for(int j = N / 2; j >= NUMTHREADS; j = j/2){
			dim3 tempGrid = GetGrid(j);
			SumOverLargeBuffer<<<tempGrid, threadsFull, 0, *stream>>>(dev_workingBuffer,j,N);
			cudaMemcpyAsync( &(hostSummationBuffer[x]), dev_workingBuffer, sizeof(float), cudaMemcpyDeviceToHost, *stream );
			cudaStreamSynchronize(*stream);
		}
		SumData( min(NUMTHREADS,N), min(NUMTHREADS,N), 1, dev_workingBuffer, stream );

		//place summation on host summation buffer
		cudaMemcpyAsync( &(hostSummationBuffer[x]), dev_workingBuffer, sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaStreamSynchronize(*stream);

		//update denominator information
		Denominator += log( hostSummationBuffer[x] );

	}



	//-----------------------------------------------------------------------------------//
	//      Start PAGMM model from the seed points                                       //
	//-----------------------------------------------------------------------------------//
		
	//zero out the estimate coefficient buffer
	SetBufferToConst<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, 0.1f, N*L);
	//ZeroOutBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, N*L);
	//TranslateBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, 0.0f, 1.0f, N*L);

	//estimate coefficients
	for( int x = 0; x < V; x++){

		//find seed number
		char seedNumber = seededImage[x] - 1;
		if( seedNumber < 0 ) continue;

		//find GMM activation and place in working buffers
		KSOMLL_ProcessSample<<<gridN, threadsFull, 0, *stream>>>(x, dev_inputImage, dev_GMMOrig, dev_workingBuffer, scale);
		//CopyBuffers<<<gridN, threadsFull, 0, *stream>>>(dev_workingBuffer2, dev_workingBuffer, N);

		//create the pro-part
		//TranslateBuffer<<<gridN, threadsFull, 0, *stream>>>(dev_workingBuffer, (float) (N-K) / (float) (K*(N-1)),
		//													(float)(K-1)*hostSummationBuffer[x] / (float) (K*(N-1)), N );
		//LogBuffer<<<gridN, threadsFull, 0, *stream>>>(dev_workingBuffer, N);
		SumBuffers<<<gridN, threadsFull, 0, *stream>>>(dev_outputGMM+seedNumber*N, dev_workingBuffer, N);

		//create the con-part
		//TranslateBuffer<<<gridN, threadsFull, 0, *stream>>>(dev_workingBuffer2, -1.0f / (float)(N-1), (float) hostSummationBuffer[x] / (float)(N-1), N );
		//LogBuffer<<<gridN, threadsFull, 0, *stream>>>(dev_workingBuffer2, N);
		//for(int curl = 0; curl < L; curl++){
		//	if( curl == seedNumber ) continue;
		//	SumBuffers<<<gridN, threadsFull, 0, *stream>>>(dev_outputGMM+curl*N, dev_workingBuffer2, N);
		//}

	}

	//adjust coefficients by region size and denominator
	//for(int curl = 0; curl < L; curl++)
	//	TranslateBuffer<<<gridN, threadsFull, 0, *stream>>>(dev_outputGMM+curl*N, -1.0f / (float) (NumberOfSeeds[curl] * TotalNumberOfSeeds), 0.0f, N);
	LogBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, N*L);
	TranslateBuffer<<<gridNL, threadsFull, 0, *stream>>>(dev_outputGMM, -1.0f, log((float)N), N*L);
	
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
	delete[] hostSummationBuffer;

	//deallocate working buffer (size N*M)
	cudaFree(dev_workingBuffer);
	cudaFree(dev_workingBuffer2);

	//deallocate estimate coefficient buffer (size N*M*L)
	cudaFree(dev_outputGMM);
	
	//deallocate Gaussian mixture (size N*D)
	cudaFree(dev_GMMOrig);

	//deallocate buffer of input data (size V*D)
	cudaFree(dev_inputImage);

	//make sure all memcpy's are done before leaving
	cudaStreamSynchronize(*stream);

}