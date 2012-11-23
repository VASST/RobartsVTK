#include "CUDA_kohonenapplication.h"
#include <float.h>
#include <stdio.h>

__constant__ Kohonen_Application_Information info;
texture<float, 3, cudaReadModeElementType> Kohonen_Map;

#define NUM_THREADS 256


__global__ void ProcessImageToMapFirst(float* KohonenMap, float* InputData, float2* OutputData, float* OutputWeight, float2* OutputNearest, float* OutputDistance ){

	float currWeight = 0.0f;
	float2 weightedIndex = {0.0f, 0.0f};
	float2 minIndex = {0.0f, 0.0f};
	float minDistance = FLT_MAX;

	//save off the index of the closest map point
	int kOffset = threadIdx.x + blockDim.x * blockIdx.x;
	int bufferSize = info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2];
	if( kOffset < bufferSize ) OutputData[kOffset] = weightedIndex;
	if( kOffset < bufferSize ) OutputWeight[kOffset] = currWeight;
	if( kOffset < bufferSize ) OutputNearest[kOffset] = minIndex;
	if( kOffset < bufferSize ) OutputDistance[kOffset] = minDistance;

}

__global__ void ProcessImageToMapRepeat(float* KohonenMap, float* InputData, float2* OutputData, float* OutputWeight, float2* OutputNearest, float* OutputDistance, int KohonenRow){

	__shared__ float MapPoint[NUM_THREADS];

	int kOffset = threadIdx.x + blockDim.x * blockIdx.x;

	float currWeight = OutputWeight[kOffset];
	float2 weightedIndex = OutputData[kOffset];
	float2 minIndex = OutputNearest[kOffset];
	float minDistance = OutputDistance[kOffset];
	
	int bufferSize = info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2];
	int count = 0;
	for(int i = 0; i < info.KohonenMapSize[0]; i++){
		
		//load data into shared memory
		__syncthreads();
		if(count == 0){
			MapPoint[threadIdx.x] = KohonenMap[i*info.NumberOfDimensions+threadIdx.x];
		}
		__syncthreads();

		//find the distance to the map point
		float currDistance = 0.0f;
		for(int j = 0; j < info.NumberOfDimensions; j++){
			//float value = info.Weights[j]*(InputData[j*bufferSize+kOffset] - MapPoint[j]);
			float value = info.Weights[j]*(InputData[j*bufferSize+kOffset] - MapPoint[count*info.NumberOfDimensions+j]);
			//float value = info.Weights[j]*(InputData[j*bufferSize+kOffset] - KohonenMap[i*info.NumberOfDimensions+j]);
			currDistance += value*value;
		}

		//tell if we need to update the shared memory buffer
		count++;
		if( (count+1)*info.NumberOfDimensions > NUM_THREADS-1)
			count = 0;

		//if less than the minimum, save it
		weightedIndex.x += exp( -1.0f * currDistance ) * (float) i;
		weightedIndex.y += exp( -1.0f * currDistance ) * (float) KohonenRow;
		currWeight += exp( -1.0f * currDistance );
		minIndex.x = (minDistance <= currDistance) ? minIndex.x : (float) i;
		minIndex.y = (minDistance <= currDistance) ? minIndex.y : (float) KohonenRow;
		minDistance = (minDistance <= currDistance) ? minDistance : currDistance;

	}

	//save off the index of the closest map point
	if( kOffset < bufferSize ) OutputData[kOffset] = weightedIndex;
	if( kOffset < bufferSize ) OutputWeight[kOffset] = currWeight;
	if( kOffset < bufferSize ) OutputNearest[kOffset] = minIndex;
	if( kOffset < bufferSize ) OutputDistance[kOffset] = minDistance;

}

__global__ void NormalizeImage( float2* OutputData, float* OutputWeight, float2* OutputNearest, float* OutputDistance ){
	int kOffset = threadIdx.x + blockDim.x * blockIdx.x;
	float currWeight = OutputWeight[kOffset];
	float2 appIndex = OutputData[kOffset];
	float2 minIndex = OutputNearest[kOffset];
	appIndex.x = (currWeight > (appIndex.x / (float) info.KohonenMapSize[0]) ) ? appIndex.x / currWeight : minIndex.x;
	appIndex.y = (currWeight > (appIndex.y / (float) info.KohonenMapSize[1]) ) ? appIndex.y / currWeight : minIndex.y;
	int bufferSize = info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2];
	if( kOffset < bufferSize ) OutputData[kOffset] = appIndex;
}

void CUDAalgo_applyKohonenMap( float* inputData, float* inputKohonen, float* outputData,
								Kohonen_Application_Information& information,
								cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Application_Information) );

	//translate data onto device
	float* device_KohonenMap = 0;
	cudaMalloc( (void**) &device_KohonenMap, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions );
	cudaMemcpyAsync( device_KohonenMap, inputKohonen, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );

	//rearrange image data to be easier to work with (should parallelize)
	float* device_InputData = 0;
	cudaMalloc( (void**) &device_InputData, sizeof(float)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*information.NumberOfDimensions );
	float* inputTransposed = new float[information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*information.NumberOfDimensions];
	for( int i = 0; i < information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]; i++ ){
		for( int j = 0; j < information.NumberOfDimensions; j++ ){
			int inIndex = i * information.NumberOfDimensions + j;
			int outIndex = j * information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2] + i;
			inputTransposed[outIndex] = inputData[inIndex];
		}
	}
	cudaMemcpyAsync( device_InputData, inputTransposed, sizeof(float)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );
	delete[] inputTransposed;
	
	float2* device_OutputData = 0;
	cudaMalloc( (void**) &device_OutputData, sizeof(float2)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2] );
	float* device_OutputWeight = 0;
	cudaMalloc( (void**) &device_OutputWeight, sizeof(float)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2] );
	float2* device_OutputNearest = 0;
	cudaMalloc( (void**) &device_OutputNearest, sizeof(float2)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2] );
	float* device_OutputDistance = 0;
	cudaMalloc( (void**) &device_OutputDistance, sizeof(float)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2] );

	//apply the map
	dim3 grid((information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2] + NUM_THREADS - 1) / NUM_THREADS,1,1);
	dim3 threads(NUM_THREADS,1,1);
	ProcessImageToMapFirst<<<grid, threads, 0, *stream >>>(device_KohonenMap, device_InputData, device_OutputData, device_OutputWeight, device_OutputNearest, device_OutputDistance );
	cudaStreamSynchronize(*stream);
	for( int i = 0; i < information.KohonenMapSize[1]; i++ ){
		ProcessImageToMapRepeat<<<grid, threads, 0, *stream >>>(device_KohonenMap + i*information.KohonenMapSize[0]*information.NumberOfDimensions,
									device_InputData, device_OutputData, device_OutputWeight, device_OutputNearest, device_OutputDistance, i );
		cudaStreamSynchronize(*stream);
	}
	NormalizeImage<<<grid, threads, 0, *stream >>>(device_OutputData, device_OutputWeight, device_OutputNearest, device_OutputDistance );
		
	printf( "Apply Map: " );
	cudaStreamSynchronize(*stream);
	printf( cudaGetErrorString( cudaGetLastError() ) );
	printf( "\n" );

	//copy results back
	cudaMemcpyAsync( outputData, device_OutputData, sizeof(float2)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2], cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);

	//remove allocated memory
	cudaFree(device_KohonenMap);
	cudaFree(device_InputData);
	cudaFree(device_OutputData);
	cudaFree(device_OutputWeight);
	cudaFree(device_OutputNearest);
	cudaFree(device_OutputDistance);
}