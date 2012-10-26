#include "CUDA_kohonenapplication.h"
#include <float.h>

__constant__ Kohonen_Application_Information info;


__global__ void ProcessImageToMapFirst(float* KohonenMap, float* InputData, short2* OutputData, float* OutputDistance ){

	__shared__ float MapPoint[MAX_DIMENSIONALITY];
	
	float minDistance = 0.0f;
	short2 minIndex = {0, 0};

	int kOffset = threadIdx.x + blockDim.x * blockIdx.x;
	int bufferSize = info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2];
	for(int i = 0; i < 1; i++){

		//load in the map point
		if(threadIdx.x < MAX_DIMENSIONALITY){
			MapPoint[threadIdx.x] = KohonenMap[i*info.NumberOfDimensions+threadIdx.x];
		}
		
		//find the distance to the map point
		float currDistance = 0.0f;
		for(int j = 0; j < info.NumberOfDimensions; j++){
			float value = InputData[j*bufferSize+kOffset] - MapPoint[j];
			currDistance += info.Weights[j]*value*value;
		}

		//if less than the minimum, save it
		minDistance = currDistance;

	}
	for(int i = 1; i < info.KohonenMapSize[0]; i++){

		//load in the map point
		if(threadIdx.x < MAX_DIMENSIONALITY){
			MapPoint[threadIdx.x] = KohonenMap[i*info.NumberOfDimensions+threadIdx.x];
		}
		
		//find the distance to the map point
		float currDistance = 0.0f;
		for(int j = 0; j < info.NumberOfDimensions; j++){
			float value = InputData[j*bufferSize+kOffset] - MapPoint[j];
			currDistance += info.Weights[j]*value*value;
		}

		//if less than the minimum, save it
		minIndex.x = (currDistance < minDistance) ? i : minIndex.x;
		minDistance = (currDistance < minDistance) ? currDistance : minDistance;

	}

	//save off the index of the closest map point
	if( kOffset < bufferSize ) OutputData[kOffset] = minIndex;
	if( kOffset < bufferSize ) OutputDistance[kOffset] = minDistance;

}

__global__ void ProcessImageToMapRepeat(float* KohonenMap, float* InputData, short2* OutputData, float* OutputDistance, int KohonenRow){

	__shared__ float MapPoint[MAX_DIMENSIONALITY];

	int kOffset = threadIdx.x + blockDim.x * blockIdx.x;

	float minDistance = OutputDistance[kOffset];
	short2 minIndex = OutputData[kOffset];

	int bufferSize = info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2];
	for(int i = 0; i < info.KohonenMapSize[0]; i++){

		//load in the map point
		int iApp = i+KohonenRow*info.KohonenMapSize[0];
		if(threadIdx.x < MAX_DIMENSIONALITY){
			MapPoint[threadIdx.x] = KohonenMap[iApp*info.NumberOfDimensions+threadIdx.x];
		}
		
		//find the distance to the map point
		float currDistance = 0.0f;
		for(int j = 0; j < info.NumberOfDimensions; j++){
			float value = InputData[j*bufferSize+kOffset] - MapPoint[j];
			currDistance += info.Weights[j]*value*value;
		}

		//if less than the minimum, save it
		minIndex.x = (currDistance < minDistance) ? i : minIndex.x;
		minIndex.y = (currDistance < minDistance) ? KohonenRow : minIndex.y;
		minDistance = (currDistance < minDistance) ? currDistance : minDistance;

	}

	//save off the index of the closest map point
	if( kOffset < bufferSize ) OutputData[kOffset] = minIndex;
	if( kOffset < bufferSize ) OutputDistance[kOffset] = minDistance;

}

void CUDAalgo_applyKohonenMap( float* inputData, float* inputKohonen, short* outputData,
								Kohonen_Application_Information& information,
								cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Application_Information) );

	//translate data onto device
	float* device_KohonenMap = 0;
	cudaMalloc( (void**) &device_KohonenMap, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions );
	float* device_InputData = 0;
	cudaMalloc( (void**) &device_InputData, sizeof(float)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*information.NumberOfDimensions );
	cudaMemcpyAsync( device_KohonenMap, inputKohonen, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );
	
	//rearrange image data to be easier to work with (should parallelize)
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
	
	short* device_OutputData = 0;
	cudaMalloc( (void**) &device_OutputData, sizeof(short)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*2 );
	float* device_OutputDistance = 0;
	cudaMalloc( (void**) &device_OutputDistance, sizeof(float)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2] );

	//apply the map

	dim3 grid((information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2] + 255) / 256,1,1);
	dim3 threads(256,1,1);
	ProcessImageToMapFirst<<<grid, threads, 0, *stream >>>(device_KohonenMap, device_InputData, (short2*) device_OutputData, device_OutputDistance );
	cudaStreamSynchronize(*stream);
	for( int i = 1; i < information.KohonenMapSize[1]; i++ ){
		ProcessImageToMapRepeat<<<grid, threads, 0, *stream >>>(device_KohonenMap, device_InputData, (short2*) device_OutputData, device_OutputDistance, i );
		cudaStreamSynchronize(*stream);
	}

	//copy results back
	cudaMemcpyAsync( outputData, device_OutputData, sizeof(short)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*2, cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);

	//remove allocated memory
	cudaFree(device_KohonenMap);
	cudaFree(device_InputData);
	cudaFree(device_OutputDistance);
	cudaFree(device_OutputData);
}