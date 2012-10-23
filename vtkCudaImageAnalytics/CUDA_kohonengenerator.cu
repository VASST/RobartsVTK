#include "CUDA_kohonengenerator.h"
#include <float.h>

//parameters held in constant memory
__constant__ Kohonen_Generator_Information info;
__constant__ float SamplePoint[MAX_DIMENSIONALITY];

__global__ void ProcessSample(float* KohonenMap, float* DistanceBuffer, short2* IndexBuffer ){

	__shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

	//get sample co-ordinates in buffer
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadIdx.x < MAX_DIMENSIONALITY){
		SamplePointLocal[threadIdx.x] = SamplePoint[threadIdx.x];
	}
	
	//calculate the distance
	float distance = 0.0f;
	int bufferSize = info.KohonenMapSize[0] * info.KohonenMapSize[1];
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = KohonenMap[i*bufferSize+kOffset] - SamplePointLocal[i];
		distance += info.Weights[i]*value*value;
		__syncthreads();
	}

	DistanceBuffer[kOffset] = distance;
	short2 index = {kOffset % info.KohonenMapSize[0], kOffset / info.KohonenMapSize[0] };
	IndexBuffer[kOffset] = index;

}

__global__ void FindMinSample( float* DistanceBuffer, short2* IndexBuffer, int spread ){
	
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	float distance1 = DistanceBuffer[kOffset];
	float distance2 = DistanceBuffer[kOffset+spread];
	short2 index1 = IndexBuffer[kOffset];
	short2 index2 = IndexBuffer[kOffset+spread];

	if( kOffset+spread < info.KohonenMapSize[0] * info.KohonenMapSize[1] ){
		DistanceBuffer[kOffset] = (distance1 < distance2) ? distance1 : distance2;
		IndexBuffer[kOffset] = (distance1 < distance2) ? index1 : index2;
	}

}

__global__ void UpdateWeights( float* KohonenMap, short2 minIndex, float alpha, float neigh ){

	__shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

	//get sample co-ordinates in buffer
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	short2 currIndex = {kOffset % info.KohonenMapSize[0], kOffset / info.KohonenMapSize[0] };
	if(threadIdx.x < MAX_DIMENSIONALITY){
		SamplePointLocal[threadIdx.x] = SamplePoint[threadIdx.x];
	}

	
	float multiplier = (neigh > 0.125f ) ? alpha * exp( -((currIndex.x-minIndex.x)*(currIndex.x-minIndex.x) + (currIndex.y-minIndex.y)*(currIndex.y-minIndex.y) ) / neigh ) : 0;
	int bufferSize = info.KohonenMapSize[0] * info.KohonenMapSize[1];
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = (1.0f-multiplier)*KohonenMap[i*bufferSize+kOffset] + multiplier*SamplePointLocal[i];
		KohonenMap[i*bufferSize+kOffset] = value;
		__syncthreads();
	}

}

void CUDAalgo_generateKohonenMap( float* inputData, float* outputKohonen,
									Kohonen_Generator_Information& information,
									float alpha, float alphaDecay,
									float neighbourhood, float nDecay,
									cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Generator_Information) );

	//create buffer for the Kohonen map
	float* device_KohonenMap = 0;
	cudaMalloc( (void**) &device_KohonenMap, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions );
	for(int i = 0; i < information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions; i++)
		outputKohonen[i] = (float)rand()/(float)RAND_MAX;
	cudaMemcpyAsync(device_KohonenMap, outputKohonen, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );

	//allocate a distance buffer
	float* device_DistanceBuffer = 0;
	cudaMalloc( (void**) &device_DistanceBuffer, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );
	short2* device_IndexBuffer = 0;
	cudaMalloc( (void**) &device_IndexBuffer, sizeof(short2)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );

	//train kohonen map
	dim3 grid(information.KohonenMapSize[0]*information.KohonenMapSize[1]/256, 1, 1);
	dim3 threads(256, 1, 1);
	for( int epoch = 0; epoch < information.MaxEpochs; epoch++ ){
		for( int batch = 0; batch < information.BatchSize; batch++ ){
			int x = (rand() % information.VolumeSize[0])+ information.VolumeSize[0] * (
					(rand() % information.VolumeSize[1])+ information.VolumeSize[1] * (
					(rand() % information.VolumeSize[2])							  )) * information.NumberOfDimensions;
			cudaMemcpyToSymbolAsync(SamplePoint, &(inputData[x]), sizeof(float)*information.NumberOfDimensions );
			cudaStreamSynchronize(*stream);
			ProcessSample<<<grid, threads, 0, *stream>>>(device_KohonenMap, device_DistanceBuffer, device_IndexBuffer);
			for(int i = information.KohonenMapSize[0]*information.KohonenMapSize[1] / 2; i > 0; i = i/2){
				dim3 tempGrid( i>256 ? i/256 : 1, 1, 1);
				FindMinSample<<<tempGrid, threads, 0, *stream>>>(device_DistanceBuffer, device_IndexBuffer, i);
			}
			short2 minIndex;
			cudaStreamSynchronize(*stream);
			cudaMemcpy( &minIndex, device_IndexBuffer, sizeof(short2), cudaMemcpyDeviceToHost );
			UpdateWeights<<<grid, threads, 0, *stream>>>(device_KohonenMap, minIndex, alpha, neighbourhood);
		}
		alpha *= alphaDecay;
		neighbourhood *= nDecay;
	}

	//remove distance buffer
	cudaFree(device_DistanceBuffer);
	cudaFree(device_IndexBuffer);

	//translate back data
	float* tempKohonen = new float[information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions];
	cudaMemcpyAsync( tempKohonen, device_KohonenMap, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions, cudaMemcpyDeviceToHost, *stream );
	cudaFree(device_KohonenMap);
	cudaStreamSynchronize(*stream);

	int bufferJump = information.KohonenMapSize[0]*information.KohonenMapSize[1];
	for(int i = 0; i < information.KohonenMapSize[0]*information.KohonenMapSize[1]; i++)
		for( int j = 0; j < information.NumberOfDimensions; j++ )
			outputKohonen[i*information.NumberOfDimensions+j] = tempKohonen[j*bufferJump+i];
	delete[] tempKohonen;

}