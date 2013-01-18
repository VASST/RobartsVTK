#include "CUDA_kohonengenerator.h"
#include <float.h>
#include <stdio.h>

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
	__syncthreads();
	
	//calculate the distance
	float distance = 0.0f;
	int bufferSize = info.KohonenMapSize[0] * info.KohonenMapSize[1];
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = info.Weights[i]*(KohonenMap[i*bufferSize+kOffset] - SamplePointLocal[i]);
		distance += value*value;
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

	
	float multiplier = alpha * exp( -((currIndex.x-minIndex.x)*(currIndex.x-minIndex.x) + (currIndex.y-minIndex.y)*(currIndex.y-minIndex.y) ) / neigh );
	int bufferSize = info.KohonenMapSize[0] * info.KohonenMapSize[1];
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = (1.0f-multiplier)*KohonenMap[i*bufferSize+kOffset] + multiplier*SamplePointLocal[i];
		KohonenMap[i*bufferSize+kOffset] = value;
		__syncthreads();
	}

}

void CUDAalgo_generateKohonenMap(	float** inputData, float* outputKohonen, char** maskData, double* range,
									int* VolumeSize, int NumVolumes,
									Kohonen_Generator_Information& information,
									int MaxEpochs, int BatchSize,
									float alpha, float alphaDecay,
									float neighbourhood, float nDecay,
									cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Generator_Information) );

	//create buffer for the Kohonen map
	float* device_KohonenMap = 0;
	cudaMalloc( (void**) &device_KohonenMap, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions );
	int c = 0;
	for(int i = 0; i < information.KohonenMapSize[0]*information.KohonenMapSize[1]; i++)
		for(int j = 0; j < information.NumberOfDimensions; j++ )
			outputKohonen[c++] = (float)((double)rand()/(double)RAND_MAX);
	cudaMemcpyAsync(device_KohonenMap, outputKohonen, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );

	//allocate a distance buffer
	float* device_DistanceBuffer = 0;
	cudaMalloc( (void**) &device_DistanceBuffer, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );
	short2* device_IndexBuffer = 0;
	cudaMalloc( (void**) &device_IndexBuffer, sizeof(short2)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );

	//pre-train kohonen map

	//train kohonen map
	dim3 grid(information.KohonenMapSize[0]*information.KohonenMapSize[1]/256, 1, 1);
	dim3 threads(256, 1, 1);
	if( BatchSize == -1 ){
		for( int epoch = 0; epoch < MaxEpochs; epoch++ ){

			for( int picture = 0; picture < NumVolumes; picture++ ){
				for( int sampleOffset = 0; sampleOffset < VolumeSize[3*picture]*VolumeSize[3*picture+1]*VolumeSize[3*picture+2]; sampleOffset++){

					int sampleDimensionalOffset = information.NumberOfDimensions * sampleOffset;

					//if this is not a valid sample (ie: masked out) then try again
					if( maskData && (maskData[picture])[sampleOffset] == 0 )
						continue;

					//find the distance between each centroid and the sample
					cudaMemcpyToSymbolAsync(SamplePoint, &((inputData[picture])[sampleDimensionalOffset]), sizeof(float)*information.NumberOfDimensions );
					cudaStreamSynchronize(*stream);
					ProcessSample<<<grid, threads, 0, *stream>>>(device_KohonenMap, device_DistanceBuffer, device_IndexBuffer);

					//find the winning centroid
					for(int i = information.KohonenMapSize[0]*information.KohonenMapSize[1] / 2; i > 0; i = i/2){
						dim3 tempGrid( i>256 ? i/256 : 1, 1, 1);
						FindMinSample<<<tempGrid, threads, 0, *stream>>>(device_DistanceBuffer, device_IndexBuffer, i);
					}

					//update the weights of each centroid
					short2 minIndex;
					cudaMemcpyAsync( &minIndex, device_IndexBuffer, sizeof(short2), cudaMemcpyDeviceToHost, *stream );
					cudaStreamSynchronize(*stream);
					UpdateWeights<<<grid, threads, 0, *stream>>>(device_KohonenMap, minIndex, alpha, neighbourhood);
				}
			}

			//update the weight updaters
			alpha *= alphaDecay;
			neighbourhood *= nDecay;
		}

	//if we are randomly sampling from the images
	}else{
		for( int epoch = 0; epoch < MaxEpochs; epoch++ ){
			for( int batch = 0; batch < BatchSize; batch++ ){

				int sampleP = rand() % NumVolumes;
				int sampleX = rand() % VolumeSize[3*sampleP];
				int sampleY = rand() % VolumeSize[3*sampleP+1];
				int sampleZ = rand() % VolumeSize[3*sampleP+2];
				int sampleOffset = (sampleX + VolumeSize[3*sampleP] *( sampleY + VolumeSize[3*sampleP+1] * sampleZ ) );
				int sampleDimensionalOffset = information.NumberOfDimensions * sampleOffset;

				//if this is not a valid sample (ie: masked out) then try again
				if( maskData && (maskData[sampleP])[sampleOffset] == 0 ){
					batch--;
					continue;
				}

				//find the distance between each centroid and the sample
				cudaMemcpyToSymbolAsync(SamplePoint, &((inputData[sampleP])[sampleDimensionalOffset]), sizeof(float)*information.NumberOfDimensions );
				cudaStreamSynchronize(*stream);
				ProcessSample<<<grid, threads, 0, *stream>>>(device_KohonenMap, device_DistanceBuffer, device_IndexBuffer);

				//find the winning centroid
				for(int i = information.KohonenMapSize[0]*information.KohonenMapSize[1] / 2; i > 0; i = i/2){
					dim3 tempGrid( i>256 ? i/256 : 1, 1, 1);
					FindMinSample<<<tempGrid, threads, 0, *stream>>>(device_DistanceBuffer, device_IndexBuffer, i);
				}

				//update the weights of each centroid
				short2 minIndex;
				cudaMemcpyAsync( &minIndex, device_IndexBuffer, sizeof(short2), cudaMemcpyDeviceToHost, *stream );
				cudaStreamSynchronize(*stream);
				UpdateWeights<<<grid, threads, 0, *stream>>>(device_KohonenMap, minIndex, alpha, neighbourhood);
			}
		}

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