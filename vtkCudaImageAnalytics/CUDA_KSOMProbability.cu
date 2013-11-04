#include "CUDA_KSOMProbability.h"
#include "CUDA_commonKernels.h"
#include <float.h>
#include <stdio.h>

__constant__ Kohonen_Probability_Information info;

__global__ void ProcessSample(float* InputData, float* KohonenMap, float* Buffer){

	__shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

	//get sample co-ordinates in buffer
	int kOffset = CUDASTDOFFSET;
	if(threadIdx.x < MAX_DIMENSIONALITY){
		SamplePointLocal[threadIdx.x] = InputData[threadIdx.x];
	}
	__syncthreads();
	
	//calculate the distance
	float distance = 0.0f;
	float penalty = 1.0f;
	int bufferSize = info.KohonenMapSize[0] * info.KohonenMapSize[1];
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float var = KohonenMap[(2*i+1)*bufferSize+kOffset];
		float value = (KohonenMap[(2*i)*bufferSize+kOffset] - SamplePointLocal[i]);
		distance += value * value * info.Scale / var;
		penalty *= var;
	}
	distance += 0.5 * log(penalty);

	//output weight
	//float weight = exp( -1.0f * distance );
	//if(kOffset < bufferSize) Buffer[kOffset] = weight;
	if(kOffset < bufferSize) Buffer[kOffset] = distance;
	
}

__global__ void SumOverLargeBufferLogBased( float* buffer, int spread, int size ){
	
	int offset = CUDASTDOFFSET;
	float value1 = buffer[offset];
	float value2 = buffer[offset+spread];
	
	float x = max(value1,value2);
	float n = min(value1,value2);
	float value = x - log( 1+exp(x-n) );

	if( offset+spread < size )
		buffer[offset] = value;

}

void CUDAalgo_applyProbabilityMaps( float* inputData, char* inputMask, float* inputKohonen, float** probabilityData,
									float** outputData, bool useProbData, bool useEntropy,
									Kohonen_Probability_Information& information, cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Probability_Information) );

	//translate data onto device (need to transpose KSOM)
	int VolumeSize = information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2];
	int MapSize = information.KohonenMapSize[0]*information.KohonenMapSize[1];
	float* tempKohonen = new float[2*MapSize*information.NumberOfDimensions];
	int bufferJump = MapSize;
	for(int i = 0; i < MapSize; i++)
		for( int j = 0; j < 2*information.NumberOfDimensions; j++ )
			tempKohonen[j*bufferJump+i] = inputKohonen[i*2*information.NumberOfDimensions+j];
	float* device_KohonenMap = 0;
	cudaMalloc( (void**) &device_KohonenMap, sizeof(float)*MapSize*2*information.NumberOfDimensions );
	cudaMemcpy( device_KohonenMap, tempKohonen, sizeof(float)*MapSize*2*information.NumberOfDimensions, cudaMemcpyHostToDevice );
	delete[] tempKohonen;

	//allocate a distance buffer
	float* device_BaseBuffer = 0;
	cudaMalloc( (void**) &device_BaseBuffer, sizeof(float)*MapSize );
	float* device_WorkingBuffer = 0;
	cudaMalloc( (void**) &device_WorkingBuffer, sizeof(float)*MapSize );

	//rearrange image data to be easier to work with (should parallelize)
	float* device_InputData = 0;
	cudaMalloc( (void**) &device_InputData, sizeof(float)*VolumeSize*information.NumberOfDimensions );
	cudaMemcpyAsync( device_InputData, inputData, sizeof(float)*VolumeSize*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );
	
	//copy probability buffers
	float* device_ProbabilityBuffer = 0;
	if( useProbData ){
		cudaMalloc( (void**) &device_ProbabilityBuffer, sizeof(float)*MapSize*information.NumberOfLabels );
		for( int i = 0; i < information.NumberOfLabels; i++)
			cudaMemcpyAsync( device_ProbabilityBuffer+i*MapSize, probabilityData[i], sizeof(float)*MapSize, cudaMemcpyHostToDevice, *stream );
	}

	//apply the map
	dim3 grid = GetGrid(MapSize);
	dim3 threads(NUMTHREADS,1,1);
	for( int voxel = 0; voxel < VolumeSize; voxel++ ){
		
		//if we are not in the mask, ignore this voxel
		if( inputMask != 0 && inputMask[voxel] == 0 ){
			for( int i = 0; i < information.NumberOfLabels; i++)
				(outputData[i])[voxel] = FLT_MAX;
			continue;
		}

		//else, process it over the entire map
		int InputBufferOffset = voxel*information.NumberOfDimensions;
		ProcessSample<<<grid, threads, 0, *stream>>>(device_InputData+InputBufferOffset, device_KohonenMap, device_BaseBuffer );
		
		for( int i = 0; i < information.NumberOfLabels; i++){

			//multiply the basic amount with the probability buffer into the working buffer
			if( useProbData )
				MultiplyAndStoreBuffer<<<grid, threads, 0, *stream>>>(device_BaseBuffer, device_ProbabilityBuffer+i*MapSize, device_WorkingBuffer, MapSize );
			else
				CopyBuffers<<<grid, threads, 0, *stream>>>(device_WorkingBuffer, device_BaseBuffer, MapSize);

			//reduce working buffer by summation
			int j = 1;
			while( j < MapSize ) j += j;
			for(; j >= 1; j = j/2){
				dim3 tempGrid = GetGrid(j);
				SumOverLargeBufferLogBased<<<tempGrid, threads, 0, *stream>>>(device_WorkingBuffer,j,MapSize);
				cudaStreamSynchronize(*stream);
			}

			//store resulting cost
			cudaMemcpyAsync( (outputData[i])+voxel, device_WorkingBuffer, sizeof(float), cudaMemcpyDeviceToHost, *stream );

		}
	}
	cudaStreamSynchronize(*stream);

	//switch to entropy
	if( !useEntropy )
		for( int voxel = 0; voxel < VolumeSize; voxel++ )
			for( int i = 0; i < information.NumberOfLabels; i++)
				(outputData[i])[voxel] =  exp(-((outputData[i])[voxel]) );

	//remove allocated memory
	cudaFree(device_KohonenMap);
	cudaFree(device_InputData);
	cudaFree(device_BaseBuffer);
	cudaFree(device_WorkingBuffer);
	cudaFree(device_ProbabilityBuffer);
}