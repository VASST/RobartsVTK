#include "CUDA_kohonenapplication.h"
#include "CUDA_commonKernels.h"
#include <float.h>
#include <stdio.h>

__constant__ Kohonen_Application_Information info;

__global__ void ProcessSampleAccumulate(float* InputData, float* KohonenMap, float* Accumulator, float2* AccumulatorLoc, short CompLocX, short CompLocY){

	__shared__ float ComponentLocal[2*MAX_DIMENSIONALITY+1];

	//get sample co-ordinates in buffer
	int kOffset = CUDASTDOFFSET;
	if(threadIdx.x < 2*MAX_DIMENSIONALITY+1)
		ComponentLocal[threadIdx.x] = KohonenMap[threadIdx.x];
	__syncthreads();
	
	//calculate the distance
	//float distance = -log(ComponentLocal[0]);
	float distance = 0.0f;
	float penalty = 1.0f;
	int VolumeSize = info.BufferSize;
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = InputData[i*VolumeSize+kOffset];
		distance += 0.5f * (ComponentLocal[2*i+1]-value) * (ComponentLocal[2*i+1]-value) * info.Scale / ComponentLocal[2*i+2];
		penalty *= ComponentLocal[2*i+2];
	}
	distance += 0.5 * log(penalty);

	//accumulate information
	float accumulator = Accumulator[kOffset];
	float2 location = AccumulatorLoc[kOffset];
	location.x = (distance > accumulator) ? location.x : (float) CompLocX;
	location.y = (distance > accumulator) ? location.y : (float) CompLocY;
	if(kOffset < VolumeSize) AccumulatorLoc[kOffset] = location;
	accumulator = min( accumulator, distance );
	if(kOffset < VolumeSize) Accumulator[kOffset] = accumulator;

}

__global__ void ProcessSampleOverwrite(float* InputData, float* KohonenMap, float* Accumulator, float2* AccumulatorLoc,
							  short CompLocX, short CompLocY){

	__shared__ float ComponentLocal[2*MAX_DIMENSIONALITY+1];

	//get sample co-ordinates in buffer
	int kOffset = CUDASTDOFFSET;
	if(threadIdx.x < 2*MAX_DIMENSIONALITY+1)
		ComponentLocal[threadIdx.x] = KohonenMap[threadIdx.x];
	__syncthreads();
	
	//calculate the distance
	float distance = 0.0f;//-log(ComponentLocal[0]);
	float penalty = 1.0f;
	int VolumeSize = info.BufferSize;
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = InputData[i*VolumeSize+kOffset];
		distance += 0.5f * (ComponentLocal[2*i+1]-value) * (ComponentLocal[2*i+1]-value) * info.Scale / ComponentLocal[2*i+2];
		penalty *= ComponentLocal[2*i+2];
	}
	distance += 0.5 * log(penalty);

	//accumulate information
	float2 location;
	location.x = (float) CompLocX;
	location.y = (float) CompLocY;
	if(kOffset < VolumeSize) AccumulatorLoc[kOffset] = location;
	if(kOffset < VolumeSize) Accumulator[kOffset] = distance;

}

void CUDAalgo_applyKohonenMap( float* inputData, char* inputMask, float* inputKohonen, float* outputData,
								Kohonen_Application_Information& information,
								cudaStream_t* stream ){


	//translate data onto device (need to transpose KSOM)
	int VolumeSize = information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2];
	int MapSize = information.KohonenMapSize[0]*information.KohonenMapSize[1];
	
	//copy kohonen data to GPU
	float* device_KohonenMap = 0;
	cudaMalloc( (void**) &device_KohonenMap, sizeof(float)*MapSize*(2*information.NumberOfDimensions+1) );
	cudaMemcpy( device_KohonenMap, inputKohonen, sizeof(float)*MapSize*(2*information.NumberOfDimensions+1), cudaMemcpyHostToDevice );

	//partition image into affordable sizes
	size_t freeMemory, totalMemory;
	cudaMemGetInfo(&freeMemory, &totalMemory);
	int SizeAllowed = (int) ((double)freeMemory / (double)(sizeof(float)*(information.NumberOfDimensions+3)) );
	SizeAllowed -= SizeAllowed % NUMTHREADS;
	if(SizeAllowed > VolumeSize) SizeAllowed = VolumeSize;
	while(VolumeSize % SizeAllowed < NUMTHREADS && VolumeSize % SizeAllowed > 0)
		SizeAllowed -= NUMTHREADS;
	information.BufferSize = SizeAllowed;

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Application_Information) );

	//create necessary GPU buffers
	float* device_InputData = 0;
	cudaMalloc( (void**) &device_InputData, sizeof(float)*SizeAllowed*information.NumberOfDimensions );
	float* device_Accumulator = 0;
	cudaMalloc( (void**) &device_Accumulator, sizeof(float)*SizeAllowed );
	float2* device_AccumulatorLoc = 0;
	cudaMalloc( (void**) &device_AccumulatorLoc, sizeof(float2)*SizeAllowed );

	//create necessary CPU buffers
	float* tempImage = new float[SizeAllowed*information.NumberOfDimensions];

	//go over each partition
	int pConsumed = 0;
	while(pConsumed < VolumeSize){

		//figure out sizes and starting points
		int pSize = (VolumeSize-pConsumed > SizeAllowed) ? SizeAllowed : VolumeSize-pConsumed;
		float* inputDataStart = inputData + pConsumed*information.NumberOfDimensions;
		float* outputDataStart = outputData + pConsumed*2;

		//rearrange image data to be easier to work with (should parallelize)
		for( int j = 0; j < information.NumberOfDimensions; j++ )
			for(int i = 0; i < pSize; i++)
				tempImage[j*pSize+i] = inputDataStart[i*information.NumberOfDimensions+j];
		cudaMemcpyAsync( device_InputData, tempImage, sizeof(float)*pSize*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );

		//apply the map
		dim3 grid = GetGrid(pSize);
		dim3 threads(NUMTHREADS,1,1);
		ProcessSampleOverwrite<<<grid, threads, 0, *stream>>>(device_InputData, device_KohonenMap,
															  device_Accumulator, device_AccumulatorLoc,0,0);
		for( int component = 1; component < MapSize; component++ )
 			ProcessSampleAccumulate<<<grid, threads, 0, *stream>>>(device_InputData, device_KohonenMap+component*(2*information.NumberOfDimensions+1),
																   device_Accumulator, device_AccumulatorLoc,
																   component%information.KohonenMapSize[0], component/information.KohonenMapSize[0]);
	

		//move result to CPU
		cudaMemcpyAsync( outputDataStart, device_AccumulatorLoc, sizeof(float2)*pSize, cudaMemcpyDeviceToHost, *stream );
		cudaStreamSynchronize(*stream);
	
		//update the amount of consumed volume
		pConsumed+=pSize;

	}

	//remove allocated memory
	delete[] tempImage;
	cudaFree(device_KohonenMap);
	cudaFree(device_InputData);
	cudaFree(device_Accumulator);

}