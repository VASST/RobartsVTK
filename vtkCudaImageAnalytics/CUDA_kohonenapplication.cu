#include "CUDA_kohonenapplication.h"
#include "CUDA_commonKernels.h"
#include <float.h>
#include <stdio.h>

__constant__ Kohonen_Application_Information info;

__global__ void ProcessSample(float* InputData, float* KohonenMap, float* Accumulator, float2* AccumulatorLoc,
							  short CompLocX, short CompLocY){

	__shared__ float ComponentLocal[2*MAX_DIMENSIONALITY+1];

	//get sample co-ordinates in buffer
	int kOffset = CUDASTDOFFSET;
	if(threadIdx.x < 2*MAX_DIMENSIONALITY+1)
		ComponentLocal[threadIdx.x] = KohonenMap[threadIdx.x];
	__syncthreads();
	
	//calculate the distance
	float distance = 0.0f;
	float penalty = ComponentLocal[0]*ComponentLocal[0];
	int VolumeSize = info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2];
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = InputData[i*VolumeSize+kOffset];
		distance += (ComponentLocal[2*i+1]-value) * (ComponentLocal[2*i+1]-value) * info.Scale / ComponentLocal[2*i+2];
		penalty *= ComponentLocal[2*i+2];
	}
	distance += 0.5 * log(penalty);

	//accumulate information
	float accumulator = Accumulator[kOffset];
	accumulator += exp( -distance );
	if(kOffset < VolumeSize) Accumulator[kOffset] = accumulator;
	float2 location = AccumulatorLoc[kOffset];
	location.x += exp( -distance ) * (float) CompLocX;
	location.y += exp( -distance ) * (float) CompLocY;
	if(kOffset < VolumeSize) AccumulatorLoc[kOffset] = location;

}

__global__ void NormalizeResults(float* Accumulator, float2* AccumulatorLoc){
	int kOffset = CUDASTDOFFSET;
	int VolumeSize = info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2];
	float accumulator = Accumulator[kOffset];
	float2 location = AccumulatorLoc[kOffset];
	location.x /= accumulator;
	location.y /= accumulator;
	if(kOffset < VolumeSize) AccumulatorLoc[kOffset] = location;
}

void CUDAalgo_applyKohonenMap( float* inputData, char* inputMask, float* inputKohonen, float* outputData,
								Kohonen_Application_Information& information,
								cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Application_Information) );

	//translate data onto device (need to transpose KSOM)
	int VolumeSize = information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2];
	int MapSize = information.KohonenMapSize[0]*information.KohonenMapSize[1];
	
	//copy kohonen data to GPU
	float* device_KohonenMap = 0;
	cudaMalloc( (void**) &device_KohonenMap, sizeof(float)*MapSize*(2*information.NumberOfDimensions+1) );
	cudaMemcpy( device_KohonenMap, inputKohonen, sizeof(float)*MapSize*(2*information.NumberOfDimensions+1), cudaMemcpyHostToDevice );

	//rearrange image data to be easier to work with (should parallelize)
	float* tempImage = new float[2*VolumeSize*information.NumberOfDimensions];
	int bufferJump = MapSize;
	for( int j = 0; j < 2*information.NumberOfDimensions; j++ )
		for(int i = 0; i < VolumeSize; i++)
			tempImage[j*bufferJump+i] = inputData[i*2*information.NumberOfDimensions+j];
	float* device_InputData = 0;
	cudaMalloc( (void**) &device_InputData, sizeof(float)*VolumeSize*information.NumberOfDimensions );
	cudaMemcpyAsync( device_InputData, tempImage, sizeof(float)*VolumeSize*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );
	delete[] tempImage;

	//allocate an accumulation buffer
	dim3 grid = GetGrid(VolumeSize);
	dim3 threads(NUMTHREADS,1,1);
	float* device_Accumulator = 0;
	cudaMalloc( (void**) &device_Accumulator, sizeof(float)*VolumeSize );
	ZeroOutBuffer<float><<<grid, threads, 0, *stream>>>(device_Accumulator, VolumeSize);
	float2* device_AccumulatorLoc = 0;
	cudaMalloc( (void**) &device_AccumulatorLoc, sizeof(float2)*VolumeSize );
	grid = GetGrid(2*VolumeSize);
	ZeroOutBuffer<float><<<grid, threads, 0, *stream>>>((float*)device_AccumulatorLoc, 2*VolumeSize);
	grid = GetGrid(VolumeSize);

	//apply the map
	for( int component = 0; component < MapSize; component++ )
		ProcessSample<<<grid, threads, 0, *stream>>>(device_InputData, device_KohonenMap+component*(2*information.NumberOfDimensions+1),
													 device_Accumulator, device_AccumulatorLoc,
													 component%information.KohonenMapSize[0], component/information.KohonenMapSize[0]);
	NormalizeResults<<<grid, threads, 0, *stream>>>(device_Accumulator, device_AccumulatorLoc);

	//move entropy to CPU
	cudaMemcpyAsync( outputData, device_AccumulatorLoc, sizeof(float2)*VolumeSize, cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);


	//remove allocated memory
	cudaFree(device_KohonenMap);
	cudaFree(device_InputData);
	cudaFree(device_Accumulator);

}