#include "CUDA_KSOMProbability.h"
#include "CUDA_commonKernels.h"
#include <float.h>
#include <stdio.h>

__constant__ Kohonen_Probability_Information info;

__global__ void ProcessSample(float* InputData, float* KohonenMap, float* Accumulator){

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

	//accumulate entropy
	float oldEntropy = Accumulator[kOffset];
	float x = max(oldEntropy, distance);
	float n = min(oldEntropy, distance);
	float newEntropy = (exp(n-x) > 0.0f) ? n + log(1+exp(n-x)): -log( exp(-x) + exp(-n));
	newEntropy = (newEntropy < n) ? newEntropy : n;
	if(kOffset < VolumeSize) Accumulator[kOffset] = newEntropy;

}

void CUDAalgo_applyProbabilityMaps( float* inputData, char* inputMask, float* inputKohonen, float** probabilityData,
									float** outputData, bool useProbData, bool useEntropy,
									Kohonen_Probability_Information& information, cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Probability_Information) );

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
	SetBufferToConst<<<grid, threads, 0, *stream>>>(device_Accumulator, FLT_MAX, VolumeSize);

	//apply the map
	for( int i = 0; i < information.NumberOfLabels; i++){

		for( int component = 0; component < MapSize; component++ )
			ProcessSample<<<grid, threads, 0, *stream>>>(device_InputData, device_KohonenMap+component*(2*information.NumberOfDimensions+1),
														 device_Accumulator);
		
		if( !useEntropy )
			NegExpBuffer<<<grid, threads, 0, *stream>>>(device_Accumulator, VolumeSize);

		//move entropy to CPU
		cudaMemcpyAsync( (outputData[i]), device_Accumulator, sizeof(float)*VolumeSize, cudaMemcpyDeviceToHost, *stream );
	}
	cudaStreamSynchronize(*stream);

	//remove allocated memory
	cudaFree(device_KohonenMap);
	cudaFree(device_InputData);
	cudaFree(device_Accumulator);

}