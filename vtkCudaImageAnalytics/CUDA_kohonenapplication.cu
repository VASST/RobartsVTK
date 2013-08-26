#include "CUDA_kohonenapplication.h"
#include "CUDA_commonKernels.h"
#include <float.h>
#include <stdio.h>

__constant__ Kohonen_Application_Information info;
texture<float, 3, cudaReadModeElementType> Kohonen_Map;

__global__ void ProcessSample(int samplePointLoc, float* InputData, float* KohonenMap, float* DistanceBuffer, float* WeightBuffer, float2* IndexBuffer, float2* WeightedIndexBuffer){

	__shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

	//get sample co-ordinates in buffer
	int kOffset = CUDASTDOFFSET;
	if(threadIdx.x < MAX_DIMENSIONALITY){
		SamplePointLocal[threadIdx.x] = InputData[info.NumberOfDimensions*samplePointLoc+threadIdx.x];
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

	//output distance and weight
	if(kOffset < bufferSize) DistanceBuffer[kOffset] = distance;
	float weight = exp( -1.0f * distance );
	if(kOffset < bufferSize) WeightBuffer[kOffset] = weight;
	
	//output index and weighted index
	float2 index = {(float)(kOffset % info.KohonenMapSize[0]), (float)(kOffset / info.KohonenMapSize[0]) };
	if(kOffset < bufferSize) IndexBuffer[kOffset] = index;
	index.x *= weight;
	index.y *= weight;
	if(kOffset < bufferSize) WeightedIndexBuffer[kOffset] = index;
	
}

__global__ void ReduceSample( float* DistanceBuffer, float* WeightBuffer, float2* IndexBuffer, float2* WeightedIndexBuffer, int spread ){
	
	//collect samples
	int kOffset = CUDASTDOFFSET;
	float distance1 = DistanceBuffer[kOffset];
	float distance2 = DistanceBuffer[kOffset+spread];
	float weight1 = WeightBuffer[kOffset];
	float weight2 = WeightBuffer[kOffset+spread];
	float2 index1 = IndexBuffer[kOffset];
	float2 index2 = IndexBuffer[kOffset+spread];
	float2 weightedIndex1 = WeightedIndexBuffer[kOffset];
	float2 weightedIndex2 = WeightedIndexBuffer[kOffset+spread];

	//reduce between these two samples
	index1 = (distance1 < distance2) ? index1 : index2;
	distance1  = (distance1 < distance2) ? distance1 : distance2;
	weight1 += weight2;
	weightedIndex1.x += weightedIndex2.x;
	weightedIndex1.y += weightedIndex2.y;

	//output results
	if( kOffset+spread < info.KohonenMapSize[0] * info.KohonenMapSize[1] ){
		DistanceBuffer[kOffset] = distance1;
		WeightBuffer[kOffset] = weight1;
		IndexBuffer[kOffset] = index1;
		WeightedIndexBuffer[kOffset] = weightedIndex1;
	}

}

template <unsigned int blockSize>
__global__ void reduce6(float* DistanceBuffer, float* WeightBuffer, float2* IndexBuffer, float2* WeightedIndexBuffer, float* OutputValues, unsigned int n)
{
	__shared__ float minDist[blockSize];
	__shared__ float sumDist[blockSize];
	__shared__ float2 minIndex[blockSize];
	__shared__ float2 sumIndex[blockSize];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;

	minDist[tid] = FLT_MAX;
	sumDist[tid] = 0;
	minIndex[tid].x = 0;
	minIndex[tid].y = 0;
	sumIndex[tid].x = 0;
	sumIndex[tid].y = 0;
	
	while (i < n) {

		sumDist[tid] += WeightBuffer[i];
		sumIndex[tid].x += WeightedIndexBuffer[i].x;
		sumIndex[tid].y += WeightedIndexBuffer[i].y;
		if( minDist[tid] > DistanceBuffer[i] ){
			minDist[tid] = DistanceBuffer[i];
			minIndex[tid] = IndexBuffer[i];
		}

		sumDist[tid] += WeightBuffer[i+blockSize];
		sumIndex[tid].x += WeightedIndexBuffer[i+blockSize].x;
		sumIndex[tid].y += WeightedIndexBuffer[i+blockSize].y;
		if( minDist[tid] > DistanceBuffer[i+blockSize] ){
			minDist[tid] = DistanceBuffer[i+blockSize];
			minIndex[tid] = IndexBuffer[i+blockSize];
		}

		i += gridSize;
		__syncthreads();
	}
	
	if (blockSize >= 512) { if (tid < 256) {
			sumDist[tid] += sumDist[tid + 256];
			sumIndex[tid].x += sumIndex[tid + 256].x;
			sumIndex[tid].y += sumIndex[tid + 256].y;
			if( minDist[tid] > minDist[tid + 256] ){
				minDist[tid] = minDist[tid + 256];
				minIndex[tid] = minIndex[tid + 256];
			}
	} __syncthreads(); }

	if (blockSize >= 256) { if (tid < 128) {
			sumDist[tid] += sumDist[tid + 128];
			sumIndex[tid].x += sumIndex[tid + 128].x;
			sumIndex[tid].y += sumIndex[tid + 128].y;
			if( minDist[tid] > minDist[tid + 128] ){
				minDist[tid] = minDist[tid + 128];
				minIndex[tid] = minIndex[tid + 128];
			}
	} __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) {
			sumDist[tid] += sumDist[tid + 64];
			sumIndex[tid].x += sumIndex[tid + 64].x;
			sumIndex[tid].y += sumIndex[tid + 64].y;
			if( minDist[tid] > minDist[tid + 64] ){
				minDist[tid] = minDist[tid + 64];
				minIndex[tid] = minIndex[tid + 64];
			}
	} __syncthreads(); }
	
	if (tid < 32) {
		if (blockSize >= 64){
			sumDist[tid] += sumDist[tid + 32];
			sumIndex[tid].x += sumIndex[tid + 32].x;
			sumIndex[tid].y += sumIndex[tid + 32].y;
			if( minDist[tid] > minDist[tid + 32] ){
				minDist[tid] = minDist[tid + 32];
				minIndex[tid] = minIndex[tid + 32];
			}
			__syncthreads();
		}
		if (blockSize >= 32){
			sumDist[tid] += sumDist[tid + 16];
			sumIndex[tid].x += sumIndex[tid + 16].x;
			sumIndex[tid].y += sumIndex[tid + 16].y;
			if( minDist[tid] > minDist[tid + 16] ){
				minDist[tid] = minDist[tid + 16];
				minIndex[tid] = minIndex[tid + 16];
			}
			__syncthreads();
		}
		if (blockSize >= 16){
			sumDist[tid] += sumDist[tid + 8];
			sumIndex[tid].x += sumIndex[tid + 8].x;
			sumIndex[tid].y += sumIndex[tid + 8].y;
			if( minDist[tid] > minDist[tid + 8] ){
				minDist[tid] = minDist[tid + 8];
				minIndex[tid] = minIndex[tid + 8];
			}
			__syncthreads();
		}
		if (blockSize >=  8){
			sumDist[tid] += sumDist[tid + 4];
			sumIndex[tid].x += sumIndex[tid + 4].x;
			sumIndex[tid].y += sumIndex[tid + 4].y;
			if( minDist[tid] > minDist[tid + 4] ){
				minDist[tid] = minDist[tid + 4];
				minIndex[tid] = minIndex[tid + 4];
			}
			__syncthreads();
		}
		if (blockSize >=  4){
			sumDist[tid] += sumDist[tid + 2];
			sumIndex[tid].x += sumIndex[tid + 2].x;
			sumIndex[tid].y += sumIndex[tid + 2].y;
			if( minDist[tid] > minDist[tid + 2] ){
				minDist[tid] = minDist[tid + 2];
				minIndex[tid] = minIndex[tid + 2];
			}
			__syncthreads();
		}
		if (blockSize >=  2){
			sumDist[tid] += sumDist[tid + 1];
			sumIndex[tid].x += sumIndex[tid + 1].x;
			sumIndex[tid].y += sumIndex[tid + 1].y;
			if( minDist[tid] > minDist[tid + 1] ){
				minDist[tid] = minDist[tid + 1];
				minIndex[tid] = minIndex[tid + 1];
			}
			__syncthreads();
		}
	}
	if (tid == 0){
		OutputValues[6*blockIdx.x] = minDist[0];
		OutputValues[6*blockIdx.x+1] = minIndex[0].x;
		OutputValues[6*blockIdx.x+2] = minIndex[0].y;
		OutputValues[6*blockIdx.x+3] = sumDist[0];
		OutputValues[6*blockIdx.x+4] = sumIndex[0].x;
		OutputValues[6*blockIdx.x+5] = sumIndex[0].y;
	}
}

void reduceData(int size, int threads, int blocks, float* DistanceBuffer, float* WeightBuffer, float2* IndexBuffer, float2* WeightedIndexBuffer, float* OutputValues, cudaStream_t* stream ){

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * (sizeof(float)+sizeof(short2)) : threads * (sizeof(float)+sizeof(short2));
	switch (threads)
	{
	case 512:
		reduce6<512><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	case 256:
		reduce6<256><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	case 128:
		reduce6<128><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	case 64:
		reduce6< 64><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	case 32:
		reduce6< 32><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	case 16:
		reduce6< 16><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	case 8:
		reduce6< 8><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	case 4:
		reduce6< 4><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	case 2:
		reduce6< 2><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	case 1:
		reduce6< 1><<< dimGrid, dimBlock, smemSize, *stream >>>(DistanceBuffer, WeightBuffer, IndexBuffer, WeightedIndexBuffer, OutputValues, size); break;
	}

}

void CUDAalgo_applyKohonenMap( float* inputData, char* inputMask, float* inputKohonen, float* outputData,
								Kohonen_Application_Information& information,
								cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Application_Information) );

	//translate data onto device (need to transpose KSOM)
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
	float* device_DistanceBufferMin = 0;
	float* device_DistanceBufferTot = 0;
	cudaMalloc( (void**) &device_DistanceBufferMin, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );
	cudaMalloc( (void**) &device_DistanceBufferTot, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );
	float2* device_IndexBufferMin = 0;
	float2* device_IndexBufferTot = 0;
	cudaMalloc( (void**) &device_IndexBufferMin, sizeof(float2)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );
	cudaMalloc( (void**) &device_IndexBufferTot, sizeof(float2)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );
	float* device_reducedOutput = 0;
	cudaMalloc( (void**) &device_reducedOutput, sizeof(float2)*6 );

	//rearrange image data to be easier to work with (should parallelize)
	float* device_InputData = 0;
	cudaMalloc( (void**) &device_InputData, sizeof(float)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*information.NumberOfDimensions );
	//float* inputTransposed = new float[information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*information.NumberOfDimensions];
	//for( int i = 0; i < information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]; i++ ){
	//	for( int j = 0; j < information.NumberOfDimensions; j++ ){
	//		int inIndex = i * information.NumberOfDimensions + j;
	//		int outIndex = j * information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2] + i;
	//		inputTransposed[outIndex] = inputData[inIndex];
	//	}
	//}
	//cudaMemcpyAsync( device_InputData, inputTransposed, sizeof(float)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );
	cudaMemcpyAsync( device_InputData, inputData, sizeof(float)*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );
	//delete[] inputTransposed;
	
	//apply the map
	dim3 grid = GetGrid(information.KohonenMapSize[0]*information.KohonenMapSize[1]);
	dim3 threads(NUMTHREADS,1,1);
	for( int voxel = 0; voxel < information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]; voxel++ ){
		
		//if we are not in the mask, ignore this voxel
		if( inputMask != 0 && inputMask[voxel] == 0 ){
			outputData[2*voxel] = -1.0f;
			outputData[2*voxel+1] = -1.0f;
			continue;
		}

		//else, process it over the entire map
		ProcessSample<<<grid, threads, 0, *stream>>>(voxel, device_InputData, device_KohonenMap, device_DistanceBufferMin, device_DistanceBufferTot, device_IndexBufferMin, device_IndexBufferTot );

		//summarize into a few key elements
		for(int i = information.KohonenMapSize[0]*information.KohonenMapSize[1] / 2; i >= NUMTHREADS; i = i/2){
			dim3 tempGrid = GetGrid(i);
			ReduceSample<<<tempGrid, threads, 0, *stream>>>(device_DistanceBufferMin, device_DistanceBufferTot, device_IndexBufferMin, device_IndexBufferTot, i );
		}
		reduceData( min(NUMTHREADS,information.KohonenMapSize[0]*information.KohonenMapSize[1]), min(NUMTHREADS,information.KohonenMapSize[0]*information.KohonenMapSize[1]), 1,
					device_DistanceBufferMin, device_DistanceBufferTot, device_IndexBufferMin, device_IndexBufferTot, device_reducedOutput, stream );
					
		//copy results back
		float results[6];
		cudaMemcpyAsync( results, device_reducedOutput, sizeof(float)*6, cudaMemcpyDeviceToHost, *stream );
		cudaStreamSynchronize(*stream);
		if( results[4] / results[3] < information.KohonenMapSize[0] &&  results[5] / results[3] < information.KohonenMapSize[1] ){
			outputData[2*voxel] = results[4] / results[3];
			outputData[2*voxel+1] = results[5] / results[3];
		}else{
			outputData[2*voxel] = results[1];
			outputData[2*voxel+1] = results[2];
		}


	}

	//remove allocated memory
	cudaFree(device_KohonenMap);
	cudaFree(device_InputData);
	cudaFree(device_DistanceBufferMin);
	cudaFree(device_DistanceBufferTot);
	cudaFree(device_IndexBufferMin);
	cudaFree(device_IndexBufferTot);
	cudaFree(device_reducedOutput);
}