#include "CUDA_kohonengenerator.h"
#include <float.h>
#include <stdio.h>
#include <time.h>

#define DEBUGGING

//parameters held in constant memory
__constant__ Kohonen_Generator_Information info;
__constant__ float SamplePoint[MAX_DIMENSIONALITY];

__global__ void ProcessSample(float* KohonenMap, float* DistanceBuffer, short2* IndexBuffer, int mapSizeX, int mapSizeY ){

	__shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

	//get sample co-ordinates in buffer
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadIdx.x < MAX_DIMENSIONALITY){
		SamplePointLocal[threadIdx.x] = SamplePoint[threadIdx.x];
	}
	__syncthreads();
	
	//calculate the distance
	float distance = 0.0f;
	int bufferSize = mapSizeX * mapSizeY;
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = info.Weights[i]*(KohonenMap[i*bufferSize+kOffset] - SamplePointLocal[i]);
		distance += value*value;
	}
	__syncthreads();

	DistanceBuffer[kOffset] = distance;
	short2 index = {kOffset % mapSizeX, kOffset / mapSizeX };
	IndexBuffer[kOffset] = index;

}

__global__ void DoubleMapSizeInX( float* KohonenMap, float* tempStore, int currMapSizeX, int currMapSizeY ){

	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;

	//double size in X direction
	int bufferSize = currMapSizeX * currMapSizeY;
	int xIndex = kOffset % currMapSizeX;
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float valueOld = KohonenMap[i*bufferSize+kOffset];
		float valueNeighbour = KohonenMap[i*bufferSize+kOffset+1];
		float difference = (xIndex != currMapSizeX-1) ? valueNeighbour - valueOld : 0.0f;
		
		float2 outputValue = {valueOld, valueOld + 0.5f * difference};
		if( kOffset < bufferSize )((float2*) tempStore)[i*bufferSize+kOffset] = outputValue;
	}
}

__global__ void DoubleMapSizeInY( float* KohonenMap, float* tempStore, int currMapSizeX, int currMapSizeY ){
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	int bufferSize = currMapSizeX * currMapSizeY;
	
	//double size in Y direction
	int xIndex = kOffset % currMapSizeX;
	int yIndex = kOffset / currMapSizeX;
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float valueOld = tempStore[i*bufferSize+kOffset];
		float valueNeighbour = tempStore[i*bufferSize+kOffset+currMapSizeX];
		float difference = (yIndex != currMapSizeY-1) ? valueNeighbour - valueOld : 0.0f;

		if( kOffset < bufferSize ) KohonenMap[i*2*bufferSize+xIndex+currMapSizeX*2*yIndex] = valueOld;
		if( kOffset < bufferSize ) KohonenMap[i*2*bufferSize+xIndex+currMapSizeX*(2*yIndex+1)] = valueOld + 0.5f * difference;
	}

}

template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata, short2 * i_idata, short2 *i_odata, unsigned int n)
{
	__shared__ float sdata[blockSize];
	__shared__ short2 sindex[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = FLT_MAX;
	sindex[tid].x = 0;
	sindex[tid].y = 0;
	
	while (i < n) {
		if( sdata[tid] >= g_idata[i] ){
			sdata[tid] = g_idata[i];
			sindex[tid] = i_idata[i];
		}
		if( sdata[tid] >= g_idata[i+blockSize] ){
			sdata[tid] = g_idata[i+blockSize];
			sindex[tid] = i_idata[i+blockSize];
		}
		i += gridSize;
		__syncthreads();
	}
	
	if (blockSize >= 512) { if (tid < 256) {
			if( sdata[tid] >= sdata[tid + 256] ){
				sdata[tid] = sdata[tid + 256];
				sindex[tid] = sindex[tid + 256];
			}
	} __syncthreads(); }

	if (blockSize >= 256) { if (tid < 128) {
			if( sdata[tid] >= sdata[tid + 128] ){
				sdata[tid] = sdata[tid + 128];
				sindex[tid] = sindex[tid + 128];
			}
	} __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) {
			if( sdata[tid] >= sdata[tid + 64] ){
				sdata[tid] = sdata[tid + 64];
				sindex[tid] = sindex[tid + 64];
			}
	} __syncthreads(); }
	
	if (tid < 32) {
		if (blockSize >= 64){
			if( sdata[tid] >= sdata[tid + 32] ){
				sdata[tid] = sdata[tid + 32];
				sindex[tid] = sindex[tid + 32];
			}
			__syncthreads();
		}
		if (blockSize >= 32){
			if( sdata[tid] >= sdata[tid + 16] ){
				sdata[tid] = sdata[tid + 16];
				sindex[tid] = sindex[tid + 16];
			}
			__syncthreads();
		}
		if (blockSize >= 16){
			if( sdata[tid] >= sdata[tid + 8] ){
				sdata[tid] = sdata[tid + 8];
				sindex[tid] = sindex[tid + 8];
			}
			__syncthreads();
		}
		if (blockSize >=  8){
			if( sdata[tid] >= sdata[tid + 4] ){
				sdata[tid] = sdata[tid + 4];
				sindex[tid] = sindex[tid + 4];
			}
			__syncthreads();
		}
		if (blockSize >=  4){
			if( sdata[tid] >= sdata[tid + 2] ){
				sdata[tid] = sdata[tid + 2];
				sindex[tid] = sindex[tid + 2];
			}
			__syncthreads();
		}
		if (blockSize >=  2){
			if( sdata[tid] >= sdata[tid + 1] ){
				sdata[tid] = sdata[tid + 1];
				sindex[tid] = sindex[tid + 1];
			}
			__syncthreads();
		}
	}
	if (tid == 0){
		g_odata[0] = sdata[0];
		i_odata[0] = sindex[0];
	}
}

__global__ void FindMinSample( float* DistanceBuffer, short2* IndexBuffer, int spread, int mapSizeX, int mapSizeY ){
	
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	float distance1 = DistanceBuffer[kOffset];
	float distance2 = DistanceBuffer[kOffset+spread];
	short2 index1 = IndexBuffer[kOffset];
	short2 index2 = IndexBuffer[kOffset+spread];

	if( kOffset+spread < mapSizeX * mapSizeY ){
		DistanceBuffer[kOffset] = (distance1 < distance2) ? distance1 : distance2;
		IndexBuffer[kOffset] = (distance1 < distance2) ? index1 : index2;
	}

}

__global__ void UpdateWeights( float* KohonenMap, short2 minIndex, float alpha, float neigh, int mapSizeX, int mapSizeY ){

	__shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

	//get sample co-ordinates in buffer
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	short2 currIndex = {kOffset % mapSizeX, kOffset / mapSizeX };
	if(threadIdx.x < MAX_DIMENSIONALITY){
		SamplePointLocal[threadIdx.x] = SamplePoint[threadIdx.x];
	}

	
	float multiplier = alpha * exp( -((currIndex.x-minIndex.x)*(currIndex.x-minIndex.x) + (currIndex.y-minIndex.y)*(currIndex.y-minIndex.y) ) / neigh );
	int bufferSize = mapSizeX * mapSizeY;
	for(int i = 0; i < info.NumberOfDimensions; i++){
		float value = (1.0f-multiplier)*KohonenMap[i*bufferSize+kOffset] + multiplier*SamplePointLocal[i];
		//float value = SamplePointLocal[i];
		KohonenMap[i*bufferSize+kOffset] = value;
		__syncthreads();
	}

}

void getMinimum(int size, int threads, int blocks, float *d_idata, float *d_odata, short2* d_iindex, short2* d_oindex, cudaStream_t* stream ){

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * (sizeof(float)+sizeof(short2)) : threads * (sizeof(float)+sizeof(short2));
	switch (threads)
	{
	case 512:
		reduce6<512><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	case 256:
		reduce6<256><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	case 128:
		reduce6<128><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	case 64:
		reduce6< 64><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	case 32:
		reduce6< 32><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	case 16:
		reduce6< 16><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	case 8:
		reduce6< 8><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	case 4:
		reduce6< 4><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	case 2:
		reduce6< 2><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	case 1:
		reduce6< 1><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size); break;
	}

}

#define NUMTHREADS 512

void CUDAalgo_generateKohonenMap(	float** inputData, float* outputKohonen, char** maskData, double* range,
									int* VolumeSize, int NumVolumes,
									Kohonen_Generator_Information& information,
									int MaxEpochs, int BatchSize,
									float alphaVMult, float alphaVShift, float alphaHMult, float alphaHShift,
									float nVMult, float nVShift, float nHMult, float nHShift,
									cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Generator_Information) );


	//find minimum starting size
	float neighbourhood = nVShift + nVMult / (1 + exp( nHShift ) );
	int currentMapSize[3] = {2, 2, 1};
	while( neighbourhood * (double) currentMapSize[0] < 2.0 && currentMapSize[0] < information.KohonenMapSize[0] ) currentMapSize[0] += currentMapSize[0];
	if( currentMapSize[0] > information.KohonenMapSize[0] ) currentMapSize[0] = information.KohonenMapSize[0];
	while( neighbourhood * (double) currentMapSize[1] < 2.0 && currentMapSize[1] < information.KohonenMapSize[1] ) currentMapSize[1] += currentMapSize[1];
	if( currentMapSize[1] > information.KohonenMapSize[1] ) currentMapSize[1] = information.KohonenMapSize[1];

	//allocate a distance buffer
	float* device_DistanceBuffer1 = 0;
	cudaMalloc( (void**) &device_DistanceBuffer1, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );
	short2* device_IndexBuffer1 = 0;
	cudaMalloc( (void**) &device_IndexBuffer1, sizeof(short2)*information.KohonenMapSize[0]*information.KohonenMapSize[1] );
	
	//create buffer for the Kohonen map
	float* device_KohonenMap = 0;
	cudaMalloc( (void**) &device_KohonenMap, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions );
	float* device_tempSpace = 0;
	cudaMalloc( (void**) &device_tempSpace, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions );
	int c = 0;
	for(int j = 0; j < information.NumberOfDimensions; j++ )
		for(int i = 0; i < currentMapSize[0]*currentMapSize[1]; i++)
			outputKohonen[c++] = (float)( (range[2*j+1]-range[2*j]) * ((double) rand() / (double)RAND_MAX) + range[2*j] );
	cudaMemcpyAsync(device_KohonenMap, outputKohonen, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );

	//train kohonen map
	dim3 grid((currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);
	dim3 threads(NUMTHREADS, 1, 1);
	if( BatchSize == -1 ){
	
		BatchSize = 0;
		for( int p = 0; p < NumVolumes; p++ )
			for( int sampleOffset = 0; sampleOffset < VolumeSize[3*p]*VolumeSize[3*p+1]*VolumeSize[3*p+2]; sampleOffset++)
				if( maskData && (maskData[p])[sampleOffset] != 0 ) BatchSize++;

		float alpha = (alphaVShift + alphaVMult / (1 + exp( alphaHShift ) ));

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
					ProcessSample<<<grid, threads, 0, *stream>>>(device_KohonenMap, device_DistanceBuffer1, device_IndexBuffer1, currentMapSize[0], currentMapSize[1]);

					
					//update the weights of each centroid
					short2 minIndex = {-1,-1};
					float distance = -1.0f;
					cudaStreamSynchronize(*stream);
					cudaMemcpy( &minIndex, device_IndexBuffer1, sizeof(short2), cudaMemcpyDeviceToHost );
					cudaMemcpy( &distance, device_DistanceBuffer1, sizeof(float), cudaMemcpyDeviceToHost );

					//find the winning centroid
					for(int i = currentMapSize[0]*currentMapSize[1] / 2; i > NUMTHREADS; i = i/2){
						dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
						FindMinSample<<<tempGrid, threads, 0, *stream>>>(device_DistanceBuffer1, device_IndexBuffer1, i, currentMapSize[0], currentMapSize[1]);
					}
					getMinimum( min(NUMTHREADS,currentMapSize[0]*currentMapSize[1]), min(NUMTHREADS,currentMapSize[0]*currentMapSize[1]), 1,
								device_DistanceBuffer1, device_DistanceBuffer1, device_IndexBuffer1, device_IndexBuffer1, stream );

					//update the weights of each centroid
					cudaStreamSynchronize(*stream);
					cudaMemcpy( &minIndex, device_IndexBuffer1, sizeof(short2), cudaMemcpyDeviceToHost );
					cudaMemcpy( &distance, device_DistanceBuffer1, sizeof(float), cudaMemcpyDeviceToHost );
					//printf("%d, %d, %f\n", minIndex.x,  minIndex.y, distance);
					UpdateWeights<<<grid, threads, 0, *stream>>>(device_KohonenMap, minIndex, alpha, neighbourhood*sqrt((float)(currentMapSize[0]*currentMapSize[0]+currentMapSize[1]*currentMapSize[1])), currentMapSize[0], currentMapSize[1]);
				}
			}

			#ifdef DEBUGGING
				printf("Finished epoch %d with parameters (a,n) = (%f,%f) at %d\n", epoch, alpha, neighbourhood, (int) time(NULL));
			#endif

			//update the weight updaters
			alpha = (alphaVShift + alphaVMult / (1 + exp( (epoch+1) * alphaHMult + alphaHShift ) ) );
			neighbourhood = nVShift + nVMult / (1 + exp( (epoch+1) * nHMult + nHShift ) );
			
			if( ((neighbourhood * (double) currentMapSize[0] < 2.0 ) && (currentMapSize[0] < information.KohonenMapSize[0])) ){
				grid = dim3 ((2*currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);
				DoubleMapSizeInX<<<grid, threads, 0, *stream>>>( device_KohonenMap, device_tempSpace, currentMapSize[0], currentMapSize[1] );
				currentMapSize[0] *= 2;
				#ifdef DEBUGGING
					printf("Updating size to (%d,%d)\n", currentMapSize[0],currentMapSize[1]);
				#endif
			}

			if( ((neighbourhood * (double) currentMapSize[1] < 2.0) && (currentMapSize[1] < information.KohonenMapSize[1])) ){
				grid = dim3 ((2*currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);
				DoubleMapSizeInY<<<grid, threads, 0, *stream>>>( device_KohonenMap, device_tempSpace, currentMapSize[0], currentMapSize[1] );
				currentMapSize[1] *= 2;
				#ifdef DEBUGGING
					printf("Updating size to (%d,%d)\n", currentMapSize[0],currentMapSize[1]);
				#endif
			}
		}

	//if we are randomly sampling from the images
	}else{
	
		float alpha = (alphaVShift + alphaVMult / (1 + exp( alphaHShift ) ) );

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
				ProcessSample<<<grid, threads, 0, *stream>>>(device_KohonenMap, device_DistanceBuffer1, device_IndexBuffer1, currentMapSize[0], currentMapSize[1]);
				
				//find the winning centroid
				for(int i = currentMapSize[0]*currentMapSize[1] / 2; i > NUMTHREADS; i = i/2){
					dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
					FindMinSample<<<tempGrid, threads, 0, *stream>>>(device_DistanceBuffer1, device_IndexBuffer1, i, currentMapSize[0], currentMapSize[1]);
				}
				getMinimum( min(NUMTHREADS,currentMapSize[0]*currentMapSize[1]), min(NUMTHREADS,currentMapSize[0]*currentMapSize[1]), 1,
							device_DistanceBuffer1, device_DistanceBuffer1, device_IndexBuffer1, device_IndexBuffer1, stream );

				//update the weights of each centroid
				short2* minIndex = new short2[currentMapSize[0]*currentMapSize[1]];
				cudaMemcpyAsync( &minIndex, device_IndexBuffer1, sizeof(short2)*currentMapSize[0]*currentMapSize[1], cudaMemcpyDeviceToHost, *stream );
				cudaStreamSynchronize(*stream);
				UpdateWeights<<<grid, threads, 0, *stream>>>(device_KohonenMap, minIndex[0], alpha, neighbourhood*sqrt((float)(currentMapSize[0]*currentMapSize[0]+currentMapSize[1]*currentMapSize[1])), currentMapSize[0], currentMapSize[1]);
			
			}

			#ifdef DEBUGGING
				printf("Finished epoch %d with parameters (a,n) = (%f,%f) at %d\n", epoch, alpha, neighbourhood, (int) time(NULL));
			#endif

			//update the weight updaters
			alpha = (alphaVShift + alphaVMult / (1 + exp( epoch * alphaHShift + alphaHShift ) ) );
			neighbourhood = nVShift + nVMult / (1 + exp( epoch * nHShift + nHShift ) );
			
			if( ((neighbourhood * (double) currentMapSize[0] < 2.0 ) && (currentMapSize[0] < information.KohonenMapSize[0])) ){
				grid = dim3 ((2*currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);
				DoubleMapSizeInX<<<grid, threads, 0, *stream>>>( device_KohonenMap, device_tempSpace, currentMapSize[0], currentMapSize[1] );
				currentMapSize[0] *= 2;
				#ifdef DEBUGGING
					printf("Updating size to (%d,%d)\n", currentMapSize[0],currentMapSize[1]);
				#endif
			}

			if( ((neighbourhood * (double) currentMapSize[1] < 2.0) && (currentMapSize[1] < information.KohonenMapSize[1])) ){
				grid = dim3 ((2*currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);
				DoubleMapSizeInY<<<grid, threads, 0, *stream>>>( device_KohonenMap, device_tempSpace, currentMapSize[0], currentMapSize[1] );
				currentMapSize[1] *= 2;
				#ifdef DEBUGGING
					printf("Updating size to (%d,%d)\n", currentMapSize[0],currentMapSize[1]);
				#endif
			}

		}

	}

	//remove distance buffer
	cudaFree(device_DistanceBuffer1);
	cudaFree(device_IndexBuffer1);

	//translate back data
	float* tempKohonen = new float[information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions];
	cudaMemcpyAsync( tempKohonen, device_KohonenMap, sizeof(float)*information.KohonenMapSize[0]*information.KohonenMapSize[1]*information.NumberOfDimensions, cudaMemcpyDeviceToHost, *stream );
	cudaFree(device_KohonenMap);
	cudaFree(device_tempSpace);
	cudaStreamSynchronize(*stream);

	int bufferJump = information.KohonenMapSize[0]*information.KohonenMapSize[1];
	for(int i = 0; i < information.KohonenMapSize[0]*information.KohonenMapSize[1]; i++)
		for( int j = 0; j < information.NumberOfDimensions; j++ )
			outputKohonen[i*information.NumberOfDimensions+j] = tempKohonen[j*bufferJump+i];
	delete[] tempKohonen;

}