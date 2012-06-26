#include "CUDA_fuzzyconnectednessfilter.h"

#define MAX(a,b) ( a > b ? a : b )
#define MIN(a,b) ( a <= b ? a : b )
#define TNORM(a,b,n) (n == 0) ? MIN(a,b) : (n == 1) ? a*b: a*b / (2.0f - a - b + a*b);
#define SNORM(a,b,n) (n == 0) ? MAX(a,b) : (n == 1) ? a + (1.0f - a)*b: (a + b) / (1.0f + a*b);

__constant__ Fuzzy_Connectedness_Information info;

__global__ void updateConnectedness(float* connectedness, float* seed,
									float* affinityX, float* affinityY, float* affinityZ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int3 iN = info.VolumeSize;
	int3 i;
	i.x = idx % iN.x;
	i.z = idx / iN.x;
	i.y = i.z % iN.y;
	i.z = i.z / iN.y;
	char t = info.tnorm;
	char s = info.snorm;
	
	float uk = 0.0f;

	float n = connectedness[idx+1];
	float a = affinityX[idx];
	n = TNORM(a,n,t);
	uk = SNORM(uk, i.x != (iN.x-1) ? n : 0.0f, s);
   
	n = connectedness[idx+iN.x];
	a = affinityY[idx];
	n = TNORM(a,n,t);
	uk = SNORM(uk, i.y != (iN.y-1) ? n : 0.0f, s);
   
	n = connectedness[idx+iN.x*iN.y];
	a = affinityZ[idx];
	n = TNORM(a,n,t);
	uk = SNORM(uk, i.z < (iN.z-1) ? n : 0.0f, s);
  
	n = connectedness[idx-1];
	a = affinityX[idx-1];
	n = TNORM(a,n,t);
	uk = SNORM(uk, i.x != 0 ? n : 0.0f, s);
	
	n = connectedness[idx-iN.x];
	a = affinityY[idx-iN.x];
	n = TNORM(a,n,t);
	uk = SNORM(uk, i.y != 0 ? n : 0.0f, s);
	
	n = connectedness[idx-iN.x*iN.y];
	a = affinityZ[idx-iN.x*iN.y];
	n = TNORM(a,n,t);
	uk = SNORM(uk, i.z != 0 ? n : 0.0f, s);

	//write out value for others to read
	float us = seed[idx];
	if( i.z < iN.z ) connectedness[idx] = us * (1.0f - us) * uk;
	__syncthreads();
  
}

template<class T>
__global__ void calculateAffinty(T* image, float* affinityX, float* affinityY, float* affinityZ){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int3 iN = info.VolumeSize;
	float3 sp = info.Spacing;
	int3 i;
	i.x = idx % iN.x;
	i.z = idx / iN.x;
	i.y = i.z % iN.y;
	i.z = i.z / iN.y;

	//collect current value
	float imageValue = (float) image[idx];

	//calculate difference in the X direction
	float neigh = (float) image[idx+1] - imageValue;
	neigh *= info.gradientWeight * neigh;
	neigh += info.distanceWeight * sp.x;

	//output the affinity X values
	if( i.z < iN.z ) affinityX[idx] = exp( - neigh );

	//calculate weighted difference in the Y direction
	neigh = (float) image[idx+iN.x] - imageValue;
	neigh *= info.gradientWeight * neigh;
	neigh += info.distanceWeight * sp.y;

	//output the affinity Y values
	if( i.z < iN.z ) affinityY[idx] = exp( - neigh );

	//calculate difference in the Z direction
	neigh = (float) image[idx+iN.x*iN.y] - imageValue;
	neigh *= info.gradientWeight * neigh;
	neigh += info.distanceWeight * sp.z;
	
	//output the affinity Z values
	if( i.z < iN.z ) affinityZ[idx] = exp( - neigh );

}

template<class T>
void CUDAalgo_calculateConnectedness( float* connectedness, float* seed, int numIterations, T* image, int numCompo,
	Fuzzy_Connectedness_Information& information, cudaStream_t* stream ){

	//load parameter information into constant memory
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Fuzzy_Connectedness_Information), 0, cudaMemcpyHostToDevice, *stream);

	//load image into GPU
	T* dev_image = 0;
	cudaMalloc( (void**) &dev_image,  sizeof(T)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z );
	cudaMemcpyAsync( (void*) dev_image, (void*) image, sizeof(T)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z, cudaMemcpyHostToDevice, *stream );

	//allocate affinity buffers
	float* dev_affinityX = 0;
	cudaMalloc( (void**) &dev_affinityX,  sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z );
	float* dev_affinityY = 0;
	cudaMalloc( (void**) &dev_affinityY,  sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z );
	float* dev_affinityZ = 0;
	cudaMalloc( (void**) &dev_affinityZ,  sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z );

	//calculate the affinities
	int maxBlockSize = 256;
	dim3 threads( maxBlockSize, 1, 1);
	int gridSize = information.VolumeSize.x * information.VolumeSize.y * information.VolumeSize.z / maxBlockSize + ( (information.VolumeSize.x * information.VolumeSize.y * information.VolumeSize.z % maxBlockSize == 0 ) ? 0 : 1 );
	dim3 grid( gridSize, 1, 1);
	calculateAffinty<<< grid, threads, 0, *stream >>>( dev_image, dev_affinityX, dev_affinityY, dev_affinityZ );

	//deallocate image and allocate connectedness and seed image
	cudaFree( dev_image );
	float* dev_connectedness = 0;
	float* dev_seededness = 0;
	cudaMalloc( (void**) &dev_connectedness,  sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z );
	cudaMalloc( (void**) &dev_seededness,  sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z );
	cudaMemcpyAsync( (void*) dev_seededness, (void*) seed, sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z, cudaMemcpyHostToDevice, *stream );
	cudaMemcpyAsync( (void*) dev_connectedness, (void*) dev_seededness, sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z, cudaMemcpyDeviceToDevice, *stream );

	//calculate connectedness
	for(int i = 0; i < numIterations; i++){
		updateConnectedness<<< grid, threads, 0, *stream >>>( dev_connectedness, dev_seededness,
											dev_affinityX, dev_affinityY, dev_affinityZ );
	}

	//load connectedness back into CPU storage
	cudaMemcpyAsync( (void*) connectedness, (void*) dev_connectedness, sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z, cudaMemcpyDeviceToHost, *stream );
	
	//deallocate connectedness and affinities on GPU
	cudaFree( dev_connectedness );
	cudaFree( dev_seededness );
	cudaFree( dev_affinityX );
	cudaFree( dev_affinityY );
	cudaFree( dev_affinityZ );

}