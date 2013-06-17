#include "CUDA_fuzzyconnectednessfilter.h"
#include "CUDA_commonKernels.h"

#define MAX(a,b) ( a > b ? a : b )
#define MIN(a,b) ( a <= b ? a : b )
#define TNORM(a,b,n) (n == 0) ? MIN(a,b) : (n == 1) ? a*b: a*b / (2.0f - a - b + a*b);
#define SNORM(a,b,n) (n == 0) ? MAX(a,b) : (n == 1) ? a + (1.0f - a)*b: (a + b) / (1.0f + a*b);

__constant__ Fuzzy_Connectedness_Information info;

__global__ void updateConnectedness(float* connectedness, float* seed, float3* affinity){

	int idx = CUDASTDOFFSET;
	int4 iN;
	iN.x = info.VolumeSize.x;
	iN.y = info.VolumeSize.y;
	iN.z = info.VolumeSize.z;
	iN.w = info.NumObjects;
	int4 i;
	i.x = idx % iN.x;
	i.w = idx / iN.x;
	i.y = i.w % iN.y;
	i.w = i.w / iN.y;
	i.z = i.w % iN.z;
	i.w = i.w / iN.z;
	int lidx = i.x + iN.x * (i.y + iN.y*i.z);

	char t = info.tnorm;
	char s = info.snorm;

	float n = (i.x != (iN.x-1)) ? connectedness[idx+1] : 0.0f;
	float3 a = affinity[lidx];
	float uk = TNORM(a.x,n,t);
   
	n = (i.y != (iN.y-1)) ? connectedness[idx+iN.x] : 0.0f;
	n = TNORM(a.y,n,t);
	uk = SNORM(uk, n, s);
   
	n = (i.z != (iN.z-1)) ? connectedness[idx+iN.x*iN.y] : 0.0f;
	a = affinity[3*lidx+2];
	n = TNORM(a.z,n,t);
	uk = SNORM(uk, n, s);
  
	n = (i.x != 0) ? connectedness[idx-1] : 0.0f;
	a = (i.x != 0) ? affinity[lidx-1] : a;
	n = TNORM((i.x != 0) ? a.x : 0.0f, n, t);
	uk = SNORM(uk, n, s);
	
	n = (i.y != 0) ? connectedness[idx-iN.x] : 0.0f;
	a = (i.y != 0) ? affinity[lidx-iN.x] : a;
	n = TNORM((i.y != 0) ? a.y : 0.0f, n,t);
	uk = SNORM(uk, n, s);
	
	n = (i.z != 0) ? connectedness[idx-iN.x*iN.y] : 0.0f;
	a = (i.z != 0) ? affinity[lidx-iN.x*iN.y] : a;
	n = TNORM((i.z != 0) ? a.z : 0.0f, n,t);
	uk = SNORM(uk, n, s);

	//write out value for others to read
	float us = seed[idx];
	if( i.w < iN.w ) connectedness[idx] = us * (1.0f - us) * uk;
	__syncthreads();
  
}

void CUDAalgo_calculateConnectedness( float* connectedness, float* seed, float* affinity, int numIterations,
	Fuzzy_Connectedness_Information& information, cudaStream_t* stream ){

	//load parameter information into constant memory
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Fuzzy_Connectedness_Information), 0, cudaMemcpyHostToDevice, *stream);

	//allocate affinity buffers
	float3* dev_affinity = 0;
	cudaMalloc( (void**) &dev_affinity,  sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z*3 );
	cudaMemcpyAsync( (void*) dev_affinity, (void*) affinity, sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z*information.NumObjects, cudaMemcpyHostToDevice, *stream );
	
	//allocate connectedness and seed image
	float* dev_connectedness = 0;
	float* dev_seededness = 0;
	cudaMalloc( (void**) &dev_connectedness,  sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z*information.NumObjects );
	cudaMalloc( (void**) &dev_seededness,  sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z*information.NumObjects );
	cudaMemcpyAsync( (void*) dev_seededness, (void*) seed, sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z*information.NumObjects, cudaMemcpyHostToDevice, *stream );
	cudaMemcpyAsync( (void*) dev_connectedness, (void*) dev_seededness, sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z*information.NumObjects, cudaMemcpyDeviceToDevice, *stream );

	//calculate connectedness
	dim3 threads(NUMTHREADS);
	dim3 grid = GetGrid(information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z*information.NumObjects);
	for(int i = 0; i < numIterations; i++){
		updateConnectedness<<< grid, threads, 0, *stream >>>( dev_connectedness, dev_seededness, dev_affinity );
	}

	//load connectedness back into CPU storage and deallocate GPU storage
	cudaFree( dev_seededness );
	cudaFree( dev_affinity );
	cudaMemcpyAsync( (void*) connectedness, (void*) dev_connectedness, sizeof(float)*information.VolumeSize.x*information.VolumeSize.y*information.VolumeSize.z, cudaMemcpyDeviceToHost, *stream );
	cudaFree( dev_connectedness );

}