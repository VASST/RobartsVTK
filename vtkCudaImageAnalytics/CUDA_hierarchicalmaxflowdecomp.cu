#include "CUDA_hierarchicalmaxflowdecomp.h"
#include "stdio.h"
#include "cuda.h"

#define NUMTHREADS 512
//#define DEBUG_VTKCUDAHMFD

__global__ void kern_GHMFD_MultiplyBuffers(float* storage, float* multiplicand, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value1 = storage[idx];
	float value2 = multiplicand[idx];
	value1 *= value2;
	if( idx < size ) storage[idx] = value1;
}

__global__ void kern_GHMFD_AccumBuffers(float* storage, float* addend, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value1 = storage[idx];
	float value2 = addend[idx];
	value1 += value2;
	if( idx < size ) storage[idx] = value1;
}

__global__ void kern_GHMFD_ZeroOutBuffer(float* storage, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx < size ) storage[idx] = 0.0f;
}

__global__ void kern_GHMFD_GradientBuffer(float* buffer, float* gradBuffer, int3 dims, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int3 idxN = { idx % dims.x, (idx / dims.x) % dims.y, idx / (dims.x*dims.y) };
	
	float gradMagSquared = 0.0f;
	float cur = buffer[idx];
	float XHi = buffer[idx+1];				XHi = (idxN.x == dims.x-1)	? cur: XHi;
	float XLo = buffer[idx-1];				XLo = (idxN.x == 0)			? cur: XLo;
	gradMagSquared += (XHi-XLo)*(XHi-XLo);
	float YHi = buffer[idx+dims.x];			YHi = (idxN.y == dims.y-1)	? cur: YHi;
	float YLo = buffer[idx-dims.x];			YLo = (idxN.y == 0)			? cur: YLo;
	gradMagSquared += (YHi-YLo)*(YHi-YLo);
	float ZHi = buffer[idx+dims.x*dims.y];	ZHi = (idxN.z == dims.z-1)	? cur: ZHi;
	float ZLo = buffer[idx-dims.x*dims.y];	ZLo = (idxN.z == 0)			? cur: ZLo;
	gradMagSquared += (ZHi-ZLo)*(ZHi-ZLo);

	if(idx < size) gradBuffer[idx] = 0.5 * sqrt( gradMagSquared );
}

__global__ void ReduceBySummation( float* Buffer, int spread, int size ){
	
	//collect samples
	int kOffset = blockDim.x * blockIdx.x + threadIdx.x;
	float value1 = Buffer[kOffset];
	float value2 = Buffer[kOffset+spread];

	//reduce between these two samples
	value1 += value2;

	//output results
	if( kOffset+spread < size ) Buffer[kOffset] = value1;

}

template <unsigned int blockSize>
__global__ void reduceSum(float* Buffer, unsigned int n)
{
	__shared__ float sumDist[blockSize];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;

	sumDist[tid] = 0;
	while (i < n) {
		sumDist[tid] += Buffer[i];
		sumDist[tid] += Buffer[i+blockSize];
		i += gridSize;
		__syncthreads();
	}
	
	if (blockSize >= 512) { if (tid < 256) {
			sumDist[tid] += sumDist[tid + 256];
	} __syncthreads(); }

	if (blockSize >= 256) { if (tid < 128) {
			sumDist[tid] += sumDist[tid + 128];
	} __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) {
			sumDist[tid] += sumDist[tid + 64];
	} __syncthreads(); }
	
	if (tid < 32) {
		if (blockSize >= 64){
			sumDist[tid] += sumDist[tid + 32];
			__syncthreads();
		}
		if (blockSize >= 32){
			sumDist[tid] += sumDist[tid + 16];
			__syncthreads();
		}
		if (blockSize >= 16){
			sumDist[tid] += sumDist[tid + 8];
			__syncthreads();
		}
		if (blockSize >=  8){
			sumDist[tid] += sumDist[tid + 4];
			__syncthreads();
		}
		if (blockSize >=  4){
			sumDist[tid] += sumDist[tid + 2];
			__syncthreads();
		}
		if (blockSize >=  2){
			sumDist[tid] += sumDist[tid + 1];
			__syncthreads();
		}
	}

	if (tid == 0) Buffer[blockIdx.x] = sumDist[0];
}

void reduceData(int size, int threads, int blocks, float* Buffer, cudaStream_t* stream ){

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * (sizeof(float)+sizeof(short2)) : threads * (sizeof(float)+sizeof(short2));
	switch (threads)
	{
	case 512:
		reduceSum<512><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	case 256:
		reduceSum<256><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	case 128:
		reduceSum<128><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	case 64:
		reduceSum< 64><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	case 32:
		reduceSum< 32><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	case 16:
		reduceSum< 16><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	case 8:
		reduceSum< 8><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	case 4:
		reduceSum< 4><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	case 2:
		reduceSum< 2><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	case 1:
		reduceSum< 1><<< dimGrid, dimBlock, smemSize, *stream >>>(Buffer, size); break;
	}

}

double CUDA_GHMFD_DataTermForLabel(float* data, float* label, int size, cudaStream_t* stream){
	//allocate GPU buffers
	float* devDataTermBuffer = 0;
	float* devLabelBuffer = 0;
	cudaMalloc( &devDataTermBuffer, sizeof(float)*size );
	cudaMalloc( &devLabelBuffer, sizeof(float)*size );
	cudaMemcpyAsync( devDataTermBuffer, data,  sizeof(float)*size, cudaMemcpyHostToDevice, *stream );
	cudaMemcpyAsync( devLabelBuffer,    label, sizeof(float)*size, cudaMemcpyHostToDevice, *stream );
	float retVal = 0.0f;

	//multiply buffers
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_GHMFD_MultiplyBuffers<<<grid,threads,0,*stream>>>(devDataTermBuffer, devLabelBuffer, size);

	//reduce buffer by summation
	int i = 1;
	while(i < size/2) i+=i;
	for(; i > NUMTHREADS; i = i/2){
		dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
		ReduceBySummation<<<tempGrid, threads, 0, *stream>>>(devDataTermBuffer, i, size );

		#ifdef DEBUG_VTKCUDAHMFD
			cudaThreadSynchronize();
			printf( "Reduce: " );
			printf( cudaGetErrorString( cudaGetLastError() ) );
			printf( "\n" );
		#endif

	}
	reduceData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devDataTermBuffer, stream );

	//return result to CPU
	cudaMemcpyAsync( &retVal, devDataTermBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);

	//deallocate GPU buffers
	cudaFree( devDataTermBuffer );
	cudaFree( devLabelBuffer );

	//return to main class
	return (double) retVal;
}


double CUDA_GHMFD_LeafSmoothnessForLabel(float* smoothness, float* label, int x, int y, int z, int size, float* GPUParentLabel, cudaStream_t* stream){
	//allocate GPU buffers
	float* devSmoothnessBuffer = 0;
	float* devLabelBuffer = 0;
	float* devGradBuffer = 0;
	cudaMalloc( &devSmoothnessBuffer, sizeof(float)*size );
	cudaMalloc( &devLabelBuffer, sizeof(float)*size );
	cudaMalloc( &devGradBuffer, sizeof(float)*size );
	cudaMemcpyAsync( devSmoothnessBuffer, smoothness,	sizeof(float)*size, cudaMemcpyHostToDevice, *stream );
	cudaMemcpyAsync( devLabelBuffer,	  label,		sizeof(float)*size, cudaMemcpyHostToDevice, *stream );

	//find gradient
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	int3 dims = {x,y,z};
	kern_GHMFD_GradientBuffer<<<grid,threads,0,*stream>>>(devLabelBuffer, devGradBuffer, dims, size);

	//multiply buffers
	kern_GHMFD_MultiplyBuffers<<<grid,threads,0,*stream>>>(devSmoothnessBuffer, devGradBuffer, size);

	//reduce buffer by summation
	int i = 1;
	while(i < size/2) i+=i;
	for(; i > NUMTHREADS; i = i/2){
		dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
		ReduceBySummation<<<tempGrid, threads, 0, *stream>>>(devSmoothnessBuffer, i, size );
	}
	reduceData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devSmoothnessBuffer, stream );

	//return result to CPU
	float retVal = 0.0f;
	cudaMemcpyAsync( &retVal, devSmoothnessBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);

	//add to parent buffer if present
	if( GPUParentLabel )
		kern_GHMFD_AccumBuffers<<<grid,threads,0,*stream>>>(GPUParentLabel, devLabelBuffer, size);
		
	//deallocate GPU buffers
	cudaFree( devSmoothnessBuffer );
	cudaFree( devLabelBuffer );
	cudaFree( devGradBuffer );

	//return to main class
	return (double) retVal;

}

double CUDA_GHMFD_LeafNoSmoothnessForLabel(float* label, int x, int y, int z, int size, float* GPUParentLabel, cudaStream_t* stream){
	//allocate GPU buffers
	float* devLabelBuffer = 0;
	float* devGradBuffer = 0;
	cudaMalloc( &devLabelBuffer, sizeof(float)*size );
	cudaMalloc( &devGradBuffer, sizeof(float)*size );
	cudaMemcpyAsync( devLabelBuffer,	  label,		sizeof(float)*size, cudaMemcpyHostToDevice, *stream );
	
	//find gradient
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	int3 dims = {x,y,z};
	kern_GHMFD_GradientBuffer<<<grid,threads,0,*stream>>>(devLabelBuffer, devGradBuffer, dims, size);

	//reduce buffer by summation
	int i = 1;
	while(i < size/2) i+=i;
	for(; i > NUMTHREADS; i = i/2){
		dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
		ReduceBySummation<<<tempGrid, threads, 0, *stream>>>(devGradBuffer, i, size );
	}
	reduceData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devGradBuffer, stream );

	//return result to CPU
	float retVal = 0.0f;
	cudaMemcpyAsync( &retVal, devGradBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);
	
	//add to parent buffer if present
	if( GPUParentLabel )
		kern_GHMFD_AccumBuffers<<<grid,threads,0,*stream>>>(GPUParentLabel, devLabelBuffer, size);

	//deallocate GPU buffers
	cudaFree( devLabelBuffer );
	cudaFree( devGradBuffer );

	//return to main class
	return (double) retVal;

}


double CUDA_GHMFD_BranchSmoothnessForLabel(float* smoothness, float* devLabelBuffer, int x, int y, int z, int size, float* GPUParentLabel, cudaStream_t* stream){
	//allocate GPU buffers
	float* devSmoothnessBuffer = 0;
	float* devGradBuffer = 0;
	cudaMalloc( &devSmoothnessBuffer, sizeof(float)*size );
	cudaMalloc( &devGradBuffer, sizeof(float)*size );
	cudaMemcpyAsync( devSmoothnessBuffer, smoothness,	sizeof(float)*size, cudaMemcpyHostToDevice, *stream );

	//find gradient
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	int3 dims = {x,y,z};
	kern_GHMFD_GradientBuffer<<<grid,threads,0,*stream>>>(devLabelBuffer, devGradBuffer, dims, size);

	//multiply buffers
	kern_GHMFD_MultiplyBuffers<<<grid,threads,0,*stream>>>(devSmoothnessBuffer, devGradBuffer, size);

	//reduce buffer by summation
	int i = 1;
	while(i < size/2) i+=i;
	for(; i > NUMTHREADS; i = i/2){
		dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
		ReduceBySummation<<<tempGrid, threads, 0, *stream>>>(devSmoothnessBuffer, i, size );
	}
	reduceData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devSmoothnessBuffer, stream );

	//return result to CPU
	float retVal = 0.0f;
	cudaMemcpyAsync( &retVal, devSmoothnessBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);

	//add to parent buffer if present
	if( GPUParentLabel )
		kern_GHMFD_AccumBuffers<<<grid,threads,0,*stream>>>(GPUParentLabel, devLabelBuffer, size);
		
	//deallocate GPU buffers
	cudaFree( devSmoothnessBuffer );
	cudaFree( devGradBuffer );

	//return to main class
	return (double) retVal;

}

double CUDA_GHMFD_BranchNoSmoothnessForLabel(float* devLabelBuffer, int x, int y, int z, int size, float* GPUParentLabel, cudaStream_t* stream){
	//allocate GPU buffers
	float* devGradBuffer = 0;
	cudaMalloc( &devGradBuffer, sizeof(float)*size );
	
	//find gradient
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	int3 dims = {x,y,z};
	kern_GHMFD_GradientBuffer<<<grid,threads,0,*stream>>>(devLabelBuffer, devGradBuffer, dims, size);

	//reduce buffer by summation
	int i = 1;
	while(i < size/2) i+=i;
	for(; i > NUMTHREADS; i = i/2){
		dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
		ReduceBySummation<<<tempGrid, threads, 0, *stream>>>(devGradBuffer, i, size );
	}
	reduceData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devGradBuffer, stream );

	//return result to CPU
	float retVal = 0.0f;
	cudaMemcpyAsync( &retVal, devGradBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);
	
	//add to parent buffer if present
	if( GPUParentLabel )
		kern_GHMFD_AccumBuffers<<<grid,threads,0,*stream>>>(GPUParentLabel, devLabelBuffer, size);

	//deallocate GPU buffers
	cudaFree( devGradBuffer );

	//return to main class
	return (double) retVal;

}

float* CUDA_GHMFD_GetBuffer(int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	float* buffer = 0;
	cudaMalloc(&buffer,size*sizeof(float));
	kern_GHMFD_ZeroOutBuffer<<<grid,threads,0,*stream>>>(buffer, size);
	return buffer;
}

void CUDA_GHMFD_ReturnBuffer(float* buffer){
	cudaFree(buffer);
}