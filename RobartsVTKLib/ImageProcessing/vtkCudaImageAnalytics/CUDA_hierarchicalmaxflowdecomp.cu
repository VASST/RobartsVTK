#include "CUDA_hierarchicalmaxflowdecomp.h"
#include "CUDA_commonKernels.h"
#include "stdio.h"
#include "cuda.h"

//#define DEBUG_VTKCUDAHMFD

__global__ void kern_GHMFD_GradientBuffer(float* buffer, float* gradBuffer, int3 dims, int size){
  int idx = CUDASTDOFFSET;
  int3 idxN = { idx % dims.x, (idx / dims.x) % dims.y, idx / (dims.x*dims.y) };
  
  float gradMagSquared = 0.0f;
  float cur = buffer[idx];
  float XHi = (idxN.x == dims.x-1)  ? cur: buffer[idx+1];
  float XLo = (idxN.x == 0)      ? cur: buffer[idx-1];        
  gradMagSquared += (XHi-XLo)*(XHi-XLo);
  float YHi = (idxN.y == dims.y-1)  ? cur: buffer[idx+dims.x];
  float YLo = (idxN.y == 0)      ? cur: buffer[idx-dims.x];
  gradMagSquared += (YHi-YLo)*(YHi-YLo);
  float ZHi = (idxN.z == dims.z-1)  ? cur: buffer[idx+dims.x*dims.y];
  float ZLo = (idxN.z == 0)      ? cur: buffer[idx-dims.x*dims.y];
  gradMagSquared += (ZHi-ZLo)*(ZHi-ZLo);

  if(idx < size) gradBuffer[idx] = 0.5f * sqrt( gradMagSquared );
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
  dim3 grid = GetGrid(size);
  MultiplyBuffers<<<grid,threads,0,*stream>>>(devDataTermBuffer, devLabelBuffer, size);

  //reduce buffer by summation
  int i = 1;
  while(i < size/2) i+=i;
  for(; i >= NUMTHREADS; i = i/2){
    dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
    SumOverLargeBuffer<<<tempGrid, threads, 0, *stream>>>(devDataTermBuffer, i, size );

    #ifdef DEBUG_VTKCUDAHMFD
      cudaThreadSynchronize();
      printf( "Reduce: " );
      printf( cudaGetErrorString( cudaGetLastError() ) );
      printf( "\n" );
    #endif

  }
  SumData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devDataTermBuffer, stream );

  //return result to CPU
  cudaMemcpyAsync( &retVal, devDataTermBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
  cudaStreamSynchronize(*stream);

  //deallocate GPU buffers
  cudaFree( devDataTermBuffer );
  cudaFree( devLabelBuffer );

  //return to main class
  return (double) retVal;
}


double CUDA_GHMFD_LeafSmoothnessForLabel(float* smoothness, float* label, int x, int y, int z, int size, float* GPUParentLabel, float* devGradBuffer, cudaStream_t* stream){
  //allocate GPU buffers
  float* devSmoothnessBuffer = 0;
  float* devLabelBuffer = 0;
  cudaMalloc( &devSmoothnessBuffer, sizeof(float)*size );
  cudaMalloc( &devLabelBuffer, sizeof(float)*size );
  cudaMemcpyAsync( devSmoothnessBuffer, smoothness,  sizeof(float)*size, cudaMemcpyHostToDevice, *stream );
  cudaMemcpyAsync( devLabelBuffer,    label,    sizeof(float)*size, cudaMemcpyHostToDevice, *stream );

  //find gradient
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  int3 dims = {x,y,z};
  kern_GHMFD_GradientBuffer<<<grid,threads,0,*stream>>>(devLabelBuffer, devGradBuffer, dims, size);

  //multiply buffers
  MultiplyBuffers<<<grid,threads,0,*stream>>>(devSmoothnessBuffer, devGradBuffer, size);

  //reduce buffer by summation
  int i = 1;
  while(i < size/2) i+=i;
  for(; i >= NUMTHREADS; i = i/2){
    dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
    SumOverLargeBuffer<<<tempGrid, threads, 0, *stream>>>(devSmoothnessBuffer, i, size );
  }
  SumData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devSmoothnessBuffer, stream );

  //return result to CPU
  float retVal = 0.0f;
  cudaMemcpyAsync( &retVal, devSmoothnessBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
  cudaStreamSynchronize(*stream);

  //add to parent buffer if present
  if( GPUParentLabel )
    SumBuffers<<<grid,threads,0,*stream>>>(GPUParentLabel, devLabelBuffer, size);
    
  //deallocate GPU buffers
  cudaFree( devSmoothnessBuffer );
  cudaFree( devLabelBuffer );

  //return to main class
  return (double) retVal;

}

double CUDA_GHMFD_LeafNoSmoothnessForLabel(float* label, int x, int y, int z, int size, float* GPUParentLabel, float* devGradBuffer, cudaStream_t* stream){
  //allocate GPU buffers
  float* devLabelBuffer = 0;
  cudaMalloc( &devLabelBuffer, sizeof(float)*size );
  cudaMemcpyAsync( devLabelBuffer,    label,    sizeof(float)*size, cudaMemcpyHostToDevice, *stream );
  
  //find gradient
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  int3 dims = {x,y,z};
  kern_GHMFD_GradientBuffer<<<grid,threads,0,*stream>>>(devLabelBuffer, devGradBuffer, dims, size);

  //reduce buffer by summation
  int i = 1;
  while(i < size/2) i+=i;
  for(; i >= NUMTHREADS; i = i/2){
    dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
    SumOverLargeBuffer<<<tempGrid, threads, 0, *stream>>>(devGradBuffer, i, size );
  }
  SumData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devGradBuffer, stream );

  //return result to CPU
  float retVal = 0.0f;
  cudaMemcpyAsync( &retVal, devGradBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
  cudaStreamSynchronize(*stream);
  
  //add to parent buffer if present
  if( GPUParentLabel )
    SumBuffers<<<grid,threads,0,*stream>>>(GPUParentLabel, devLabelBuffer, size);

  //deallocate GPU buffers
  cudaFree( devLabelBuffer );

  //return to main class
  return (double) retVal;

}


double CUDA_GHMFD_BranchSmoothnessForLabel(float* smoothness, float* devLabelBuffer, int x, int y, int z, int size, float* GPUParentLabel, float* devGradBuffer, cudaStream_t* stream){
  //allocate GPU buffers
  float* devSmoothnessBuffer = 0;
  cudaMalloc( &devSmoothnessBuffer, sizeof(float)*size );
  cudaMemcpyAsync( devSmoothnessBuffer, smoothness,  sizeof(float)*size, cudaMemcpyHostToDevice, *stream );

  //find gradient
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  int3 dims = {x,y,z};
  kern_GHMFD_GradientBuffer<<<grid,threads,0,*stream>>>(devLabelBuffer, devGradBuffer, dims, size);

  //multiply buffers
  MultiplyBuffers<<<grid,threads,0,*stream>>>(devSmoothnessBuffer, devGradBuffer, size);

  //reduce buffer by summation
  int i = 1;
  while(i < size/2) i+=i;
  for(; i >= NUMTHREADS; i = i/2){
    dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
    SumOverLargeBuffer<<<tempGrid, threads, 0, *stream>>>(devSmoothnessBuffer, i, size );
  }
  SumData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devSmoothnessBuffer, stream );

  //return result to CPU
  float retVal = 0.0f;
  cudaMemcpyAsync( &retVal, devSmoothnessBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
  cudaStreamSynchronize(*stream);

  //add to parent buffer if present
  if( GPUParentLabel )
    SumBuffers<<<grid,threads,0,*stream>>>(GPUParentLabel, devLabelBuffer, size);
    
  //deallocate GPU buffers
  cudaFree( devSmoothnessBuffer );

  //return to main class
  return (double) retVal;

}

double CUDA_GHMFD_BranchNoSmoothnessForLabel(float* devLabelBuffer, int x, int y, int z, int size, float* GPUParentLabel, float* devGradBuffer, cudaStream_t* stream){
  
  //find gradient
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  int3 dims = {x,y,z};
  kern_GHMFD_GradientBuffer<<<grid,threads,0,*stream>>>(devLabelBuffer, devGradBuffer, dims, size);

  //reduce buffer by summation
  int i = 1;
  while(i < size/2) i+=i;
  for(; i >= NUMTHREADS; i = i/2){
    dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
    SumOverLargeBuffer<<<tempGrid, threads, 0, *stream>>>(devGradBuffer, i, size );
  }
  SumData( min(NUMTHREADS,size), min(NUMTHREADS,size), 1, devGradBuffer, stream );

  //return result to CPU
  float retVal = 0.0f;
  cudaMemcpyAsync( &retVal, devGradBuffer,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
  cudaStreamSynchronize(*stream);
  
  //add to parent buffer if present
  if( GPUParentLabel )
    SumBuffers<<<grid,threads,0,*stream>>>(GPUParentLabel, devLabelBuffer, size);

  //return to main class
  return (double) retVal;

}

float* CUDA_GHMFD_GetBuffer(int size, cudaStream_t* stream){
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  float* buffer = 0;
  cudaMalloc(&buffer,size*sizeof(float));
  ZeroOutBuffer<<<grid,threads,0,*stream>>>(buffer, size);
  return buffer;
}

void CUDA_GHMFD_ReturnBuffer(float* buffer){
  cudaFree(buffer);
}