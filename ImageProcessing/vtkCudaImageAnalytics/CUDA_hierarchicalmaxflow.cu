/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    CUDA_hierarchicalmaxflow.cu

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file CUDA_hierarchicalmaxflow.cu
 *
 *  @brief Implementation file with definitions of GPU kernels used predominantly in GHMF segmentation
 *      These are used only by vtkHierarchicalMaxFlowSegmentation and vtkHierarchicalMaxFlowSegmentation2.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#include "CUDA_commonKernels.h"
#include "CUDA_hierarchicalmaxflow.h"
#include "cuda.h"
#include "stdio.h"
#include "vtkCudaCommon.h"

//#define DEBUG_VTKCUDAHMF

void CUDA_GetGPUBuffers( int maxNumber, double maxPercent, float** buffer, int pad, int volSize, int* numberAcquired, double* percentAcquired )
{
  size_t freeMemory, totalMemory;
  cudaError_t nErr = cudaSuccess;
  cudaMemGetInfo(&freeMemory, &totalMemory);

  int maxAllowed = (int) ( (((double) totalMemory * maxPercent) - (double)(2*pad)) / (double) (4 * volSize) );
  maxNumber = (maxNumber > maxAllowed) ? maxAllowed : maxNumber;
  //printf("===========================================================\n");
  //printf("Free/Total(kB): %f/%f\n", (float)freeMemory/1024.0f, (float)totalMemory/1024.0f);

  while( maxNumber > 0 )
  {
    nErr = cudaMalloc((void**) buffer, sizeof(float)*(maxNumber*volSize+2*pad));
    if( nErr == cudaSuccess )
    {
      break;
    }
    maxNumber--;
  }

  cudaMemGetInfo(&freeMemory, &totalMemory);
  //printf("===========================================================\n");
  //printf("Free/Total(kB): %f/%f\n", (float)freeMemory/1024.0f, (float)totalMemory/1024.0f);

  *numberAcquired = maxNumber;
  *percentAcquired = (double) sizeof(float)*(maxNumber*volSize+2*pad) / (double) totalMemory;
}

void CUDA_ReturnGPUBuffers(float* buffer)
{
  cudaFree(buffer);
}

void CUDA_CopyBufferToCPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream)
{
  cudaMemcpyAsync( CPUBuffer, GPUBuffer, sizeof(float)*size, cudaMemcpyDeviceToHost, *stream );
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  printf( "\t\tCUDA_CopyBufferToCPU: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\t%p to %p ", GPUBuffer, CPUBuffer );
  printf( "\n" );
#endif
}

void CUDA_CopyBufferToGPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream)
{
  cudaMemcpyAsync( GPUBuffer, CPUBuffer, sizeof(float)*size, cudaMemcpyHostToDevice, *stream );
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  printf( "\t\tCUDA_CopyBufferToGPU: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\t%p to %p ", CPUBuffer, GPUBuffer );
  printf( "\n" );
#endif
}

void CUDA_zeroOutBuffer(float* GPUBuffer, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  ZeroOutBuffer<<<grid,threads,0,*stream>>>(GPUBuffer,size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_zeroOutBuffer: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

void CUDA_SetBufferToValue(float* GPUBuffer, float value, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  SetBufferToConst<<<grid,threads,0,*stream>>>(GPUBuffer,value,size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_SetBufferToValue: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

void CUDA_divideAndStoreBuffer(float* inBuffer, float* outBuffer, float number, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  MultiplyAndStoreBuffer<<<grid,threads,0,*stream>>>(inBuffer,outBuffer,1.0f/number,size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_divideAndStoreBuffer: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_FindSinkPotentialAndStore(float* workingBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float iCC, int size)
{
  int idx = CUDASTDOFFSET;
  float value = workingBuffer[idx] + incBuffer[idx] - divBuffer[idx] + labelBuffer[idx] * iCC;
  if( idx < size )
  {
    workingBuffer[idx] = value;
  }
}

void CUDA_storeSinkFlowInBuffer(float* workingBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  kern_FindSinkPotentialAndStore<<<grid,threads,0,*stream>>>(workingBuffer,incBuffer,divBuffer,labelBuffer,1.0f/CC,size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_storeSinkFlowInBuffer: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_FindSourcePotentialAndStore(float* workingBuffer, float* sinkBuffer, float* divBuffer, float* labelBuffer, float iCC, int size)
{
  int idx = CUDASTDOFFSET;
  float value = workingBuffer[idx] + sinkBuffer[idx] + divBuffer[idx] - labelBuffer[idx] * iCC;
  if( idx < size )
  {
    workingBuffer[idx] = value;
  }
}

void CUDA_storeSourceFlowInBuffer(float* workingBuffer, float* sinkBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_FindSourcePotentialAndStore<<<grid,threads,0,*stream>>>(workingBuffer,sinkBuffer,divBuffer,labelBuffer,1.0f/CC,size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_storeSourceFlowInBuffer: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_FindLeafSinkPotential(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float iCC, int size)
{
  int idx = CUDASTDOFFSET;
  float value = incBuffer[idx] - divBuffer[idx] + labelBuffer[idx] * iCC;
  if( idx < size )
  {
    sinkBuffer[idx] = value;
  }
}

void CUDA_updateLeafSinkFlow(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream )
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_FindLeafSinkPotential<<<grid,threads,0,*stream>>>(sinkBuffer,incBuffer,divBuffer,labelBuffer,1.0f/CC,size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_updateLeafSinkFlow: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_ApplyCapacity(float* sinkBuffer, float* capBuffer, int size)
{
  int idx = CUDASTDOFFSET;
  float value = sinkBuffer[idx];
  float cap = capBuffer[idx];
  value = (value < 0.0f) ? 0.0f: value;
  value = (value > cap) ? cap: value;
  if( idx < size )
  {
    sinkBuffer[idx] = value;
  }
}

void CUDA_constrainLeafSinkFlow(float* sinkBuffer, float* capBuffer, int size, cudaStream_t* stream )
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_ApplyCapacity<<<grid,threads,0,*stream>>>(sinkBuffer,capBuffer,size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_constrainLeafSinkFlow: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_UpdateLabel(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size)
{
  int idx = CUDASTDOFFSET;
  float value = labelBuffer[idx] + CC*(incBuffer[idx] - divBuffer[idx] - sinkBuffer[idx]);
  value = saturate(value);
  if( idx < size )
  {
    labelBuffer[idx] = value;
  }
}

void CUDA_updateLabel(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream )
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_UpdateLabel<<<grid,threads,0,*stream>>>(sinkBuffer,incBuffer,divBuffer,labelBuffer,CC,size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_updateLabel: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_CalcGradStep(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float stepSize, float iCC, int size)
{
  int idx = CUDASTDOFFSET;
  float value = stepSize*(sinkBuffer[idx] + divBuffer[idx] - incBuffer[idx] - labelBuffer[idx] * iCC);
  if( idx < size )
  {
    divBuffer[idx] = value;
  }
}

void CUDA_flowGradientStep(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float stepSize, float CC, int size, cudaStream_t* stream )
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_CalcGradStep<<<grid,threads,0,*stream>>>(sinkBuffer,incBuffer,divBuffer,labelBuffer,stepSize,1.0f/CC,size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_flowGradientStep: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_DescentSpatialFlow(float* allowed, float* flowX, float* flowY, float* flowZ, const int2 dims, const int size)
{
  int idx = CUDASTDOFFSET;
  int3 idxN;
  idxN.y = idx / dims.x;
  idxN.x = idx % dims.x;
  idxN.z = idxN.y / dims.y;
  idxN.y = idxN.y % dims.y;
  float currAllowed = allowed[idx];

  float xAllowed = allowed[idx-1];
  float yAllowed = allowed[idx-dims.x];
  float zAllowed = allowed[idx-dims.x*dims.y];

  float newFlowX = flowX[idx] - (currAllowed - xAllowed);
  if( idx < size )
  {
    flowX[idx] = idxN.x ? newFlowX : 0.0f;
  }

  float newFlowY = flowY[idx] - (currAllowed - yAllowed);
  if( idx < size )
  {
    flowY[idx] = idxN.y ? newFlowY : 0.0f;
  }

  float newFlowZ = flowZ[idx] - (currAllowed - zAllowed);
  if( idx < size )
  {
    flowZ[idx] = idxN.z ? newFlowZ : 0.0f;
  }
}

void CUDA_applyStep(float* divBuffer, float* flowX, float* flowY, float* flowZ, int X, int Y, int Z, int size, cudaStream_t* stream )
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  int2 vDims = {X,Y};
  kern_DescentSpatialFlow<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, vDims, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_applyStep: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_ComputeFlowMagVariSmooth(float* amount, float* flowX, float* flowY, float* flowZ, float* smooth, const float alpha, const int2 dims, const int size)
{
  int idx = CUDASTDOFFSET;
  int2 idxN;
  idxN.y = idx / dims.x;
  idxN.x = idx % dims.x;
  idxN.y = idxN.y % dims.y;

  //compute flow in X
  float AmountUp = flowX[idx];
  AmountUp *= AmountUp;
  float AmountDown = flowX[idx+1];
  AmountDown = (idxN.x != dims.x-1 ? AmountDown*AmountDown : 0.0f);
  float FlowMag = AmountUp + AmountDown;

  //compute flow in Y
  AmountUp = flowY[idx];
  AmountUp *= AmountUp;
  AmountDown = flowY[idx+dims.x];
  AmountDown = (idxN.y != dims.y-1 ? AmountDown*AmountDown : 0.0f);
  FlowMag += AmountUp + AmountDown;

  //compute flow in Z
  AmountUp = flowZ[idx];
  AmountUp *= AmountUp;
  AmountDown = flowZ[idx+dims.x*dims.y];
  AmountDown = (idx+dims.x*dims.y < size ? AmountDown*AmountDown : 0.0f);
  FlowMag += AmountUp + AmountDown;

  //adjust to be proper
  FlowMag = sqrt( 0.5f * FlowMag );

  //find the constraint on the flow
  float smoothness = alpha * smooth[idx];

  //find the multiplier and output to buffer
  float multiplier = (FlowMag > smoothness) ? smoothness / FlowMag : 1.0f;
  if( idx < size )
  {
    amount[idx] = multiplier;
  }
}

__global__ void kern_ComputeFlowMagConstSmooth(float* amount, float* flowX, float* flowY, float* flowZ, const float alpha, const int2 dims, const int size)
{
  int idx = CUDASTDOFFSET;
  int2 idxN;
  idxN.y = idx / dims.x;
  idxN.x = idx % dims.x;
  idxN.y = idxN.y % dims.y;

  //compute flow in X
  float AmountUp = flowX[idx];
  AmountUp *= AmountUp;
  float AmountDown = flowX[idx+1];
  AmountDown = (idxN.x != dims.x-1 ? AmountDown*AmountDown : 0.0f);
  float FlowMag = AmountUp + AmountDown;

  //compute flow in Y
  AmountUp = flowY[idx];
  AmountUp *= AmountUp;
  AmountDown = flowY[idx+dims.x];
  AmountDown = (idxN.y != dims.y-1 ? AmountDown*AmountDown : 0.0f);
  FlowMag += AmountUp + AmountDown;

  //compute flow in Z
  AmountUp = flowZ[idx];
  AmountUp *= AmountUp;
  AmountDown = flowZ[idx+dims.x*dims.y];
  AmountDown = (idx+dims.x*dims.y < size ? AmountDown*AmountDown : 0.0f);
  FlowMag += AmountUp + AmountDown;

  //adjust to be proper
  FlowMag = sqrt( 0.5f * FlowMag );

  //find the multiplier and output to buffer
  float multiplier = (FlowMag > alpha) ? alpha / FlowMag : 1.0f;
  if( idx < size )
  {
    amount[idx] = multiplier;
  }
}

void CUDA_computeFlowMag(float* divBuffer, float* flowX, float* flowY, float* flowZ, float* smoothnessTerm, float smoothnessConstant, int X, int Y, int Z, int size, cudaStream_t* stream )
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  int2 vDims = {X,Y};
  if(smoothnessTerm)
  {
    kern_ComputeFlowMagVariSmooth<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, smoothnessTerm, smoothnessConstant, vDims, size);
  }
  else
  {
    kern_ComputeFlowMagConstSmooth<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, smoothnessConstant, vDims, size);
  }
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_computeFlowMag: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_Project(float* div, float* flowX, float* flowY, float* flowZ, const int2 dims, const int size)
{
  int idx = CUDASTDOFFSET;
  int3 idxN;
  idxN.y = idx / dims.x;
  idxN.x = idx % dims.x;
  idxN.z = idxN.y / dims.y;
  idxN.y = idxN.y % dims.y;

  float currAllowed = div[idx];
  float xAllowed = div[idx-1];
  float yAllowed = div[idx-dims.x];
  float zAllowed = div[idx-dims.x*dims.y];

  float newFlowX = flowX[idx] * 0.5f * (currAllowed + xAllowed);
  if( idx < size )
  {
    flowX[idx] = idxN.x ? newFlowX : 0.0f;
  }

  float newFlowY = flowY[idx] * 0.5f * (currAllowed + yAllowed);
  if( idx < size )
  {
    flowY[idx] = idxN.y ? newFlowY : 0.0f;
  }

  float newFlowZ = flowZ[idx] * 0.5f * (currAllowed + zAllowed);
  if( idx < size )
  {
    flowZ[idx] = idxN.z ? newFlowZ : 0.0f;
  }
}

__global__ void kern_Divergence(float* div, float* flowX, float* flowY, float* flowZ, const int2 dims, const int size)
{
  int idx = CUDASTDOFFSET;
  int3 idxN;
  idxN.y = idx / dims.x;
  idxN.x = idx % dims.x;
  idxN.z = idxN.y / dims.y;
  idxN.y = idxN.y % dims.y;

  float xAllowed = flowX[idx+1];
  float yAllowed = flowY[idx+dims.x];
  float zAllowed = flowZ[idx+dims.x*dims.y];

  float divergence = flowX[idx]+flowY[idx]+flowZ[idx];
  divergence -= (idxN.x != dims.x-1) ? xAllowed : 0.0f;
  divergence -= (idxN.y != dims.y-1) ? yAllowed : 0.0f;
  divergence -= (idx < size-dims.x*dims.y) ? zAllowed : 0.0f;

  if( idx < size )
  {
    div[idx] = divergence;
  }
}

void CUDA_projectOntoSet(float* divBuffer, float* flowX, float* flowY, float* flowZ, int X, int Y, int Z, int size, cudaStream_t* stream )
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  int2 vDims = {X,Y};
  kern_Project<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, vDims, size);
  kern_Divergence<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, vDims, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_projectOntoSet: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

void CUDA_CopyBuffer(float* dst, float* src, int size, cudaStream_t* stream)
{
  if( dst == src )
  {
    return;
  }
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  CopyBuffers<<<grid,threads,0,*stream>>>(dst, src, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  printf( "\t\tCUDA_CopyBuffer: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\t%p to %p ", src, dst );
  printf( "\n" );
#endif
}

__global__ void kern_MinBuffers(float* b1, float* b2, int size)
{
  int idx = CUDASTDOFFSET;
  float value1 = b1[idx];
  float value2 = b2[idx];
  float minVal =  (value1 < value2) ? value1 : value2;
  if( idx < size )
  {
    b1[idx] = minVal;
  }
}

void CUDA_MinBuffer(float* dst, float* src, int size, cudaStream_t* stream)
{
  if( dst == src )
  {
    return;
  }
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_MinBuffers<<<grid,threads,0,*stream>>>(dst, src, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  printf( "\t\tCUDA_MinBuffer: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\t%p to %p ", src, dst );
  printf( "\n" );
#endif
}

__global__ void kern_Lbl(float* lbl, float* flo, float* cap, const int size)
{
  int idx = CUDASTDOFFSET;
  float value1 = cap[idx];
  float value2 = flo[idx];
  float minVal =  (value2 == value1) ? 1.0f : 0.0f;
  if( idx < size )
  {
    lbl[idx] = minVal;
  }
}

void CUDA_LblBuffer(float* lbl, float* flo, float* cap, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_Lbl<<<grid,threads,0,*stream>>>(lbl, flo, cap, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_LblBuffer: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

void CUDA_SumBuffer(float* dst, float* src, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  SumBuffers<<<grid,threads,0,*stream>>>(dst, src, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  printf( "\t\tCUDA_SumBuffer: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\t%p to %p ", src, dst );
  printf( "\n" );
#endif
}

void CUDA_SumScaledBuffer(float* dst, float* src, float scale, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  SumScaledBuffers<<<grid,threads,0,*stream>>>(dst, src, scale, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  printf( "\t\tCUDA_SumScaledBuffer: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\t%p to %p ", src, dst );
  printf( "\n" );
#endif
}

void CUDA_ShiftBuffer(float* buf, float shift, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  TranslateBuffer<<<grid,threads,0,*stream>>>(buf, 1.0f, shift, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  printf( "\t\tCUDA_ShiftBuffer: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\t%p ", buf );
  printf( "\n" );
#endif
}

__global__ void kern_DivideBuffers(float* dst, float* src, const int size)
{
  int idx = CUDASTDOFFSET;
  float value1 = src[idx];
  float value2 = dst[idx];
  float minVal =  value2 / value1;
  if( idx < size )
  {
    dst[idx] = minVal;
  }
}

void CUDA_DivBuffer(float* dst, float* src, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_DivideBuffers<<<grid,threads,0,*stream>>>(dst, src, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  printf( "\t\tCUDA_DivBuffer: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\t%p to %p ", src, dst );
  printf( "\n" );
#endif
}

__global__ void kern_ResetSinkBuffer(float* sink, float* source, float* div, float* label, float ik, float iCC, int size)
{
  int idx = CUDASTDOFFSET;
  float value = (1.0f-ik)*sink[idx] + ik*(source[idx] - div[idx] + label[idx] * iCC);
  if( idx < size )
  {
    sink[idx] = value;
  }
}

void CUDA_ResetSinkBuffer(float* sink, float* source, float* div, float* label, float ik, float iCC, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_ResetSinkBuffer<<<grid,threads,0,*stream>>>(sink, source, div, label, ik, iCC, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_ResetSinkBuffer: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_PushUpSourceFlows(float* psink, float* sink, float* source, float* div, float* label, float w, float iCC, int size)
{
  int idx = CUDASTDOFFSET;
  float value = psink[idx] + w*(sink[idx] - source[idx] + div[idx] - label[idx] * iCC);
  if( idx < size )
  {
    psink[idx] = value;
  }
}

void CUDA_PushUpSourceFlows(float* psink, float* sink, float* source, float* div, float* label, float w, float iCC, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_PushUpSourceFlows<<<grid,threads,0,*stream>>>(psink, sink, source, div, label, w, iCC, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_PushUpSourceFlows: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_Copy2Buffers(float* fIn, float* fOut1, float* fOut2, int size)
{
  int idx = CUDASTDOFFSET;
  float value = fIn[idx];
  if( idx < size )
  {
    fOut1[idx] = value;
  }
  if( idx < size )
  {
    fOut2[idx] = value;
  }
}

void CUDA_Copy2Buffers(float* fIn, float* fOut1, float* fOut2, int size, cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  kern_Copy2Buffers<<<grid,threads,0,*stream>>>(fIn, fOut1, fOut2, size);
#ifdef DEBUG_VTKCUDAHMF
  cudaThreadSynchronize();
  std::cout << "\t\tCUDA_Copy2Buffers: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}