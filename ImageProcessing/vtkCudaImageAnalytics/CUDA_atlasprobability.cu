/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    CUDA_atlasprobability.cu

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file CUDA_atlasprobability.cu
 *
 *  @brief Implementation file with definitions of GPU kernels used for the 'atlas probability'
 *      prior.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#include "CUDA_atlasprobability.h"
#include "CUDA_commonKernels.h"
#include "cuda.h"
#include "stdio.h"
#include "vtkCudaCommon.h"

template void CUDA_IncrementInformation<float>(float* labelData, float desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<double>(double* labelData, double desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<long>(long* labelData, long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned long>(unsigned long* labelData, unsigned long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<long long>(long long* labelData, long long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned long long>(unsigned long long* labelData, unsigned long long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<int>(int* labelData, int desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned int>(unsigned int* labelData, unsigned int desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<short>(short* labelData, short desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned short>(unsigned short* labelData, unsigned short desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<char>(char* labelData, char desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<signed char>(signed char* labelData, signed char desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_IncrementInformation<unsigned char>(unsigned char* labelData, unsigned char desiredValue, short* agreement, int size, cudaStream_t* stream);

template< class T >
void CUDA_IncrementInformation(T* labelData, T desiredValue, short* agreement, int size, cudaStream_t* stream)
{
  T* GPUBuffer = 0;

  cudaMalloc((void**) &GPUBuffer, sizeof(T)*size);
  cudaMemcpyAsync( GPUBuffer, labelData, sizeof(T)*size, cudaMemcpyHostToDevice, *stream );

#ifdef DEBUG_VTKCUDA_IALP
  cudaThreadSynchronize();
  std::cout << "CUDA_IncrementInformation: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif

  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);
  IncrementBuffer<T><<<grid,threads,0,*stream>>>(GPUBuffer, desiredValue, agreement, size);
  cudaFree(GPUBuffer);

#ifdef DEBUG_VTKCUDA_IALP
  cudaThreadSynchronize();
  std::cout << "CUDA_IncrementInformation: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

void CUDA_GetRelevantBuffers(short** agreement, float** output, int size, cudaStream_t* stream)
{
  cudaMalloc((void**) agreement, sizeof(short)*size);
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
  ZeroOutBuffer<<<grid,threads,0,*stream>>>(*agreement,size);
  cudaMalloc((void**) output, sizeof(float)*size);

#ifdef DEBUG_VTKCUDA_IALP
  cudaThreadSynchronize();
  std::cout << "CUDA_GetRelevantBuffers: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

void CUDA_CopyBackResult(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream)
{
  cudaThreadSynchronize();
  cudaMemcpy( CPUBuffer, GPUBuffer, sizeof(float)*size, cudaMemcpyDeviceToHost );
  cudaFree(GPUBuffer);

#ifdef DEBUG_VTKCUDA_IALP
  cudaThreadSynchronize();
  std::cout << "CUDA_CopyBackResult: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}

__global__ void kern_ConvertBuffer(short* agreement, float* output, int size )
{
  int idx = CUDASTDOFFSET;
  float locAgreement = (float) agreement[idx];
  if( idx < size )
  {
    output[idx] = locAgreement;
  }
}

__global__ void kern_LogBuffer(float* agreement, float* output, float maxOut, int size, short max)
{
  int idx = CUDASTDOFFSET;
  float locAgreement = (float) agreement[idx];
  float logValue = (locAgreement > 0.0f) ? log((float)max)-log(locAgreement): maxOut;
  logValue = (logValue > 0.0f) ? logValue : 0.0f;
  logValue = (logValue < maxOut) ? logValue: maxOut;
  if( idx < size )
  {
    output[idx] = logValue;
  }
}

__global__ void kern_NormLogBuffer(float* agreement, float* output, float maxOut, int size, short max)
{
  int idx = CUDASTDOFFSET;
  float locAgreement = (float) agreement[idx];
  float logValue = (locAgreement > 0.0f) ? log((float)max)-log(locAgreement): maxOut;
  logValue = (logValue > 0.0f) ? logValue : 0.0f;
  logValue = (logValue < maxOut) ? logValue / maxOut: 1.0f;
  if( idx < size )
  {
    output[idx] = logValue;
  }
}

__global__ void kern_ProbBuffer(float* agreement, float* output, int size, short max)
{
  int idx = CUDASTDOFFSET;
  float locAgreement = agreement[idx];
  float probValue = (float) locAgreement / (float) max;
  probValue = (probValue < 1.0f) ? probValue: 1.0f;
  if( idx < size )
  {
    output[idx] = probValue;
  }
}

__global__ void kern_BlurBuffer(float* input, float* output, int size, int spread, int dim)
{
  int idx = CUDASTDOFFSET;
  int x = (idx / spread) % dim;
  float curr = input[idx];
  float down = (idx-spread >= 0)   ? input[idx-spread] : 0;
  float up   = (idx+spread < size) ? input[idx+spread] : 0;
  float newVal = 0.7865707f * curr + 0.1064508f * ((x > 0 ? down : curr) + (x < dim-1 ? up : curr));
  __syncthreads();
  if( idx < size )
  {
    output[idx] = newVal;
  }
}

void CUDA_ConvertInformation(short* agreement, float* output, float maxOut, int size, short max, short flags, int gaussWidth[], int imageDims[], cudaStream_t* stream)
{
  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);

  //Gaussian smooth results
  float* floatAgreement = 0;
  cudaMalloc( &floatAgreement, sizeof(float)*(size+2*imageDims[0]*imageDims[1]) );
  float* floatAgreementUsed = floatAgreement + imageDims[0]*imageDims[1];

  kern_ConvertBuffer<<<grid,threads,0,*stream>>>(agreement, floatAgreementUsed, size);


  if( flags & 4 )
  {
    while( gaussWidth[0] > 0 || gaussWidth[1] > 0 || gaussWidth[2] > 0 )
    {
      if( gaussWidth[0] > 0 )
      {
        kern_BlurBuffer<<<grid,threads,0,*stream>>>(floatAgreementUsed, floatAgreementUsed, size, 1, imageDims[0] );
        gaussWidth[0]--;
        cudaThreadSynchronize();
        // check for error
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
          printf("CUDA error: %s\n", cudaGetErrorString(error));
          //exit(-1);
        }

      }
      if( gaussWidth[1] > 0 )
      {
        kern_BlurBuffer<<<grid,threads,0,*stream>>>(floatAgreementUsed, floatAgreementUsed, size, imageDims[0], imageDims[1] );
        gaussWidth[1]--;
        cudaThreadSynchronize();
        // check for error
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
          printf("CUDA error: %s\n", cudaGetErrorString(error));
          //exit(-1);
        }
      }
      if( gaussWidth[2] > 0 )
      {
        kern_BlurBuffer<<<grid,threads,0,*stream>>>(floatAgreementUsed, floatAgreementUsed, size, imageDims[0]*imageDims[1], imageDims[2] );
        gaussWidth[2]--;
        cudaThreadSynchronize();
        // check for error
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
          printf("CUDA error: %s\n", cudaGetErrorString(error));
          //exit(-1);
        }
      }
    }
  }

  if( flags & 1 )
    if( flags & 2)
    {
      kern_NormLogBuffer<<<grid,threads,0,*stream>>>(floatAgreementUsed, output, maxOut, size, max);
    }
    else
    {
      kern_LogBuffer<<<grid,threads,0,*stream>>>(floatAgreementUsed, output, maxOut, size, max);
    }
  else
  {
    kern_ProbBuffer<<<grid,threads,0,*stream>>>(floatAgreementUsed, output, size, max);
  }

  cudaFree(agreement);
  cudaFree(floatAgreement);

#ifdef DEBUG_VTKCUDA_IALP
  cudaThreadSynchronize();
  std::cout << "CUDA_ConvertInformation: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif
}