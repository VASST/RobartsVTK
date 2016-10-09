/*=========================================================================

  Program:   Robarts Visualization Toolkit

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "CUDA_KSOMProbability.h"
#include "CUDA_commonKernels.h"
#include "vtkCudaCommon.h"
#include <float.h>
#include <iostream>

__constant__ Kohonen_Probability_Information info;

__global__ void ProcessSample(float* InputData, float* KohonenMap, float* Accumulator)
{
  __shared__ float ComponentLocal[2*MAX_DIMENSIONALITY+1];

  //get sample co-ordinates in buffer
  int kOffset = CUDASTDOFFSET;
  if(threadIdx.x < 2*MAX_DIMENSIONALITY+1)
  {
    ComponentLocal[threadIdx.x] = KohonenMap[threadIdx.x];
  }
  __syncthreads();

  //calculate the distance
  float distance = -log(ComponentLocal[0]) + 0.918939f * (float) info.NumberOfDimensions;
  float penalty = 1.0f;
  int VolumeSize = info.BufferSize;
  for(int i = 0; i < info.NumberOfDimensions; i++)
  {
    float value = InputData[i*VolumeSize+kOffset];
    distance += 0.5f * (ComponentLocal[2*i+1]-value) * (ComponentLocal[2*i+1]-value) * info.Scale / ComponentLocal[2*i+2];
    penalty *= ComponentLocal[2*i+2];
  }
  distance += 0.5 * log(penalty);

  //accumulate entropy
  float oldEntropy = Accumulator[kOffset];
  float x = max(oldEntropy, distance);
  float n = min(oldEntropy, distance);
  float newEntropy = (exp(n-x) > 0.0f) ? n + log(1+exp(n-x)): -log( exp(-x) + exp(-n));
  newEntropy = (newEntropy < n) ? newEntropy : n;
  if(kOffset < VolumeSize)
  {
    Accumulator[kOffset] = newEntropy;
  }

}

void CUDAalgo_applyProbabilityMaps( float* inputData, float* inputKohonen, float** probabilityData,
                                    float** outputData, bool useProbData, bool useEntropy,
                                    Kohonen_Probability_Information& information, cudaStream_t* stream )
{
  //translate data onto device (need to transpose KSOM)
  int VolumeSize = information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2];
  int MapSize = information.KohonenMapSize[0]*information.KohonenMapSize[1];

  //copy kohonen data to GPU
  float* device_KohonenMap = 0;
  cudaMalloc( (void**) &device_KohonenMap, sizeof(float)*MapSize*(2*information.NumberOfDimensions+1) );
  cudaMemcpy( device_KohonenMap, inputKohonen, sizeof(float)*MapSize*(2*information.NumberOfDimensions+1), cudaMemcpyHostToDevice );

  //partition image into affordable sizes
  size_t freeMemory, totalMemory;
  cudaMemGetInfo(&freeMemory, &totalMemory);
  int SizeAllowed = (int) ((double)freeMemory / (double)(sizeof(float)*(information.NumberOfDimensions+3)) );
  SizeAllowed -= SizeAllowed % NUMTHREADS;
  if(SizeAllowed > VolumeSize)
  {
    SizeAllowed = VolumeSize;
  }
  while(VolumeSize % SizeAllowed < NUMTHREADS && VolumeSize % SizeAllowed > 0)
  {
    SizeAllowed -= NUMTHREADS;
  }
  information.BufferSize = SizeAllowed;

  //copy information to GPU
  cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Probability_Information) );

  //create necessary GPU buffers
  float* device_InputData = 0;
  cudaMalloc( (void**) &device_InputData, sizeof(float)*SizeAllowed*information.NumberOfDimensions );
  float* device_Accumulator = 0;
  cudaMalloc( (void**) &device_Accumulator, sizeof(float)*SizeAllowed );

  //create necessary CPU buffers
  float* tempImage = new float[SizeAllowed*information.NumberOfDimensions];

  //go over each partition
  int pConsumed = 0;
  while(pConsumed < VolumeSize)
  {
    //figure out sizes and starting points
    int pSize = SizeAllowed;
    pConsumed = (VolumeSize-pConsumed > SizeAllowed) ? pConsumed : VolumeSize-SizeAllowed;
    //int pSize = (VolumeSize-pConsumed > SizeAllowed) ? SizeAllowed : VolumeSize-pConsumed;
    float* inputDataStart = inputData + pConsumed*information.NumberOfDimensions;

    //rearrange image data to be easier to work with (should parallelize)
    //if(pConsumed==0)
    for( int j = 0; j < information.NumberOfDimensions; j++ )
      for(int i = 0; i < pSize; i++)
      {
        tempImage[j*pSize+i] = inputDataStart[i*information.NumberOfDimensions+j];
      }
    cudaMemcpyAsync( device_InputData, tempImage, sizeof(float)*pSize*information.NumberOfDimensions, cudaMemcpyHostToDevice, *stream );
    cudaStreamSynchronize(*stream);

    //get the appropriate grid size
    dim3 grid = GetGrid(pSize);
    dim3 threads(NUMTHREADS,1,1);

    //apply for each label
    for( int label = 0; label < information.NumberOfLabels; label++)
    {

      //clear the accumulator
      SetBufferToConst<<<GetGrid(SizeAllowed), threads, 0, *stream>>>(device_Accumulator, FLT_MAX, SizeAllowed);

      for( int component = 0; component < MapSize; component++ )
        ProcessSample<<<grid, threads, 0, *stream>>>(device_InputData, device_KohonenMap+component*(2*information.NumberOfDimensions+1),
            device_Accumulator);
      if( !useEntropy )
      {
        NegExpBuffer<<<grid, threads, 0, *stream>>>(device_Accumulator, pSize);
      }

      //move entropy to CPU
      float* outputDataStart = (outputData[label]) + pConsumed;
      cudaMemcpyAsync( outputDataStart, device_Accumulator, sizeof(float)*pSize, cudaMemcpyDeviceToHost, *stream );
      cudaStreamSynchronize(*stream);
    }

    //update the amount of consumed volume
    pConsumed+=pSize;
  }

  //remove allocated memory
  delete[] tempImage;
  cudaFree(device_KohonenMap);
  cudaFree(device_InputData);
  cudaFree(device_Accumulator);
}