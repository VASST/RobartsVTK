#include "CUDA_commonKernels.h"
#include "CUDA_kohonengenerator.h"
#include "vtkCudaCommon.h"
#include <float.h>
#include <iostream>
#include <iostream>
#include <time.h>

//parameters held in constant memory
__constant__ Kohonen_Generator_Information info;
__constant__ float SamplePoint[MAX_DIMENSIONALITY];

__global__ void ProcessSample(float* KohonenMap, float* DistanceBuffer, short2* IndexBuffer, float* WeightBuffer, int mapSizeX, int mapSizeY )
{
  __shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

  //get sample co-ordinates in buffer
  int kOffset = CUDASTDOFFSET;
  if(threadIdx.x < MAX_DIMENSIONALITY)
  {
    SamplePointLocal[threadIdx.x] = SamplePoint[threadIdx.x];
  }
  __syncthreads();

  //calculate the distance
  float distance = 0.0f;
  float penalty = 1.0f;
  int bufferSize = mapSizeX * mapSizeY;
  for(int i = 0; i < info.NumberOfDimensions; i++)
  {
    float var = KohonenMap[(2*i+2)*bufferSize+kOffset];
    float value = (KohonenMap[(2*i+1)*bufferSize+kOffset] - SamplePointLocal[i]);
    float valSquared = value * value;

    distance += (valSquared > 0.0f) ? 0.5f * valSquared / var : 0.0f;
    penalty *= var;
  }
  distance += 0.5f * log(penalty);
  float weight = KohonenMap[kOffset];
  __syncthreads();

  if( kOffset < bufferSize )
  {
    DistanceBuffer[kOffset] = distance - log(weight);
  }
  if( kOffset < bufferSize )
  {
    WeightBuffer[kOffset] = weight * exp(-distance);
  }
  short2 index = {kOffset % mapSizeX, kOffset / mapSizeX };
  if( kOffset < bufferSize )
  {
    IndexBuffer[kOffset] = index;
  }
}

__global__ void DoubleMapSizeInX( float* KohonenMap, float* tempStore, int currMapSizeX, int currMapSizeY )
{
  int kOffset = CUDASTDOFFSET;

  //double size in X direction
  int bufferSize = currMapSizeX * currMapSizeY;
  int xIndex = kOffset % currMapSizeX;
  for(int i = 0; i < 2*info.NumberOfDimensions+1; i++)
  {
    float valueOld = KohonenMap[i*bufferSize+kOffset];
    float valueNeighbour = KohonenMap[i*bufferSize+kOffset+1];
    float difference = (xIndex != currMapSizeX-1) ? valueNeighbour - valueOld : 0.0f;

    if(i)
    {
      float2 outputValue = {valueOld, valueOld + 0.5f * difference};
      if( kOffset < bufferSize )
      {
        ((float2*) tempStore)[i*bufferSize+kOffset] = outputValue;
      }
    }
    else
    {
      float2 outputValue = {valueOld*0.5f, valueOld*0.5f};
      if( kOffset < bufferSize )
      {
        ((float2*) tempStore)[i*bufferSize+kOffset] = outputValue;
      }
    }
  }
}

__global__ void DoubleMapSizeInY( float* KohonenMap, float* tempStore, int currMapSizeX, int currMapSizeY )
{
  int kOffset = CUDASTDOFFSET;
  int bufferSize = currMapSizeX * currMapSizeY;

  //double size in Y direction
  int xIndex = kOffset % currMapSizeX;
  int yIndex = kOffset / currMapSizeX;
  for(int i = 0; i < 2*info.NumberOfDimensions+1; i++)
  {
    float valueOld = tempStore[i*bufferSize+kOffset];
    float valueNeighbour = tempStore[i*bufferSize+kOffset+currMapSizeX];
    float difference = (yIndex != currMapSizeY-1) ? valueNeighbour - valueOld : 0.0f;

    if(i)
    {
      if( kOffset < bufferSize )
      {
        KohonenMap[i*2*bufferSize+xIndex+currMapSizeX*2*yIndex] = valueOld;
      }
      if( kOffset < bufferSize )
      {
        KohonenMap[i*2*bufferSize+xIndex+currMapSizeX*(2*yIndex+1)] = valueOld + 0.5f * difference;
      }
    }
    else
    {
      if( kOffset < bufferSize )
      {
        KohonenMap[xIndex+currMapSizeX*2*yIndex] = valueOld * 0.5f;
      }
      if( kOffset < bufferSize )
      {
        KohonenMap[xIndex+currMapSizeX*(2*yIndex+1)] = valueOld *0.5f;
      }
    }
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

  while (i < n)
  {
    if( sdata[tid] >= g_idata[i] )
    {
      sdata[tid] = g_idata[i];
      sindex[tid] = i_idata[i];
    }
    if( sdata[tid] >= g_idata[i+blockSize] )
    {
      sdata[tid] = g_idata[i+blockSize];
      sindex[tid] = i_idata[i+blockSize];
    }
    i += gridSize;
    __syncthreads();
  }

  if (blockSize >= 512)
  {
    if (tid < 256)
    {
      if( sdata[tid] >= sdata[tid + 256] )
      {
        sdata[tid] = sdata[tid + 256];
        sindex[tid] = sindex[tid + 256];
      }
    }
    __syncthreads();
  }

  if (blockSize >= 256)
  {
    if (tid < 128)
    {
      if( sdata[tid] >= sdata[tid + 128] )
      {
        sdata[tid] = sdata[tid + 128];
        sindex[tid] = sindex[tid + 128];
      }
    }
    __syncthreads();
  }
  if (blockSize >= 128)
  {
    if (tid <  64)
    {
      if( sdata[tid] >= sdata[tid + 64] )
      {
        sdata[tid] = sdata[tid + 64];
        sindex[tid] = sindex[tid + 64];
      }
    }
    __syncthreads();
  }

  if (tid < 32)
  {
    if (blockSize >= 64)
    {
      if( sdata[tid] >= sdata[tid + 32] )
      {
        sdata[tid] = sdata[tid + 32];
        sindex[tid] = sindex[tid + 32];
      }
      __syncthreads();
    }
    if (blockSize >= 32)
    {
      if( sdata[tid] >= sdata[tid + 16] )
      {
        sdata[tid] = sdata[tid + 16];
        sindex[tid] = sindex[tid + 16];
      }
      __syncthreads();
    }
    if (blockSize >= 16)
    {
      if( sdata[tid] >= sdata[tid + 8] )
      {
        sdata[tid] = sdata[tid + 8];
        sindex[tid] = sindex[tid + 8];
      }
      __syncthreads();
    }
    if (blockSize >=  8)
    {
      if( sdata[tid] >= sdata[tid + 4] )
      {
        sdata[tid] = sdata[tid + 4];
        sindex[tid] = sindex[tid + 4];
      }
      __syncthreads();
    }
    if (blockSize >=  4)
    {
      if( sdata[tid] >= sdata[tid + 2] )
      {
        sdata[tid] = sdata[tid + 2];
        sindex[tid] = sindex[tid + 2];
      }
      __syncthreads();
    }
    if (blockSize >=  2)
    {
      if( sdata[tid] >= sdata[tid + 1] )
      {
        sdata[tid] = sdata[tid + 1];
        sindex[tid] = sindex[tid + 1];
      }
      __syncthreads();
    }
  }
  if (tid == 0)
  {
    g_odata[0] = sdata[0];
    i_odata[0] = sindex[0];
  }
}

__global__ void FindMinSample( float* DistanceBuffer, short2* IndexBuffer, int spread, int mapSizeX, int mapSizeY )
{
  int kOffset = CUDASTDOFFSET;
  float distance1 = DistanceBuffer[kOffset];
  float distance2 = DistanceBuffer[kOffset+spread];
  short2 index1 = IndexBuffer[kOffset];
  short2 index2 = IndexBuffer[kOffset+spread];

  if( kOffset+spread < mapSizeX * mapSizeY )
  {
    DistanceBuffer[kOffset] = (distance1 < distance2) ? distance1 : distance2;
    IndexBuffer[kOffset] = (distance1 < distance2) ? index1 : index2;
  }
}

__global__ void UpdateWeights( float* KohonenMap, short2 minIndex, float weightTot, float mAlpha, float mNeigh, float vAlpha, float vNeigh, float wAlpha, float wNeigh, int mapSizeX, int mapSizeY )
{
  __shared__ float SamplePointLocal[MAX_DIMENSIONALITY];

  //get sample co-ordinates in buffer
  int kOffset = CUDASTDOFFSET;
  short2 currIndex = {kOffset % mapSizeX, kOffset / mapSizeX };
  if(threadIdx.x < MAX_DIMENSIONALITY)
  {
    SamplePointLocal[threadIdx.x] = SamplePoint[threadIdx.x];
  }

  //figure out the multipliers
  float mMultiplier = mAlpha * exp( -((currIndex.x-minIndex.x)*(currIndex.x-minIndex.x) + (currIndex.y-minIndex.y)*(currIndex.y-minIndex.y) ) / mNeigh );
  float vMultiplier = vAlpha * exp( -((currIndex.x-minIndex.x)*(currIndex.x-minIndex.x) + (currIndex.y-minIndex.y)*(currIndex.y-minIndex.y) ) / vNeigh );
  float wMultiplier = wAlpha;

  //adjust the weights
  float distance = 0.0f;
  float penalty = 1.0f;
  int bufferSize = mapSizeX * mapSizeY;
  for(int i = 0; i < info.NumberOfDimensions; i++)
  {
    float mean = KohonenMap[(2*i+1)*bufferSize+kOffset];
    float value = SamplePointLocal[i]-KohonenMap[(2*i+1)*bufferSize+kOffset];
    float valSquared = value * value;
    float variance = KohonenMap[(2*i+2)*bufferSize+kOffset];

    distance += (valSquared > 0.0f) ? 0.5f * valSquared / variance : 0.0f;
    penalty *= variance;

    float newMean = (1.0f-mMultiplier)*mean + mMultiplier*SamplePointLocal[i];
    float newVariance = (1.0f-vMultiplier)*variance + vMultiplier*(SamplePointLocal[i]-mean)*(SamplePointLocal[i]-mean);

    //float value = SamplePointLocal[i];
    if( kOffset < bufferSize )
    {
      KohonenMap[(2*i+1)*bufferSize+kOffset] = newMean;
    }
    if( kOffset < bufferSize )
    {
      KohonenMap[(2*i+2)*bufferSize+kOffset] = newVariance;
    }
    __syncthreads();
  }
  distance += 0.5f * log(penalty);
  float weight = KohonenMap[kOffset];
  float newWeight = exp(-0.5f*( (currIndex.x-minIndex.x)*(currIndex.x-minIndex.x) + (currIndex.y-minIndex.y)*(currIndex.y-minIndex.y) )/(wNeigh*wNeigh)) / weightTot;
  newWeight = isfinite(newWeight) ? newWeight : 1.0f;
  if( kOffset < bufferSize )
  {
    KohonenMap[kOffset] = weight + wMultiplier * (newWeight - weight) + FLT_MIN;
  }
}

void getMinimum(int size, int threads, int blocks, float *d_idata, float *d_odata, short2* d_iindex, short2* d_oindex, cudaStream_t* stream )
{
  dim3 dimBlock(threads, 1, 1);
  dim3 dimGrid(blocks, 1, 1);

  int smemSize = (threads <= 32) ? 2 * threads * (sizeof(float)+sizeof(short2)) : threads * (sizeof(float)+sizeof(short2));
  switch (threads)
  {
  case 512:
    reduce6<512><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  case 256:
    reduce6<256><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  case 128:
    reduce6<128><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  case 64:
    reduce6< 64><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  case 32:
    reduce6< 32><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  case 16:
    reduce6< 16><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  case 8:
    reduce6< 8><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  case 4:
    reduce6< 4><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  case 2:
    reduce6< 2><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  case 1:
    reduce6< 1><<< dimGrid, dimBlock, smemSize, *stream >>>(d_idata, d_odata, d_iindex, d_oindex, size);
    break;
  }
}

unsigned int gcd(unsigned int r, unsigned int n)
{
  if( r < n )
  {
    return gcd( n, r );
  }
  if( n == 0 )
  {
    return r;
  }
  return gcd( n, r % n );
}

unsigned int random_prime(unsigned int n)
{
  unsigned int r = rand();
  unsigned int t;
  while ((t = gcd(r, n)) > 1)
  {
    r /= t;
  }
  return r;
}

void CUDAalgo_KSOMInitialize( double* Means, double* Covariance, double* Eig1, double* Eig2,
                              Kohonen_Generator_Information& information, int* currentMapSize,
                              float** device_KohonenMap, float** device_tempSpace,
                              float** device_DistanceBuffer, short2** device_IndexBuffer, float** device_WeightBuffer,
                              float meansWidth, float varsWidth, float weiWidth,
                              cudaStream_t* stream )
{
  //make sure parametes are in reasonable range
  meansWidth = max( meansWidth, FLT_MIN );
  varsWidth = max( varsWidth, FLT_MIN );

  //copy information to GPU
  cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Generator_Information) );

  //find minimum starting size
  float neighbourhood = min( meansWidth, varsWidth );
  currentMapSize[0] = currentMapSize[1] = 2;
  currentMapSize[2] = 1;
  while( neighbourhood * (double) currentMapSize[0] <= 8.0 && currentMapSize[0] < information.KohonenMapSize[0] )
  {
    currentMapSize[0] += currentMapSize[0];
  }
  if( currentMapSize[0] > information.KohonenMapSize[0] )
  {
    currentMapSize[0] = information.KohonenMapSize[0];
  }
  while( neighbourhood * (double) currentMapSize[1] <= 8.0 && currentMapSize[1] < information.KohonenMapSize[1] )
  {
    currentMapSize[1] += currentMapSize[1];
  }
  if( currentMapSize[1] > information.KohonenMapSize[1] )
  {
    currentMapSize[1] = information.KohonenMapSize[1];
  }
  int MapSize = information.KohonenMapSize[0]*information.KohonenMapSize[1];
  int currMapSize = currentMapSize[0]*currentMapSize[1];
#ifdef DEBUG
  printf("Updating size to (%d,%d)\n", currentMapSize[0],currentMapSize[1]);
#endif

  //allocate a distance buffer
  cudaMalloc( (void**) device_DistanceBuffer, sizeof(float)*MapSize );
  cudaMalloc( (void**) device_IndexBuffer, sizeof(short2)*MapSize );
  cudaMalloc( (void**) device_WeightBuffer, sizeof(float)*MapSize );

  //create buffer for the Kohonen map
  cudaMalloc( (void**) device_KohonenMap, sizeof(float)*MapSize*(2*information.NumberOfDimensions+1) );
  cudaMalloc( (void**) device_tempSpace, sizeof(float)*MapSize*(2*information.NumberOfDimensions+1) );

  //initialize weights
  dim3 grid((currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);
  dim3 threads(NUMTHREADS, 1, 1);
  int N = information.NumberOfDimensions;
  float* hostWeights = new float[currMapSize];
  double accumulator = 0.0;
  for(int i = 0; i < currentMapSize[0]; i++) for(int j = 0; j < currentMapSize[1]; j++)
    {
      hostWeights[i*currentMapSize[1]+j] = (float) ( exp( - 4.0 * ( (double) i - 0.5 * (double) currentMapSize[0] - 0.5 ) / (double) (currentMapSize[0]-1) ) *
                                           exp( - 4.0 * ( (double) j - 0.5 * (double) currentMapSize[1] - 0.5 ) / (double) (currentMapSize[1]-1) ) );
      accumulator += (double) hostWeights[i*currentMapSize[1]+j];
    }
  for(int i = 0; i < currentMapSize[0]; i++) for(int j = 0; j < currentMapSize[1]; j++)
    {
      hostWeights[i*currentMapSize[1]+j] /= (float) accumulator;
    }
  cudaMemcpy(*device_KohonenMap, hostWeights, sizeof(float)*currMapSize, cudaMemcpyHostToDevice);
  delete hostWeights;

  //initialize means
  float* hostMeans = new float[currMapSize];
  for(int n = 0; n < N; n++)
  {
    for(int i = 0; i < currentMapSize[0]; i++) for(int j = 0; j < currentMapSize[1]; j++)
      {
        hostMeans[i*currentMapSize[1]+j] = (float) ( Means[n] +
                                           4.0 * ( (double) i - 0.5 * (double) currentMapSize[0] - 0.5 ) / (double) (currentMapSize[0]-1) * Eig1[n] +
                                           4.0 * ( (double) j - 0.5 * (double) currentMapSize[1] - 0.5 ) / (double) (currentMapSize[1]-1) * Eig2[n] );

      }
    cudaMemcpy((*device_KohonenMap)+(2*n+1)*currMapSize, hostMeans, sizeof(float)*currMapSize, cudaMemcpyHostToDevice);
  }
  delete hostMeans;

  //initialize variances
  for(int n = 0; n < N; n++ )
    SetBufferToConst<float><<<grid, threads, 0, *stream>>>((*device_KohonenMap)+(2*n+2)*currMapSize,
        (float)(Covariance[n*N+n] - Eig1[n]*Eig1[n] - Eig2[n]*Eig2[n]),
        currMapSize);
}

void CUDAalgo_KSOMIteration( float** inputData,  char** maskData, int epoch,
                             int* currentMapSize,
                             float** device_KohonenMap, float** device_tempSpace,
                             float** device_DistanceBuffer, short2** device_IndexBuffer, float** device_WeightBuffer,
                             int* VolumeSize, int NumVolumes,
                             Kohonen_Generator_Information& information,
                             int BatchSize,
                             float meansAlpha, float meansWidth,
                             float varsAlpha, float varsWidth,
                             float weiAlpha, float weiWidth,
                             cudaStream_t* stream )
{
  //make sure parameters are in a reasonable range
  meansAlpha = max( meansAlpha, FLT_MIN );
  meansWidth = max( meansWidth, FLT_MIN );
  varsAlpha = max( varsAlpha, FLT_MIN );
  varsWidth = max( varsWidth, FLT_MIN );
  weiAlpha = max( weiAlpha, FLT_MIN );
  weiWidth = max( weiWidth, FLT_MIN );

  dim3 grid((currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);
  dim3 threads(NUMTHREADS, 1, 1);

  //make sure map is large enough
  float neighbourhood = min( meansWidth, min( weiWidth, varsWidth ) ) * (currentMapSize[0]+currentMapSize[1]) / 2;
  if( ((neighbourhood <= 8.0 ) && (currentMapSize[0] < information.KohonenMapSize[0])) )
  {
    grid = dim3 ((2*currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);
    DoubleMapSizeInX<<<grid, threads, 0, *stream>>>( *device_KohonenMap, *device_tempSpace, currentMapSize[0], currentMapSize[1] );
    currentMapSize[0] *= 2;
#ifdef DEBUG
    printf("Updating size to (%d,%d)\n", currentMapSize[0],currentMapSize[1]);
#endif
  }
  if( ((neighbourhood <= 8.0) && (currentMapSize[1] < information.KohonenMapSize[1])) )
  {
    grid = dim3 ((2*currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);
    DoubleMapSizeInY<<<grid, threads, 0, *stream>>>( *device_KohonenMap, *device_tempSpace, currentMapSize[0], currentMapSize[1] );
    currentMapSize[1] *= 2;
#ifdef DEBUG
    printf("Updating size to (%d,%d)\n", currentMapSize[0],currentMapSize[1]);
#endif
  }

  float meansNeigh = meansWidth * (currentMapSize[0]+currentMapSize[1]) / 2;
  float varsNeigh = varsWidth * (currentMapSize[0]+currentMapSize[1]) / 2;
  float weiNeigh = weiWidth * (currentMapSize[0]+currentMapSize[1]) / 2;

  //update grid size
  grid = dim3((currentMapSize[0]*currentMapSize[1]-1)/NUMTHREADS+1, 1, 1);

  float* cpuWeights = new float[currentMapSize[0]*currentMapSize[1]];

  //train kohonen map
  if( BatchSize == -1 )
  {

    //generate a random iterator through [0,NumVolumes-1]
    int pictureIncrement = random_prime( NumVolumes ) % NumVolumes;
    int pictureInUse = rand() % NumVolumes;

    for( int picture = 0; picture < NumVolumes; picture++ )
    {

      //figure out what pseudo-random picture to grab
      pictureInUse = (pictureInUse + pictureIncrement) % NumVolumes;
      int NumVoxels = VolumeSize[3*pictureInUse]*VolumeSize[3*pictureInUse+1]*VolumeSize[3*pictureInUse+2];

      //generate a random iterator through [0,NumVolumes-1]
      int offsetIncrement = random_prime( NumVoxels ) % NumVoxels;
      int offsetInUse = rand() % NumVoxels;

      for( int sampleOffset = 0; sampleOffset < NumVoxels; sampleOffset++)
      {

        //figure out what pseudo-random offset to grab
        offsetInUse = (offsetInUse + offsetIncrement) % NumVoxels;
        int sampleDimensionalOffset = information.NumberOfDimensions * offsetInUse;

        //if this is not a valid sample (ie: masked out) then try again
        if( maskData && (maskData[pictureInUse])[offsetInUse] == 0 )
        {
          continue;
        }

        //find the distance between each centroid and the sample
        cudaMemcpyToSymbolAsync(SamplePoint, &((inputData[pictureInUse])[sampleDimensionalOffset]),
                                sizeof(float)*information.NumberOfDimensions );
        cudaStreamSynchronize(*stream);
        ProcessSample<<<grid, threads, 0, *stream>>>(*device_KohonenMap, *device_DistanceBuffer, *device_IndexBuffer,
            *device_WeightBuffer, currentMapSize[0], currentMapSize[1]);


        //update the weights of each centroid
        short2 minIndex = {-1,-1};
        float distance = -1.0f;
        cudaStreamSynchronize(*stream);
        //cudaMemcpy( &minIndex, *device_IndexBuffer, sizeof(short2), cudaMemcpyDeviceToHost );
        //cudaMemcpy( &distance, *device_DistanceBuffer, sizeof(float), cudaMemcpyDeviceToHost );

        //find the winning centroid
        for(int i = currentMapSize[0]*currentMapSize[1] / 2; i > 0; i = i/2)
        {
          dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
          FindMinSample<<<tempGrid, threads, 0, *stream>>>(*device_DistanceBuffer, *device_IndexBuffer, i,
              currentMapSize[0], currentMapSize[1]);
        }
        //getMinimum( min(NUMTHREADS,currentMapSize[0]*currentMapSize[1]), min(NUMTHREADS,currentMapSize[0]*currentMapSize[1]), 1,
        //      *device_DistanceBuffer, *device_DistanceBuffer, *device_IndexBuffer, *device_IndexBuffer, stream );

        //update the weights of each centroid
        cudaStreamSynchronize(*stream);
        cudaMemcpy( &minIndex, *device_IndexBuffer, sizeof(short2), cudaMemcpyDeviceToHost );
        cudaMemcpy( &distance, *device_DistanceBuffer, sizeof(float), cudaMemcpyDeviceToHost );
        long double weightTot = 0.0;
        for(int i = 0; i < currentMapSize[0]; i++)
          for(int j = 0; j < currentMapSize[1]; j++)
          {
            weightTot+= (long double) exp(-0.5*( (i-minIndex.x)*(i-minIndex.x) + (j-minIndex.y)*(j-minIndex.y) )/(weiNeigh*weiNeigh));
          }

        UpdateWeights<<<grid, threads, 0, *stream>>>(*device_KohonenMap, minIndex, (float) weightTot, meansAlpha, meansNeigh,
            varsAlpha, varsNeigh, weiAlpha, weiNeigh, currentMapSize[0], currentMapSize[1]);
      }
    }
    //if we are randomly sampling from the images
  }
  else
  {
    float* tmpOld = new float[currentMapSize[0]*currentMapSize[1]];

    for( int batch = 0; batch < BatchSize; batch++ )
    {
      int sampleP = rand() % NumVolumes;
      int sampleX = rand() % VolumeSize[3*sampleP];
      int sampleY = rand() % VolumeSize[3*sampleP+1];
      int sampleZ = rand() % VolumeSize[3*sampleP+2];
      int sampleOffset = (sampleX + VolumeSize[3*sampleP] *( sampleY + VolumeSize[3*sampleP+1] * sampleZ ) );
      int sampleDimensionalOffset = information.NumberOfDimensions * sampleOffset;

      //if this is not a valid sample (ie: masked out) then try again
      if( maskData && (maskData[sampleP])[sampleOffset] == 0 )
      {
        batch--;
        continue;
      }

      //find the distance between each centroid and the sample
      cudaMemcpyToSymbolAsync(SamplePoint, &((inputData[sampleP])[sampleDimensionalOffset]), sizeof(float)*information.NumberOfDimensions );
      cudaStreamSynchronize(*stream);
      ProcessSample<<<grid, threads, 0, *stream>>>(*device_KohonenMap, *device_DistanceBuffer, *device_IndexBuffer, *device_WeightBuffer, currentMapSize[0], currentMapSize[1]);

      //find the winning centroid
      for(int i = currentMapSize[0]*currentMapSize[1] / 2; i > 0; i = i/2)
      {
        dim3 tempGrid( i>NUMTHREADS ? i/NUMTHREADS : 1, 1, 1);
        FindMinSample<<<tempGrid, threads, 0, *stream>>>(*device_DistanceBuffer, *device_IndexBuffer, i, currentMapSize[0], currentMapSize[1]);
      }
      //getMinimum( min(NUMTHREADS,currentMapSize[0]*currentMapSize[1]), min(NUMTHREADS,currentMapSize[0]*currentMapSize[1]), 1,
      //      *device_DistanceBuffer, *device_DistanceBuffer, *device_IndexBuffer, *device_IndexBuffer, stream );

      //update the weights of each centroid
      short2 minIndex;
      float distance;
      cudaMemcpyAsync( &minIndex, *device_IndexBuffer, sizeof(short2), cudaMemcpyDeviceToHost, *stream );
      cudaMemcpy( &distance, *device_DistanceBuffer, sizeof(float), cudaMemcpyDeviceToHost );
      cudaStreamSynchronize(*stream);
      long double weightTot = 0.0;
      for(int i = 0; i < currentMapSize[0]; i++)
        for(int j = 0; j < currentMapSize[1]; j++)
        {
          weightTot+= (long double) exp(-0.5*( (i-minIndex.x)*(i-minIndex.x) + (j-minIndex.y)*(j-minIndex.y) )/(weiNeigh*weiNeigh));
        }


      UpdateWeights<<<grid, threads, 0, *stream>>>(*device_KohonenMap, minIndex, (float) weightTot, meansAlpha, meansNeigh,
          varsAlpha, varsNeigh, weiAlpha, weiNeigh, currentMapSize[0], currentMapSize[1]);
    }

    delete[] tmpOld;
  }

#ifdef DEBUG
  printf("Finished epoch %d with:\n",epoch);
  printf("%d  M:(a,n) = (%f,%f,%f)\n",(int) time(NULL),meansAlpha, meansWidth, meansNeigh );
  printf("            V:(a,n) = (%f,%f,%f)\n",varsAlpha, varsWidth, varsNeigh);
  printf("            W:(a,n) = (%f,%f,%f)\n",weiAlpha,  weiWidth,  weiNeigh );
#endif

  delete cpuWeights;
}

void CUDAalgo_KSOMOffLoad( float* outputKohonen, float** device_KohonenMap,
                           float** device_tempSpace,
                           float** device_DistanceBuffer, short2** device_IndexBuffer, float** device_WeightBuffer,
                           Kohonen_Generator_Information& information,
                           cudaStream_t* stream )
{
  //remove distance buffer
  cudaFree(*device_DistanceBuffer);
  cudaFree(*device_IndexBuffer);
  cudaFree(*device_WeightBuffer);

  //translate back data
  int MapSize = information.KohonenMapSize[0]*information.KohonenMapSize[1];
  float* tempKohonen = new float[MapSize*(2*information.NumberOfDimensions+1)];
  cudaMemcpyAsync( tempKohonen, *device_KohonenMap, sizeof(float)*MapSize*(2*information.NumberOfDimensions+1),
                   cudaMemcpyDeviceToHost, *stream );
  cudaStreamSynchronize(*stream);
  cudaFree(*device_KohonenMap);
  cudaFree(*device_tempSpace);
  cudaStreamSynchronize(*stream);

  int bufferJump = information.KohonenMapSize[0]*information.KohonenMapSize[1];
  for(int i = 0; i < information.KohonenMapSize[0]*information.KohonenMapSize[1]; i++)
    for( int j = 0; j < 2*information.NumberOfDimensions+1; j++ )
    {
      outputKohonen[i*(2*information.NumberOfDimensions+1)+j] = tempKohonen[j*bufferJump+i];
    }
  delete[] tempKohonen;
}