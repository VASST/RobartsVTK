#ifndef _CUDA_VTKCUDAVOLUMEMAPPER_RENDERALGO_H
#define _CUDA_VTKCUDAVOLUMEMAPPER_RENDERALGO_H

#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include <cuda.h>

#include <iostream>

#define BLOCK_DIM2D 16 //16 is optimal, 4 is the minimum and 16 is the maximum

//execution parameters and general information
__constant__ cudaVolumeInformation        volInfo;
__constant__ cudaRendererInformation      renInfo;
__constant__ cudaOutputImageInformation      outInfo;
__constant__ float dRandomRayOffsets[BLOCK_DIM2D*BLOCK_DIM2D];

//texture element information for the ZBuffer
cudaArray* ZBufferArray = 0;
texture<float, 2, cudaReadModeElementType> zbuffer_texture;

#define bindSingle2DTexture( textureToBind, value) textureToBind.normalized = true;       \
                           textureToBind.filterMode = cudaFilterModePoint;                \
                           textureToBind.addressMode[0] = cudaAddressModeClamp;           \
                           textureToBind.addressMode[1] = cudaAddressModeClamp;           \
                           cudaBindTextureToArray(textureToBind, value, channelDesc);

#define load2DArray(array, values, s, tr) if(array) cudaFreeArray(array);                 \
                      cudaMallocArray( &array, &channelDesc, s, s);                       \
                      cudaMemcpyToArrayAsync(array, 0, 0, values, sizeof(float)*s*s,      \
                      cudaMemcpyHostToDevice, tr);

#define unloadArray(a) if(a); cudaFreeArray(a); a = 0;

//channel for loading input data and transfer functions
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float2>();

__device__ void CUDAkernel_FindKeyholeValues(float3 rayStart, float3 rayInc,
    float& numSteps, float& excludeStart, float& excludeEnd )
{

  __syncthreads();
  const int numPlanes = renInfo.NumberOfKeyholePlanes;
  __syncthreads();

  //create a rayEnd holder
  float3 oldRayStart = rayStart;
  float3 rayDir;
  rayDir.x = numSteps * rayInc.x;
  rayDir.y = numSteps * rayInc.y;
  rayDir.z = numSteps * rayInc.z;
  float3 rayEnd;
  rayEnd.x = rayStart.x + rayDir.x;
  rayEnd.y = rayStart.y + rayDir.y;
  rayEnd.z = rayStart.z + rayDir.z;

  //default to some safe values
  excludeStart = 1.0f;
  excludeEnd = -1.0f;

  // loop through all provided clipping planes
  if(!numPlanes)
  {
    return;
  }
  int flag = 0;
  for ( int i = 0; i < numPlanes; i++ )
  {

    //refine the ray direction to account for any changes in starting or ending position
    rayDir.x = rayEnd.x - rayStart.x;
    rayDir.y = rayEnd.y - rayStart.y;
    rayDir.z = rayEnd.z - rayStart.z;

    //collect all the information about the current clipping plane
    float4 keyholePlane;
    __syncthreads();
    keyholePlane.x  = renInfo.KeyholePlanes[4*i];
    keyholePlane.y  = renInfo.KeyholePlanes[4*i+1];
    keyholePlane.z  = renInfo.KeyholePlanes[4*i+2];
    keyholePlane.w  = renInfo.KeyholePlanes[4*i+3];
    __syncthreads();

    const float dp = keyholePlane.x*rayDir.x +
                     keyholePlane.y*rayDir.y +
                     keyholePlane.z*rayDir.z;
    const float t = -(keyholePlane.x*rayStart.x +
                      keyholePlane.y*rayStart.y +
                      keyholePlane.z*rayStart.z +
                      keyholePlane.w) / dp;

    const float point0 = rayStart.x + t*rayDir.x;
    const float point1 = rayStart.y + t*rayDir.y;
    const float point2 = rayStart.z + t*rayDir.z;

    //if the ray intersects the plane, set the start or end point to the intersection point
    if ( t > 0.0f && t < 1.0f )
    {

      dp > 0.0f ? rayStart.x = point0 : rayEnd.x = point0;
      dp > 0.0f ? rayStart.y = point1 : rayEnd.y = point1;
      dp > 0.0f ? rayStart.z = point2 : rayEnd.z = point2;

    }

    //flag this ray if it is outside the plane entirely
    flag |= (dp > 0.0f && t > 1.0f);
    flag |= (dp < 0.0f && t < 0.0f);

  }//for

  rayStart.x -= oldRayStart.x;
  rayStart.y -= oldRayStart.y;
  rayStart.z -= oldRayStart.z;
  rayEnd.x -= oldRayStart.x;
  rayEnd.y -= oldRayStart.y;
  rayEnd.z -= oldRayStart.z;

  //if the ray is not inside the clipping planes, make the ray zero length
  float invRayLengthSquared = 1.0f / (rayInc.x*rayInc.x + rayInc.y*rayInc.y + rayInc.z*rayInc.z);
  excludeStart = flag ? -1.0f : (rayStart.x * rayInc.x +
                                 rayStart.y * rayInc.y +
                                 rayStart.z * rayInc.z ) * invRayLengthSquared;
  excludeEnd = flag ?  -1.0f : (rayEnd.x * rayInc.x +
                                rayEnd.y * rayInc.y +
                                rayEnd.z * rayInc.z ) * invRayLengthSquared;

}

__device__ void CUDAkernel_ClipRayAgainstClippingPlanes(float3& rayStart, float3& rayEnd, float3& rayDir)
{

  __syncthreads();
  const int numPlanes = renInfo.NumberOfClippingPlanes;
  __syncthreads();

  // loop through all 6 clipping planes
  if(!numPlanes)
  {
    return;
  }
  int flag = 0;
#pragma unroll 1
  for ( int i = 0; i < numPlanes; i++ )
  {

    //refine the ray direction to account for any changes in starting or ending position
    rayDir.x = rayEnd.x - rayStart.x;
    rayDir.y = rayEnd.y - rayStart.y;
    rayDir.z = rayEnd.z - rayStart.z;

    //collect all the information about the current clipping plane
    float4 clippingPlane;
    __syncthreads();
    clippingPlane.x  = renInfo.ClippingPlanes[4*i];
    clippingPlane.y  = renInfo.ClippingPlanes[4*i+1];
    clippingPlane.z  = renInfo.ClippingPlanes[4*i+2];
    clippingPlane.w  = renInfo.ClippingPlanes[4*i+3];
    __syncthreads();

    const float dp = clippingPlane.x*rayDir.x +
                     clippingPlane.y*rayDir.y +
                     clippingPlane.z*rayDir.z;
    const float t = -(clippingPlane.x*rayStart.x +
                      clippingPlane.y*rayStart.y +
                      clippingPlane.z*rayStart.z +
                      clippingPlane.w) / dp;

    const float point0 = rayStart.x + t*rayDir.x;
    const float point1 = rayStart.y + t*rayDir.y;
    const float point2 = rayStart.z + t*rayDir.z;

    //if the ray intersects the plane, set the start or end point to the intersection point
    if ( t > 0.0f && t < 1.0f )
    {

      dp > 0.0f ? rayStart.x = point0 : rayEnd.x = point0;
      dp > 0.0f ? rayStart.y = point1 : rayEnd.y = point1;
      dp > 0.0f ? rayStart.z = point2 : rayEnd.z = point2;

    }

    //flag this ray if it is outside the plane entirely
    flag |= (dp > 0.0f && t > 1.0f);
    flag |= (dp < 0.0f && t < 0.0f);

  }//for

  //if the ray is not inside the clipping planes, make the ray zero length
  if(flag)
  {
    rayStart.x = rayEnd.x;
    rayStart.y = rayEnd.y;
    rayStart.z = rayEnd.z;
  }

}

__device__ void CUDAkernel_ClipRayAgainstVolume(float3& rayStart, float3& rayEnd, float3& rayDir)
{

  //define the ray's length and direction to account for any changes in starting and ending position
  rayDir.x = rayEnd.x - rayStart.x;
  rayDir.y = rayEnd.y - rayStart.y;
  rayDir.z = rayEnd.z - rayStart.z;

  //collect the information about the bounds of the volume in voxels from the volume information
  __syncthreads();
  const float bounds0 = volInfo.Bounds[0]+1.0f;
  const float bounds1 = volInfo.Bounds[1]-1.0f;
  const float bounds2 = volInfo.Bounds[2]+1.0f;
  const float bounds3 = volInfo.Bounds[3]-1.0f;
  const float bounds4 = volInfo.Bounds[4]+1.0f;
  const float bounds5 = volInfo.Bounds[5]-1.0f;
  __syncthreads();

  float diffS;
  float diffE;

  //find the intersection of the ray and the volume (in the x direction)
  if (rayDir.x > 0.0f)
  {
    diffS = rayStart.x < bounds0 ? bounds0 - rayStart.x : 0.0f;
    diffE = rayEnd.x > bounds1 ? bounds1 - rayEnd.x : 0.0f;
  }
  else
  {
    diffS = rayStart.x > bounds1 ? bounds1 - rayStart.x : 0.0f;
    diffE = rayEnd.x < bounds0 ? bounds0 - rayEnd.x : 0.0f;
  }
  diffS /= rayDir.x;
  diffE /= rayDir.x;

  //crop the ray to fit the x direction if possible
  if(isfinite(diffS))
  {
    rayStart.x += rayDir.x * diffS;
    rayStart.y += rayDir.y * diffS;
    rayStart.z += rayDir.z * diffS;
    rayEnd.x += rayDir.x * diffE;
    rayEnd.y += rayDir.y * diffE;
    rayEnd.z += rayDir.z * diffE;
  }

  //find the intersection of the ray and the volume (in the y direction)
  if(rayDir.y > 0.0f)
  {
    diffS = rayStart.y < bounds2 ? bounds2 - rayStart.y : 0.0f;
    diffE = rayEnd.y > bounds3 ? bounds3 - rayEnd.y : 0.0f;
  }
  else
  {
    diffS = rayStart.y > bounds3 ? bounds3 - rayStart.y : 0.0f;
    diffE = rayEnd.y < bounds2 ? bounds2 - rayEnd.y : 0.0f;
  }
  diffS /= rayDir.y;
  diffE /= rayDir.y;

  //crop the ray to fit the y direction if possible
  if(isfinite(diffS))
  {
    rayStart.x += rayDir.x * diffS;
    rayStart.y += rayDir.y * diffS;
    rayStart.z += rayDir.z * diffS;
    rayEnd.x += rayDir.x * diffE;
    rayEnd.y += rayDir.y * diffE;
    rayEnd.z += rayDir.z * diffE;
  }

  //find the intersection of the ray and the volume (in the z direction)
  if(rayDir.z > 0.0f)
  {
    diffS = rayStart.z < bounds4 ? bounds4 - rayStart.z : 0.0f;
    diffE = rayEnd.z > bounds5 ? bounds5 - rayEnd.z : 0.0f;
  }
  else
  {
    diffS = rayStart.z > bounds5 ? bounds5 - rayStart.z : 0.0f;
    diffE = rayEnd.z < bounds4 ? bounds4 - rayEnd.z : 0.0f;
  }
  diffS /= rayDir.z;
  diffE /= rayDir.z;

  //crop the ray to fit the z direction if possible
  if(isfinite(diffS))
  {
    rayStart.x += rayDir.x * diffS;
    rayStart.y += rayDir.y * diffS;
    rayStart.z += rayDir.z * diffS;
    rayEnd.x += rayDir.x * diffE;
    rayEnd.y += rayDir.y * diffE;
    rayEnd.z += rayDir.z * diffE;
  }

  // If the voxel still isn't inside the volume, then this ray
  // doesn't really intersect the volume, thus, make it all zero
  if (rayEnd.x > bounds1 + 1.0f ||
      rayEnd.y > bounds3 + 1.0f ||
      rayEnd.z > bounds5 + 1.0f ||
      rayEnd.x < bounds0 - 1.0f ||
      rayEnd.y < bounds2 - 1.0f ||
      rayEnd.z < bounds4 - 1.0f||
      rayStart.x > bounds1 + 1.0f ||
      rayStart.y > bounds3 + 1.0f ||
      rayStart.z > bounds5 + 1.0f ||
      rayStart.x < bounds0 - 1.0f ||
      rayStart.y < bounds2 - 1.0f ||
      rayStart.z < bounds4 - 1.0f )
  {
    rayStart = rayEnd;
  }

  //refine the ray's length and direction to reflect any changes in the starting and ending co-ordinates
  rayDir.x = rayEnd.x - rayStart.x;
  rayDir.y = rayEnd.y - rayStart.y;
  rayDir.z = rayEnd.z - rayStart.z;

}

__device__ void CUDAkernel_SetRayEnds(const int2& index, float3& rayStart, float3& rayDir, const int& outIndex)
{
  //set the original estimates of the starting and ending co-ordinates in the co-ordinates of the view (not voxels)
  //note: viewRayZ = 0 for start and viewRayZ = 1 for end
  __syncthreads();
  float viewRayX =  outInfo.flipped ? ( ((float) index.x) / (float) outInfo.resolution.x ) :
                    1.0f - ( ((float) index.x) / (float) outInfo.resolution.x );
  float viewRayY =  ( ((float) index.y) / (float) outInfo.resolution.y );
  __syncthreads();
  float endDepth = tex2D(zbuffer_texture, 1.0f-viewRayX, viewRayY );

  //multiply the start co-ordinate in the view by the view to voxels matrix to get the co-ordinate in voxels (NOT YET NORMALIZED)
  __syncthreads();
  rayStart.x = viewRayX*renInfo.ViewToVoxelsMatrix[0] + viewRayY*renInfo.ViewToVoxelsMatrix[1] + renInfo.ViewToVoxelsMatrix[3];
  rayStart.y = viewRayX*renInfo.ViewToVoxelsMatrix[4] + viewRayY*renInfo.ViewToVoxelsMatrix[5] + renInfo.ViewToVoxelsMatrix[7];
  rayStart.z = viewRayX*renInfo.ViewToVoxelsMatrix[8] + viewRayY*renInfo.ViewToVoxelsMatrix[9] + renInfo.ViewToVoxelsMatrix[11];
  float startNorm = viewRayX*renInfo.ViewToVoxelsMatrix[12] + viewRayY*renInfo.ViewToVoxelsMatrix[13] + renInfo.ViewToVoxelsMatrix[15];

  //multiply the equivalent for the end ray, noting that much of the pre-normalized computation is the same as the start ray
  __syncthreads();
  float3 rayEnd;
  float3 rayFull;
  rayEnd.x = rayStart.x + endDepth*renInfo.ViewToVoxelsMatrix[2];
  rayEnd.y = rayStart.y + endDepth*renInfo.ViewToVoxelsMatrix[6];
  rayEnd.z = rayStart.z + endDepth*renInfo.ViewToVoxelsMatrix[10];
  float endNorm = startNorm + endDepth*renInfo.ViewToVoxelsMatrix[14];
  __syncthreads();
  rayFull.x = rayStart.x + renInfo.ViewToVoxelsMatrix[2];
  rayFull.y = rayStart.y + renInfo.ViewToVoxelsMatrix[6];
  rayFull.z = rayStart.z + renInfo.ViewToVoxelsMatrix[10];
  float fullNorm = startNorm + renInfo.ViewToVoxelsMatrix[14];
  __syncthreads();

  //normalize (and ergo finish) the start ray's matrix multiplication
  rayStart.x /= startNorm;
  rayStart.y /= startNorm;
  rayStart.z /= startNorm;

  //normalize (and ergo finish) the end ray's matrix multiplication
  rayEnd.x /= endNorm;
  rayEnd.y /= endNorm;
  rayEnd.z /= endNorm;
  rayFull.x /= fullNorm;
  rayFull.y /= fullNorm;
  rayFull.z /= fullNorm;

  //put the maximum depth in the buffer
  float3 oldStart = rayStart;
  rayDir.x = rayFull.x - rayStart.x;
  rayDir.y = rayFull.y - rayStart.y;
  rayDir.z = rayFull.z - rayStart.z;
  __syncthreads();
  float maxDepth = __fsqrt_rz( rayDir.x*rayDir.x + rayDir.y*rayDir.y + rayDir.z*rayDir.z );
  outInfo.maxDepthBuffer[outIndex] = maxDepth;
  __syncthreads();

  //refine the ray to only include areas that are both within the volume, and within the clipping planes of said volume
  //note that ClipRayAgainstVolume calculate the ray's correct length and direction and returns it in rayInc
  CUDAkernel_ClipRayAgainstClippingPlanes(rayStart, rayEnd, rayDir);
  CUDAkernel_ClipRayAgainstVolume(rayStart, rayEnd, rayDir);

  //put the maximum depth in the buffer
  __syncthreads();
  float rayLength = __fsqrt_rz( rayDir.x*rayDir.x + rayDir.y*rayDir.y + rayDir.z*rayDir.z );
  float minDepth = __fsqrt_rz( (rayStart.x-oldStart.x)*(rayStart.x-oldStart.x) +
                               (rayStart.y-oldStart.y)*(rayStart.y-oldStart.y) +
                               (rayStart.z-oldStart.z)*(rayStart.z-oldStart.z) );
  outInfo.minDepthBuffer[outIndex] = (rayLength > 0.0f) ? minDepth : maxDepth;
  __syncthreads();
}

__global__ void CUDAkernel_renderAlgo_formRays( )
{

  //index in the output image (2D)
  int2 index;
  index.x = blockDim.x * blockIdx.x + threadIdx.x;
  index.y = blockDim.y * blockIdx.y + threadIdx.y;

  //index in the output image (1D)
  int outindex = index.x + index.y * outInfo.resolution.x;

  float3 rayStart; //ray starting point
  float3 rayInc; // ray sample increment
  float numSteps; //maximum number of samples along this ray

  // Calculate the starting and ending points of the ray, as well as the direction vector
  CUDAkernel_SetRayEnds(index, rayStart, rayInc, outindex);

  //determine the maximum number of steps the ray should sample and determine the length of each step
  __syncthreads();
  float3 spacing = volInfo.Spacing;
  float minSpacing = volInfo.MinSpacing;
  __syncthreads();
  numSteps = __fsqrt_ru(rayInc.x*rayInc.x*spacing.x*spacing.x+
                        rayInc.y*rayInc.y*spacing.y*spacing.y+
                        rayInc.z*rayInc.z*spacing.z*spacing.z) / minSpacing;
  rayInc.x /= numSteps;
  rayInc.y /= numSteps;
  rayInc.z /= numSteps;

  //find the information regarding the exclusion area
  float excludeStart = 0.0;
  float excludeEnd = 0.0;
  CUDAkernel_FindKeyholeValues( rayStart, rayInc, numSteps, excludeStart, excludeEnd );


  //write out data
  __syncthreads();
  outInfo.rayStartX[outindex] = rayStart.x;
  __syncthreads();
  outInfo.rayStartY[outindex] = rayStart.y;
  __syncthreads();
  outInfo.rayStartZ[outindex] = rayStart.z;
  __syncthreads();
  outInfo.rayIncX[outindex] = rayInc.x;
  __syncthreads();
  outInfo.rayIncY[outindex] = rayInc.y;
  __syncthreads();
  outInfo.rayIncZ[outindex] = rayInc.z;
  __syncthreads();
  outInfo.numSteps[outindex] = numSteps;
  __syncthreads();
  outInfo.excludeStart[outindex] = excludeStart;
  __syncthreads();
  outInfo.excludeEnd[outindex] = excludeEnd;
  __syncthreads();
}

__global__ void CUDAkernel_shadeAlgo_normBuffer( )
{
  int outIndex = threadIdx.x + blockDim.x * blockIdx.x; // index of result image
  float curr = outInfo.depthBuffer[outIndex];
  float max = outInfo.maxDepthBuffer[outIndex];
  float min = outInfo.minDepthBuffer[outIndex];
  curr = (max > 0.0f) ? (curr + min) / max : 1.0f;
  outInfo.depthBuffer[outIndex] = curr;
}

__global__ void CUDAkernel_shadeAlgo_doCelShade( )
{
  //index in the output image
  int outindex = threadIdx.x + blockDim.x * blockIdx.x; // index of result image

  //get the depth information from the buffer and the colour information from the output image
  float2 depthDiffX;
  float2 depthDiffY;

  __syncthreads();
  depthDiffY.y = outInfo.depthBuffer[outindex+outInfo.resolution.x];
  __syncthreads();
  depthDiffY.x = outInfo.depthBuffer[outindex];
  __syncthreads();
  depthDiffX.y = outInfo.depthBuffer[outindex+1];
  __syncthreads();
  depthDiffX.x = depthDiffY.x;

  //compute the gradient magnitude
  float gradMag = __fsqrt_rz( (depthDiffX.y - depthDiffX.x)*(depthDiffX.y - depthDiffX.x) +
                              (depthDiffY.y - depthDiffY.x)*(depthDiffY.y - depthDiffY.x) );

  //grab cel shading parameters
  __syncthreads();
  float darkness = renInfo.celr;
  float a = renInfo.cela;
  float c = renInfo.celc;
  __syncthreads();

  //multiply by the cel-shading factor
  gradMag = 1.0f - darkness * saturate( (gradMag - a) * c );

  //grab distance shading parameters
  __syncthreads();
  darkness = renInfo.disr;
  a = renInfo.disa;
  c = renInfo.disc;
  __syncthreads();

  //multiply by the depth factor
  gradMag *= 1.0f - darkness * saturate( (depthDiffX.x - a) * c );

  uchar4 colour;
  __syncthreads();
  colour = outInfo.deviceOutputImage[outindex];
  __syncthreads();

  colour.x = gradMag * ((float) colour.x);
  colour.y = gradMag * ((float) colour.y);
  colour.z = gradMag * ((float) colour.z);

  __syncthreads();
  outInfo.deviceOutputImage[outindex] = colour;
}

bool CUDA_vtkCudaVolumeMapper_renderAlgo_loadZBuffer(const float* zBuffer, const int zBufferSizeX, const int zBufferSizeY, cudaStream_t* stream)
{

  if(ZBufferArray)
  {
    cudaFreeArray(ZBufferArray);
  }

  //load the zBuffer from the host to the array
  cudaMallocArray(&ZBufferArray, &channelDesc, zBufferSizeX, zBufferSizeY);
  cudaMemcpyToArrayAsync(ZBufferArray, 0, 0, zBuffer, sizeof(float)*zBufferSizeX*zBufferSizeY, cudaMemcpyHostToDevice, *stream);

  //define the texture parameters and bind the texture to the array
  zbuffer_texture.normalized = true;
  zbuffer_texture.filterMode = cudaFilterModePoint;
  zbuffer_texture.addressMode[0] = cudaAddressModeClamp;
  zbuffer_texture.addressMode[1] = cudaAddressModeClamp;
  cudaBindTextureToArray(zbuffer_texture, ZBufferArray, channelDesc);

#ifdef DEBUG_VTKCUDAVISUALIZATION
  cudaThreadSynchronize();
  std::cout << "Load Z-Buffer: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif

  return (cudaGetLastError() == 0);

}

bool CUDA_vtkCudaVolumeMapper_renderAlgo_unloadZBuffer(cudaStream_t* stream)
{
  if(ZBufferArray)
  {
    cudaFreeArray(ZBufferArray);
  }
  ZBufferArray = 0;

#ifdef DEBUG_VTKCUDAVISUALIZATION
  cudaThreadSynchronize();
  std::cout << "Unload Z-Buffer: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif

  return (cudaGetLastError() == 0);
}

//load in a random 16x16 noise array to deartefact the image in real time
bool CUDA_vtkCudaVolumeMapper_renderAlgo_loadrandomRayOffsets(const float* randomRayOffsets, cudaStream_t* stream)
{

  cudaMemcpyToSymbolAsync(dRandomRayOffsets, randomRayOffsets, BLOCK_DIM2D*BLOCK_DIM2D*sizeof(float), 0, cudaMemcpyHostToDevice, *stream);

#ifdef DEBUG_VTKCUDAVISUALIZATION
  cudaThreadSynchronize();
  std:cout << "Load ray offsets: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;
#endif

  return (cudaGetLastError() == 0);
}

bool CUDA_vtkCudaVolumeMapper_renderAlgo_unloadrandomRayOffsets(cudaStream_t* stream)
{

#ifdef DEBUG_VTKCUDAVISUALIZATION
  cudaThreadSynchronize();
  std::cout << "Unload ray offsets: " << cudaGetErrorString( cudaGetLastError() ) << std:endl;
#endif

  return (cudaGetLastError() == 0);
}


template<typename T, typename S>
__global__ void CUDAkernel_convertUnit( T* hostBuffer, S* deviceBuffer, int bufferSize )
{

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  T value = hostBuffer[index];
  if( index < bufferSize )
  {
    deviceBuffer[index] = (S) value;
  }

}

#define CUDA_castBuffer_OptimalThreadSize 512

template<typename T, typename S>
void CUDA_castBuffer(T* hostBuffer, S** deviceBuffer, int bufferSize)
{

  //allocate required device memory buffers
  T* deviceBufferOrgType;
  S* deviceBufferNewType;
  cudaMalloc( (void**) &deviceBufferOrgType, sizeof(T)*bufferSize );
  cudaMalloc( (void**) &deviceBufferNewType, sizeof(S)*bufferSize );

  //copy Org buffer
  cudaMemcpy(deviceBufferOrgType, hostBuffer, sizeof(T)*bufferSize, cudaMemcpyHostToDevice );

  //create size thread structure
  dim3 threads (CUDA_castBuffer_OptimalThreadSize, 1, 1);
  dim3 grid ( (bufferSize-1)/CUDA_castBuffer_OptimalThreadSize+1, 1, 1 );

  //cast on GPU
  CUDAkernel_convertUnit<T,S><<<grid,threads>>>(deviceBufferOrgType,deviceBufferNewType,bufferSize);

  //deallocate buffer of type T and return new buffer
  cudaFree( deviceBufferOrgType );
  *deviceBuffer = deviceBufferNewType;

}

template<typename T>
void CUDA_allocBuffer(T* hostBuffer, T** deviceBuffer, int bufferSize)
{

  //allocate required device memory buffers
  cudaMalloc( (void**) deviceBuffer, sizeof(T)*bufferSize );

  //copy Org buffer
  cudaMemcpy(*deviceBuffer, hostBuffer, sizeof(T)*bufferSize, cudaMemcpyHostToDevice );

}

template void CUDA_allocBuffer<float>(float* hostBuffer, float** deviceBuffer, int bufferSize);
template void CUDA_castBuffer<char,float>(char* hostBuffer, float** deviceBuffer, int bufferSize);
template void CUDA_castBuffer<unsigned char,float>(unsigned char* hostBuffer, float** deviceBuffer, int bufferSize);
template void CUDA_castBuffer<short,float>(short* hostBuffer, float** deviceBuffer, int bufferSize);
template void CUDA_castBuffer<unsigned short,float>(unsigned short* hostBuffer, float** deviceBuffer, int bufferSize);
template void CUDA_castBuffer<int,float>(int* hostBuffer, float** deviceBuffer, int bufferSize);
template void CUDA_castBuffer<unsigned int,float>(unsigned int* hostBuffer, float** deviceBuffer, int bufferSize);
template void CUDA_castBuffer<long,float>(long* hostBuffer, float** deviceBuffer, int bufferSize);
template void CUDA_castBuffer<unsigned long,float>(unsigned long* hostBuffer, float** deviceBuffer, int bufferSize);
template void CUDA_castBuffer<float,float>(float* hostBuffer, float** deviceBuffer, int bufferSize);

void CUDA_deallocateMemory(void* ptr)
{
  cudaFree(ptr);
}

// Because __constants__ can't be extern'd across compilation units, include these files into this compilation unit
#include "CUDA_vtkCuda1DVolumeMapper_renderAlgo.cuh"
#include "CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.cuh"
#include "CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.cuh"
#include "CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.cuh"
#include "CUDA_vtkCuda2DVolumeMapper_renderAlgo.cuh"

#endif
