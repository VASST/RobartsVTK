#include "CUDA_voxelclassifier.h"
#include "vtkCudaCommon.h"
#include <float.h>
#include <iostream>

__constant__ Voxel_Classifier_Information info;
texture<short, 2, cudaReadModeElementType> ClassifyPrimaryTexture;
texture<short, 2, cudaReadModeElementType> ClassifyKeyholeTexture;

cudaChannelFormatDesc Voxel_Classifier_ChannelDesc = cudaCreateChannelDesc<short>();

__device__ bool WithinPlanes(const float* ConstantPlanes, const int NumPlanes, const int3& index)
{
  bool flag = false;
#pragma unroll 1
  for ( int i = 0; i < NumPlanes; i++ )
  {

    //collect all the information about the current clipping plane
    float4 clippingPlane;
    __syncthreads();
    clippingPlane.x  = ConstantPlanes[4*i];
    clippingPlane.y  = ConstantPlanes[4*i+1];
    clippingPlane.z  = ConstantPlanes[4*i+2];
    clippingPlane.w  = ConstantPlanes[4*i+3];
    __syncthreads();

    const float t = -(clippingPlane.x*index.x +
                      clippingPlane.y*index.y +
                      clippingPlane.z*index.z +
                      clippingPlane.w);

    //if the ray intersects the plane, set the start or end point to the intersection point
    flag |= (t > 0.0f);

  }//for

  return !flag;
}

__global__ void ClassifyVolume( const float2* inputVolume, short* outputVolume )
{
  //get the index of the thread in the volume
  int inIndex = CUDASTDOFFSET;
  int3 index;
  index.x = inIndex % info.VolumeSize[0];
  index.z = inIndex / info.VolumeSize[0];
  index.y = index.z % info.VolumeSize[1];
  index.z = index.z / info.VolumeSize[1];

  //get the values from the volume
  float2 value = inputVolume[inIndex];
  value.x = (float) info.TextureSize * (value.x - info.Intensity1Low) * info.Intensity1Multiplier;
  value.y = (float) info.TextureSize * (value.y - info.Intensity2Low) * info.Intensity2Multiplier;
  __syncthreads();

  //check if we are in the clipping and keyhole planes
  bool inClipping = (info.NumberOfClippingPlanes == 0 || WithinPlanes(info.ClippingPlanes, info.NumberOfClippingPlanes, index));
  bool inKeyhole = (info.NumberOfKeyholePlanes > 0 && WithinPlanes(info.KeyholePlanes, info.NumberOfKeyholePlanes, index));
  __syncthreads();

  //find the primary classification
  short classification = inClipping ? tex2D(ClassifyPrimaryTexture, value.x, value.y) : 0;
  classification = (inClipping && inKeyhole) ? - tex2D(ClassifyKeyholeTexture, value.x, value.y) : classification;

  //output the final classification
  if( inIndex < info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2] )
  {
    outputVolume[inIndex] = classification;
  }
}

void CUDAalgo_classifyVoxels( float* inputData, short* inputPrimaryTexture, short* inputKeyholeTexture, int textureSize,
                              short* outputData, Voxel_Classifier_Information& information,
                              cudaStream_t* stream )
{
  //copy information to GPU
  cudaMemcpyToSymbolAsync(info, &information, sizeof(Voxel_Classifier_Information) );
  int VolumeSize = information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2];

  //translate input onto device
  float* dev_InputData = 0;
  cudaMalloc( (void**) &dev_InputData, 2*sizeof(float)*VolumeSize );
  cudaMemcpyAsync(dev_InputData,inputData, 2*sizeof(float)*VolumeSize,
                  cudaMemcpyHostToDevice, *stream);

  //translate classification textures onto the device
  cudaArray* PrimaryTextureArray = 0;
  cudaMallocArray( &PrimaryTextureArray, &Voxel_Classifier_ChannelDesc, textureSize, textureSize);
  cudaMemcpyToArrayAsync(PrimaryTextureArray, 0, 0, inputPrimaryTexture,
                         sizeof(short)*textureSize*textureSize, cudaMemcpyHostToDevice, *stream);
  cudaArray* KeyholeTextureArray = 0;
  cudaMallocArray( &KeyholeTextureArray, &Voxel_Classifier_ChannelDesc, textureSize, textureSize);
  cudaMemcpyToArrayAsync(KeyholeTextureArray, 0, 0, inputKeyholeTexture,
                         sizeof(short)*textureSize*textureSize, cudaMemcpyHostToDevice, *stream);
  cudaThreadSynchronize();
  ClassifyPrimaryTexture.normalized = false;
  ClassifyPrimaryTexture.filterMode = cudaFilterModePoint;
  ClassifyPrimaryTexture.addressMode[0] = cudaAddressModeClamp;
  ClassifyPrimaryTexture.addressMode[1] = cudaAddressModeClamp;
  cudaBindTextureToArray(ClassifyPrimaryTexture, PrimaryTextureArray, Voxel_Classifier_ChannelDesc);
  ClassifyKeyholeTexture.normalized = false;
  ClassifyKeyholeTexture.filterMode = cudaFilterModePoint;
  ClassifyKeyholeTexture.addressMode[0] = cudaAddressModeClamp;
  ClassifyKeyholeTexture.addressMode[1] = cudaAddressModeClamp;
  cudaBindTextureToArray(ClassifyKeyholeTexture, KeyholeTextureArray, Voxel_Classifier_ChannelDesc);

  cudaThreadSynchronize();
  std::cout << "Load textures: " << std::endl << cudaGetErrorString( cudaGetLastError() ) << std::endl;

  //allocate working memory for the output
  short* dev_OutputData = 0;
  cudaMalloc( (void**) &dev_OutputData, sizeof(short)*VolumeSize );

  //classify the volume
  dim3 grid = GetGrid( VolumeSize );
  dim3 threads(NUMTHREADS,1,1);
  ClassifyVolume<<< grid, threads, 0, *stream >>>((float2*)dev_InputData, dev_OutputData);

  cudaThreadSynchronize();
  std::cout << "Classify: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;

  //retrieve classified output
  cudaMemcpyAsync( outputData, dev_OutputData, sizeof(short)*VolumeSize,
                   cudaMemcpyDeviceToHost, *stream);
  cudaStreamSynchronize(*stream);

  cudaThreadSynchronize();
  std::cout << "Memcpy: " << cudaGetErrorString( cudaGetLastError() ) << std::endl;

  //deallocate textures and image memory
  cudaFree( dev_InputData );
  cudaUnbindTexture( ClassifyPrimaryTexture );
  cudaUnbindTexture( ClassifyKeyholeTexture );
  cudaFreeArray( PrimaryTextureArray );
  cudaFreeArray( KeyholeTextureArray );
  cudaFree( dev_OutputData );
}
