#include "CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include <cuda.h>

//execution parameters and general information
__constant__ float ctParams[3];

//3D input data (read-only texture with corresponding opague device memory back)
texture<float, 3, cudaReadModeElementType> CUDA_vtkCudaDRRImageVolumeMapper_input_texture;

__device__ void CUDA_vtkCudaDRRImageVolumeMapper_CUDAkernel_CastRays(float3& rayStart,
    const float& numSteps,
    int& excludeStart,
    int& excludeEnd,
    const float3& rayInc,
    float& outputVal)
{

  //set the default values for the output (note A is currently the remaining opacity, not the output opacity)
  outputVal = log(256.0f*ctParams[2]); //A

  //fetch the required information about the size and range of the transfer function from memory to registers
  __syncthreads();
  float ctIntercept = ctParams[0];
  float ctSlope = ctParams[1];
  const float3 incSpace = volInfo.Spacing;
  __syncthreads();

  //apply a randomized offset to the ray
  float retDepth = dRandomRayOffsets[threadIdx.x + BLOCK_DIM2D * threadIdx.y];
  __syncthreads();
  int maxSteps = __float2int_rd(numSteps - retDepth);
  rayStart.x += retDepth*rayInc.x;
  rayStart.y += retDepth*rayInc.y;
  rayStart.z += retDepth*rayInc.z;
  float rayLength = sqrtf(rayInc.x*rayInc.x*incSpace.x*incSpace.x +
                          rayInc.y*rayInc.y*incSpace.y*incSpace.y +
                          rayInc.z*rayInc.z*incSpace.z*incSpace.z);
  ctIntercept *= -rayLength;
  ctSlope *= -rayLength;

  //allocate flags
  char2 step;
  step.x = 0;
  step.y = 0;

  //reformat the exclusion indices to use the same ordering (counting downwards rather than upwards)
  excludeStart = maxSteps - excludeStart;
  excludeEnd = maxSteps - excludeEnd;

  //loop as long as we are still *roughly* in the range of the clipped and cropped volume
  while( maxSteps > 0 )
  {

    //if we are in the exclusion area, leave
    bool inKeyhole = (excludeStart >= maxSteps && excludeEnd < maxSteps);

    // fetching the opacity
    float alpha = ctSlope * tex3D(CUDA_vtkCudaDRRImageVolumeMapper_input_texture,
                                  rayStart.x, rayStart.y, rayStart.z) + ctIntercept;

    //filter out objects with too low opacity (deemed unimportant, and this saves time and reduces cloudiness)
    if(alpha < 0.0f && !inKeyhole)
    {

      //determine which kind of step to make
      step.x = step.y;
      step.y = 0;

      //move to the next sample point (may involve moving backward)
      rayStart.x = rayStart.x + (step.x ? -rayInc.x : rayInc.x);
      rayStart.y = rayStart.y + (step.x ? -rayInc.y : rayInc.y);
      rayStart.z = rayStart.z + (step.x ? -rayInc.z : rayInc.z);
      maxSteps = maxSteps + (step.x ? 1 : -1);

      //accumulate the opacity for this sample point
      if(!step.x)
      {
        outputVal += alpha;
      }

      //determine whether or not we've hit an opacity where further sampling becomes neglible
      if(outputVal <= 0.0f)
      {
        break;
      }

    }
    else
    {

      //if we aren't backstepping, we can skip a sample
      if(!step.x)
      {
        rayStart.x += rayInc.x;
        rayStart.y += rayInc.y;
        rayStart.z += rayInc.z;
        maxSteps--;
      }
      step.y = !(step.x);

      //move to the next sample
      rayStart.x += rayInc.x;
      rayStart.y += rayInc.y;
      rayStart.z += rayInc.z;
      maxSteps--;
      step.x = 0;

    }

  }//while

  //adjust the opacity output to reflect the collected opacity, and not the remaining opacity
  outputVal = 256.0f-exp(outputVal);
  outputVal = (outputVal < 0.0f) ? 0.0f : outputVal;

}

__global__ void CUDA_vtkCudaDRRImageVolumeMapper_CUDAkernel_Composite( )
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
  int excludeStart; //where to start excluding
  int excludeEnd; //where to end excluding
  float outputVal; //alpha value of this ray (calculated in castRays, used in WriteData)

  //load in the rays
  __syncthreads();
  rayStart.x = outInfo.rayStartX[outindex];
  __syncthreads();
  rayStart.y = outInfo.rayStartY[outindex];
  __syncthreads();
  rayStart.z = outInfo.rayStartZ[outindex];
  __syncthreads();
  rayInc.x = outInfo.rayIncX[outindex];
  __syncthreads();
  rayInc.y = outInfo.rayIncY[outindex];
  __syncthreads();
  rayInc.z = outInfo.rayIncZ[outindex];
  __syncthreads();
  numSteps = outInfo.numSteps[outindex];
  __syncthreads();
  excludeStart = __float2int_ru(outInfo.excludeStart[outindex]);
  __syncthreads();
  excludeEnd = __float2int_rd(outInfo.excludeEnd[outindex]);
  __syncthreads();

  // trace along the ray (composite)
  CUDA_vtkCudaDRRImageVolumeMapper_CUDAkernel_CastRays(rayStart, numSteps, excludeStart, excludeEnd, rayInc, outputVal);

  //convert output to uchar, adjusting it to be valued from [0,256) rather than [0,1]
  uchar4 temp;
  temp.w = (float) outInfo.tint.w + (1.0f - (float) outInfo.tint.w / 255.0f) * outputVal;
  temp.x = (float) outInfo.tint.w / ((float) outInfo.tint.w + outputVal) * (float) outInfo.tint.x;
  temp.y = (float) outInfo.tint.w / ((float) outInfo.tint.w + outputVal) * (float) outInfo.tint.y;
  temp.z = (float) outInfo.tint.w / ((float) outInfo.tint.w + outputVal) * (float) outInfo.tint.z;

  //place output in the image buffer
  __syncthreads();
  outInfo.deviceOutputImage[outindex] = temp;
}

bool CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_changeFrame(const cudaArray* frame, cudaStream_t* stream)
{

  // set the texture to the correct image
  CUDA_vtkCudaDRRImageVolumeMapper_input_texture.normalized = false;          // access with unnormalized texture coordinates
  CUDA_vtkCudaDRRImageVolumeMapper_input_texture.filterMode = cudaFilterModeLinear;   // linear interpolation
  CUDA_vtkCudaDRRImageVolumeMapper_input_texture.addressMode[0] = cudaAddressModeClamp; // wrap texture coordinates
  CUDA_vtkCudaDRRImageVolumeMapper_input_texture.addressMode[1] = cudaAddressModeClamp;
  CUDA_vtkCudaDRRImageVolumeMapper_input_texture.addressMode[2] = cudaAddressModeClamp;

  // bind array to 3D texture
  cudaBindTextureToArray(CUDA_vtkCudaDRRImageVolumeMapper_input_texture, frame, channelDesc);

#ifdef DEBUG_VTKCUDAVISUALIZATION
  cudaThreadSynchronize();
  printf( "Change Frame Status: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\n" );
#endif

  return (cudaGetLastError() == 0);
}

//pre: the resolution of the image has been processed such that it's x and y size are both multiples of 16 (enforced automatically) and y > 256 (enforced automatically)
//post: the OutputImage pointer will hold the ray casted information
bool CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_doRender(const cudaOutputImageInformation& outputInfo,
    const cudaRendererInformation& rendererInfo,
    const cudaVolumeInformation& volumeInfo,
    const float CTIntercept, const float CTSlope, const float CTOffset,
    cudaArray* frame,
    cudaStream_t* stream)
{

  // setup execution parameters - staggered to improve parallelism
  cudaMemcpyToSymbolAsync(volInfo, &volumeInfo, sizeof(cudaVolumeInformation), 0, cudaMemcpyHostToDevice, *stream);
  cudaMemcpyToSymbolAsync(renInfo, &rendererInfo, sizeof(cudaRendererInformation), 0, cudaMemcpyHostToDevice, *stream);
  cudaMemcpyToSymbolAsync(outInfo, &outputInfo, sizeof(cudaOutputImageInformation), 0, cudaMemcpyHostToDevice, *stream);
  float CTParams[3] = {CTIntercept, CTSlope, CTOffset};
  cudaMemcpyToSymbolAsync(ctParams, &CTParams, 3*sizeof(float), 0, cudaMemcpyHostToDevice, *stream);

  //bind the input data texture to the provided frame
  CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_changeFrame(frame,stream);


  //create the necessary execution amount parameters from the block sizes and calculate the volume rendering integral
  int blockX = outputInfo.resolution.x / BLOCK_DIM2D ;
  int blockY = outputInfo.resolution.y / BLOCK_DIM2D ;

  dim3 grid(blockX, blockY, 1);
  dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);
  CUDAkernel_renderAlgo_formRays <<< grid, threads, 0, *stream >>>();

#ifdef DEBUG_VTKCUDAVISUALIZATION
  cudaThreadSynchronize();
  printf( "2D Rendering Error Status 1: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\n" );
#endif

  CUDA_vtkCudaDRRImageVolumeMapper_CUDAkernel_Composite <<< grid, threads, 0, *stream >>>();
  \

#ifdef DEBUG_VTKCUDAVISUALIZATION
  cudaThreadSynchronize();
  printf( "2D Rendering Error Status 2: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\n" );
#endif

  //shade the image
  grid.x = outputInfo.resolution.x*outputInfo.resolution.y / 256;
  grid.y = 1;
  threads.x = 256;
  threads.y = 1;
  CUDAkernel_shadeAlgo_normBuffer <<< grid, threads, 0, *stream >>>();

#ifdef DEBUG_VTKCUDAVISUALIZATION
  cudaThreadSynchronize();
  printf( "2D Rendering Error Status 3: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\n" );
#endif

  return (cudaGetLastError() == 0);
}

//pre:  the data has been preprocessed by the volumeInformationHandler such that it is float data
//    the index is between 0 and 100
//post: the input_texture will map to the source data in voxel coordinate space
bool CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_loadImageInfo(const float* data, const cudaVolumeInformation& volumeInfo, cudaArray** frame, cudaStream_t* stream)
{

  // if the array is already populated with information, free it to prevent leaking
  if(*frame)
  {
    cudaFreeArray(*frame);
  }

  //define the size of the data, retrieved from the volume information
  cudaExtent volumeSize;
  volumeSize.width = volumeInfo.VolumeSize.x;
  volumeSize.height = volumeInfo.VolumeSize.y;
  volumeSize.depth = volumeInfo.VolumeSize.z;

  // create 3D array to store the image data in
  cudaMalloc3DArray(frame, &channelDesc, volumeSize);

  // copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr( (void*) data, volumeSize.width*sizeof(float),
                        volumeSize.width, volumeSize.height);
  copyParams.dstArray = *frame;
  copyParams.extent   = volumeSize;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3DAsync(&copyParams, *stream);

#ifdef DEBUG_VTKCUDAVISUALIZATION
  cudaThreadSynchronize();
  printf( "Load volume information: " );
  printf( cudaGetErrorString( cudaGetLastError() ) );
  printf( "\n" );
#endif

  return (cudaGetLastError() == 0);
}

void CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo_clearImageArray(cudaArray** frame, cudaStream_t* stream)
{
  if(*frame)
  {
    cudaFreeArray(*frame);
  }
  *frame = 0;
}