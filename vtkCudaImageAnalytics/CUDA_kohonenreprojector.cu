#include "CUDA_kohonenreprojector.h"
#include "CUDA_commonKernels.h"
#include <float.h>

__constant__ Kohonen_Reprojection_Information info;
texture<float, 3, cudaReadModeElementType> Kohonen_Map;

__global__ void ApplyReprojection(float2* InputBuffer, float* OutputBuffer){

	//shared memory
	__shared__ float2 InputIndices[NUMTHREADS];

	//get volume and output dimensions
	int outBufferSize = info.VolumeSize[0]*info.VolumeSize[1]*info.VolumeSize[2]*info.NumberOfDimensions;

	//find index in current block
	int individualInputIndex = CUDASTDOFFSET;

	//fetch indexing information from input buffer into shared memory
	InputIndices[threadIdx.x] = InputBuffer[individualInputIndex];
	__syncthreads();

	//find the starting ouput index coallesced for output
	int individualOutputIndex = threadIdx.x + NUMTHREADS * blockDim.x * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));

	//for each component
	for( int i = 0; i < info.NumberOfDimensions; i++ ){

		//fetch input index from shared memory
		int currInputIndex = (i*NUMTHREADS + threadIdx.x) / info.NumberOfDimensions;
		float2 currLoc = InputIndices[currInputIndex];
		int currComponent = 2*((i*NUMTHREADS + threadIdx.x) % info.NumberOfDimensions)+1;

		//fetch information from texture
		float reprojValue = tex3D(Kohonen_Map, (float) currComponent, currLoc.x, currLoc.y);

		//write to output
		if( individualOutputIndex < outBufferSize )
			OutputBuffer[individualOutputIndex] = reprojValue;

		//find new output index for next iteration, coallesced for output
		individualOutputIndex += NUMTHREADS;

	}
}

void CUDAalgo_reprojectKohonenMap( float* inputData, float* inputKohonen, float* outputData,
								Kohonen_Reprojection_Information& information,
								cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Reprojection_Information) );

	//allocate output image and load the input indices into the device
	int s = information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2];
	float* dev_OutputImage = 0;
	cudaMalloc( &dev_OutputImage, sizeof(float)*s*information.NumberOfDimensions );
	float2* dev_InputImage = 0;
	cudaMalloc( &dev_InputImage, sizeof(float2)*s );
	cudaMemcpyAsync(dev_InputImage, inputData, sizeof(float2)*s, cudaMemcpyHostToDevice, *stream);

	//create device memory for the map and load Kohonen map into device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* dev_KSOM = 0;
	cudaExtent KSOMSize;
	KSOMSize.width = 2*information.NumberOfDimensions+1;
	KSOMSize.height = information.KohonenMapSize[0];
	KSOMSize.depth = information.KohonenMapSize[1];
	cudaMalloc3DArray(&dev_KSOM, &channelDesc, KSOMSize);
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr( (void*) inputKohonen, KSOMSize.width*sizeof(float),
												KSOMSize.width, KSOMSize.height);
	copyParams.dstArray = dev_KSOM;
	copyParams.extent   = KSOMSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3DAsync(&copyParams, *stream);
	cudaStreamSynchronize(*stream);

	//bind KSOM array to 3D texture
	Kohonen_Map.normalized = false; // access with unnormalized texture coordinates
	Kohonen_Map.filterMode = cudaFilterModePoint;
	Kohonen_Map.addressMode[0] = cudaAddressModeClamp;
	Kohonen_Map.addressMode[1] = cudaAddressModeClamp;
	Kohonen_Map.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(Kohonen_Map, dev_KSOM, channelDesc);

	//translate input indices
	dim3 grid = GetGrid(information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2]);
	dim3 threads(NUMTHREADS,1,1);
	cudaThreadSynchronize();
	ApplyReprojection<<<grid, threads, 0, *stream >>>(dev_InputImage, dev_OutputImage);
	cudaThreadSynchronize();

	//deallocate the device memory for the Kohonen map
	cudaUnbindTexture(Kohonen_Map);
	cudaFreeArray(dev_KSOM);

	//deallocate the memory for the input indices
	cudaFree(dev_InputImage);

	//copy out the output values and deallocate remaining device mempory
	cudaMemcpyAsync(outputData, dev_OutputImage, sizeof(float)*s*information.NumberOfDimensions, cudaMemcpyDeviceToHost, *stream);
	cudaFree( dev_OutputImage );
	cudaStreamSynchronize(*stream);
}