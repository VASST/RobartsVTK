#include "CUDA_kohonenreprojector.h"
#include <float.h>

__constant__ Kohonen_Reprojection_Information info;
texture<float, 3, cudaReadModeElementType> Kohonen_Map;

void CUDAalgo_reprojectKohonenMap( float* inputData, float* inputKohonen, short* outputData,
								Kohonen_Reprojection_Information& information,
								cudaStream_t* stream ){

	//copy information to GPU
	cudaMemcpyToSymbolAsync(info, &information, sizeof(Kohonen_Reprojection_Information) );

	//allocate output image and load the input indices into the device
	float* dev_OutputImage = 0;
	int s = information.NumberOfDimensions*information.VolumeSize[0]*information.VolumeSize[1]*information.VolumeSize[2];
	cudaMalloc( &dev_OutputImage, sizeof(float)*s );

	//create device memory for the map and load Kohonen map into device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray* dev_KSOM = 0;
	cudaExtent KSOMSize;
	KSOMSize.width = information.NumberOfDimensions;
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

	//bind KSOM array to 3D texture
	Kohonen_Map.normalized = false; // access with unnormalized texture coordinates
	Kohonen_Map.filterMode = cudaFilterModeLinear;
	Kohonen_Map.addressMode[0] = cudaAddressModeClamp;
	Kohonen_Map.addressMode[1] = cudaAddressModeClamp;
	Kohonen_Map.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(Kohonen_Map, dev_KSOM, channelDesc);

	//translate input indices

	//deallocate the device memory for the Kohonen map

	//deallocate the memory for the input indices

	//copy out the output values and deallocate remaining device mempory
}