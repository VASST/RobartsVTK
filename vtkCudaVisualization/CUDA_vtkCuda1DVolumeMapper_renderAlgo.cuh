#include "CUDA_vtkCuda1DVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include <cuda.h>

//execution parameters and general information
__constant__ cuda1DTransferFunctionInformation	CUDA_vtkCuda1DVolumeMapper_trfInfo;

//transfer function as read-only textures
texture<float, 1, cudaReadModeElementType> alpha_texture_1D;
texture<float, 1, cudaReadModeElementType> colorR_texture_1D;
texture<float, 1, cudaReadModeElementType> colorG_texture_1D;
texture<float, 1, cudaReadModeElementType> colorB_texture_1D;

//opague memory back for the transfer function
cudaArray* alphaTransferArray1D = 0;
cudaArray* colorRTransferArray1D = 0;
cudaArray* colorGTransferArray1D = 0;
cudaArray* colorBTransferArray1D = 0;

//3D input data (read-only texture with corresponding opague device memory back)
texture<float, 3, cudaReadModeElementType> CUDA_vtkCuda1DVolumeMapper_input_texture;
cudaArray* CUDA_vtkCuda1DVolumeMapper_sourceDataArray[100];

__device__ void CUDA_vtkCuda1DVolumeMapper_CUDAkernel_CastRays1D(float3& rayStart,
									const float& numSteps,
									float& excludeStart,
									float& excludeEnd,
									const float3& rayInc,
									float4& outputVal,
									float& retDepth) {

	//set the default values for the output (note A is currently the remaining opacity, not the output opacity)
	outputVal.x = 0.0f; //R
	outputVal.y = 0.0f; //G
	outputVal.z = 0.0f; //B
	outputVal.w = 1.0f; //A
		
	//fetch the required information about the size and range of the transfer function from memory to registers
	__syncthreads();
	const float functRangeLow = CUDA_vtkCuda1DVolumeMapper_trfInfo.intensityLow;
	const float functRangeMulti = CUDA_vtkCuda1DVolumeMapper_trfInfo.intensityMultiplier;
	const float spaceX = volInfo.SpacingReciprocal.x;
	const float spaceY = volInfo.SpacingReciprocal.y;
	const float spaceZ = volInfo.SpacingReciprocal.z;
	const float shadeMultiplier = renInfo.gradShadeScale;
	const float shadeShift = renInfo.gradShadeShift;
	__syncthreads();

	//apply a randomized offset to the ray
	retDepth = dRandomRayOffsets[threadIdx.x + BLOCK_DIM2D * threadIdx.y];
	__syncthreads();
	rayStart.x += retDepth*rayInc.x;
	rayStart.y += retDepth*rayInc.y;
	rayStart.z += retDepth*rayInc.z;
	retDepth += __float2int_rd(numSteps);

	//calculate the number of times this can go through the loop
	int maxSteps = __float2int_rd(numSteps);
	bool skipStep = false;
	bool backStep = false;

	//reformat the exclusion indices to use the same ordering (counting downwards rather than upwards)
	excludeStart = maxSteps - excludeStart;
	excludeEnd = maxSteps - excludeEnd;

	//loop as long as we are still *roughly* in the range of the clipped and cropped volume
	while( maxSteps > 0 ){

		//if we are in the exclusion area, leave
		if( excludeStart >= maxSteps && excludeEnd <= maxSteps ){
			rayStart.x += rayInc.x;
			rayStart.y += rayInc.y;
			rayStart.z += rayInc.z;
			maxSteps--;
			continue;
		}

		// fetching the intensity index into the transfer function
		const float tempIndex = functRangeMulti * (tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z) - functRangeLow);
			
		//fetching the gradient index into the transfer function
		float3 gradient;
		gradient.x = ( tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x+1.0f, rayStart.y, rayStart.z)
					 - tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x-1.0f, rayStart.y, rayStart.z) ) * spaceX;
		gradient.y = ( tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y+1.0f, rayStart.z)
					 - tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y-1.0f, rayStart.z) ) * spaceY;
		gradient.z = ( tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z+1.0f)
					 - tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z-1.0f) ) * spaceZ;
	
		//fetching the opacity value of the sampling point (apply transfer function in stages to minimize work)
		float alpha = tex1D(alpha_texture_1D, tempIndex);

		//filter out objects with too low opacity (deemed unimportant, and this saves time and reduces cloudiness)
		if(alpha > 0.0f && tempIndex >= 0.0f && tempIndex <= 1.0f){

			//collect the alpha difference (if we sample now) as well as the colour multiplier (with photorealistic shading)
			float multiplier = outputVal.w * alpha *
								(shadeShift + shadeMultiplier * abs(gradient.x*rayInc.x + gradient.y*rayInc.y + gradient.z*rayInc.z)
								* rsqrtf(gradient.x*gradient.x+gradient.y*gradient.y+gradient.z*gradient.z));
			alpha = (1.0f - alpha);

			//determine which kind of step to make
			backStep = skipStep;
			skipStep = false;
			
			//move to the next sample point (may involve moving backward)
			rayStart.x = rayStart.x + (backStep ? -rayInc.x : rayInc.x);
			rayStart.y = rayStart.y + (backStep ? -rayInc.y : rayInc.y);
			rayStart.z = rayStart.z + (backStep ? -rayInc.z : rayInc.z);
			maxSteps = maxSteps + (backStep ? -1 : 1);

			if(!backStep){
				//accumulate the opacity for this sample point
				outputVal.w *= alpha;

				//accumulate the colour information from this sample point
				outputVal.x += multiplier * tex1D(colorR_texture_1D, tempIndex);
				outputVal.y += multiplier * tex1D(colorG_texture_1D, tempIndex);
				outputVal.z += multiplier * tex1D(colorB_texture_1D, tempIndex);
			}
			
			//determine whether or not we've hit an opacity where further sampling becomes neglible
			if(outputVal.w < 0.03125f){
				outputVal.w = 0.0f;
				break;
			}

		}else{

			//if we aren't backstepping, we can skip a sample
			if(!backStep){
				rayStart.x += rayInc.x;
				rayStart.y += rayInc.y;
				rayStart.z += rayInc.z;
				maxSteps--;
			}
			skipStep = !(backStep);

			//move to the next sample
			rayStart.x += rayInc.x;
			rayStart.y += rayInc.y;
			rayStart.z += rayInc.z;
			maxSteps--;
			backStep = false;

		}
		
	}//while

	//find the length of the ray unused and update the ray termination distance
	retDepth -= maxSteps;

	//adjust the opacity output to reflect the collected opacity, and not the remaining opacity
	outputVal.w = 1.0f - outputVal.w;

}

__global__ void CUDA_vtkCuda1DVolumeMapper_CUDAkernel_Composite( ) {
	
	//index in the output image (2D)
	int2 index;
	index.x = blockDim.x * blockIdx.x + threadIdx.x;
	index.y = blockDim.y * blockIdx.y + threadIdx.y;

	//index in the output image (1D)
	int outindex = index.x + index.y * outInfo.resolution.x;
	
	float3 rayStart; //ray starting point
	float3 rayInc; // ray sample increment
	float numSteps; //maximum number of samples along this ray
	float excludeStart; //where to start excluding
	float excludeEnd; //where to end excluding
	float4 outputVal; //rgba value of this ray (calculated in castRays, used in WriteData)
	float outputDepth; //depth to put in the cel shading array

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
	excludeStart = outInfo.excludeStart[outindex];
	__syncthreads();
	excludeEnd = outInfo.excludeEnd[outindex];
	__syncthreads();

	// trace along the ray (composite)
	CUDA_vtkCuda1DVolumeMapper_CUDAkernel_CastRays1D(rayStart, numSteps, excludeStart, excludeEnd, rayInc, outputVal, outputDepth);

	//convert output to uchar, adjusting it to be valued from [0,256) rather than [0,1]
	uchar4 temp;
	temp.x = 255.0f * outputVal.x;
	temp.y = 255.0f * outputVal.y;
	temp.z = 255.0f * outputVal.z;
	temp.w = 255.0f * outputVal.w;
	
	//place output in the image buffer
	__syncthreads();
	outInfo.deviceOutputImage[outindex] = temp;

	//write out the depth
	__syncthreads();
	outInfo.depthBuffer[outindex + outInfo.resolution.x] = outputDepth;
}

//pre: the resolution of the image has been processed such that it's x and y size are both multiples of 16 (enforced automatically) and y > 256 (enforced automatically)
//post: the OutputImage pointer will hold the ray casted information
bool CUDA_vtkCuda1DVolumeMapper_renderAlgo_doRender(const cudaOutputImageInformation& outputInfo,
							 const cudaRendererInformation& rendererInfo,
							 const cudaVolumeInformation& volumeInfo,
							 const cuda1DTransferFunctionInformation& transInfo,
							 cudaStream_t* stream)
{

	// setup execution parameters - staggered to improve parallelism
	cudaMemcpyToSymbolAsync(volInfo, &volumeInfo, sizeof(cudaVolumeInformation) );
	cudaMemcpyToSymbolAsync(renInfo, &rendererInfo, sizeof(cudaRendererInformation));
	cudaMemcpyToSymbolAsync(outInfo, &outputInfo, sizeof(cudaOutputImageInformation));
	cudaMemcpyToSymbolAsync(CUDA_vtkCuda1DVolumeMapper_trfInfo, &transInfo, sizeof(cuda1DTransferFunctionInformation));
	
	//create the necessary execution amount parameters from the block sizes and calculate th volume rendering integral
	int blockX = outputInfo.resolution.x / BLOCK_DIM2D ;
	int blockY = outputInfo.resolution.y / BLOCK_DIM2D ;

	dim3 grid(blockX, blockY, 1);
	dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);
	CUDAkernel_renderAlgo_formRays <<< grid, threads, 0, *stream >>>();
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "1D Rendering Error Status 1: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	CUDA_vtkCuda1DVolumeMapper_CUDAkernel_Composite <<< grid, threads, 0, *stream >>>();
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "1D Rendering Error Status 2: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	//shade the image
	grid.x = outputInfo.resolution.x*outputInfo.resolution.y / 256;
	grid.y = 1;
	threads.x = 256;
	threads.y = 1;
	CUDAkernel_shadeAlgo_normBuffer <<< grid, threads, 0, *stream >>>();
	CUDAkernel_shadeAlgo_doCelShade <<< grid, threads, 0, *stream >>>();
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "1D Rendering Error Status 3: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);
}

bool CUDA_vtkCuda1DVolumeMapper_renderAlgo_changeFrame(const int frame, cudaStream_t* stream){

	// set the texture to the correct image
	CUDA_vtkCuda1DVolumeMapper_input_texture.normalized = false;					// access with unnormalized texture coordinates
	CUDA_vtkCuda1DVolumeMapper_input_texture.filterMode = cudaFilterModeLinear;		// linear interpolation
	CUDA_vtkCuda1DVolumeMapper_input_texture.addressMode[0] = cudaAddressModeClamp;	// wrap texture coordinates
	CUDA_vtkCuda1DVolumeMapper_input_texture.addressMode[1] = cudaAddressModeClamp;
	CUDA_vtkCuda1DVolumeMapper_input_texture.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	cudaBindTextureToArray(CUDA_vtkCuda1DVolumeMapper_input_texture,
							CUDA_vtkCuda1DVolumeMapper_sourceDataArray[frame], channelDesc);
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "Change Frame Status: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);

}

//pre: the transfer functions are all of type float and are all of size FunctionSize
//post: the alpha, colorR, G and B 1D textures will map to each transfer function
bool CUDA_vtkCuda1DVolumeMapper_renderAlgo_loadTextures(const cuda1DTransferFunctionInformation& transInfo,
								  float* redTF, float* greenTF, float* blueTF, float* alphaTF,
								  cudaStream_t* stream){

	//retrieve the size of the transer functions
	size_t size = sizeof(float) * transInfo.functionSize;
		
	//define the texture mapping for the alpha component after copying information from host to device array
	if(alphaTransferArray1D)
		cudaFreeArray(alphaTransferArray1D);
	cudaMallocArray( &alphaTransferArray1D, &channelDesc, transInfo.functionSize, 1);
	cudaMemcpyToArrayAsync(alphaTransferArray1D, 0, 0, alphaTF, size, cudaMemcpyHostToDevice, *stream);
	alpha_texture_1D.normalized = true;
	alpha_texture_1D.filterMode = cudaFilterModeLinear;
	alpha_texture_1D.addressMode[0] = cudaAddressModeClamp;
	cudaBindTextureToArray(alpha_texture_1D, alphaTransferArray1D);
		
	//define the texture mapping for the red component after copying information from host to device array
	if(colorRTransferArray1D)
		cudaFreeArray(colorRTransferArray1D);
	cudaMallocArray( &colorRTransferArray1D, &channelDesc, transInfo.functionSize, 1);
	cudaMemcpyToArrayAsync(colorRTransferArray1D, 0, 0, redTF, size, cudaMemcpyHostToDevice, *stream);
	colorR_texture_1D.normalized = true;
	colorR_texture_1D.filterMode = cudaFilterModeLinear;
	colorR_texture_1D.addressMode[0] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorR_texture_1D, colorRTransferArray1D);
	
	//define the texture mapping for the green component after copying information from host to device array
	if(colorGTransferArray1D)
		cudaFreeArray(colorGTransferArray1D);
	cudaMallocArray( &colorGTransferArray1D, &channelDesc, transInfo.functionSize, 1);
	cudaMemcpyToArrayAsync(colorGTransferArray1D, 0, 0, greenTF, size, cudaMemcpyHostToDevice, *stream);
	colorG_texture_1D.normalized = true;
	colorG_texture_1D.filterMode = cudaFilterModeLinear;
	colorG_texture_1D.addressMode[0] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorG_texture_1D, colorGTransferArray1D);
	
	//define the texture mapping for the blue component after copying information from host to device array
	if(colorBTransferArray1D)
		cudaFreeArray(colorBTransferArray1D);
	cudaMallocArray( &colorBTransferArray1D, &channelDesc, transInfo.functionSize, 1);
	cudaMemcpyToArrayAsync(colorBTransferArray1D, 0, 0, blueTF, size, cudaMemcpyHostToDevice, *stream);
	colorB_texture_1D.normalized = true;
	colorB_texture_1D.filterMode = cudaFilterModeLinear;
	colorB_texture_1D.addressMode[0] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorB_texture_1D, colorBTransferArray1D);

	return (cudaGetLastError() == 0);

}

bool CUDA_vtkCuda1DVolumeMapper_renderAlgo_UnloadTextures(cudaStream_t* stream){

	if(colorRTransferArray1D)
		cudaFreeArray(colorRTransferArray1D);
	colorRTransferArray1D = 0;
	if(colorGTransferArray1D)
		cudaFreeArray(colorGTransferArray1D);
	colorGTransferArray1D = 0;
	if(colorBTransferArray1D)
		cudaFreeArray(colorBTransferArray1D);
	colorBTransferArray1D = 0;
	if(alphaTransferArray1D)
		cudaFreeArray(alphaTransferArray1D);
	alphaTransferArray1D = 0;

	return (cudaGetLastError() == 0);
}

//pre:	the data has been preprocessed by the volumeInformationHandler such that it is float data
//		the index is between 0 and 100
//post: the input_texture will map to the source data in voxel coordinate space
bool CUDA_vtkCuda1DVolumeMapper_renderAlgo_loadImageInfo(const float* data, const cudaVolumeInformation& volumeInfo,
														 const int index, cudaStream_t* stream){

	// if the array is already populated with information, free it to prevent leaking
	if(CUDA_vtkCuda1DVolumeMapper_sourceDataArray[index]){
		cudaFreeArray(CUDA_vtkCuda1DVolumeMapper_sourceDataArray[index]);
	}
	
	//define the size of the data, retrieved from the volume information
	cudaExtent volumeSize;
	volumeSize.width = volumeInfo.VolumeSize.x;
	volumeSize.height = volumeInfo.VolumeSize.y;
	volumeSize.depth = volumeInfo.VolumeSize.z;
	
	// create 3D array to store the image data in
	cudaMalloc3DArray(&(CUDA_vtkCuda1DVolumeMapper_sourceDataArray[index]), &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr( (void*) data, volumeSize.width*sizeof(float),
												volumeSize.width, volumeSize.height);
	copyParams.dstArray = CUDA_vtkCuda1DVolumeMapper_sourceDataArray[index];
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	return (cudaGetLastError() == 0);

}

void CUDA_vtkCuda1DVolumeMapper_renderAlgo_initImageArray(cudaStream_t* stream){
	for(int i = 0; i < 100; i++)
		CUDA_vtkCuda1DVolumeMapper_sourceDataArray[i] = 0;
}

void CUDA_vtkCuda1DVolumeMapper_renderAlgo_clearImageArray(cudaStream_t* stream){
	for(int i = 0; i < 100; i++){
		
		// if the array is already populated with information, free it to prevent leaking
		if(CUDA_vtkCuda1DVolumeMapper_sourceDataArray[i]){
			cudaFreeArray(CUDA_vtkCuda1DVolumeMapper_sourceDataArray[i]);
		}
		
		//null the pointer
		CUDA_vtkCuda1DVolumeMapper_sourceDataArray[i] = 0;
	}
}