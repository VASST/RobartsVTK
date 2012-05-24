#include "CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "CUDA_container2DInExTransferFunctionInformation.h"
#include <cuda.h>

//execution parameters and general information
__constant__ cuda2DInExTransferFunctionInformation	CUDA_vtkCuda2DInExVolumeMapper_trfInfo;

//3D input data (read-only texture with corresponding opague device memory back)
texture<float, 3, cudaReadModeElementType> CUDA_vtkCuda2DInExVolumeMapper_input_texture;
cudaArray* CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[100];

//transfer function as read-only textures
texture<float, 2, cudaReadModeElementType> alpha_texture_2DInEx;
texture<float, 2, cudaReadModeElementType> colorR_texture_2DInEx;
texture<float, 2, cudaReadModeElementType> colorG_texture_2DInEx;
texture<float, 2, cudaReadModeElementType> colorB_texture_2DInEx;
texture<float, 2, cudaReadModeElementType> inExLogic_texture_2DInEx;

__device__ void CUDA_vtkCuda2DInExVolumeMapper_CUDAkernel_CastRays(float3& rayStart,
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
	outputVal.w = 1.0f; //remaining A
		
	//fetch the required information about the size and range of the transfer function from memory to registers
	__syncthreads();
	const float functRangeLow = CUDA_vtkCuda2DInExVolumeMapper_trfInfo.intensityLow;
	const float functRangeMulti = CUDA_vtkCuda2DInExVolumeMapper_trfInfo.intensityMultiplier;
	const float gradRangeLow = CUDA_vtkCuda2DInExVolumeMapper_trfInfo.gradientLow;
	const float gradRangeMulti = CUDA_vtkCuda2DInExVolumeMapper_trfInfo.gradientMultiplier;
	const float gradRangeOffset = CUDA_vtkCuda2DInExVolumeMapper_trfInfo.gradientOffset;
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
	bool special = false;

	//reformat the exclusion indices to use the same ordering (counting downwards rather than upwards)
	excludeStart = maxSteps - excludeStart;
	excludeEnd = maxSteps - excludeEnd;

	//loop as long as we are still *roughly* in the range of the clipped and cropped volume
	while( maxSteps > 0 ){
	
		// fetching the intensity index into the transfer function
		const float tempIndex = functRangeMulti * (tex3D(CUDA_vtkCuda2DInExVolumeMapper_input_texture,
							 rayStart.x, rayStart.y, rayStart.z) - functRangeLow);
			
		//fetching the gradient index into the transfer function
		float3 gradient;
		gradient.x = ( tex3D(CUDA_vtkCuda2DInExVolumeMapper_input_texture, rayStart.x+1.0f, rayStart.y, rayStart.z)
					 - tex3D(CUDA_vtkCuda2DInExVolumeMapper_input_texture, rayStart.x-1.0f, rayStart.y, rayStart.z) ) * spaceX;
		gradient.y = ( tex3D(CUDA_vtkCuda2DInExVolumeMapper_input_texture, rayStart.x, rayStart.y+1.0f, rayStart.z)
					 - tex3D(CUDA_vtkCuda2DInExVolumeMapper_input_texture, rayStart.x, rayStart.y-1.0f, rayStart.z) ) * spaceY;
		gradient.z = ( tex3D(CUDA_vtkCuda2DInExVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z+1.0f)
					 - tex3D(CUDA_vtkCuda2DInExVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z-1.0f) ) * spaceZ;
		const float gradMag = gradRangeMulti * (__log2f(gradient.x*gradient.x+gradient.y*gradient.y
														+gradient.z*gradient.z+gradRangeOffset) + gradRangeLow);
	
		//fetching the opacity value of the sampling point (apply transfer function in stages to minimize work)
		float inEx = tex2D(inExLogic_texture_2DInEx, tempIndex, gradMag);
		float alpha = tex2D(alpha_texture_2DInEx, tempIndex, gradMag);

		//filter out objects with too low opacity (deemed unimportant, and this saves time and reduces cloudiness)
		if(alpha > 0.0f && tempIndex >= 0.0f && tempIndex <= 1.0f && gradMag >= 0.0f && gradMag <= 1.0f){

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

				//if we are being excluded, leave before accumulating the sample
				if( !special && !( excludeStart >= maxSteps && excludeEnd <= maxSteps )
						&& inEx > 0.5f ) break;
				special = true;

				//accumulate the opacity for this sample point
				outputVal.w *= alpha;

				//accumulate the colour information from this sample point
				outputVal.x += multiplier * tex2D(colorR_texture_2DInEx, tempIndex, gradMag);
				outputVal.y += multiplier * tex2D(colorG_texture_2DInEx, tempIndex, gradMag);
				outputVal.z += multiplier * tex2D(colorB_texture_2DInEx, tempIndex, gradMag);

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
	
	//if we have some alpha information, make the background black and opague
	outputVal.w = (outputVal.w < 1.0f - 0.03125f) ? 1.0f : 0.0f;
	
}

__global__ void CUDA_vtkCuda2DInExVolumeMapper_CUDAkernel_shadeAlgo_doCelShade()
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
	float gradMag = ( (depthDiffX.y - depthDiffX.x)*(depthDiffX.y - depthDiffX.x) +
								(depthDiffY.y - depthDiffY.x)*(depthDiffY.y - depthDiffY.x) );
	
	
	//grab cel shading parameters
	__syncthreads();
	float darkness = renInfo.celr;
	float a = renInfo.cela;
	float c = renInfo.celc;
	__syncthreads();
	
	//multiply by the depth factor
	gradMag = 1.0f - darkness * saturate( (gradMag - a) * c );
	
	//grab distance shading parameters
	__syncthreads();
	darkness = renInfo.disr;
	a = renInfo.disa;
	c = renInfo.disc;
	__syncthreads();
	
	//multiply by the depth factor
	gradMag = 1.0f - darkness * saturate( (depthDiffX.x - a) * c );
	
	uchar4 colour;
	__syncthreads();
	colour = outInfo.deviceOutputImage[outindex];
	__syncthreads();

	colour.x = gradMag * ((float) colour.x);
	colour.y = gradMag * ((float) colour.y);
	colour.z = gradMag * ((float) colour.z);
	colour.w = (colour.w > 0) ? colour.w : (char) (255.0f - 255.0f * gradMag);
	
	__syncthreads();
	outInfo.deviceOutputImage[outindex] = colour;
	
	return;
}

__global__ void CUDA_vtkCuda2DInExVolumeMapper_CUDAkernel_Composite( ) {
	
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
	CUDA_vtkCuda2DInExVolumeMapper_CUDAkernel_CastRays(rayStart, numSteps, excludeStart, excludeEnd,
														rayInc, outputVal, outputDepth);

	//convert output to uchar, adjusting it to be valued from [0,256) rather than [0,1]
	uchar4 temp;
	temp.x = 255.0f * saturate(outputVal.x);
	temp.y = 255.0f * saturate(outputVal.y);
	temp.z = 255.0f * saturate(outputVal.z);
	temp.w = 255.0f * saturate(outputVal.w);
	
	//place output in the image buffer
	__syncthreads();
	outInfo.deviceOutputImage[outindex] = temp;

	//write out the depth
	__syncthreads();
	outInfo.depthBuffer[outindex + outInfo.resolution.x] = outputDepth;
}

#include <stdio.h>

//pre: the resolution of the image has been processed such that it's x and y size are both multiples of 16 (enforced automatically) and y > 256 (enforced automatically)
//post: the OutputImage pointer will hold the ray casted information
bool CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_doRender(const cudaOutputImageInformation& outputInfo,
							 const cudaRendererInformation& rendererInfo,
							 const cudaVolumeInformation& volumeInfo,
							 const cuda2DInExTransferFunctionInformation& transInfo,
							 cudaStream_t* stream)
{

	// setup execution parameters - staggered to improve parallelism
	cudaMemcpyToSymbolAsync(volInfo, &volumeInfo, sizeof(cudaVolumeInformation), 0, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(renInfo, &rendererInfo, sizeof(cudaRendererInformation), 0, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(outInfo, &outputInfo, sizeof(cudaOutputImageInformation), 0, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(CUDA_vtkCuda2DInExVolumeMapper_trfInfo, &transInfo, sizeof(cuda2DInExTransferFunctionInformation), 0, cudaMemcpyHostToDevice, *stream);

	//bind the transfer functions to the used textures
	alpha_texture_2DInEx.normalized = true;
	alpha_texture_2DInEx.filterMode = cudaFilterModePoint;
	alpha_texture_2DInEx.addressMode[0] = cudaAddressModeClamp;
	alpha_texture_2DInEx.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(alpha_texture_2DInEx, transInfo.alphaTransferArray2D, channelDesc);
	colorR_texture_2DInEx.normalized = true;
	colorR_texture_2DInEx.filterMode = cudaFilterModePoint;
	colorR_texture_2DInEx.addressMode[0] = cudaAddressModeClamp;
	colorR_texture_2DInEx.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorR_texture_2DInEx, transInfo.colorRTransferArray2D, channelDesc);
	colorG_texture_2DInEx.normalized = true;
	colorG_texture_2DInEx.filterMode = cudaFilterModePoint;
	colorG_texture_2DInEx.addressMode[0] = cudaAddressModeClamp;
	colorG_texture_2DInEx.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorG_texture_2DInEx, transInfo.colorGTransferArray2D, channelDesc);
	colorB_texture_2DInEx.normalized = true;
	colorB_texture_2DInEx.filterMode = cudaFilterModePoint;
	colorB_texture_2DInEx.addressMode[0] = cudaAddressModeClamp;
	colorB_texture_2DInEx.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorB_texture_2DInEx, transInfo.colorBTransferArray2D, channelDesc);
	inExLogic_texture_2DInEx.normalized = true;
	inExLogic_texture_2DInEx.filterMode = cudaFilterModePoint;
	inExLogic_texture_2DInEx.addressMode[0] = cudaAddressModeClamp;
	inExLogic_texture_2DInEx.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(inExLogic_texture_2DInEx, transInfo.inExLogicTransferArray2D, channelDesc);
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "2D Rendering Error Status 0: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	//create the necessary execution amount parameters from the block sizes and calculate th volume rendering integral
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

	CUDA_vtkCuda2DInExVolumeMapper_CUDAkernel_Composite <<< grid, threads, 0, *stream >>>();
	
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
	CUDA_vtkCuda2DInExVolumeMapper_CUDAkernel_shadeAlgo_doCelShade <<< grid, threads, 0, *stream >>>();
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "2D Rendering Error Status 3: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);
}

bool CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_changeFrame(const int frame, cudaStream_t* stream){

	// set the texture to the correct image
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.normalized = false;						// access with unnormalized texture coordinates
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.filterMode = cudaFilterModeLinear;		// linear interpolation
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.addressMode[0] = cudaAddressModeClamp;	// wrap texture coordinates
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.addressMode[1] = cudaAddressModeClamp;
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	cudaBindTextureToArray(CUDA_vtkCuda2DInExVolumeMapper_input_texture,
							CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[frame], channelDesc);
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "Change Frame Status: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);
}

//pre: the transfer functions are all of type float and are all of size FunctionSize
//post: the alpha, colorR, G and B 2D textures will map to each transfer function
bool CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_loadTextures(cuda2DInExTransferFunctionInformation& transInfo,
								  float* redTF, float* greenTF, float* blueTF, float* alphaTF, float* inExTF, cudaStream_t* stream){

	//retrieve the size of the transer functions
	size_t size = sizeof(float) * transInfo.functionSize;
	
	//define the texture mapping for the alpha component after copying information from host to device array
	if(transInfo.alphaTransferArray2D)
		cudaFreeArray(transInfo.alphaTransferArray2D);
	cudaMallocArray( &(transInfo.alphaTransferArray2D), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.alphaTransferArray2D, 0, 0, alphaTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);
		
	//define the texture mapping for the red component after copying information from host to device array
	if(transInfo.colorRTransferArray2D)
		cudaFreeArray(transInfo.colorRTransferArray2D);
	cudaMallocArray( &(transInfo.colorRTransferArray2D), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.colorRTransferArray2D, 0, 0, redTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);
	
	//define the texture mapping for the green component after copying information from host to device array
	if(transInfo.colorGTransferArray2D)
		cudaFreeArray(transInfo.colorGTransferArray2D);
	cudaMallocArray( &(transInfo.colorGTransferArray2D), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.colorGTransferArray2D, 0, 0, greenTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);
	
	//define the texture mapping for the blue component after copying information from host to device array
	if(transInfo.colorBTransferArray2D)
		cudaFreeArray(transInfo.colorGTransferArray2D);
	cudaMallocArray( &(transInfo.colorBTransferArray2D), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.colorBTransferArray2D, 0, 0, blueTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);
	
	//define the texture mapping for the blue component after copying information from host to device array
	if(transInfo.inExLogicTransferArray2D)
		cudaFreeArray(transInfo.colorBTransferArray2D);
	cudaMallocArray( &(transInfo.inExLogicTransferArray2D), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.inExLogicTransferArray2D, 0, 0, inExTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);

	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "Bind transfer functions: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);
}

bool CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_unloadTextures(cuda2DInExTransferFunctionInformation& transInfo, cudaStream_t* stream){
	
	//define the texture mapping for the alpha component after copying information from host to device array
	if(transInfo.alphaTransferArray2D)
		cudaFreeArray(transInfo.alphaTransferArray2D);
	transInfo.alphaTransferArray2D = 0;
		
	//define the texture mapping for the red component after copying information from host to device array
	if(transInfo.colorRTransferArray2D)
		cudaFreeArray(transInfo.colorRTransferArray2D);
	transInfo.colorRTransferArray2D = 0;
	
	//define the texture mapping for the green component after copying information from host to device array
	if(transInfo.colorGTransferArray2D)
		cudaFreeArray(transInfo.colorGTransferArray2D);
	transInfo.colorGTransferArray2D = 0;
	
	//define the texture mapping for the blue component after copying information from host to device array
	if(transInfo.colorBTransferArray2D)
		cudaFreeArray(transInfo.colorGTransferArray2D);
	transInfo.colorGTransferArray2D = 0;
	
	//define the texture mapping for the blue component after copying information from host to device array
	if(transInfo.inExLogicTransferArray2D)
		cudaFreeArray(transInfo.colorBTransferArray2D);
	transInfo.colorBTransferArray2D = 0;
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "Bind transfer functions: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);
}

//pre:	the data has been preprocessed by the volumeInformationHandler such that it is float data
//		the index is between 0 and 100
//post: the input_texture will map to the source data in voxel coordinate space
bool CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_loadImageInfo(const float* data, const cudaVolumeInformation& volumeInfo, const int index, cudaStream_t* stream){

	// if the array is already populated with information, free it to prevent leaking
	if(CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[index])
		cudaFreeArray(CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[index]);
	
	//define the size of the data, retrieved from the volume information
	cudaExtent volumeSize;
	volumeSize.width = volumeInfo.VolumeSize.x;
	volumeSize.height = volumeInfo.VolumeSize.y;
	volumeSize.depth = volumeInfo.VolumeSize.z;
	
	// create 3D array to store the image data in
	cudaMalloc3DArray(&(CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[index]), &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr( (void*) data, volumeSize.width*sizeof(float),
												volumeSize.width, volumeSize.height);
	copyParams.dstArray = CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[index];
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3DAsync(&copyParams, *stream);

	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "Load volume information: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);

}

void CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_initImageArray(cudaStream_t* stream){
	for(int i = 0; i < 100; i++)
		CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[i] = 0;
}

void CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_clearImageArray(cudaStream_t* stream){
	for(int i = 0; i < 100; i++){
		
		// if the array is already populated with information, free it to prevent leaking
		if(CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[i])
			cudaFreeArray(CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[i]);
		
		//null the pointer
		CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[i] = 0;
	}
}