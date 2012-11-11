#include "CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "CUDA_containerDualImageTransferFunctionInformation.h"
#include <cuda.h>

//execution parameters and general information
__constant__ cudaDualImageTransferFunctionInformation	CUDA_vtkCudaDualImageVolumeMapper_trfInfo;

//transfer function as read-only textures
texture<float, 2, cudaReadModeElementType> alpha_texture_DualImage;
texture<float, 2, cudaReadModeElementType> ambient_texture_DualImage;
texture<float, 2, cudaReadModeElementType> diffuse_texture_DualImage;
texture<float, 2, cudaReadModeElementType> specular_texture_DualImage;
texture<float, 2, cudaReadModeElementType> specularPower_texture_DualImage;
texture<float, 2, cudaReadModeElementType> colorR_texture_DualImage;
texture<float, 2, cudaReadModeElementType> colorG_texture_DualImage;
texture<float, 2, cudaReadModeElementType> colorB_texture_DualImage;

//3D input data (read-only texture with corresponding opague device memory back)
texture<float2, 3, cudaReadModeElementType> CUDA_vtkCudaDualImageVolumeMapper_input_texture;
cudaArray* CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[100];

__device__ void CUDA_vtkCudaDualImageVolumeMapper_CUDAkernel_CastRays(float3& rayStart,
									const float& numSteps,
									int& excludeStart,
									int& excludeEnd,
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
	const float funct1RangeLow = CUDA_vtkCudaDualImageVolumeMapper_trfInfo.intensity1Low;
	const float funct1RangeMulti = CUDA_vtkCudaDualImageVolumeMapper_trfInfo.intensity1Multiplier;
	const float funct2RangeLow = CUDA_vtkCudaDualImageVolumeMapper_trfInfo.intensity2Low;
	const float funct2RangeMulti = CUDA_vtkCudaDualImageVolumeMapper_trfInfo.intensity2Multiplier;
	const float3 space = volInfo.SpacingReciprocal;
	const float3 incSpace = volInfo.Spacing;
	__syncthreads();

	//apply a randomized offset to the ray
	retDepth = dRandomRayOffsets[threadIdx.x + BLOCK_DIM2D * threadIdx.y];
	__syncthreads();
	int maxSteps = __float2int_rd(numSteps) - retDepth;
	rayStart.x += retDepth*rayInc.x;
	rayStart.y += retDepth*rayInc.y;
	rayStart.z += retDepth*rayInc.z;
	retDepth = maxSteps;
	float rayLength = sqrtf(rayInc.x*rayInc.x*incSpace.x*incSpace.x +
							rayInc.y*rayInc.y*incSpace.y*incSpace.y +
							rayInc.z*rayInc.z*incSpace.z*incSpace.z);

	//allocate flags
	char2 step;
	step.x = 0;
	step.y = 0;

	//reformat the exclusion indices to use the same ordering (counting downwards rather than upwards)
	excludeStart = maxSteps - excludeStart;
	excludeEnd = maxSteps - excludeEnd;

	//loop as long as we are still *roughly* in the range of the clipped and cropped volume
	while( maxSteps > 0 ){

		//if we are in the exclusion area, leave
		if( excludeStart > maxSteps && excludeEnd < maxSteps ){
			rayStart.x += (maxSteps-excludeEnd) * rayInc.x;
			rayStart.y += (maxSteps-excludeEnd) * rayInc.y;
			rayStart.z += (maxSteps-excludeEnd) * rayInc.z;
			maxSteps = excludeEnd;
			step.x = 0;
			step.y = 0;
			continue;
		}

		// fetching the intensity index into the transfer function
		float2 tempIndex = tex3D(CUDA_vtkCudaDualImageVolumeMapper_input_texture,
					rayStart.x, rayStart.y, rayStart.z);
		tempIndex.x = funct1RangeMulti * (tempIndex.x - funct1RangeLow);
		tempIndex.y = funct2RangeMulti * (tempIndex.y - funct2RangeLow);
			
		//fetching the gradient
		float3 gradient1;
		float3 gradient2;
		float2 tempGradient;
		tempGradient = tex3D(CUDA_vtkCudaDualImageVolumeMapper_input_texture, rayStart.x+1.0f, rayStart.y, rayStart.z);
		gradient1.x = tempGradient.x;
		gradient2.x = tempGradient.y;
		tempGradient = tex3D(CUDA_vtkCudaDualImageVolumeMapper_input_texture, rayStart.x-1.0f, rayStart.y, rayStart.z);
		gradient1.x = (gradient1.x - tempGradient.x) * space.x;
		gradient2.x = (gradient2.x - tempGradient.y) * space.x;
		tempGradient = tex3D(CUDA_vtkCudaDualImageVolumeMapper_input_texture, rayStart.x, rayStart.y+1.0f, rayStart.z);
		gradient1.y = tempGradient.x;
		gradient2.y = tempGradient.y;
		tempGradient = tex3D(CUDA_vtkCudaDualImageVolumeMapper_input_texture, rayStart.x, rayStart.y-1.0f, rayStart.z);
		gradient1.y = (gradient1.y - tempGradient.x) * space.y;
		gradient2.y = (gradient2.y - tempGradient.y) * space.y;
		tempGradient = tex3D(CUDA_vtkCudaDualImageVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z+1.0f);
		gradient1.z = tempGradient.x;
		gradient2.z = tempGradient.y;
		tempGradient = tex3D(CUDA_vtkCudaDualImageVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z-1.0f);
		gradient1.z = (gradient1.z - tempGradient.x) * space.z;
		gradient2.z = (gradient2.z - tempGradient.y) * space.z;
		float gradMag = sqrtf(gradient1.x*gradient1.x+gradient1.y*gradient1.y+gradient1.z*gradient1.z);
		float inverseGradMag1 = rsqrtf(gradient1.x*gradient1.x+gradient1.y*gradient1.y+gradient1.z*gradient1.z);
		float inverseGradMag2 = rsqrtf(gradient2.x*gradient2.x+gradient2.y*gradient2.y+gradient2.z*gradient2.z);
		float dot = gradient1.x*gradient2.x+gradient1.y*gradient2.y+gradient1.z*gradient2.z;
		dot *= inverseGradMag1*inverseGradMag2;
		
		//adjust shading
		float phongLambert = saturate( abs ( gradient1.x*rayInc.x*incSpace.x + 
											gradient1.y*rayInc.y*incSpace.y +
											gradient1.z*rayInc.z*incSpace.z 	) / (gradMag * rayLength) );
		float shadeD = tex2D(ambient_texture_DualImage, tempIndex.x, tempIndex.y)
						+ tex2D(diffuse_texture_DualImage, tempIndex.x, tempIndex.y) * phongLambert;
		float shadeS = tex2D(specular_texture_DualImage, tempIndex.x, tempIndex.y) * 
						pow( phongLambert, tex2D(specularPower_texture_DualImage, tempIndex.x, tempIndex.y) );

		//fetching the opacity value of the sampling point (apply transfer function in stages to minimize work)
		float alpha = tex2D(alpha_texture_DualImage, tempIndex.x, tempIndex.y);

		
		//filter out objects with too low opacity (deemed unimportant, and this saves time and reduces cloudiness)
		if(alpha > 0.0f){

			//collect the alpha difference (if we sample now)
			const float multiplier = outputVal.w * alpha;
			alpha = (1.0f - alpha);

			//determine which kind of step to make
			step.x = step.y;
			step.y = 0;

			//move to the next sample point (may involve moving backward)
			rayStart.x = rayStart.x + (step.x ? -rayInc.x : rayInc.x);
			rayStart.y = rayStart.y + (step.x ? -rayInc.y : rayInc.y);
			rayStart.z = rayStart.z + (step.x ? -rayInc.z : rayInc.z);
			maxSteps = maxSteps + (step.x ? 1 : -1);

			if(!step.x){
				//accumulate the opacity for this sample point
				outputVal.w *= alpha;

				//accumulate the colour information from this sample point
				outputVal.x += multiplier * saturate(shadeD*tex2D(colorR_texture_DualImage, tempIndex.x, tempIndex.y)+shadeS);
				outputVal.y += multiplier * saturate(shadeD*tex2D(colorG_texture_DualImage, tempIndex.x, tempIndex.y)+shadeS);
				outputVal.z += multiplier * saturate(shadeD*tex2D(colorB_texture_DualImage, tempIndex.x, tempIndex.y)+shadeS);
			}
			
			//determine whether or not we've hit an opacity where further sampling becomes neglible
			if(outputVal.w < 0.03125f){
				outputVal.w = 0.0f;
				break;
			}


		}else{

			//if we aren't backstepping, we can skip a sample
			if(!step.x){
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

	//find the length of the ray unused and update the ray termination distance
	retDepth -= maxSteps;

	//adjust the opacity output to reflect the collected opacity, and not the remaining opacity
	outputVal.w = 1.0f - outputVal.w;

}

__global__ void CUDA_vtkCudaDualImageVolumeMapper_CUDAkernel_Composite( ) {
	
	//index in the output image (DualImage)
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
	excludeStart = __float2int_ru(outInfo.excludeStart[outindex]);
	__syncthreads();
	excludeEnd = __float2int_rd(outInfo.excludeEnd[outindex]);
	__syncthreads();

	// trace along the ray (composite)
	CUDA_vtkCudaDualImageVolumeMapper_CUDAkernel_CastRays(rayStart, numSteps, excludeStart, excludeEnd, rayInc, outputVal, outputDepth);

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
bool CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_doRender(const cudaOutputImageInformation& outputInfo,
							 const cudaRendererInformation& rendererInfo,
							 const cudaVolumeInformation& volumeInfo,
							 const cudaDualImageTransferFunctionInformation& transInfo,
							 cudaStream_t* stream)
{

	// setup execution parameters - staggered to improve parallelism
	cudaMemcpyToSymbolAsync(volInfo, &volumeInfo, sizeof(cudaVolumeInformation), 0, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(renInfo, &rendererInfo, sizeof(cudaRendererInformation), 0, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(outInfo, &outputInfo, sizeof(cudaOutputImageInformation), 0, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(CUDA_vtkCudaDualImageVolumeMapper_trfInfo, &transInfo, sizeof(cudaDualImageTransferFunctionInformation), 0, cudaMemcpyHostToDevice, *stream);
	
	//bind transfer function textures
	alpha_texture_DualImage.normalized = true;
	alpha_texture_DualImage.filterMode = cudaFilterModePoint;
	alpha_texture_DualImage.addressMode[0] = cudaAddressModeClamp;
	alpha_texture_DualImage.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(alpha_texture_DualImage, transInfo.alphaTransferArrayDualImage, channelDesc);
	colorR_texture_DualImage.normalized = true;
	colorR_texture_DualImage.filterMode = cudaFilterModePoint;
	colorR_texture_DualImage.addressMode[0] = cudaAddressModeClamp;
	colorR_texture_DualImage.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorR_texture_DualImage, transInfo.colorRTransferArrayDualImage, channelDesc);
	colorG_texture_DualImage.normalized = true;
	colorG_texture_DualImage.filterMode = cudaFilterModePoint;
	colorG_texture_DualImage.addressMode[0] = cudaAddressModeClamp;
	colorG_texture_DualImage.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorG_texture_DualImage, transInfo.colorGTransferArrayDualImage, channelDesc);
	colorB_texture_DualImage.normalized = true;
	colorB_texture_DualImage.filterMode = cudaFilterModePoint;
	colorB_texture_DualImage.addressMode[0] = cudaAddressModeClamp;
	colorB_texture_DualImage.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorB_texture_DualImage, transInfo.colorBTransferArrayDualImage, channelDesc);
	
	ambient_texture_DualImage.normalized = true;
	ambient_texture_DualImage.filterMode = cudaFilterModePoint;
	ambient_texture_DualImage.addressMode[0] = cudaAddressModeClamp;
	ambient_texture_DualImage.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(ambient_texture_DualImage, transInfo.ambientTransferArrayDualImage, channelDesc);
	diffuse_texture_DualImage.normalized = true;
	diffuse_texture_DualImage.filterMode = cudaFilterModePoint;
	diffuse_texture_DualImage.addressMode[0] = cudaAddressModeClamp;
	diffuse_texture_DualImage.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(diffuse_texture_DualImage, transInfo.diffuseTransferArrayDualImage, channelDesc);
	specular_texture_DualImage.normalized = true;
	specular_texture_DualImage.filterMode = cudaFilterModePoint;
	specular_texture_DualImage.addressMode[0] = cudaAddressModeClamp;
	specular_texture_DualImage.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(specular_texture_DualImage, transInfo.specularTransferArrayDualImage, channelDesc);
	specularPower_texture_DualImage.normalized = true;
	specularPower_texture_DualImage.filterMode = cudaFilterModePoint;
	specularPower_texture_DualImage.addressMode[0] = cudaAddressModeClamp;
	specularPower_texture_DualImage.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(specularPower_texture_DualImage, transInfo.specularPowerTransferArrayDualImage, channelDesc);

	//create the necessary execution amount parameters from the block sizes and calculate the volume rendering integral
	int blockX = outputInfo.resolution.x / BLOCK_DIM2D;
	int blockY = outputInfo.resolution.y / BLOCK_DIM2D;

	dim3 grid(blockX, blockY, 1);
	dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);
	CUDAkernel_renderAlgo_formRays <<< grid, threads, 0, *stream >>>();
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "DualImage Rendering Error Status 1: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	CUDA_vtkCudaDualImageVolumeMapper_CUDAkernel_Composite <<< grid, threads, 0, *stream >>>();
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "DualImage Rendering Error Status 2: " );
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
		printf( "DualImage Rendering Error Status 3: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
	
	return (cudaGetLastError() == 0);
}

bool CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_changeFrame(const int frame, cudaStream_t* stream){

	// set the texture to the correct image
	CUDA_vtkCudaDualImageVolumeMapper_input_texture.normalized = false;					// access with unnormalized texture coordinates
	CUDA_vtkCudaDualImageVolumeMapper_input_texture.filterMode = cudaFilterModeLinear;		// linear interpolation
	CUDA_vtkCudaDualImageVolumeMapper_input_texture.addressMode[0] = cudaAddressModeClamp;	// wrap texture coordinates
	CUDA_vtkCudaDualImageVolumeMapper_input_texture.addressMode[1] = cudaAddressModeClamp;
	CUDA_vtkCudaDualImageVolumeMapper_input_texture.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	cudaBindTextureToArray(CUDA_vtkCudaDualImageVolumeMapper_input_texture,
							CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[frame], channelDesc2);
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "Change Frame Status: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);
}

//pre: the transfer functions are all of type float and are all of size FunctionSize
//post: the alpha, colorR, G and B DualImage textures will map to each transfer function
bool CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_loadTextures(cudaDualImageTransferFunctionInformation& transInfo,
								  float* redTF, float* greenTF, float* blueTF, float* alphaTF,
								  float* ambTF, float* diffTF, float* specTF, float* powTF,
								  cudaStream_t* stream){

	//retrieve the size of the transer functions
	size_t size = sizeof(float) * transInfo.functionSize;
	
	//define the texture mapping for the alpha component after copying information from host to device array
	if(transInfo.alphaTransferArrayDualImage)
		cudaFreeArray(transInfo.alphaTransferArrayDualImage);
	cudaMallocArray( &(transInfo.alphaTransferArrayDualImage), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.alphaTransferArrayDualImage, 0, 0, alphaTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);

	//define the texture mapping for the red component after copying information from host to device array
	if(transInfo.colorRTransferArrayDualImage)
		cudaFreeArray(transInfo.colorRTransferArrayDualImage);
	cudaMallocArray( &(transInfo.colorRTransferArrayDualImage), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.colorRTransferArrayDualImage, 0, 0, redTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);

	//define the texture mapping for the green component after copying information from host to device array
	if(transInfo.colorGTransferArrayDualImage)
		cudaFreeArray(transInfo.colorGTransferArrayDualImage);
	cudaMallocArray( &(transInfo.colorGTransferArrayDualImage), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.colorGTransferArrayDualImage, 0, 0, greenTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);

	//define the texture mapping for the blue component after copying information from host to device array
	if(transInfo.colorBTransferArrayDualImage)
		cudaFreeArray(transInfo.colorBTransferArrayDualImage);
	cudaMallocArray( &(transInfo.colorBTransferArrayDualImage), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.colorBTransferArrayDualImage, 0, 0, blueTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);
	
	//define the texture mapping for the ambient component after copying information from host to device array
	if(transInfo.ambientTransferArrayDualImage)
		cudaFreeArray(transInfo.ambientTransferArrayDualImage);
	cudaMallocArray( &(transInfo.ambientTransferArrayDualImage), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.ambientTransferArrayDualImage, 0, 0, ambTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);

	//define the texture mapping for the ambient component after copying information from host to device array
	if(transInfo.diffuseTransferArrayDualImage)
		cudaFreeArray(transInfo.diffuseTransferArrayDualImage);
	cudaMallocArray( &(transInfo.diffuseTransferArrayDualImage), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.diffuseTransferArrayDualImage, 0, 0, diffTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);
	
	//define the texture mapping for the ambient component after copying information from host to device array
	if(transInfo.specularTransferArrayDualImage)
		cudaFreeArray(transInfo.specularTransferArrayDualImage);
	cudaMallocArray( &(transInfo.specularTransferArrayDualImage), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.specularTransferArrayDualImage, 0, 0, specTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);

	//define the texture mapping for the specular power component after copying information from host to device array
	if(transInfo.specularPowerTransferArrayDualImage)
		cudaFreeArray(transInfo.specularPowerTransferArrayDualImage);
	cudaMallocArray( &(transInfo.specularPowerTransferArrayDualImage), &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMemcpyToArrayAsync(transInfo.specularPowerTransferArrayDualImage, 0, 0, powTF, size*transInfo.functionSize, cudaMemcpyHostToDevice, *stream);

	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "Bind transfer functions: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);
}

bool CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_unloadTextures(cudaDualImageTransferFunctionInformation& transInfo, cudaStream_t* stream ){
	
	//define the texture mapping for the alpha component after copying information from host to device array
	if(transInfo.alphaTransferArrayDualImage)
		cudaFreeArray(transInfo.alphaTransferArrayDualImage);
	transInfo.alphaTransferArrayDualImage = 0;

	//define the texture mapping for the red component after copying information from host to device array
	if(transInfo.colorRTransferArrayDualImage)
		cudaFreeArray(transInfo.colorRTransferArrayDualImage);
	transInfo.colorRTransferArrayDualImage = 0;

	//define the texture mapping for the green component after copying information from host to device array
	if(transInfo.colorGTransferArrayDualImage)
		cudaFreeArray(transInfo.colorGTransferArrayDualImage);
	transInfo.colorGTransferArrayDualImage = 0;

	//define the texture mapping for the blue component after copying information from host to device array
	if(transInfo.colorBTransferArrayDualImage)
		cudaFreeArray(transInfo.colorBTransferArrayDualImage);
	transInfo.colorBTransferArrayDualImage = 0;
	
	//define the texture mapping for the ambient component after copying information from host to device array
	if(transInfo.ambientTransferArrayDualImage)
		cudaFreeArray(transInfo.ambientTransferArrayDualImage);
	transInfo.ambientTransferArrayDualImage = 0;

	//define the texture mapping for the ambient component after copying information from host to device array
	if(transInfo.diffuseTransferArrayDualImage)
		cudaFreeArray(transInfo.diffuseTransferArrayDualImage);
	transInfo.diffuseTransferArrayDualImage = 0;

	//define the texture mapping for the ambient component after copying information from host to device array
	if(transInfo.specularTransferArrayDualImage)
		cudaFreeArray(transInfo.specularTransferArrayDualImage);
	transInfo.specularTransferArrayDualImage = 0;

	//define the texture mapping for the specular power component after copying information from host to device array
	if(transInfo.specularPowerTransferArrayDualImage)
		cudaFreeArray(transInfo.specularPowerTransferArrayDualImage);
	transInfo.specularPowerTransferArrayDualImage = 0;

	return (cudaGetLastError() == 0);
}

//pre:	the data has been preprocessed by the volumeInformationHandler such that it is float data
//		the index is between 0 and 100
//post: the input_texture will map to the source data in voxel coordinate space
bool CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_loadImageInfo(const float* data, const cudaVolumeInformation& volumeInfo, const int index, cudaStream_t* stream){

	// if the array is already populated with information, free it to prevent leaking
	if(CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[index])
		cudaFreeArray(CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[index]);
	
	//define the size of the data, retrieved from the volume information
	cudaExtent volumeSize;
	volumeSize.width = volumeInfo.VolumeSize.x;
	volumeSize.height = volumeInfo.VolumeSize.y;
	volumeSize.depth = volumeInfo.VolumeSize.z;
	
	// create 3D array to store the image data in
	cudaMalloc3DArray(&(CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[index]), &channelDesc2, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr( (void*) data, volumeSize.width*sizeof(float2),
												volumeSize.width, volumeSize.height);
	copyParams.dstArray = CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[index];
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

void CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_initImageArray(cudaStream_t* stream){
	for(int i = 0; i < 100; i++)
		CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[i] = 0;
}

void CUDA_vtkCudaDualImageVolumeMapper_renderAlgo_clearImageArray(cudaStream_t* stream){
	for(int i = 0; i < 100; i++){
		
		// if the array is already populated with information, free it to prevent leaking
		if(CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[i])
			cudaFreeArray(CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[i]);
		
		//null the pointer
		CUDA_vtkCudaDualImageVolumeMapper_sourceDataArray[i] = 0;
	}
}