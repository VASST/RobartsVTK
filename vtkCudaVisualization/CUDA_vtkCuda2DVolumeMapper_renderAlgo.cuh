#include "CUDA_vtkCuda2DVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "CUDA_container2DTransferFunctionInformation.h"
#include <cuda.h>

//execution parameters and general information
__constant__ cuda2DTransferFunctionInformation	CUDA_vtkCuda2DVolumeMapper_trfInfo;
	
//transfer function as read-only textures
texture<float, 2, cudaReadModeElementType> alpha_texture_2D;
texture<float, 2, cudaReadModeElementType> colorR_texture_2D;
texture<float, 2, cudaReadModeElementType> colorG_texture_2D;
texture<float, 2, cudaReadModeElementType> colorB_texture_2D;
texture<float, 2, cudaReadModeElementType> ambient_texture_2D;
texture<float, 2, cudaReadModeElementType> diffuse_texture_2D;
texture<float, 2, cudaReadModeElementType> specular_texture_2D;
texture<float, 2, cudaReadModeElementType> specularPower_texture_2D;
texture<float, 2, cudaReadModeElementType> k_alpha_texture_2D;
texture<float, 2, cudaReadModeElementType> k_colorR_texture_2D;
texture<float, 2, cudaReadModeElementType> k_colorG_texture_2D;
texture<float, 2, cudaReadModeElementType> k_colorB_texture_2D;
texture<float, 2, cudaReadModeElementType> k_ambient_texture_2D;
texture<float, 2, cudaReadModeElementType> k_diffuse_texture_2D;
texture<float, 2, cudaReadModeElementType> k_specular_texture_2D;
texture<float, 2, cudaReadModeElementType> k_specularPower_texture_2D;

//3D input data (read-only texture with corresponding opague device memory back)
texture<float, 3, cudaReadModeElementType> CUDA_vtkCuda2DVolumeMapper_input_texture;

__device__ void CUDA_vtkCuda2DVolumeMapper_CUDAkernel_CastRays_WithDual(float3& rayStart,
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
	const float functRangeLow = CUDA_vtkCuda2DVolumeMapper_trfInfo.intensityLow;
	const float functRangeMulti = CUDA_vtkCuda2DVolumeMapper_trfInfo.intensityMultiplier;
	const float gradRangeLow = CUDA_vtkCuda2DVolumeMapper_trfInfo.gradientLow;
	const float gradRangeMulti = CUDA_vtkCuda2DVolumeMapper_trfInfo.gradientMultiplier;
	const float gradRangeOffset = CUDA_vtkCuda2DVolumeMapper_trfInfo.gradientOffset;
	const float3 space = volInfo.SpacingReciprocal;
	const float3 incSpace = volInfo.Spacing;
	__syncthreads();

	//apply a randomized offset to the ray
	retDepth = dRandomRayOffsets[threadIdx.x + BLOCK_DIM2D * threadIdx.y];
	__syncthreads();
	int maxSteps = __float2int_rd(numSteps - retDepth);
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
		bool inKeyhole = (excludeStart >= maxSteps && excludeEnd < maxSteps);

		// fetching the intensity index into the transfer function
		const float tempIndex = functRangeMulti * (tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture,
					rayStart.x, rayStart.y, rayStart.z) - functRangeLow);
			
		//fetching the gradient index into the transfer function
		float3 gradient;
		gradient.x = ( tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x+1.0f, rayStart.y, rayStart.z)
					 - tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x-1.0f, rayStart.y, rayStart.z) ) * space.x;
		gradient.y = ( tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x, rayStart.y+1.0f, rayStart.z)
					 - tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x, rayStart.y-1.0f, rayStart.z) ) * space.y;
		gradient.z = ( tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z+1.0f)
					 - tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z-1.0f) ) * space.z;
		float gradMag = gradient.x*gradient.x+gradient.y*gradient.y+gradient.z*gradient.z;
		const float gradMagIndex = gradRangeMulti * (__log2f(gradMag+gradRangeOffset) + gradRangeLow);
		gradMag = sqrtf( gradMag );
		
		//fetching the opacity value of the sampling point (apply transfer function in stages to minimize work)
		float alpha = inKeyhole ? tex2D(k_alpha_texture_2D, tempIndex, gradMagIndex) :
								  tex2D(alpha_texture_2D, tempIndex, gradMagIndex) ;

		//adjust shading
		float phongLambert = saturate( abs ( gradient.x*rayInc.x*incSpace.x + 
											gradient.y*rayInc.y*incSpace.y +
											gradient.z*rayInc.z*incSpace.z 	) / (gradMag * rayLength) );
		float shadeD = ( inKeyhole ? tex2D(k_ambient_texture_2D, tempIndex, gradMagIndex) :
								     tex2D(ambient_texture_2D, tempIndex, gradMagIndex) ) +
					   ( inKeyhole ? tex2D(k_diffuse_texture_2D, tempIndex, gradMagIndex) :
								     tex2D(diffuse_texture_2D, tempIndex, gradMagIndex) ) * phongLambert;
		float shadeS = ( inKeyhole ? tex2D(k_specular_texture_2D, tempIndex, gradMagIndex) :
								     tex2D(specular_texture_2D, tempIndex, gradMagIndex) ) * 
						pow( phongLambert, ( inKeyhole ? tex2D(k_specularPower_texture_2D, tempIndex, gradMagIndex) :
														 tex2D(specularPower_texture_2D, tempIndex, gradMagIndex) ) );

		//filter out objects with too low opacity (deemed unimportant, and this saves time and reduces cloudiness)
		if(alpha > 0.0f){

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
				float multiplier = outputVal.w * alpha;
				outputVal.w *= (1.0f - alpha);

				//accumulate the colour information from this sample point
				if( inKeyhole ){
					outputVal.x += multiplier * saturate( shadeD*tex2D(k_colorR_texture_2D, tempIndex, gradMagIndex) + shadeS );
					outputVal.y += multiplier * saturate( shadeD*tex2D(k_colorG_texture_2D, tempIndex, gradMagIndex) + shadeS );
					outputVal.z += multiplier * saturate( shadeD*tex2D(k_colorB_texture_2D, tempIndex, gradMagIndex) + shadeS );
				}else{
					outputVal.x += multiplier * saturate( shadeD*tex2D(colorR_texture_2D, tempIndex, gradMagIndex) + shadeS );
					outputVal.y += multiplier * saturate( shadeD*tex2D(colorG_texture_2D, tempIndex, gradMagIndex) + shadeS );
					outputVal.z += multiplier * saturate( shadeD*tex2D(colorB_texture_2D, tempIndex, gradMagIndex) + shadeS );
				}
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

__device__ void CUDA_vtkCuda2DVolumeMapper_CUDAkernel_CastRays_WithoutDual(float3& rayStart,
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
	const float functRangeLow = CUDA_vtkCuda2DVolumeMapper_trfInfo.intensityLow;
	const float functRangeMulti = CUDA_vtkCuda2DVolumeMapper_trfInfo.intensityMultiplier;
	const float gradRangeLow = CUDA_vtkCuda2DVolumeMapper_trfInfo.gradientLow;
	const float gradRangeMulti = CUDA_vtkCuda2DVolumeMapper_trfInfo.gradientMultiplier;
	const float gradRangeOffset = CUDA_vtkCuda2DVolumeMapper_trfInfo.gradientOffset;
	const float3 space = volInfo.SpacingReciprocal;
	const float3 incSpace = volInfo.Spacing;
	__syncthreads();

	//apply a randomized offset to the ray
	retDepth = dRandomRayOffsets[threadIdx.x + BLOCK_DIM2D * threadIdx.y];
	__syncthreads();
	int maxSteps = __float2int_rd(numSteps - retDepth);
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
		if( excludeStart >= maxSteps && excludeEnd < maxSteps ){
			rayStart.x += (maxSteps-excludeEnd) * rayInc.x;
			rayStart.y += (maxSteps-excludeEnd) * rayInc.y;
			rayStart.z += (maxSteps-excludeEnd) * rayInc.z;
			maxSteps = excludeEnd;
			step.x = 0;
			step.y = 0;
			continue;
		}

		// fetching the intensity index into the transfer function
		const float tempIndex = functRangeMulti * (tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture,
					rayStart.x, rayStart.y, rayStart.z) - functRangeLow);
			
		//fetching the gradient index into the transfer function
		float3 gradient;
		gradient.x = ( tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x+1.0f, rayStart.y, rayStart.z)
					 - tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x-1.0f, rayStart.y, rayStart.z) ) * space.x;
		gradient.y = ( tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x, rayStart.y+1.0f, rayStart.z)
					 - tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x, rayStart.y-1.0f, rayStart.z) ) * space.y;
		gradient.z = ( tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z+1.0f)
					 - tex3D(CUDA_vtkCuda2DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z-1.0f) ) * space.z;
		float gradMag = gradient.x*gradient.x+gradient.y*gradient.y+gradient.z*gradient.z;
		const float gradMagIndex = gradRangeMulti * (__log2f(gradMag+gradRangeOffset) + gradRangeLow);
		gradMag = sqrtf( gradMag );
		
		//fetching the opacity value of the sampling point (apply transfer function in stages to minimize work)
		float alpha = tex2D(alpha_texture_2D, tempIndex, gradMagIndex);

		//adjust shading
		float phongLambert = saturate( abs ( gradient.x*rayInc.x*incSpace.x + 
											gradient.y*rayInc.y*incSpace.y +
											gradient.z*rayInc.z*incSpace.z 	) / (gradMag * rayLength) );
		float shadeD = tex2D(ambient_texture_2D, tempIndex, gradMagIndex)
						+ tex2D(diffuse_texture_2D, tempIndex, gradMagIndex) * phongLambert;
		float shadeS = tex2D(specular_texture_2D, tempIndex, gradMagIndex) * 
						pow( phongLambert, tex2D(specularPower_texture_2D, tempIndex, gradMagIndex) );

		//filter out objects with too low opacity (deemed unimportant, and this saves time and reduces cloudiness)
		if(alpha > 0.0f){

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
				float multiplier = outputVal.w * alpha;
				outputVal.w *= (1.0f - alpha);

				//accumulate the colour information from this sample point
				outputVal.x += multiplier * saturate( shadeD*tex2D(colorR_texture_2D, tempIndex, gradMagIndex) + shadeS );
				outputVal.y += multiplier * saturate( shadeD*tex2D(colorG_texture_2D, tempIndex, gradMagIndex) + shadeS );
				outputVal.z += multiplier * saturate( shadeD*tex2D(colorB_texture_2D, tempIndex, gradMagIndex) + shadeS );
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

__global__ void CUDA_vtkCuda2DVolumeMapper_CUDAkernel_Composite_WithoutDual( ) {
	
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
	excludeStart = __float2int_rd(outInfo.excludeStart[outindex]);
	__syncthreads();
	excludeEnd = __float2int_ru(outInfo.excludeEnd[outindex]);
	__syncthreads();

	// trace along the ray (composite)
	CUDA_vtkCuda2DVolumeMapper_CUDAkernel_CastRays_WithoutDual(rayStart, numSteps, excludeStart, excludeEnd, rayInc, outputVal, outputDepth);

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


__global__ void CUDA_vtkCuda2DVolumeMapper_CUDAkernel_Composite_WithDual( ) {
	
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
	CUDA_vtkCuda2DVolumeMapper_CUDAkernel_CastRays_WithDual(rayStart, numSteps, excludeStart, excludeEnd, rayInc, outputVal, outputDepth);

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

bool CUDA_vtkCuda2DVolumeMapper_renderAlgo_changeFrame(const cudaArray* frame, cudaStream_t* stream){

	// set the texture to the correct image
	CUDA_vtkCuda2DVolumeMapper_input_texture.normalized = false;					// access with unnormalized texture coordinates
	CUDA_vtkCuda2DVolumeMapper_input_texture.filterMode = cudaFilterModeLinear;		// linear interpolation
	CUDA_vtkCuda2DVolumeMapper_input_texture.addressMode[0] = cudaAddressModeClamp;	// wrap texture coordinates
	CUDA_vtkCuda2DVolumeMapper_input_texture.addressMode[1] = cudaAddressModeClamp;
	CUDA_vtkCuda2DVolumeMapper_input_texture.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	cudaBindTextureToArray(CUDA_vtkCuda2DVolumeMapper_input_texture, frame, channelDesc);
	
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
bool CUDA_vtkCuda2DVolumeMapper_renderAlgo_doRender(const cudaOutputImageInformation& outputInfo,
							 const cudaRendererInformation& rendererInfo,
							 const cudaVolumeInformation& volumeInfo,
							 const cuda2DTransferFunctionInformation& transInfo,
							 cudaArray* frame,
							 cudaStream_t* stream)
{

	// setup execution parameters - staggered to improve parallelism
	cudaMemcpyToSymbolAsync(volInfo, &volumeInfo, sizeof(cudaVolumeInformation), 0, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(renInfo, &rendererInfo, sizeof(cudaRendererInformation), 0, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(outInfo, &outputInfo, sizeof(cudaOutputImageInformation), 0, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(CUDA_vtkCuda2DVolumeMapper_trfInfo, &transInfo, sizeof(cuda2DTransferFunctionInformation), 0, cudaMemcpyHostToDevice, *stream);
	
	//bind the input data texture to the provided frame
	CUDA_vtkCuda2DVolumeMapper_renderAlgo_changeFrame(frame,stream);
	
	//bind transfer function textures
	bindSingle2DTexture(alpha_texture_2D,transInfo.alphaTransferArray2D);
	bindSingle2DTexture(colorR_texture_2D,transInfo.colorRTransferArray2D);
	bindSingle2DTexture(colorG_texture_2D,transInfo.colorGTransferArray2D);
	bindSingle2DTexture(colorB_texture_2D,transInfo.colorBTransferArray2D);
	bindSingle2DTexture(ambient_texture_2D,transInfo.ambientTransferArray2D);
	bindSingle2DTexture(diffuse_texture_2D,transInfo.diffuseTransferArray2D);
	bindSingle2DTexture(specular_texture_2D,transInfo.specularTransferArray2D);
	bindSingle2DTexture(specularPower_texture_2D,transInfo.specularPowerTransferArray2D);
	
	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "DualImage Rendering Error Status 0.1: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	if( transInfo.useSecondTransferFunction ){
		bindSingle2DTexture(k_alpha_texture_2D,transInfo.K_alphaTransferArray2D);
		bindSingle2DTexture(k_colorR_texture_2D,transInfo.K_colorRTransferArray2D);
		bindSingle2DTexture(k_colorG_texture_2D,transInfo.K_colorGTransferArray2D);
		bindSingle2DTexture(k_colorB_texture_2D,transInfo.K_colorBTransferArray2D);
		bindSingle2DTexture(k_ambient_texture_2D,transInfo.K_ambientTransferArray2D);
		bindSingle2DTexture(k_diffuse_texture_2D,transInfo.K_diffuseTransferArray2D);
		bindSingle2DTexture(k_specular_texture_2D,transInfo.K_specularTransferArray2D);
		bindSingle2DTexture(k_specularPower_texture_2D,transInfo.K_specularPowerTransferArray2D);
		
		#ifdef DEBUG_VTKCUDAVISUALIZATION
			cudaThreadSynchronize();
			printf( "DualImage Rendering Error Status 0.2: " );
			printf( cudaGetErrorString( cudaGetLastError() ) );
			printf( "\n" );
		#endif
	}

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
	
	if( transInfo.useSecondTransferFunction )
		CUDA_vtkCuda2DVolumeMapper_CUDAkernel_Composite_WithDual <<< grid, threads, 0, *stream >>>();
	else
		CUDA_vtkCuda2DVolumeMapper_CUDAkernel_Composite_WithoutDual <<< grid, threads, 0, *stream >>>();
	
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
	CUDAkernel_shadeAlgo_doCelShade <<< grid, threads, 0, *stream >>>();

	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "2D Rendering Error Status 3: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
	
	return (cudaGetLastError() == 0);
}

//pre: the transfer functions are all of type float and are all of size FunctionSize
//post: the alpha, colorR, G and B 2D textures will map to each transfer function
bool CUDA_vtkCuda2DVolumeMapper_renderAlgo_loadTextures(cuda2DTransferFunctionInformation& transInfo,
								  float* redTF, float* greenTF, float* blueTF, float* alphaTF,
								  float* ambTF, float* diffTF, float* specTF, float* powTF,
								  float* kredTF, float* kgreenTF, float* kblueTF, float* kalphaTF,
								  float* kambTF, float* kdiffTF, float* kspecTF, float* kpowTF,
								  cudaStream_t* stream){
								  
	//define the texture mapping for the first TF's components after copying information from host to device array
	load2DArray(transInfo.alphaTransferArray2D, alphaTF, transInfo.functionSize, *stream);
	load2DArray(transInfo.colorRTransferArray2D, redTF, transInfo.functionSize, *stream);
	load2DArray(transInfo.colorGTransferArray2D, greenTF, transInfo.functionSize, *stream);
	load2DArray(transInfo.colorBTransferArray2D, blueTF, transInfo.functionSize, *stream);
	load2DArray(transInfo.ambientTransferArray2D, ambTF, transInfo.functionSize, *stream);
	load2DArray(transInfo.diffuseTransferArray2D, diffTF, transInfo.functionSize, *stream);
	load2DArray(transInfo.specularTransferArray2D, specTF, transInfo.functionSize, *stream);
	load2DArray(transInfo.specularPowerTransferArray2D, powTF, transInfo.functionSize, *stream);

	//define the texture mapping for the second TF's components after copying information from host to device array
	if( transInfo.useSecondTransferFunction ){
		load2DArray(transInfo.K_alphaTransferArray2D, kalphaTF, transInfo.functionSize, *stream);
		load2DArray(transInfo.K_colorRTransferArray2D, kredTF, transInfo.functionSize, *stream);
		load2DArray(transInfo.K_colorGTransferArray2D, kgreenTF, transInfo.functionSize, *stream);
		load2DArray(transInfo.K_colorBTransferArray2D, kblueTF, transInfo.functionSize, *stream);
		load2DArray(transInfo.K_ambientTransferArray2D, kambTF, transInfo.functionSize, *stream);
		load2DArray(transInfo.K_diffuseTransferArray2D, kdiffTF, transInfo.functionSize, *stream);
		load2DArray(transInfo.K_specularTransferArray2D, kspecTF, transInfo.functionSize, *stream);
		load2DArray(transInfo.K_specularPowerTransferArray2D, kpowTF, transInfo.functionSize, *stream);
	}

	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "Bind transfer functions: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);
}

bool CUDA_vtkCuda2DVolumeMapper_renderAlgo_unloadTextures( cuda2DTransferFunctionInformation& transInfo, cudaStream_t* stream ){
	
	//unload each array in the transfer function information container
	unloadArray(transInfo.alphaTransferArray2D);
	unloadArray(transInfo.colorRTransferArray2D);
	unloadArray(transInfo.colorGTransferArray2D);
	unloadArray(transInfo.colorBTransferArray2D);
	unloadArray(transInfo.ambientTransferArray2D);
	unloadArray(transInfo.diffuseTransferArray2D);
	unloadArray(transInfo.specularTransferArray2D);
	unloadArray(transInfo.specularPowerTransferArray2D);

	unloadArray(transInfo.K_alphaTransferArray2D);
	unloadArray(transInfo.K_colorRTransferArray2D);
	unloadArray(transInfo.K_colorGTransferArray2D);
	unloadArray(transInfo.K_colorBTransferArray2D);
	unloadArray(transInfo.K_ambientTransferArray2D);
	unloadArray(transInfo.K_diffuseTransferArray2D);
	unloadArray(transInfo.K_specularTransferArray2D);
	unloadArray(transInfo.K_specularPowerTransferArray2D);

	return (cudaGetLastError() == 0);
}

//pre:	the data has been preprocessed by the volumeInformationHandler such that it is float data
//		the index is between 0 and 100
//post: the input_texture will map to the source data in voxel coordinate space
bool CUDA_vtkCuda2DVolumeMapper_renderAlgo_loadImageInfo(const float* data, const cudaVolumeInformation& volumeInfo, cudaArray** frame, cudaStream_t* stream){

	// if the array is already populated with information, free it to prevent leaking
	if(*frame) cudaFreeArray(*frame);
	
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

void CUDA_vtkCuda2DVolumeMapper_renderAlgo_clearImageArray(cudaArray** frame, cudaStream_t* stream){
	if(*frame)
		cudaFreeArray(*frame);
	*frame = 0;
}