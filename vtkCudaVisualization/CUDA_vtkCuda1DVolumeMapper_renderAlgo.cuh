/** @file CUDA_vtkCuda1DVolumeMapper_renderAlgo.cu
 *
 *  @brief Underlying CUDA implementation of the 1D volume ray caster
 *
 *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
 *  @note First documented on May 11, 2012
 *
 */
 
#include "CUDA_vtkCuda1DVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include <cuda.h>

//execution parameters and general information
__constant__ cuda1DTransferFunctionInformation	CUDA_vtkCuda1DVolumeMapper_trfInfo;

//transfer function as read-only textures
texture<float, 1, cudaReadModeElementType> alpha_texture_1D;
texture<float, 1, cudaReadModeElementType> galpha_texture_1D;
texture<float, 1, cudaReadModeElementType> colorR_texture_1D;
texture<float, 1, cudaReadModeElementType> colorG_texture_1D;
texture<float, 1, cudaReadModeElementType> colorB_texture_1D;

//3D input data (read-only texture with corresponding opague device memory back)
texture<float, 3, cudaReadModeElementType> CUDA_vtkCuda1DVolumeMapper_input_texture;
cudaArray* CUDA_vtkCuda1DVolumeMapper_sourceDataArray[100];

__device__ void CUDA_vtkCuda1DVolumeMapper_CUDAkernel_CastRays1D(float3& rayStart,
									const float& numSteps,
									float& excludeStart,
									float& excludeEnd,
									const float3& rayInc,
									float4& outputVal) {

	//set the default values for the output (note A is currently the remaining opacity, not the output opacity)
	outputVal.x = 0.0f; //R
	outputVal.y = 0.0f; //G
	outputVal.z = 0.0f; //B
	outputVal.w = 1.0f; //A
		
	//fetch the required information about the size and range of the transfer function from memory to registers
	__syncthreads();
	const float functRangeLow = CUDA_vtkCuda1DVolumeMapper_trfInfo.intensityLow;
	const float functRangeMulti = CUDA_vtkCuda1DVolumeMapper_trfInfo.intensityMultiplier;
	const float gradRangeLow = CUDA_vtkCuda1DVolumeMapper_trfInfo.gradientLow;
	const float gradRangeMulti = CUDA_vtkCuda1DVolumeMapper_trfInfo.gradientMultiplier;
	const float3 space = volInfo.SpacingReciprocal;
	const float3 incSpace = volInfo.Spacing;
	const float ambient = CUDA_vtkCuda1DVolumeMapper_trfInfo.Ambient;
	const float diffuse = CUDA_vtkCuda1DVolumeMapper_trfInfo.Diffuse;
	const float2 spec = CUDA_vtkCuda1DVolumeMapper_trfInfo.Specular;
	__syncthreads();

	//apply a randomized offset to the ray
	float retDepth = dRandomRayOffsets[threadIdx.x + BLOCK_DIM2D * threadIdx.y];
	__syncthreads();
	int maxSteps = __float2int_rd(numSteps - retDepth) ;
	rayStart.x += retDepth*rayInc.x;
	rayStart.y += retDepth*rayInc.y;
	rayStart.z += retDepth*rayInc.z;
	float rayLength = sqrtf(rayInc.x*rayInc.x*incSpace.x*incSpace.x +
							rayInc.y*rayInc.y*incSpace.y*incSpace.y +
							rayInc.z*rayInc.z*incSpace.z*incSpace.z);
	//allocate flags
	char2 step;
	step.x = 0;
	step.y = 0;

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
		const float tempIndex = functRangeMulti * (tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z) - functRangeLow);
	
		//fetching the opacity value of the sampling point (apply transfer function in stages to minimize work)
		// as well as the colour multiplier (with photorealistic shading)
		float alpha = tex1D(alpha_texture_1D, tempIndex);

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

				float3 gradient;
				gradient.x = ( tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x+1.0f, rayStart.y, rayStart.z)
							 - tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x-1.0f, rayStart.y, rayStart.z) ) * space.x;
				gradient.y = ( tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y+1.0f, rayStart.z)
							 - tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y-1.0f, rayStart.z) ) * space.y;
				gradient.z = ( tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z+1.0f)
							 - tex3D(CUDA_vtkCuda1DVolumeMapper_input_texture, rayStart.x, rayStart.y, rayStart.z-1.0f) ) * space.z;
				float gradMag = sqrtf(gradient.x*gradient.x+gradient.y*gradient.y+gradient.z*gradient.z);
				alpha *= isfinite(gradRangeMulti) ? tex1D(galpha_texture_1D, gradRangeMulti*(gradMag-gradRangeLow)) : 1.0f;
				float phongLambert = saturate( abs ( gradient.x*rayInc.x*incSpace.x + 
											     gradient.y*rayInc.y*incSpace.y +
											     gradient.z*rayInc.z*incSpace.z 	) / (gradMag * rayLength) );
				float shadeD = ambient + diffuse * phongLambert;
				float shadeS = spec.x * pow(phongLambert, spec.y);

				//accumulate the opacity for this sample point
				float multiplier = outputVal.w * alpha;
				outputVal.w *= (1.0f - alpha);

				//accumulate the colour information from this sample point
				outputVal.x += multiplier * saturate(shadeD * tex1D(colorR_texture_1D, tempIndex) + shadeS);
				outputVal.y += multiplier * saturate(shadeD * tex1D(colorG_texture_1D, tempIndex) + shadeS);
				outputVal.z += multiplier * saturate(shadeD * tex1D(colorB_texture_1D, tempIndex) + shadeS);
			}
			
			//determine whether or not we've hit an opacity where further sampling becomes neglible
			if(outputVal.w < 0.015625f){
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

	//adjust the opacity output to reflect the collected opacity, and not the remaining opacity
	outputVal.w = 1.0f - outputVal.w;
	outputVal.x = saturate( outputVal.x );
	outputVal.y = saturate( outputVal.y );
	outputVal.z = saturate( outputVal.z );

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
	float4 outputVal; //rgba value of this ray (calculated in castRays, used in WriteData)
	float excludeStart; //where to start excluding
	float excludeEnd; //where to end excluding

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
	CUDA_vtkCuda1DVolumeMapper_CUDAkernel_CastRays1D(rayStart, numSteps, excludeStart, excludeEnd, rayInc, outputVal);

	//convert output to uchar, adjusting it to be valued from [0,256) rather than [0,1]
	uchar4 temp;
	temp.x = 255.0f * outputVal.x;
	temp.y = 255.0f * outputVal.y;
	temp.z = 255.0f * outputVal.z;
	temp.w = 255.0f * outputVal.w;
	
	//place output in the image buffer
	__syncthreads();
	outInfo.deviceOutputImage[outindex] = temp;

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
	
	//map the texture for the transfer function
	alpha_texture_1D.normalized = true;
	alpha_texture_1D.filterMode = cudaFilterModeLinear;
	alpha_texture_1D.addressMode[0] = cudaAddressModeClamp;
	cudaBindTextureToArray(alpha_texture_1D, transInfo.alphaTransferArray1D);
	galpha_texture_1D.normalized = true;
	galpha_texture_1D.filterMode = cudaFilterModeLinear;
	galpha_texture_1D.addressMode[0] = cudaAddressModeClamp;
	cudaBindTextureToArray(galpha_texture_1D, transInfo.galphaTransferArray1D);
	colorR_texture_1D.normalized = true;
	colorR_texture_1D.filterMode = cudaFilterModeLinear;
	colorR_texture_1D.addressMode[0] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorR_texture_1D, transInfo.colorRTransferArray1D);
	colorG_texture_1D.normalized = true;
	colorG_texture_1D.filterMode = cudaFilterModeLinear;
	colorG_texture_1D.addressMode[0] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorG_texture_1D, transInfo.colorGTransferArray1D);
	colorB_texture_1D.normalized = true;
	colorB_texture_1D.filterMode = cudaFilterModeLinear;
	colorB_texture_1D.addressMode[0] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorB_texture_1D, transInfo.colorBTransferArray1D);

	//create the necessary execution amount parameters from the block sizes and calculate th volume rendering integral
	int blockX = outputInfo.resolution.x / BLOCK_DIM2D ;
	int blockY = outputInfo.resolution.y / BLOCK_DIM2D ;

	dim3 grid(blockX, blockY, 1);
	dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);
	CUDAkernel_renderAlgo_formRays <<< grid, threads, 0, *stream >>>();
	CUDA_vtkCuda1DVolumeMapper_CUDAkernel_Composite <<< grid, threads, 0, *stream >>>();

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

	return (cudaGetLastError() == 0);

}

//pre: the transfer functions are all of type float and are all of size FunctionSize
//post: the alpha, colorR, G and B 1D textures will map to each transfer function
bool CUDA_vtkCuda1DVolumeMapper_renderAlgo_loadTextures(cuda1DTransferFunctionInformation& transInfo,
								  float* redTF, float* greenTF, float* blueTF, float* alphaTF, float* galphaTF,
								  cudaStream_t* stream){

	//retrieve the size of the transer functions
	size_t size = sizeof(float) * transInfo.functionSize;
		
	//define the texture mapping for the alpha component after copying information from host to device array
	if(transInfo.alphaTransferArray1D)
		cudaFreeArray(transInfo.alphaTransferArray1D);
	cudaMallocArray( &(transInfo.alphaTransferArray1D), &channelDesc, transInfo.functionSize, 1);
	cudaMemcpyToArrayAsync(transInfo.alphaTransferArray1D, 0, 0, alphaTF, size, cudaMemcpyHostToDevice, *stream);
	if(transInfo.galphaTransferArray1D)
		cudaFreeArray(transInfo.galphaTransferArray1D);
	cudaMallocArray( &(transInfo.galphaTransferArray1D), &channelDesc, transInfo.functionSize, 1);
	cudaMemcpyToArrayAsync(transInfo.galphaTransferArray1D, 0, 0, galphaTF, size, cudaMemcpyHostToDevice, *stream);
		
	//define the texture mapping for the red component after copying information from host to device array
	if(transInfo.colorRTransferArray1D)
		cudaFreeArray(transInfo.colorRTransferArray1D);
	cudaMallocArray( &(transInfo.colorRTransferArray1D), &channelDesc, transInfo.functionSize, 1);
	cudaMemcpyToArrayAsync(transInfo.colorRTransferArray1D, 0, 0, redTF, size, cudaMemcpyHostToDevice, *stream);
	
	//define the texture mapping for the green component after copying information from host to device array
	if(transInfo.colorGTransferArray1D)
		cudaFreeArray(transInfo.colorGTransferArray1D);
	cudaMallocArray( &(transInfo.colorGTransferArray1D), &channelDesc, transInfo.functionSize, 1);
	cudaMemcpyToArrayAsync(transInfo.colorGTransferArray1D, 0, 0, greenTF, size, cudaMemcpyHostToDevice, *stream);
	
	//define the texture mapping for the blue component after copying information from host to device array
	if(transInfo.colorBTransferArray1D)
		cudaFreeArray(transInfo.colorBTransferArray1D);
	cudaMallocArray( &(transInfo.colorBTransferArray1D), &channelDesc, transInfo.functionSize, 1);
	cudaMemcpyToArrayAsync(transInfo.colorBTransferArray1D, 0, 0, blueTF, size, cudaMemcpyHostToDevice, *stream);

	return (cudaGetLastError() == 0);

}

bool CUDA_vtkCuda1DVolumeMapper_renderAlgo_UnloadTextures(cuda1DTransferFunctionInformation& transInfo, cudaStream_t* stream){

	if(transInfo.colorRTransferArray1D)
		cudaFreeArray(transInfo.colorRTransferArray1D);
	transInfo.colorRTransferArray1D = 0;
	if(transInfo.colorGTransferArray1D)
		cudaFreeArray(transInfo.colorGTransferArray1D);
	transInfo.colorGTransferArray1D = 0;
	if(transInfo.colorBTransferArray1D)
		cudaFreeArray(transInfo.colorBTransferArray1D);
	transInfo.colorBTransferArray1D = 0;
	if(transInfo.alphaTransferArray1D)
		cudaFreeArray(transInfo.alphaTransferArray1D);
	transInfo.alphaTransferArray1D = 0;
	if(transInfo.galphaTransferArray1D)
		cudaFreeArray(transInfo.galphaTransferArray1D);
	transInfo.galphaTransferArray1D = 0;

	return (cudaGetLastError() == 0);
}

//pre:	the data has been preprocessed by the volumeInformationHandler such that it is float data
//		the index is between 0 and 100
//post: the input_texture will map to the source data in voxel coordinate space
bool CUDA_vtkCuda1DVolumeMapper_renderAlgo_loadImageInfo(const float* data, const cudaVolumeInformation& volumeInfo, const int index, cudaStream_t* stream){

	// if the array is already populated with information, free it to prevent leaking
	if(CUDA_vtkCuda1DVolumeMapper_sourceDataArray[index])
		cudaFreeArray(CUDA_vtkCuda1DVolumeMapper_sourceDataArray[index]);
	
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
	cudaMemcpy3DAsync(&copyParams, *stream);

	#ifdef DEBUG_VTKCUDAVISUALIZATION
		cudaThreadSynchronize();
		printf( "Load volume information: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	return (cudaGetLastError() == 0);

}

void CUDA_vtkCuda1DVolumeMapper_renderAlgo_initImageArray(cudaStream_t* stream){
	for(int i = 0; i < 100; i++)
		CUDA_vtkCuda1DVolumeMapper_sourceDataArray[i] = 0;
}

void CUDA_vtkCuda1DVolumeMapper_renderAlgo_clearImageArray(cudaStream_t* stream){
	for(int i = 0; i < 100; i++){
		
		// if the array is already populated with information, free it to prevent leaking
		if(CUDA_vtkCuda1DVolumeMapper_sourceDataArray[i])
			cudaFreeArray(CUDA_vtkCuda1DVolumeMapper_sourceDataArray[i]);
		
		//null the pointer
		CUDA_vtkCuda1DVolumeMapper_sourceDataArray[i] = 0;
	}
}