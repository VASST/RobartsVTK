#include "CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.h"
#include "CUDA_vtkCudaVolumeMapper_renderAlgo.h"
#include "CUDA_container2DTransferFunctionInformation.h"
#include "CUDA_containerSlicePlanesInformation.h"
#include <cuda.h>

//execution parameters and general information
__constant__ cuda2DTransferFunctionInformation	CUDA_vtkCuda2DInExVolumeMapper_trfInfo;
__constant__ cudaSlicePlanesInformation			CUDA_vtkCuda2DInExVolumeMapper_slcInfo;

//transfer function as read-only textures
texture<float, 2, cudaReadModeElementType> alpha_texture_2DInex;
texture<float, 2, cudaReadModeElementType> colorR_texture_2DInex;
texture<float, 2, cudaReadModeElementType> colorG_texture_2DInex;
texture<float, 2, cudaReadModeElementType> colorB_texture_2DInex;
texture<float, 2, cudaReadModeElementType> inExLogic_texture_2DInex;

//opague memory back for the transfer function
cudaArray* alphaTransferArray2DInex = 0;
cudaArray* colorRTransferArray2DInex = 0;
cudaArray* colorGTransferArray2DInex = 0;
cudaArray* colorBTransferArray2DInex = 0;
cudaArray* inExLogicTransferArray2DInex = 0;

//3D input data (read-only texture with corresponding opague device memory back)
texture<float, 3, cudaReadModeElementType> CUDA_vtkCuda2DInExVolumeMapper_input_texture;
cudaArray* CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[100];

__device__ void CUDA_vtkCuda2DInExVolumeMapper_CUDAkernel_CastRays(float3& rayStart,
									const float& numSteps,
									float& excludeStart,
									float& excludeEnd,
									float& includeStart,
									float& includeEnd,
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
		float inEx = tex2D(inExLogic_texture_2DInex, tempIndex, gradMag);
		float alpha = tex2D(alpha_texture_2DInex, tempIndex, gradMag);

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
				if( outputVal.w > 1.0f - 0.03125f && !( excludeStart >= maxSteps && excludeEnd <= maxSteps )
						&& inEx > 0.5 ) break;

				//if we are not explicitly included, set the alpha and multiplier to zero
				alpha = !(includeEnd < 0.0f || (includeStart >= maxSteps && includeEnd <= maxSteps) ) ? alpha : 0.0f;
				multiplier = !(includeEnd < 0.0f || (includeStart >= maxSteps && includeEnd <= maxSteps) ) ? multiplier : 0.0f;

				//accumulate the opacity for this sample point
				outputVal.w *= alpha;

				//accumulate the colour information from this sample point
				outputVal.x += multiplier * tex2D(colorR_texture_2DInex, tempIndex, gradMag);
				outputVal.y += multiplier * tex2D(colorG_texture_2DInex, tempIndex, gradMag);
				outputVal.z += multiplier * tex2D(colorB_texture_2DInex, tempIndex, gradMag);
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

__device__ void CUDA_vtkCuda2DInExVolumeMapper_FindSlicingValues(float3 rayStart, float3 rayInc,
											 float& numSteps, float& includeStart, float& includeEnd ){
	
	__syncthreads();
	const int numPlanes = CUDA_vtkCuda2DInExVolumeMapper_slcInfo.NumberOfSlicingPlanes;
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
	includeStart = 1.0f;
	includeEnd = -1.0f;

	// loop through all provided slicing planes
	if(!numPlanes) return;
	int flag = 0;
	#pragma unroll 1
	for ( int i = 0; i < numPlanes; i++ ){
		
		//refine the ray direction to account for any changes in starting or ending position
		rayDir.x = rayEnd.x - rayStart.x;
		rayDir.y = rayEnd.y - rayStart.y;
		rayDir.z = rayEnd.z - rayStart.z;
		
		//collect all the information about the current clipping plane
		float4 SlicingPlane;
		__syncthreads();
		SlicingPlane.x	= CUDA_vtkCuda2DInExVolumeMapper_slcInfo.SlicingPlanes[4*i];
		SlicingPlane.y	= CUDA_vtkCuda2DInExVolumeMapper_slcInfo.SlicingPlanes[4*i+1];
		SlicingPlane.z	= CUDA_vtkCuda2DInExVolumeMapper_slcInfo.SlicingPlanes[4*i+2];
		SlicingPlane.w	= CUDA_vtkCuda2DInExVolumeMapper_slcInfo.SlicingPlanes[4*i+3];
		__syncthreads();
		
		const float dp = SlicingPlane.x*rayDir.x +
						 SlicingPlane.y*rayDir.y +
						 SlicingPlane.z*rayDir.z;
		const float t = -(SlicingPlane.x*rayStart.x +
						SlicingPlane.y*rayStart.y + 
						SlicingPlane.z*rayStart.z + 
						SlicingPlane.w) / dp;

		const float point0 = rayStart.x + t*rayDir.x;
		const float point1 = rayStart.y + t*rayDir.y;
		const float point2 = rayStart.z + t*rayDir.z;

		//if the ray intersects the plane, set the start or end point to the intersection point
		if ( t > 0.0f && t < 1.0f ){
			
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
	includeStart = flag ?  1.0f : rayStart.x * rayInc.x +
								  rayStart.y * rayInc.y +
								  rayStart.z * rayInc.z - 0.1f;
	includeEnd = flag ?  -1.0f : rayEnd.x * rayInc.x +
								 rayEnd.y * rayInc.y +
								 rayEnd.z * rayInc.z + 0.1f;

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
	float includeStart; //where to start excluding
	float includeEnd; //where to end excluding
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

	//find the area to actually render
	CUDA_vtkCuda2DInExVolumeMapper_FindSlicingValues(rayStart, rayInc, numSteps, includeStart, includeEnd );

	// trace along the ray (composite)
	CUDA_vtkCuda2DInExVolumeMapper_CUDAkernel_CastRays(rayStart, numSteps, excludeStart, excludeEnd,
														includeStart, includeEnd, rayInc, outputVal, outputDepth);

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

#include <stdio.h>

extern "C"
//pre: the resolution of the image has been processed such that it's x and y size are both multiples of 16 (enforced automatically) and y > 256 (enforced automatically)
//post: the OutputImage pointer will hold the ray casted information
void CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_doRender(const cudaOutputImageInformation& outputInfo,
							 const cudaRendererInformation& rendererInfo,
							 const cudaVolumeInformation& volumeInfo,
							 const cuda2DTransferFunctionInformation& transInfo,
							 const cudaSlicePlanesInformation& sliceInfo)
{

	// setup execution parameters - staggered to improve parallelism
	cudaMemcpyToSymbolAsync(volInfo, &volumeInfo, sizeof(cudaVolumeInformation));
	cudaMemcpyToSymbolAsync(renInfo, &rendererInfo, sizeof(cudaRendererInformation));
	cudaMemcpyToSymbolAsync(outInfo, &outputInfo, sizeof(cudaOutputImageInformation));
	cudaMemcpyToSymbolAsync(CUDA_vtkCuda2DInExVolumeMapper_trfInfo, &transInfo, sizeof(cuda2DTransferFunctionInformation));
	cudaMemcpyToSymbolAsync(CUDA_vtkCuda2DInExVolumeMapper_slcInfo, &sliceInfo, sizeof(cudaSlicePlanesInformation));
	
	//create the necessary execution amount parameters from the block sizes and calculate th volume rendering integral
	int blockX = outputInfo.resolution.x / BLOCK_DIM2D ;
	int blockY = outputInfo.resolution.y / BLOCK_DIM2D ;

	dim3 grid(blockX, blockY, 1);
	dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);
	cudaThreadSynchronize();
	CUDAkernel_renderAlgo_formRays <<< grid, threads >>>();
	CUDA_vtkCuda2DInExVolumeMapper_CUDAkernel_Composite <<< grid, threads >>>();

	//shade the image
	grid.x = outputInfo.resolution.x*outputInfo.resolution.y / 256;
	grid.y = 1;
	threads.x = 256;
	threads.y = 1;
	cudaThreadSynchronize();
	CUDAkernel_shadeAlgo_doCelShade <<< grid, threads >>>();
	cudaThreadSynchronize();
	
	printf( "2D Rendering Error Status: " );
	printf( cudaGetErrorString( cudaGetLastError() ) );
	printf( "\n" );

	return;
}

extern "C"
void CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_changeFrame(const int frame){

	// set the texture to the correct image
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.normalized = false;						// access with unnormalized texture coordinates
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.filterMode = cudaFilterModeLinear;		// linear interpolation
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.addressMode[0] = cudaAddressModeClamp;	// wrap texture coordinates
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.addressMode[1] = cudaAddressModeClamp;
	CUDA_vtkCuda2DInExVolumeMapper_input_texture.addressMode[2] = cudaAddressModeClamp;

	// bind array to 3D texture
	cudaBindTextureToArray(CUDA_vtkCuda2DInExVolumeMapper_input_texture,
							CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[frame], channelDesc);
	
	printf( "Change Frame Status: " );
	printf( cudaGetErrorString( cudaGetLastError() ) );
	printf( "\n" );
}

extern "C"
//pre: the transfer functions are all of type float and are all of size FunctionSize
//post: the alpha, colorR, G and B 2D textures will map to each transfer function
void CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_loadTextures(const cuda2DTransferFunctionInformation& transInfo,
								  float* redTF, float* greenTF, float* blueTF, float* alphaTF, float* inExTF){

	//retrieve the size of the transer functions
	size_t size = sizeof(float) * transInfo.functionSize;
	
	if(alphaTransferArray2DInex)
		cudaFreeArray(alphaTransferArray2DInex);
	if(colorRTransferArray2DInex)
		cudaFreeArray(colorRTransferArray2DInex);
	if(colorGTransferArray2DInex)
		cudaFreeArray(colorGTransferArray2DInex);
	if(colorBTransferArray2DInex)
		cudaFreeArray(colorBTransferArray2DInex);
	if(inExLogicTransferArray2DInex)
		cudaFreeArray(colorBTransferArray2DInex);
		
	//allocate space for the arrays
	cudaMallocArray( &alphaTransferArray2DInex, &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMallocArray( &colorRTransferArray2DInex, &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMallocArray( &colorGTransferArray2DInex, &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMallocArray( &colorBTransferArray2DInex, &channelDesc, transInfo.functionSize, transInfo.functionSize);
	cudaMallocArray( &inExLogicTransferArray2DInex, &channelDesc, transInfo.functionSize, transInfo.functionSize);
		
	//define the texture mapping for the alpha component after copying information from host to device array
	cudaMemcpyToArray(alphaTransferArray2DInex, 0, 0, alphaTF, size*transInfo.functionSize, cudaMemcpyHostToDevice);
	alpha_texture_2DInex.normalized = true;
	alpha_texture_2DInex.filterMode = cudaFilterModePoint;
	alpha_texture_2DInex.addressMode[0] = cudaAddressModeClamp;
	alpha_texture_2DInex.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(alpha_texture_2DInex, alphaTransferArray2DInex, channelDesc);
		
	//define the texture mapping for the red component after copying information from host to device array
	cudaMemcpyToArray(colorRTransferArray2DInex, 0, 0, redTF, size*transInfo.functionSize, cudaMemcpyHostToDevice);
	colorR_texture_2DInex.normalized = true;
	colorR_texture_2DInex.filterMode = cudaFilterModePoint;
	colorR_texture_2DInex.addressMode[0] = cudaAddressModeClamp;
	colorR_texture_2DInex.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorR_texture_2DInex, colorRTransferArray2DInex, channelDesc);
	
	//define the texture mapping for the green component after copying information from host to device array
	cudaMemcpyToArray(colorGTransferArray2DInex, 0, 0, greenTF, size*transInfo.functionSize, cudaMemcpyHostToDevice);
	colorG_texture_2DInex.normalized = true;
	colorG_texture_2DInex.filterMode = cudaFilterModePoint;
	colorG_texture_2DInex.addressMode[0] = cudaAddressModeClamp;
	colorG_texture_2DInex.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorG_texture_2DInex, colorGTransferArray2DInex, channelDesc);
	
	//define the texture mapping for the blue component after copying information from host to device array
	cudaMemcpyToArray(colorBTransferArray2DInex, 0, 0, blueTF, size*transInfo.functionSize, cudaMemcpyHostToDevice);
	colorB_texture_2DInex.normalized = true;
	colorB_texture_2DInex.filterMode = cudaFilterModePoint;
	colorB_texture_2DInex.addressMode[0] = cudaAddressModeClamp;
	colorB_texture_2DInex.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorB_texture_2DInex, colorBTransferArray2DInex, channelDesc);
	
	//define the texture mapping for the blue component after copying information from host to device array
	cudaMemcpyToArray(inExLogicTransferArray2DInex, 0, 0, inExTF, size*transInfo.functionSize, cudaMemcpyHostToDevice);
	inExLogic_texture_2DInex.normalized = true;
	inExLogic_texture_2DInex.filterMode = cudaFilterModePoint;
	inExLogic_texture_2DInex.addressMode[0] = cudaAddressModeClamp;
	inExLogic_texture_2DInex.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(inExLogic_texture_2DInex, inExLogicTransferArray2DInex, channelDesc);

	printf( "Bind transfer functions: " );
	printf( cudaGetErrorString( cudaGetLastError() ) );
	printf( "\n" );
}

extern "C"
//pre:	the data has been preprocessed by the volumeInformationHandler such that it is float data
//		the index is between 0 and 100
//post: the input_texture will map to the source data in voxel coordinate space
void CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_loadImageInfo(const float* data, const cudaVolumeInformation& volumeInfo, const int index){

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
	cudaMemcpy3D(&copyParams);

	printf( "Load volume information: " );
	printf( cudaGetErrorString( cudaGetLastError() ) );
	printf( "\n" );

}

extern "C"
void CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_initImageArray(){
	for(int i = 0; i < 100; i++)
		CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[i] = 0;
}

extern "C"
void CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo_clearImageArray(){
	for(int i = 0; i < 100; i++){
		
		// if the array is already populated with information, free it to prevent leaking
		if(CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[i])
			cudaFreeArray(CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[i]);
		
		//null the pointer
		CUDA_vtkCuda2DInExVolumeMapper_sourceDataArray[i] = 0;
	}
}