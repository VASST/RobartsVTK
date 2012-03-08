extern "C" {
#include "CUDA_renderAlgo.h"
}

//cuda includes
#include <cuda.h>

//This is the side length of the square that the output image is broken up into
#define BLOCK_DIM2D 16     // this must be set to 4 or more, 16 is optimal

//execution parameters and general information
__constant__ cudaVolumeInformation   volInfo;
__constant__ cudaRendererInformation renInfo;
__constant__ float random[BLOCK_DIM2D*BLOCK_DIM2D];

//transfer function as read-only textures
texture<float, 2, cudaReadModeElementType> alpha_texture_2D;
texture<float, 2, cudaReadModeElementType> colorR_texture_2D;
texture<float, 2, cudaReadModeElementType> colorG_texture_2D;
texture<float, 2, cudaReadModeElementType> colorB_texture_2D;

//opague memory back for the transfer function
cudaArray* alphaTransferArray = 0;
cudaArray* colorRTransferArray = 0;
cudaArray* colorGTransferArray = 0;
cudaArray* colorBTransferArray = 0;

//3D input data (read-only texture with corresponding opague device memory back)
texture<float, 3, cudaReadModeElementType> input_texture;
cudaArray* sourceDataArray[100];

//channel for loading input data and transfer functions
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

__device__ void CUDAkernel_CastRays2D(float4& rayStart,
									const float& numSteps,
									const float3& rayInc,
									const float3& DepthReference,
									float4& outputVal,
									float& retDepth){

	//set the default values for the output (note A is currently the remaining opacity, not the output opacity)
	outputVal.x = 0.0f; //R
	outputVal.y = 0.0f; //G
	outputVal.z = 0.0f; //B
	outputVal.w = 1.0f; //A
		
	//fetch the required information about the size and range of the transfer function from memory to registers
	__syncthreads();
	const float functRangeLow = volInfo.intensityLow;
	const float functRangeMulti = volInfo.intensityMultiplier;
	const float gradRangeMulti = volInfo.twiceGradientMultiplier;
	const float spaceX = volInfo.SpacingReciprocal.x;
	const float spaceY = volInfo.SpacingReciprocal.y;
	const float spaceZ = volInfo.SpacingReciprocal.z;
	const float shadeMultiplier = renInfo.gradShadeScale;
	const float shadeShift = renInfo.gradShadeShift;
	const float depthShift = renInfo.depthShadeShift;
	const float depthScale = renInfo.depthShadeScale;
	__syncthreads();

	//apply a randomized offset to the ray
	const float randomChange = random[threadIdx.x + BLOCK_DIM2D * threadIdx.y];
	__syncthreads();
	rayStart.x += randomChange*rayInc.x;
	rayStart.y += randomChange*rayInc.y;
	rayStart.z += randomChange*rayInc.z;

	//calculate initial ray depth
	retDepth = __fsqrt_rz(	(rayStart.x-DepthReference.x)*(rayStart.x-DepthReference.x) +
							(rayStart.y-DepthReference.y)*(rayStart.y-DepthReference.y) + 
							(rayStart.z-DepthReference.z)*(rayStart.z-DepthReference.z) );

	//calculate the number of times this can go through the loop
	int maxSteps = __float2int_rd(numSteps);

	//loop as long as we are still *roughly* in the range of the clipped and cropped volume
	while( maxSteps-- ){

		// increase the depth of the penetrating ray
		retDepth += 1.0f;

		// fetching the intensity index into the transfer function
		const float valCentre = tex3D(input_texture, rayStart.x, rayStart.y, rayStart.z);
		const float tempIndex = functRangeMulti * (valCentre - functRangeLow);
			
		//fetching the gradient index into the transfer function
		float3 gradient;
		gradient.x = ( tex3D(input_texture, rayStart.x+1.0f, rayStart.y, rayStart.z) - tex3D(input_texture, rayStart.x-1.0f, rayStart.y, rayStart.z) ) * spaceX;
		gradient.y = ( tex3D(input_texture, rayStart.x, rayStart.y+1.0f, rayStart.z) - tex3D(input_texture, rayStart.x, rayStart.y-1.0f, rayStart.z) ) * spaceY;
		gradient.z = ( tex3D(input_texture, rayStart.x, rayStart.y, rayStart.z+1.0f) - tex3D(input_texture, rayStart.x, rayStart.y, rayStart.z-1.0f) ) * spaceZ;
		const float gradMag = gradRangeMulti * __log2f(gradient.x*gradient.x+gradient.y*gradient.y+gradient.z*gradient.z+1.0f);
	
		//fetching the opacity value of the sampling point (apply transfer function in stages to minimize work)
		float alpha = tex2D(alpha_texture_2D, tempIndex, gradMag);
		
		//filter out objects with too low opacity (deemed unimportant, and this saves time and reduces cloudiness)
		if(alpha > 0.0078125f){
		
			float multiplier = outputVal.w * alpha;
			alpha = (1.0f - alpha);
			outputVal.w *= alpha;

			//apply photorealistic shading
			multiplier *= (shadeShift + shadeMultiplier * abs(gradient.x*rayInc.x + gradient.y*rayInc.y + gradient.z*rayInc.z) * rsqrtf(gradient.x*gradient.x+gradient.y*gradient.y+gradient.z*gradient.z));

			//apply depth shading
			multiplier *= saturate(depthShift + depthScale * retDepth);

			//accumulate the colour information from this sample point
			outputVal.x += multiplier * tex2D(colorR_texture_2D, tempIndex, gradMag);
			outputVal.y += multiplier * tex2D(colorG_texture_2D, tempIndex, gradMag);
			outputVal.z += multiplier * tex2D(colorB_texture_2D, tempIndex, gradMag);

			//determine whether or not we've hit an opacity where further sampling becomes neglible
			if(outputVal.w < 0.03125f){

				//logarithmically interpolate to get the correct termination position of the ray
				multiplier = ( -5.0f - __log2f(outputVal.w) ) / __log2f( alpha );
				rayStart.x -= rayInc.x * multiplier;
				rayStart.y -= rayInc.y * multiplier;
				rayStart.z -= rayInc.z * multiplier;

				break;
			}
			
		}
			
		//increment to our next sampling point on the ray
		rayStart.x += rayInc.x;
		rayStart.y += rayInc.y;
		rayStart.z += rayInc.z;
		
	}//while

	//adjust the opacity output to reflect the collected opacity, and not the remaining opacity
	outputVal.w = 1.0f - outputVal.w;

}

__device__ void CUDAkernel_ClipRayAgainstClippingPlanes(float4& rayStart, float4& rayEnd, float3& rayDir)
{
	
	int flag = 0;
	
	__syncthreads();
	const int numPlanes = renInfo.NumberOfClippingPlanes;
	__syncthreads();

	// loop through all 6 clipping planes
	if(numPlanes == 0) return;
	#pragma unroll 1
	for ( int i = 0; i < 6; i++ ){
		
		//refine the ray direction to account for any changes in starting or ending position
		rayDir.x = rayEnd.x - rayStart.x;
		rayDir.y = rayEnd.y - rayStart.y;
		rayDir.z = rayEnd.z - rayStart.z;
		
		//collect all the information about the current clipping plane
		float4 clippingPlane;
		__syncthreads();
		clippingPlane.x	= renInfo.ClippingPlanes[4*i];
		clippingPlane.y	= renInfo.ClippingPlanes[4*i+1];
		clippingPlane.z	= renInfo.ClippingPlanes[4*i+2];
		clippingPlane.w	= renInfo.ClippingPlanes[4*i+3];
		__syncthreads();
		
		const float dp = clippingPlane.x*rayDir.x + clippingPlane.y*rayDir.y + clippingPlane.z*rayDir.z;
		const float t = -(clippingPlane.x*rayStart.x +
						clippingPlane.y*rayStart.y + 
						clippingPlane.z*rayStart.z + 
						clippingPlane.w) / dp;

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

	//if the ray is not inside the clipping planes, make the ray zero length
	if(flag){
		rayStart.x = rayEnd.x;
		rayStart.y = rayEnd.y;
		rayStart.z = rayEnd.z;
	}

}

__device__ void CUDAkernel_ClipRayAgainstVolume(float4& rayStart, float4& rayEnd, float3& rayDir)
{
	
	//define the ray's length and direction to account for any changes in starting and ending position
	rayDir.x = rayEnd.x - rayStart.x;
	rayDir.y = rayEnd.y - rayStart.y;
	rayDir.z = rayEnd.z - rayStart.z;
	
	//collect the information about the bounds of the volume in voxels from the volume information
	__syncthreads();
	const float bounds0 = volInfo.Bounds[0];
	const float bounds1 = volInfo.Bounds[1];
	const float bounds2 = volInfo.Bounds[2];
	const float bounds3 = volInfo.Bounds[3];
	const float bounds4 = volInfo.Bounds[4];
	const float bounds5 = volInfo.Bounds[5];
	__syncthreads();
		
	float diffS;
	float diffE;
	
	//find the intersection of the ray and the volume (in the x direction)
	if (rayDir.x > 0.0f){
		diffS = rayStart.x < bounds0 ? bounds0 - rayStart.x : 0.0f; 
		diffE = rayEnd.x > bounds1 ? bounds1 - rayEnd.x : 0.0f;
	}else{
		diffS = rayStart.x > bounds1 ? bounds1 - rayStart.x : 0.0f;
		diffE = rayEnd.x < bounds0 ? bounds0 - rayEnd.x : 0.0f;
	}
	diffS /= rayDir.x;
	diffE /= rayDir.x;

	//crop the ray to fit the x direction if possible
	if(isfinite(diffS)){
		rayStart.x += rayDir.x * diffS;
		rayStart.y += rayDir.y * diffS;
		rayStart.z += rayDir.z * diffS;
		rayEnd.x += rayDir.x * diffE;
		rayEnd.y += rayDir.y * diffE;
		rayEnd.z += rayDir.z * diffE;
	}
	
	//find the intersection of the ray and the volume (in the y direction)
	if(rayDir.y > 0.0f){
		diffS = rayStart.y < bounds2 ? bounds2 - rayStart.y : 0.0f;
		diffE = rayEnd.y > bounds3 ? bounds3 - rayEnd.y : 0.0f;
	}else{
		diffS = rayStart.y > bounds3 ? bounds3 - rayStart.y : 0.0f;
		diffE = rayEnd.y < bounds2 ? bounds2 - rayEnd.y : 0.0f;
	}
	diffS /= rayDir.y;
	diffE /= rayDir.y;

	//crop the ray to fit the y direction if possible
	if(isfinite(diffS)){
		rayStart.x += rayDir.x * diffS;
		rayStart.y += rayDir.y * diffS;
		rayStart.z += rayDir.z * diffS;
		rayEnd.x += rayDir.x * diffE;
		rayEnd.y += rayDir.y * diffE;
		rayEnd.z += rayDir.z * diffE;
	}
	
	//find the intersection of the ray and the volume (in the z direction)
	if(rayDir.z > 0.0f){
		diffS = rayStart.z < bounds4 ? bounds4 - rayStart.z : 0.0f;
		diffE = rayEnd.z > bounds5 ? bounds5 - rayEnd.z : 0.0f;
	}else{
		diffS = rayStart.z > bounds5 ? bounds5 - rayStart.z : 0.0f;
		diffE = rayEnd.z < bounds4 ? bounds4 - rayEnd.z : 0.0f;
	}
	diffS /= rayDir.z;
	diffE /= rayDir.z;

	//crop the ray to fit the z direction if possible
	if(isfinite(diffS)){
		rayStart.x += rayDir.x * diffS;
		rayStart.y += rayDir.y * diffS;
		rayStart.z += rayDir.z * diffS;
		rayEnd.x += rayDir.x * diffE;
		rayEnd.y += rayDir.y * diffE;
		rayEnd.z += rayDir.z * diffE;
	}

	//refine the ray's length and direction to reflect any changes in the starting and ending co-ordinates
	rayDir.x = rayEnd.x - rayStart.x;
	rayDir.y = rayEnd.y - rayStart.y;
	rayDir.z = rayEnd.z - rayStart.z;
	
	// If the voxel still isn't inside the volume, then this ray
	// doesn't really intersect the volume, thus, make it all zero
	if (rayEnd.x > bounds1 + 1.0f ||
		rayEnd.y > bounds3 + 1.0f ||
		rayEnd.z > bounds5 + 1.0f ||
		rayEnd.x < bounds0 - 1.0f || 
		rayEnd.y < bounds2 - 1.0f || 
		rayEnd.z < bounds4 - 1.0f||
		rayStart.x > bounds1 + 1.0f ||
		rayStart.y > bounds3 + 1.0f ||
		rayStart.z > bounds5 + 1.0f ||
		rayStart.x < bounds0 - 1.0f || 
		rayStart.y < bounds2 - 1.0f || 
		rayStart.z < bounds4 - 1.0f ){
		rayDir.x = 0.0f;
		rayDir.y = 0.0f;
		rayDir.z = 0.0f;
	}

}

__device__ void CUDAkernel_SetRayEnds(const int2& index, float4& rayStart, float3& rayInc, float3& DepthReference, float& numSteps)
{
	float4 rayEnd;
    
    //set the original estimates of the starting and ending co-ordinates in the co-ordinates of the view (not voxels)
    //note: viewRayZ = 0 for start and viewRayZ = 1 for end
    __syncthreads();
    float viewRayX = ( (float) index.x / (float) renInfo.Resolution.x );
    float viewRayY = ( (float) index.y / (float) renInfo.Resolution.y );

	//multiply the start co-ordinate in the view by the view to voxels matrix to get the co-ordinate in voxels (NOT YET NORMALIZED)
	__syncthreads();
	rayStart.x = viewRayX*renInfo.ViewToVoxelsMatrix[0] + viewRayY*renInfo.ViewToVoxelsMatrix[1] + renInfo.ViewToVoxelsMatrix[3];
	rayStart.y = viewRayX*renInfo.ViewToVoxelsMatrix[4] + viewRayY*renInfo.ViewToVoxelsMatrix[5] + renInfo.ViewToVoxelsMatrix[7];
	rayStart.z = viewRayX*renInfo.ViewToVoxelsMatrix[8] + viewRayY*renInfo.ViewToVoxelsMatrix[9] + renInfo.ViewToVoxelsMatrix[11];
	rayStart.w = viewRayX*renInfo.ViewToVoxelsMatrix[12] + viewRayY*renInfo.ViewToVoxelsMatrix[13] + renInfo.ViewToVoxelsMatrix[15];

	//multiply the equivalent for the end ray, noting that much of the pre-normalized computation is the same as the start ray
	__syncthreads();	
	rayEnd.x = rayStart.x + renInfo.ViewToVoxelsMatrix[2];
	rayEnd.y = rayStart.y + renInfo.ViewToVoxelsMatrix[6];
	rayEnd.z = rayStart.z + renInfo.ViewToVoxelsMatrix[10];
	rayEnd.w = rayStart.w + renInfo.ViewToVoxelsMatrix[14];
	__syncthreads();
	
	//normalize (and ergo finish) the start ray's matrix multiplication
	rayStart.x /= rayStart.w;
	rayStart.y /= rayStart.w;
	rayStart.z /= rayStart.w;
	
	//normalize (and ergo finish) the end ray's matrix multiplication
	rayEnd.x /= rayEnd.w;
	rayEnd.y /= rayEnd.w;
	rayEnd.z /= rayEnd.w;
    
    //collect a reference point to judge the depth on
    DepthReference.x = rayStart.x;
    DepthReference.y = rayStart.y;
    DepthReference.z = rayStart.z;
    
    //refine the ray to only include areas that are both within the volume, and within the clipping planes of said volume
	//note that ClipRayAgainstVolume calculate the ray's correct length and direction and returns it in rayInc
	CUDAkernel_ClipRayAgainstClippingPlanes(rayStart, rayEnd, rayInc);
	CUDAkernel_ClipRayAgainstVolume(rayStart, rayEnd, rayInc);
	
	//determine the maximum number of steps the ray should sample, as well as the length of each step
    numSteps = __fsqrt_rd(rayInc.x*rayInc.x+rayInc.y*rayInc.y+rayInc.z*rayInc.z) ;
	rayInc.x /= numSteps;
	rayInc.y /= numSteps;
	rayInc.z /= numSteps;
    
}

__device__ void CUDAkernel_WriteData(const int outindex,
                                const float4& outputVal)
{
	//convert output to uchar, adjusting it to be valued from [0,256) rather than [0,1]
	uchar4 temp;
	temp.x = 255.0f * outputVal.x;
	temp.y = 255.0f * outputVal.y;
	temp.z = 255.0f * outputVal.z;
	temp.w = 255.0f * outputVal.w;
	
	//place output in the image buffer
	int yval = threadIdx.y;
	for(int i = 0; i < BLOCK_DIM2D; i++){
		__syncthreads();
		if(yval == i){
			renInfo.OutputImage[outindex] = temp;
		}
	}
}

__device__ void CUDAkernel_WriteDepth(const int outindex,
                                const float& depthVal,
                                float* depthArray)
{
	//place output in the depth buffer
	int yval = threadIdx.y;
	for(int i = 0; i < BLOCK_DIM2D; i++){
		__syncthreads();
		if(yval == i){
			depthArray[outindex + renInfo.Resolution.x] = depthVal;
		}
	}
}

__global__ void CUDAkernel_renderAlgo_doIntegrationRender2D(float* depthArray)
{

	//index in the output image (2D)
    int2 index;
    index.x = blockDim.x * blockIdx.x + threadIdx.x;
    index.y = blockDim.y * blockIdx.y + threadIdx.y;

    //index in the output image (1D)
    int outindex = index.x + index.y * renInfo.Resolution.x;
    
	float4 rayStart; //ray starting point
	float3 rayInc; // ray sample increment
	float3 depthRef; //ray depth reference
	float numSteps; //maximum number of samples along this ray
    float4 outputVal; //rgba value of this ray (calculated in castRays, used in WriteData)
    float outputDepth; //depth to put in the Gooch-esque shading array

	// Calculate the starting and ending points of the ray, as well as the sampling vector and max number of samples
    CUDAkernel_SetRayEnds(index, rayStart, rayInc, depthRef, numSteps);
    

    // trace along the ray (composite)
    CUDAkernel_CastRays2D(rayStart, numSteps, rayInc, depthRef, outputVal, outputDepth);

    //write to output
    CUDAkernel_WriteData(outindex, outputVal);
    CUDAkernel_WriteDepth(outindex, outputDepth, depthArray);
}

__global__ void CUDAkernel_shadeAlgo_doGoochesqueShade(float* depthArray)
{
	//index in the output image
    int2 index;
    index.x = (blockIdx.x << 8) + threadIdx.x;
    index.y = (blockDim.y*blockIdx.y);
    int outindex = index.x + (index.y * renInfo.Resolution.x); // index of result image
    
    //get the depth information from the buffer and the colour information from the output image
    float2 depthDiffX;
    float2 depthDiffY;

	__syncthreads();
	depthDiffY.y = depthArray[outindex+renInfo.Resolution.x];
	__syncthreads();
	depthDiffY.x = depthArray[outindex-renInfo.Resolution.x];
	__syncthreads();
	depthDiffX.x = depthArray[outindex-1];
	__syncthreads();
	depthDiffX.y = depthArray[outindex+1];
	__syncthreads();

	//compute the gradient magnitude
	float gradMag = __fsqrt_rz( (depthDiffX.y - depthDiffX.x)*(depthDiffX.y - depthDiffX.x)	+
								(depthDiffY.y - depthDiffY.x)*(depthDiffY.y - depthDiffY.x)	);
	
	//grab shading parameters
	__syncthreads();
	float darkness = renInfo.darkness;
	float a = renInfo.a;
	float b = renInfo.b;
	float c = renInfo.computedShift;
	__syncthreads();
	
	//multiply by the depth factor
	gradMag = (c + darkness / ( 1.0f + __expf(a - b * gradMag) ) );
	
	uchar4 colour;
	__syncthreads();
	if(index.x < renInfo.Resolution.x){
		colour = renInfo.OutputImage[outindex];
	}
	__syncthreads();

	colour.x = gradMag * ((float) colour.x);
	colour.y = gradMag * ((float) colour.y);
	colour.z = gradMag * ((float) colour.z);
	
	__syncthreads();
	if(index.x < renInfo.Resolution.x){
		renInfo.OutputImage[outindex] = colour;
	}
	
	return;
}

extern "C"
//pre: the resolution of the image has been processed such that it's x and y size are both multiples of 16 (enforced automatically) and y > 256 (enforced automatically)
//post: the OutputImage pointer will hold the ray casted information
void CUDArenderAlgo_doRender(const cudaRendererInformation& rendererInfo,
                             const cudaVolumeInformation& volumeInfo)
{

    // setup execution parameters - staggered to improve parallelism
	cudaMemcpyToSymbolAsync(volInfo, &volumeInfo, sizeof(cudaVolumeInformation));
    
    //define the block sizes
    
    // setup execution parameters - staggered to improve parallelism
    cudaMemcpyToSymbolAsync(renInfo, &rendererInfo, sizeof(cudaRendererInformation));

    //create the depth buffer for Gooch-esque shading
    float* depth;
    cudaMalloc( (void**) &depth, rendererInfo.Resolution.x*(rendererInfo.Resolution.y+2)*sizeof(float) );

	//create the necessary execution amount parameters from the block sizes and calculate th volume rendering integral
    int blockX = rendererInfo.Resolution.x / BLOCK_DIM2D ;
    int blockY = rendererInfo.Resolution.y / BLOCK_DIM2D ;
    dim3 grid(blockX, blockY, 1);
    dim3 threads(BLOCK_DIM2D, BLOCK_DIM2D, 1);
	CUDAkernel_renderAlgo_doIntegrationRender2D <<< grid, threads >>>(depth);

	//shade the image
	grid.x = ((rendererInfo.Resolution.x-1) / 256) + 1;
	grid.y = rendererInfo.Resolution.y;
    threads.x = 256;
    threads.y = 1;
	CUDAkernel_shadeAlgo_doGoochesqueShade <<< grid, threads >>>(depth);

	//delete the depth buffer
	cudaFree(depth);

	return;
}

extern "C"
void CUDArenderAlgo_changeFrame(int frame){

	// set the texture to the correct image
	input_texture.normalized = false;						// access with normalized texture coordinates
	input_texture.filterMode = cudaFilterModeLinear;		// linear interpolation
	input_texture.addressMode[0] = cudaAddressModeClamp;	// wrap texture coordinates
	input_texture.addressMode[1] = cudaAddressModeClamp;
	input_texture.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    cudaBindTextureToArray(input_texture, sourceDataArray[frame], channelDesc);

}

extern "C"
//pre: the transfer functions are all of type float and are all of size FunctionSize
//post: the alpha, colorR, G and B textures will map 1 dimensionally to each transfer function
void CUDAkernelsetup_loadTextures(const cudaVolumeInformation& volumeInfo, int FunctionSize,
								  float* redTF, float* greenTF, float* blueTF, float* alphaTF){

	//retrieve the size of the transer functions
	size_t size = sizeof(float)*FunctionSize;
	
	if(alphaTransferArray)
		cudaFreeArray(alphaTransferArray);
	if(colorRTransferArray)
		cudaFreeArray(colorRTransferArray);
	if(colorGTransferArray)
		cudaFreeArray(colorGTransferArray);
	if(colorBTransferArray)
		cudaFreeArray(colorBTransferArray);
		
	//allocate space for the arrays
	cudaMallocArray( &alphaTransferArray, &channelDesc, size, FunctionSize);
	cudaMallocArray( &colorRTransferArray, &channelDesc, size, FunctionSize);
	cudaMallocArray( &colorGTransferArray, &channelDesc, size, FunctionSize);
	cudaMallocArray( &colorBTransferArray, &channelDesc, size, FunctionSize);
		
	//define the texture mapping for the alpha component after copying information from host to device array
	cudaMemcpyToArray(alphaTransferArray, 0, 0, alphaTF, size*FunctionSize, cudaMemcpyHostToDevice);
	alpha_texture_2D.normalized = false;
	alpha_texture_2D.filterMode = cudaFilterModeLinear;
	alpha_texture_2D.addressMode[0] = cudaAddressModeClamp;
	alpha_texture_2D.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(alpha_texture_2D, alphaTransferArray, channelDesc);
		
	//define the texture mapping for the red component after copying information from host to device array
	cudaMemcpyToArray(colorRTransferArray, 0, 0, redTF, size*FunctionSize, cudaMemcpyHostToDevice);
	colorR_texture_2D.normalized = false;
	colorR_texture_2D.filterMode = cudaFilterModeLinear;
	colorR_texture_2D.addressMode[0] = cudaAddressModeClamp;
	colorR_texture_2D.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorR_texture_2D, colorRTransferArray, channelDesc);
	
	//define the texture mapping for the green component after copying information from host to device array
	cudaMemcpyToArray(colorGTransferArray, 0, 0, greenTF, size*FunctionSize, cudaMemcpyHostToDevice);
	colorG_texture_2D.normalized = false;
	colorG_texture_2D.filterMode = cudaFilterModeLinear;
	colorG_texture_2D.addressMode[0] = cudaAddressModeClamp;
	colorG_texture_2D.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorG_texture_2D, colorGTransferArray, channelDesc);
	
	//define the texture mapping for the blue component after copying information from host to device array
	cudaMemcpyToArray(colorBTransferArray, 0, 0, blueTF, size*FunctionSize, cudaMemcpyHostToDevice);
	colorB_texture_2D.normalized = false;
	colorB_texture_2D.filterMode = cudaFilterModeLinear;
	colorB_texture_2D.addressMode[0] = cudaAddressModeClamp;
	colorB_texture_2D.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(colorB_texture_2D, colorBTransferArray, channelDesc);
}

extern "C"
//pre:	the data has been preprocessed by the volumeInformationHandler such that it is float data
//		the index is between 0 and 100
//post: the input_texture will map to the source data in voxel coordinate space
void CUDAkernelsetup_loadImageInfo(const cudaVolumeInformation& volumeInfo, int index){

	// if the array is already populated with information, free it to prevent leaking
	if(sourceDataArray[index]){
		cudaFreeArray(sourceDataArray[index]);
	}
	
	//define the size of the data, retrieved from the volume information
	cudaExtent volumeSize;
	volumeSize.width = volumeInfo.VolumeSize.x;
	volumeSize.height = volumeInfo.VolumeSize.y;
	volumeSize.depth = volumeInfo.VolumeSize.z;
	
	// create 3D array
	cudaMalloc3DArray(&(sourceDataArray[index]), &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(volumeInfo.SourceData, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = sourceDataArray[index];
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

}

extern "C"
void CUDAkernelsetup_initImageArray(){
	for(int i = 0; i < 100; i++){
		sourceDataArray[i] = 0;
	}
}

extern "C"
void CUDAkernelsetup_clearImageArray(){
	for(int i = 0; i < 100; i++){
		
		// if the array is already populated with information, free it to prevent leaking
		if(sourceDataArray[i]){
			cudaFreeArray(sourceDataArray[i]);
		}
		
		//null the pointer
		sourceDataArray[i] = 0;
	}
}

//load in a random 16x16 noise array to deartefact the image in real time
extern "C"
void CUDAkernelsetup_loadRandoms(float* randoms){
    cudaMemcpyToSymbolAsync(random, randoms, BLOCK_DIM2D*BLOCK_DIM2D*sizeof(float));
}