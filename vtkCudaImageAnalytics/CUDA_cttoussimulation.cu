#include "CUDA_cttoussimulation.h"
#include "CUDA_commonKernels.h"

//3D input data (read-only texture with corresponding opague device memory back)
texture<float, 3, cudaReadModeElementType> ct_input_texture;
cudaArray* ct_input_array = 0;
texture<unsigned char, 3, cudaReadModeElementType> us_input_texture;
cudaArray* us_input_array = 0;

//parameters held in constant memory
__constant__ CT_To_US_Information info;

//device code to collect information on each of the sample points
__device__ void CUDAkernel_CollectSamples(	int2 index,
						float3& rayStart,
						float3& rayInc,
						float* outputDensity,
						float* outputTransmission,
						float* outputReflection){

	//collect parameters from constant memory
	uint3 volumeSize = info.VolumeSize;
	float3 spacing = info.spacing;
	unsigned int numStepsToTake = info.Resolution.z;
	unsigned int xResolution = info.Resolution.x;
	unsigned int yResolution = info.Resolution.y;
	unsigned int actIndex = index.x + index.y* xResolution;
	unsigned int indexInc = xResolution*yResolution;
	bool isValid = (index.y < yResolution);
	float threshold = info.reflectionThreshold;
	float densitySlope = 0.5f*0.00025f;
	float densityIntercept = 512.0f*0.00025f;
	float HounsFieldScale = info.hounsfieldScale;
	float HounsFieldOffset = info.hounsfieldOffset;

	float directionMag = sqrt( rayInc.x*rayInc.x + rayInc.y*rayInc.y + rayInc.z*rayInc.z );
	float worldDirectionMag = 2.0f * sqrt( rayInc.x*rayInc.x/(spacing.x*spacing.x) +
										   rayInc.y*rayInc.y/(spacing.y*spacing.y) +
										   rayInc.z*rayInc.z/(spacing.z*spacing.z) );

	//set up running accumulators
	float transmission = 1.0f;

	//set up output scaling parameters
	float multiplier = info.a;
	float divisor = 1.0f / log(1.0f+multiplier);

	for(unsigned int numStepsTaken = 0; numStepsTaken < numStepsToTake; numStepsTaken++){

		//create default values for the sample point
		float density = 0.0f;
		float transmissionLost = 1.0f;
		float pointReflection = 0.0f;

		float attenuation = 0.0f;

		__syncthreads();
		if(!(rayStart.x < 0.0f || rayStart.y < 0.0f || rayStart.z < 0.0f ||
			 rayStart.x > (float)(volumeSize.x - 1) ||
			 rayStart.y > (float)(volumeSize.y - 1) ||
			 rayStart.y > (float)(volumeSize.y - 1) )){

			//get the attenuation and gradient of the attenuation in Hounsfield units
			attenuation = HounsFieldScale*tex3D(ct_input_texture, rayStart.x, rayStart.y, rayStart.z) + HounsFieldOffset;
			float gradientX = HounsFieldScale*(tex3D(ct_input_texture, rayStart.x+1.0f, rayStart.y, rayStart.z) - tex3D(ct_input_texture, rayStart.x-1.0f, rayStart.y, rayStart.z)) * spacing.x;
			float gradientY = HounsFieldScale*(tex3D(ct_input_texture, rayStart.x, rayStart.y+1.0f, rayStart.z) - tex3D(ct_input_texture, rayStart.x, rayStart.y-1.0f, rayStart.z)) * spacing.y;
			float gradientZ = HounsFieldScale*(tex3D(ct_input_texture, rayStart.x, rayStart.y, rayStart.z+1.0f) - tex3D(ct_input_texture, rayStart.x, rayStart.y, rayStart.z-1.0f)) * spacing.z;
			float gradMagSquared = gradientX*gradientX + gradientY*gradientY + gradientZ*gradientZ;
			float gradMag = sqrt( gradMagSquared );

			//calculate the reflection, density and transmission at this sample point
			transmissionLost = (gradMag < threshold) ? saturate( 1.0f - gradMagSquared * worldDirectionMag / (4.0f * attenuation * attenuation) ) : 0.0f;
			pointReflection  = transmission * (rayInc.x*gradientX + rayInc.y*gradientY + rayInc.z*gradientZ) * gradMag / ( 4.0f * attenuation * attenuation * directionMag );
			density          = (transmission > 0.0f) ? densitySlope * attenuation + densityIntercept : 0.0f;

		}

		//scale the point reflection
		pointReflection = saturate( log( 1 + multiplier * pointReflection ) * divisor );
		
		//output the reflection and density
		__syncthreads();
		if( isValid ) outputReflection[actIndex] = pointReflection;
		__syncthreads();
		if( isValid ) outputDensity[actIndex] = density;
		__syncthreads();

		//update the running values
		transmission *= transmissionLost;

		//output the transmission
		__syncthreads();
		if( isValid ) outputTransmission[actIndex] = transmission;
		__syncthreads();

		//update the sampling location
		actIndex += indexInc;
		rayStart.x += rayInc.x;
		rayStart.y += rayInc.y;
		rayStart.z += rayInc.z;

	}


}

__global__ void CUDAkernel_ColourParamOutput(float alpha, float beta, float bias,
						float* outputDensity,
						float* outputReflection,
						uchar3* outputUltrasound ){
	int actIndex = CUDASTDOFFSET;
	bool isValid = (actIndex < info.Resolution.x*info.Resolution.y*info.Resolution.z );
	
	__syncthreads();
	float pointReflection = ( isValid ) ? outputReflection[actIndex] : 0.0f;
	__syncthreads();
	float density = ( isValid ) ? outputDensity[actIndex] : 0.0f;
	__syncthreads();

	uchar3 outputImage;
	outputImage.x = 255.0f * saturate(alpha*density+beta*pointReflection+bias);
	outputImage.y = 255.0f * saturate(alpha*density+beta*pointReflection+bias);
	outputImage.z = 255.0f * saturate(alpha*density+beta*pointReflection+bias);

	//output the simulated ultrasound
	__syncthreads();
	if( isValid ) outputUltrasound[actIndex] = outputImage;
	__syncthreads();
}

//device code to determine from the parameters, the start, end and increment vectors in volume space
__device__ void CUDAkernel_FindVectors(	float2 nIndex,
										float3& rayStart,
										float3& rayInc){

	
	//find the US coordinates of this particular beam's Start point
	float3 usStart;
	usStart.x = tan( info.fanAngle.x * nIndex.x );
	usStart.y = tan( info.fanAngle.y * nIndex.y );
	usStart.z = __fsqrt_rz( info.StartDepth * info.StartDepth / 
							( 1.0f + usStart.x*usStart.x + usStart.y*usStart.y) );
	usStart.x = 0.5f * info.probeWidth.x * nIndex.x + usStart.x*usStart.z;
	usStart.y = 0.5f * info.probeWidth.y * nIndex.y + usStart.y*usStart.z;
	__syncthreads();

	//find the Start vector in world coordinates
	float4 worldStart;
	worldStart.x = info.UltraSoundToWorld[ 0] * usStart.x + info.UltraSoundToWorld[ 1] * usStart.y + info.UltraSoundToWorld[ 2] * usStart.z + info.UltraSoundToWorld[ 3];
	worldStart.y = info.UltraSoundToWorld[ 4] * usStart.x + info.UltraSoundToWorld[ 5] * usStart.y + info.UltraSoundToWorld[ 6] * usStart.z + info.UltraSoundToWorld[ 7];
	worldStart.z = info.UltraSoundToWorld[ 8] * usStart.x + info.UltraSoundToWorld[ 9] * usStart.y + info.UltraSoundToWorld[10] * usStart.z + info.UltraSoundToWorld[11];
	worldStart.w = info.UltraSoundToWorld[12] * usStart.x + info.UltraSoundToWorld[13] * usStart.y + info.UltraSoundToWorld[14] * usStart.z + info.UltraSoundToWorld[15];
	__syncthreads();
	worldStart.x /= worldStart.w; 
	worldStart.y /= worldStart.w; 
	worldStart.z /= worldStart.w;

	//transform the Start into volume co-ordinates
	__syncthreads();
	rayStart.x   = info.WorldToVolume[ 0]*worldStart.x + info.WorldToVolume[ 1]*worldStart.y + info.WorldToVolume[ 2]*worldStart.z + info.WorldToVolume[ 3];
	rayStart.y   = info.WorldToVolume[ 4]*worldStart.x + info.WorldToVolume[ 5]*worldStart.y + info.WorldToVolume[ 6]*worldStart.z + info.WorldToVolume[ 7];
	rayStart.z   = info.WorldToVolume[ 8]*worldStart.x + info.WorldToVolume[ 9]*worldStart.y + info.WorldToVolume[10]*worldStart.z + info.WorldToVolume[11];
	worldStart.w = info.WorldToVolume[12]*worldStart.x + info.WorldToVolume[13]*worldStart.y + info.WorldToVolume[14]*worldStart.z + info.WorldToVolume[15];
	__syncthreads();
	rayStart.x /= worldStart.w;
	rayStart.y /= worldStart.w;
	rayStart.z /= worldStart.w;
	

	//find the US coordinates of this particular beam's Start point
	float3 usEnd;
	usEnd.x = tan( info.fanAngle.x * nIndex.x );
	usEnd.y = tan( info.fanAngle.y * nIndex.y );
	usEnd.z = __fsqrt_rz( (info.EndDepth * info.EndDepth) / 
						( 1.0f + usEnd.x*usEnd.x + usEnd.y*usEnd.y) );
	usEnd.x = 0.5f * info.probeWidth.x * nIndex.x + usEnd.x*usEnd.z;
	usEnd.y = 0.5f * info.probeWidth.y * nIndex.y + usEnd.y*usEnd.z;
	
	//find the End vector in world coordinates
	float4 worldEnd;
	__syncthreads();
	worldEnd.x = info.UltraSoundToWorld[ 0] * usEnd.x + info.UltraSoundToWorld[ 1] * usEnd.y + info.UltraSoundToWorld[ 2] * usEnd.z + info.UltraSoundToWorld[ 3];
	worldEnd.y = info.UltraSoundToWorld[ 4] * usEnd.x + info.UltraSoundToWorld[ 5] * usEnd.y + info.UltraSoundToWorld[ 6] * usEnd.z + info.UltraSoundToWorld[ 7];
	worldEnd.z = info.UltraSoundToWorld[ 8] * usEnd.x + info.UltraSoundToWorld[ 9] * usEnd.y + info.UltraSoundToWorld[10] * usEnd.z + info.UltraSoundToWorld[11];
	worldEnd.w = info.UltraSoundToWorld[12] * usEnd.x + info.UltraSoundToWorld[13] * usEnd.y + info.UltraSoundToWorld[14] * usEnd.z + info.UltraSoundToWorld[15];
	__syncthreads();
	worldEnd.x /= worldEnd.w; 
	worldEnd.y /= worldEnd.w; 
	worldEnd.z /= worldEnd.w;

	//transform the End into volume co-ordinates
	float3 rayEnd;
	__syncthreads();
	rayEnd.x   = info.WorldToVolume[ 0]*worldEnd.x + info.WorldToVolume[ 1]*worldEnd.y + info.WorldToVolume[ 2]*worldEnd.z + info.WorldToVolume[ 3];
	rayEnd.y   = info.WorldToVolume[ 4]*worldEnd.x + info.WorldToVolume[ 5]*worldEnd.y + info.WorldToVolume[ 6]*worldEnd.z + info.WorldToVolume[ 7];
	rayEnd.z   = info.WorldToVolume[ 8]*worldEnd.x + info.WorldToVolume[ 9]*worldEnd.y + info.WorldToVolume[10]*worldEnd.z + info.WorldToVolume[11];
	worldEnd.w = info.WorldToVolume[12]*worldEnd.x + info.WorldToVolume[13]*worldEnd.y + info.WorldToVolume[14]*worldEnd.z + info.WorldToVolume[15];
	__syncthreads();
	rayEnd.x /= worldEnd.w;
	rayEnd.y /= worldEnd.w;
	rayEnd.z /= worldEnd.w;

	//calculate the increment vector
	__syncthreads();
	rayInc.x = (rayStart.x-rayEnd.x) / info.Resolution.z;
	rayInc.y = (rayStart.y-rayEnd.y) / info.Resolution.z;
	rayInc.z = (rayStart.z-rayEnd.z) / info.Resolution.z;

}

__global__ void CUDAkernel_CreateSimulatedUS(	float* outputDensity,
												float* outputTransmission,
												float* outputReflection){

	//find x index value in the simulated ultrasound image
	int2 index;
	index.x = CUDASTDOFFSET % info.Resolution.x;
	index.y = CUDASTDOFFSET / info.Resolution.x;

	//find the normalized indices
	float2 normIndex;
	normIndex.x = (float) (index.x+index.x) / (float) info.Resolution.x - 1.0f;
	normIndex.y = (float) (index.y+index.y) / (float) info.Resolution.y - 1.0f;
	
	//starting and increment vectors
	float3 rayStart;
	float3 rayInc;
	CUDAkernel_FindVectors( normIndex, rayStart, rayInc );

	//simulate the ultrasound (writing to the output buffers)
	CUDAkernel_CollectSamples( index, rayStart, rayInc, outputDensity, outputTransmission, outputReflection );

}

__global__ void CUDAkernel_sampleU( float* destination, int size ){
	int fullIndex = CUDASTDOFFSET;
	int3 index;
	index.x = fullIndex % info.Resolution.x;
	index.z = fullIndex / info.Resolution.x;
	index.y = index.z % info.Resolution.y;
	index.z = index.z / info.Resolution.y;

	if( info.Resolution.y == 1 ){
		int temp = index.z;
		index.z = index.y;
		index.y = temp;
	}

	unsigned char value = tex3D(us_input_texture, index.x, index.y, index.z);

	if( fullIndex < size ) destination[fullIndex] = (float) value / 255.0f;
}

__global__ void CUDAkernel_multiply( float* sourceA, float* sourceB, float* destination, int size ){
	int index = CUDASTDOFFSET;
	float a = sourceA[index];
	float b = sourceB[index];
	if( index < size ) destination[index] = a * b;
}

__global__ void CUDAkernel_accumulate( float* buffer, int addSize, int size ){
	int index = CUDASTDOFFSET;
	float a = buffer[index];
	float b = buffer[index+addSize];
	if( index+addSize < size ) buffer[index] = a+b;
}

void CUDAsetup_loadCTImage( float* CTImage, CT_To_US_Information& information, cudaStream_t* stream){

	//free the array if there is another image residing
	cudaStreamSynchronize( *stream );
	if(ct_input_array ) cudaFreeArray(ct_input_array );

	//find the volume size
	cudaExtent volumeSize;
	volumeSize.width = information.VolumeSize.x;
	volumeSize.height = information.VolumeSize.y;
	volumeSize.depth = information.VolumeSize.z;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&ct_input_array, &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(CTImage, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
	copyParams.dstArray = ct_input_array;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3DAsync(&copyParams, *stream);

	// set the texture parameters
	ct_input_texture.normalized = false;						// access with unnormalized texture coordinates
	ct_input_texture.filterMode = cudaFilterModeLinear;			// linear interpolation
	ct_input_texture.addressMode[0] = cudaAddressModeClamp;		// wrap texture coordinates
	ct_input_texture.addressMode[1] = cudaAddressModeClamp;
	ct_input_texture.addressMode[2] = cudaAddressModeClamp;

	//bind the texture in
	cudaBindTextureToArray(ct_input_texture, ct_input_array, channelDesc);

}

void CUDAsetup_loadUSImage( unsigned char* USImage, int resolution[3], cudaStream_t* stream){

	//free the array if there is another image residing
	cudaStreamSynchronize( *stream );
	if(us_input_array ) cudaFreeArray(us_input_array );

	//find the volume size
	cudaExtent volumeSize;
	volumeSize.width = resolution[0];
	volumeSize.height = resolution[1];
	volumeSize.depth = resolution[2];

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
	cudaMalloc3DArray(&us_input_array, &channelDesc, volumeSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(USImage, volumeSize.width*sizeof(unsigned char), volumeSize.width, volumeSize.height);
	copyParams.dstArray = ct_input_array;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3DAsync(&copyParams, *stream);

	// set the texture parameters
	us_input_texture.normalized = false;						// access with unnormalized texture coordinates
	us_input_texture.filterMode = cudaFilterModeLinear;			// linear interpolation
	us_input_texture.addressMode[0] = cudaAddressModeClamp;		// wrap texture coordinates
	us_input_texture.addressMode[1] = cudaAddressModeClamp;
	us_input_texture.addressMode[2] = cudaAddressModeClamp;

	//bind the texture in
	cudaBindTextureToArray(us_input_texture, us_input_array, channelDesc);

}

void CUDAsetup_unloadCTImage(cudaStream_t* stream){
	cudaStreamSynchronize( *stream );
	if(ct_input_array ) cudaFreeArray(ct_input_array );
	ct_input_array = 0;
}

void CUDAsetup_unloadUSImage(cudaStream_t* stream){
	cudaStreamSynchronize( *stream );
	if(us_input_array ) cudaFreeArray(us_input_array );
	us_input_array = 0;
}

inline int pow2roundup (int x) {
    if (x < 0)
        return 0;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

void CUDAalgo_simulateUltraSound(	float* outputDensity, float* outputTransmission, float* outputReflection, unsigned char* outputUltrasound,
									CT_To_US_Information& information, cudaStream_t* stream ){

	//copy the information to the device
	cudaMemcpyToSymbolAsync(info, &information, sizeof(CT_To_US_Information), 0, cudaMemcpyHostToDevice, *stream);

	//allocate the device output buffers
	float* device_output_dens;
	float* device_output_trans;
	float* device_output_refl;
	uchar3* device_output_us;
	cudaMalloc( (void**) &device_output_dens,  sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z );
	cudaMalloc( (void**) &device_output_trans, sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z );
	cudaMalloc( (void**) &device_output_refl,  sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z );
	cudaMalloc( (void**) &device_output_us,    3*sizeof(unsigned char)*information.Resolution.x*information.Resolution.y*information.Resolution.z );

	//simulate the ultrasound
	dim3 threads( NUMTHREADS, 1, 1);
	dim3 grid = GetGrid(information.Resolution.x * information.Resolution.y);
	CUDAkernel_CreateSimulatedUS<<< grid, threads, 0, *stream >>>( device_output_dens, device_output_trans, device_output_refl );
	
	//copy the preliminary results
	cudaMemcpyAsync( (void*) outputDensity,      (void*) device_output_dens,  sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z, cudaMemcpyDeviceToHost, *stream );
	cudaMemcpyAsync( (void*) outputTransmission, (void*) device_output_trans, sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z, cudaMemcpyDeviceToHost, *stream );
	cudaMemcpyAsync( (void*) outputReflection,   (void*) device_output_refl,  sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z, cudaMemcpyDeviceToHost, *stream );
	cudaFree(device_output_trans);

	//optimal parameter fiddling
	grid = GetGrid(information.Resolution.x*information.Resolution.y*information.Resolution.z);
	int largestAddSize = pow2roundup( information.Resolution.x * information.Resolution.y * information.Resolution.z );
	int actualSize = information.Resolution.x*information.Resolution.y*information.Resolution.z;
	if( information.optimalParam ){
		
		//add together all the information in the simulation and corresponding location in the ultrasound
		//to produce the M matrix and U vector
		
		//find the values of u (interpolated from given texture)
		float* device_output_u;
		cudaMalloc( (void**) &device_output_u,  sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z );
		CUDAkernel_sampleU<<< grid, threads, 0, *stream >>>( device_output_u, actualSize );
		
		//find the sum of the du terms
		float* device_output_dens_u;
		cudaMalloc( (void**) &device_output_dens_u,  sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z );
		CUDAkernel_multiply<<< grid, threads, 0, *stream >>>( device_output_dens, device_output_u, device_output_dens_u, actualSize);
		for(int i = largestAddSize; i > 0; i /= 2){
			CUDAkernel_accumulate<<< grid, threads, 0, *stream >>>(device_output_dens_u, largestAddSize, actualSize);
		}
		float sum_du = 0.0f;
		cudaMemcpyAsync((void*) &sum_du, (void*) device_output_dens_u, sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaFree(device_output_dens_u);

		//find the sum of the ru terms
		float* device_output_refl_u;
		cudaMalloc( (void**) &device_output_refl_u,  sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z );
		CUDAkernel_multiply<<< grid, threads, 0, *stream >>>( device_output_refl, device_output_u, device_output_refl_u, actualSize);
		for(int i = largestAddSize; i > 0; i /= 2){
			CUDAkernel_accumulate<<< grid, threads, 0, *stream >>>(device_output_refl_u, largestAddSize, actualSize);
		}
		float sum_ru = 0.0f;
		cudaMemcpyAsync((void*) &sum_ru, (void*) device_output_refl_u,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaFree(device_output_refl_u);
		
		//find the sum of the u terms
		for(int i = largestAddSize; i > 0; i /= 2){
			CUDAkernel_accumulate<<< grid, threads, 0, *stream >>>(device_output_u, largestAddSize, actualSize);
		}
		float sum_u = 0.0f;
		cudaMemcpyAsync((void*) &sum_u, (void*) device_output_u,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaFree(device_output_u);
		
		//find the sum of the d^2 terms
		float* device_output_dens_square;
		cudaMalloc( (void**) &device_output_dens_square,  sizeof(float)*actualSize );
		CUDAkernel_multiply<<< grid, threads, 0, *stream >>>( device_output_dens, device_output_dens, device_output_dens_square, actualSize);
		for(int i = largestAddSize; i > 0; i /= 2){
			CUDAkernel_accumulate<<< grid, threads, 0, *stream >>>(device_output_dens_square, largestAddSize, actualSize);
		}
		float sum_d2 = 0.0f;
		cudaMemcpyAsync((void*) &sum_d2, (void*) device_output_dens_square,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaFree(device_output_dens_square);
		
		//find the sum of the dr terms
		float* device_output_dens_refl;
		cudaMalloc( (void**) &device_output_dens_refl,  sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z );
		CUDAkernel_multiply<<< grid, threads, 0, *stream >>>( device_output_dens, device_output_refl, device_output_dens_refl, actualSize);
		for(int i = largestAddSize; i > 0; i /= 2){
			CUDAkernel_accumulate<<< grid, threads, 0, *stream >>>(device_output_dens_refl, largestAddSize, actualSize);
		}
		float sum_dr = 0.0f;
		cudaMemcpyAsync((void*) &sum_dr, (void*) device_output_dens_refl,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaFree(device_output_dens_refl);
		
		//find the sum of the r^2 terms
		float* device_output_refl_square;
		cudaMalloc( (void**) &device_output_refl_square,  sizeof(float)*information.Resolution.x*information.Resolution.y*information.Resolution.z );
		CUDAkernel_multiply<<< grid, threads, 0, *stream >>>( device_output_refl, device_output_refl, device_output_refl_square, actualSize);
		for(int i = largestAddSize; i > 0; i /= 2){
			CUDAkernel_accumulate<<< grid, threads, 0, *stream >>>(device_output_refl_square, largestAddSize, actualSize);
		}
		float sum_r2 = 0.0f;
		cudaMemcpyAsync((void*) &sum_r2, (void*) device_output_refl_square,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
		cudaFree(device_output_refl_square);
			
		//find the sum of the r terms
		for(int i = largestAddSize; i > 0; i /= 2){
			CUDAkernel_accumulate<<< grid, threads, 0, *stream >>>(device_output_refl, largestAddSize, actualSize);
		}
		float sum_r = 0.0f;
		cudaMemcpyAsync((void*) &sum_r, (void*) device_output_refl,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
		
		//find the sum of the d terms
		for(int i = largestAddSize; i > 0; i /= 2){
			CUDAkernel_accumulate<<< grid, threads, 0, *stream >>>(device_output_dens, largestAddSize, actualSize);
		}
		float sum_d = 0.0f;
		cudaMemcpyAsync((void*) &sum_d, (void*) device_output_dens,  sizeof(float), cudaMemcpyDeviceToHost, *stream );
		
		//find the adjoint matrix
		float m11 = (sum_r2*actualSize)-(sum_r * sum_r);
		float m12 = -1.0f * (sum_dr*actualSize)-(sum_d * sum_r);
		float m13 = (sum_dr*sum_r)-(sum_d * sum_r2);

		float m21 = -1.0f * (sum_dr*actualSize)-(sum_r * sum_d);
		float m22 = (sum_d2*actualSize)-(sum_d * sum_d);
		float m23 = -1.0f * (sum_d2*sum_r)-(sum_d * sum_dr);

		float m31 = (sum_dr*sum_r)-(sum_r2 * sum_d);
		float m32 = -1.0f * (sum_d2*sum_r)-(sum_dr * sum_d);
		float m33 = (sum_d2*sum_r2)-(sum_dr * sum_dr);

		//find the determinant
		float det = sum_d2 * m11 - sum_dr * m12 + sum_d * m13;

		//find the inverse matrix (assume non-zero determinant therefore invertible M)
		//then multiply by U vector
		float calcAlpha = (m11*sum_du+m12*sum_ru+m13*sum_u) / det;
		float calcBeta = (m21*sum_du+m22*sum_ru+m23*sum_u) / det;;
		float calcBias = (m31*sum_du+m32*sum_ru+m33*sum_u) / det;;

		//copy new linear combination to the information holder
		information.alpha = calcAlpha;
		information.beta = calcBeta;
		information.bias = calcBias;
		
	}

	//we already have parameters, so just use them
	CUDAkernel_ColourParamOutput<<< grid, threads, 0, *stream >>>( information.alpha, information.beta, information.bias, device_output_dens, device_output_refl, device_output_us );
	
	//calculate cross correlation if warranted
	//TODO
	if( information.optimalParam ){

	}

	//copy the results
	cudaMemcpyAsync( (void*) outputUltrasound,   (void*) device_output_us,    3*sizeof(unsigned char)*information.Resolution.x*information.Resolution.y*information.Resolution.z, cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize( *stream );

	//free the device buffers
	cudaFree(device_output_dens);
	cudaFree(device_output_refl);
	cudaFree(device_output_us);

}