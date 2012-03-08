extern "C" {
#include "CUDA_segmentAlgo.h"
}

//cuda includes
#include <cuda.h>

struct cudaSegmentInformation {
	//information about transfer function
	float LowI;
	float HighI;
	float LowG;
	float HighG;
	
	//information about volume size
	int SizeX;
	int SizeY;
	int SizeZ;
	
	//information about volume spacing
	float HalfRecSpaceX;
	float HalfRecSpaceY;
	float HalfRecSpaceZ;
};

__constant__ cudaSegmentInformation segInfo;

texture<short, 2, cudaReadModeElementType> classification_texture;
cudaArray* classification_texture_array = 0;
cudaChannelFormatDesc segChannelDesc = cudaCreateChannelDesc<short>();

__global__ void CUDAkernel_segmentAlgo(short* output, short* data){

	__shared__ short s_data[512];

	//load it's location in the volume
	int3 index;
	index.x = threadIdx.x;
	index.y = blockIdx.x;
	index.z = blockIdx.y;

	int VolumeSizeXDiv4 = segInfo.SizeX >> 2;
	int VolumeSizeXYDiv4 = (segInfo.SizeX * segInfo.SizeY) >> 2;
	int inIndex = index.x + VolumeSizeXDiv4 * index.y + VolumeSizeXYDiv4 * index.z ;

	__syncthreads();
	if(index.x < VolumeSizeXDiv4 ){
		short4 move = ((short4*) data)[inIndex];
		s_data[4*index.x] = move.x;
		s_data[4*index.x+1] = move.y;
		s_data[4*index.x+2] = move.z;
		s_data[4*index.x+3] = move.w;
	}

	__syncthreads();
	short curr = s_data[index.x];
	
	//calculate the value of the previous and next values in the X direction
	__syncthreads();
	short prevX = ( (index.x == 0) ? curr : s_data[index.x-1]);
	__syncthreads();
	short nextX = ( (index.x == segInfo.SizeX-1) ? curr : s_data[index.x+1]);
	
	__syncthreads();
	if(index.x < VolumeSizeXDiv4 ){
		short4 move = ((short4*) data)[inIndex - VolumeSizeXDiv4 ];
		s_data[4*index.x] = move.x;
		s_data[4*index.x+1] = move.y;
		s_data[4*index.x+2] = move.z;
		s_data[4*index.x+3] = move.w;
	}
	
	//and the Y direction
	__syncthreads();
	short prevY = ( (index.y == 0) ? curr : s_data[index.x]);
	
	__syncthreads();
	if(index.x < VolumeSizeXDiv4 ){
		short4 move = ((short4*) data)[inIndex + VolumeSizeXDiv4 ];
		s_data[4*index.x] = move.x;
		s_data[4*index.x+1] = move.y;
		s_data[4*index.x+2] = move.z;
		s_data[4*index.x+3] = move.w;
	}
	
	__syncthreads();
	short nextY = ( (index.y == segInfo.SizeY-1) ? curr : s_data[index.x]);
	
	__syncthreads();
	if(index.x < VolumeSizeXDiv4 ){
		short4 move = ((short4*) data)[inIndex - VolumeSizeXYDiv4 ];
		s_data[4*index.x] = move.x;
		s_data[4*index.x+1] = move.y;
		s_data[4*index.x+2] = move.z;
		s_data[4*index.x+3] = move.w;
	}
	
	//and the Z direction
	__syncthreads();
	short prevZ = ( (index.z == 0) ? curr : s_data[index.x]);
	
	__syncthreads();
	if(index.x < VolumeSizeXDiv4 ){
		short4 move = ((short4*) data)[inIndex + VolumeSizeXYDiv4 ];
		s_data[4*index.x] = move.x;
		s_data[4*index.x+1] = move.y;
		s_data[4*index.x+2] = move.z;
		s_data[4*index.x+3] = move.w;
	}
	
	__syncthreads();
	short nextZ = ( (index.z == segInfo.SizeZ-1) ? curr : s_data[index.x]);

	//calculate the gradient
	__syncthreads();
	float gradX = (float) (nextX - prevX) * segInfo.HalfRecSpaceX;
	float gradY = (float) (nextY - prevY) * segInfo.HalfRecSpaceY;
	float gradZ = (float) (nextZ - prevZ) * segInfo.HalfRecSpaceZ;
	
	//find the indexes and collect the classification results
	float gradIndex = ( 128.0f * ( __log2f( gradX*gradX + gradY*gradY + gradZ*gradZ + 1.0f ) ) / segInfo.HighG );
	float intensIndex = ( 256.0f * ( (float) curr - segInfo.LowI ) / (segInfo.HighI - segInfo.LowI ) );
	s_data[index.x] = tex2D(classification_texture, intensIndex, gradIndex);
	if( gradIndex < 0.0f || gradIndex > 256.0f){
		s_data[index.x] = 0;
	}else if( intensIndex < 0.0f || intensIndex > 256.0f){
		s_data[index.x] = 0;
	}
	
	__syncthreads();
	if(index.x < VolumeSizeXDiv4 ){
		short4 move;
		move.x = s_data[4*index.x];
		move.y = s_data[4*index.x+1];
		move.z = s_data[4*index.x+2];
		move.w = s_data[4*index.x+3];
		((short4*) output)[inIndex] = move;
	}
	
}

void CUDAsegmentsetup_loadTexture(short* functionTable){

	//allocate space for the arrays
	if(classification_texture_array){
		cudaFreeArray(classification_texture_array);
	}
	cudaMallocArray( &classification_texture_array, &segChannelDesc, 256*sizeof(short), 256);
		
	//define the texture mapping for the alpha component after copying information from host to device array
	cudaMemcpyToArray(classification_texture_array, 0, 0, functionTable, 256*256*sizeof(short), cudaMemcpyHostToDevice);
	classification_texture.normalized = false;
	classification_texture.filterMode = cudaFilterModePoint;
	classification_texture.addressMode[0] = cudaAddressModeClamp;
	classification_texture.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(classification_texture, classification_texture_array, segChannelDesc);
	
}

void CUDAsegmentAlgo_doSegment(short* output, void* data, int* dims, float* spaces, short* function,
							   float LowI, float HighI, float LowG, float HighG)
{
	//load the classification function into a texture for use
	CUDAsegmentsetup_loadTexture(function);
	delete function;

	//copy the required parameters for segmentation
	cudaSegmentInformation tempInfo;
	tempInfo.LowI = LowI;
	tempInfo.HighI = HighI;
	tempInfo.LowG = LowG;
	tempInfo.HighG = HighG;
	tempInfo.SizeX = dims[0];
	tempInfo.SizeY = dims[1];
	tempInfo.SizeZ = dims[2];
	tempInfo.HalfRecSpaceX = 0.5f / spaces[0];
	tempInfo.HalfRecSpaceY = 0.5f / spaces[1];
	tempInfo.HalfRecSpaceZ = 0.5f / spaces[2];
    cudaMemcpyToSymbolAsync(segInfo, &tempInfo, sizeof(cudaSegmentInformation));
	
	//define the output area
	int memorySize = dims[0] * dims[1] * dims[2];
	short* d_output;
	short* d_input;
	
	cudaMalloc( (void**) &d_output, memorySize*sizeof(short) );
	cudaMalloc( (void**) &d_input,  memorySize*sizeof(short) );
	cudaMemcpy( (void*) d_input, data, memorySize*sizeof(short), cudaMemcpyHostToDevice );
	
	//create the necessary execution amount parameters from the block sizes
    dim3 grid(dims[1], dims[2], 1);
    dim3 threads(dims[0], 1, 1);
    
    CUDAkernel_segmentAlgo<<< grid, threads >>>(d_output, d_input);
    
    cudaMemcpy( (void*) output, (void*) d_output, memorySize*sizeof(short), cudaMemcpyDeviceToHost );
    cudaFree( (void*) d_output );
    cudaFree( (void*) d_input );
    
}