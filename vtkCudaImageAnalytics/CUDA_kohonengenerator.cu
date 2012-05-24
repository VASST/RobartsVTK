#include "CUDA_kohonengenerator.h"

//3D input data (read-only texture with corresponding opague device memory back)
texture<float, 3, cudaReadModeElementType> kohonen_input_texture;
cudaArray* kohonen_input_array = 0;

//parameters held in constant memory
__constant__ Kohonen_Generator_Information info;

void CUDAsetup_loadNDImage( cudaStream_t* stream ){

}

void CUDAsetup_loadNDImage( float* image, Kohonen_Generator_Information& information, cudaStream_t* stream){

}

void CUDAalgo_generateKohonenMap( float* outputKohonen, Kohonen_Generator_Information& information, cudaStream_t* stream ){

}
