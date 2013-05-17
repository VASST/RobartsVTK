#ifndef __CUDA_IMAGEVOTE_H__
#define __CUDA_IMAGEVOTE_H__

#include "vector_types.h"

template<typename IT, typename OT>
void CUDA_CIV_COMPUTE( IT** inputBuffers, int inputNum, OT* outputBuffer, OT* map, int size, cudaStream_t* stream);



#endif