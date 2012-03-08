#ifndef CUDA_SEGMENTALGO_H
#define CUDA_SEGMENTALGO_H

extern "C"
void CUDAsegmentAlgo_doSegment(short* output, void* data, int* dims, float* spaces, short* function,
							   float LowI, float HighI, float LowG, float HighG);

#endif