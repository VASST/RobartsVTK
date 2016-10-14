/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    CUDA_imagevote.cu

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file CUDA_imagevote.cu
 *
 *  @brief Implementation file with definitions of GPU kernels used predominantly in performing a voting
 *      operation to merge probabilistic labellings
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#include "CUDA_commonKernels.h"
#include "CUDA_imagevote.h"
#include "vtkCudaCommon.h"

template<typename IT, typename OT>
__global__ void CUDA_CIV_kernMinWithMap(IT* inputBuffer, IT* currentMax, OT* outputBuffer, OT newMapVal, int size)
{
  int idx = CUDASTDOFFSET;

  IT inputValue = inputBuffer[idx];
  IT previValue = currentMax[idx];
  OT previMap   = outputBuffer[idx];

  OT newMap = (inputValue >= previValue) ? newMapVal: previMap;
  IT newVal = (inputValue >= previValue) ? inputValue: previValue;

  if( idx < size )
  {
    currentMax[idx] = newVal;
    outputBuffer[idx] = newMap;
  }
}

template void CUDA_CIV_COMPUTE<double,double>( double** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,double>( long** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,double>( unsigned long** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,double>( long long** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,double>( unsigned long long** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,double>( int** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,double>( unsigned int** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,double>( short** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,double>( unsigned short** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,double>( char** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,double>( unsigned char** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,double>( signed char** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,double>( float** inputBuffers, int inputNum, double* outputBuffer, double* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,long>( double** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,long>( long** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,long>( unsigned long** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,long>( long long** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,long>( unsigned long long** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,long>( int** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,long>( unsigned int** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,long>( short** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,long>( unsigned short** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,long>( char** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,long>( unsigned char** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,long>( signed char** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,long>( float** inputBuffers, int inputNum, long* outputBuffer, long* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,unsigned long>( double** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,unsigned long>( long** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,unsigned long>( unsigned long** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,unsigned long>( long long** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,unsigned long>( unsigned long long** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,unsigned long>( int** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,unsigned long>( unsigned int** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,unsigned long>( short** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,unsigned long>( unsigned short** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,unsigned long>( char** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,unsigned long>( unsigned char** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,unsigned long>( signed char** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,unsigned long>( float** inputBuffers, int inputNum, unsigned long* outputBuffer, unsigned long* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,long long>( double** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,long long>( long** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,long long>( unsigned long** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,long long>( long long** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,long long>( unsigned long long** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,long long>( int** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,long long>( unsigned int** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,long long>( short** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,long long>( unsigned short** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,long long>( char** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,long long>( unsigned char** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,long long>( signed char** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,long long>( float** inputBuffers, int inputNum, long long* outputBuffer, long long* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,unsigned long long>( double** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,unsigned long long>( long** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,unsigned long long>( unsigned long** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,unsigned long long>( long long** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,unsigned long long>( unsigned long long** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,unsigned long long>( int** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,unsigned long long>( unsigned int** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,unsigned long long>( short** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,unsigned long long>( unsigned short** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,unsigned long long>( char** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,unsigned long long>( unsigned char** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,unsigned long long>( signed char** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,unsigned long long>( float** inputBuffers, int inputNum, unsigned long long* outputBuffer, unsigned long long* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,int>( double** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,int>( long** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,int>( unsigned long** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,int>( long long** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,int>( unsigned long long** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,int>( int** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,int>( unsigned int** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,int>( short** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,int>( unsigned short** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,int>( char** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,int>( unsigned char** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,int>( signed char** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,int>( float** inputBuffers, int inputNum, int* outputBuffer, int* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,unsigned int>( double** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,unsigned int>( long** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,unsigned int>( unsigned long** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,unsigned int>( long long** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,unsigned int>( unsigned long long** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,unsigned int>( int** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,unsigned int>( unsigned int** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,unsigned int>( short** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,unsigned int>( unsigned short** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,unsigned int>( char** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,unsigned int>( unsigned char** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,unsigned int>( signed char** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,unsigned int>( float** inputBuffers, int inputNum, unsigned int* outputBuffer, unsigned int* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,short>( double** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,short>( long** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,short>( unsigned long** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,short>( long long** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,short>( unsigned long long** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,short>( int** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,short>( unsigned int** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,short>( short** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,short>( unsigned short** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,short>( char** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,short>( unsigned char** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,short>( signed char** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,short>( float** inputBuffers, int inputNum, short* outputBuffer, short* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,unsigned short>( double** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,unsigned short>( long** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,unsigned short>( unsigned long** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,unsigned short>( long long** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,unsigned short>( unsigned long long** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,unsigned short>( int** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,unsigned short>( unsigned int** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,unsigned short>( short** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,unsigned short>( unsigned short** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,unsigned short>( char** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,unsigned short>( unsigned char** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,unsigned short>( signed char** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,unsigned short>( float** inputBuffers, int inputNum, unsigned short* outputBuffer, unsigned short* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,char>( double** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,char>( long** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,char>( unsigned long** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,char>( long long** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,char>( unsigned long long** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,char>( int** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,char>( unsigned int** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,char>( short** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,char>( unsigned short** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,char>( char** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,char>( unsigned char** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,char>( signed char** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,char>( float** inputBuffers, int inputNum, char* outputBuffer, char* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,unsigned char>( double** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,unsigned char>( long** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,unsigned char>( unsigned long** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,unsigned char>( long long** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,unsigned char>( unsigned long long** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,unsigned char>( int** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,unsigned char>( unsigned int** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,unsigned char>( short** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,unsigned char>( unsigned short** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,unsigned char>( char** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,unsigned char>( unsigned char** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,unsigned char>( signed char** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,unsigned char>( float** inputBuffers, int inputNum, unsigned char* outputBuffer, unsigned char* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,signed char>( double** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,signed char>( long** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,signed char>( unsigned long** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,signed char>( long long** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,signed char>( unsigned long long** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,signed char>( int** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,signed char>( unsigned int** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,signed char>( short** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,signed char>( unsigned short** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,signed char>( char** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,signed char>( unsigned char** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,signed char>( signed char** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,signed char>( float** inputBuffers, int inputNum, signed char* outputBuffer, signed char* map, int size, cudaStream_t* stream);

template void CUDA_CIV_COMPUTE<double,float>( double** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long,float>( long** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long,float>( unsigned long** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<long long,float>( long long** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned long long,float>( unsigned long long** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<int,float>( int** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned int,float>( unsigned int** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<short,float>( short** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned short,float>( unsigned short** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<char,float>( char** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<unsigned char,float>( unsigned char** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<signed char,float>( signed char** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);
template void CUDA_CIV_COMPUTE<float,float>( float** inputBuffers, int inputNum, float* outputBuffer, float* map, int size, cudaStream_t* stream);

template<typename IT, typename OT>
void CUDA_CIV_COMPUTE( IT** inputBuffers, int inputNum, OT* outputBuffer, OT* map, int size, cudaStream_t* stream)
{

  dim3 threads(NUMTHREADS,1,1);
  dim3 grid = GetGrid(size);

  //allocate GPU output buffer and maximum value buffer
  IT* gpuMaxBuffer = 0;
  IT* gpuInBuffer  = 0;
  OT* gpuOutBuffer = 0;
  cudaMalloc( &gpuMaxBuffer, sizeof(IT)*size );
  cudaMalloc( &gpuInBuffer,  sizeof(IT)*size );
  cudaMalloc( &gpuOutBuffer, sizeof(OT)*size );

  //initialize max buffer
  cudaMemcpyAsync( gpuMaxBuffer, inputBuffers[0], sizeof(IT)*size, cudaMemcpyHostToDevice, *stream);

  for(int i = 0; i < inputNum; i++)
  {

    //copy current input in
    cudaMemcpyAsync( gpuInBuffer, inputBuffers[i], sizeof(IT)*size, cudaMemcpyHostToDevice, *stream);

    //perform kernel
    CUDA_CIV_kernMinWithMap<IT,OT><<<grid,threads,0,*stream>>>(gpuInBuffer, gpuMaxBuffer, gpuOutBuffer, map[i], size);

  }

  //copy output back
  cudaMemcpyAsync( outputBuffer, gpuOutBuffer, sizeof(OT)*size, cudaMemcpyDeviceToHost, *stream);

  //sync everything
  cudaStreamSynchronize(*stream);

  //deallocate buffers
  cudaFree(gpuMaxBuffer);
  cudaFree(gpuInBuffer);
  cudaFree(gpuOutBuffer);
}