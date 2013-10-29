/*=========================================================================

  Program:   Visualization Toolkit
  Module:    CUDA_loglikelihoodterm.cu

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file CUDA_loglikelihoodterm.cu
 *
 *  @brief Implementation file with definitions of GPU kernels used predominantly in calculating
 *			log likelihood components of data terms for GHMF.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *	
 *	@note August 27th 2013 - Documentation first compiled.
 *
 */
 
#include "CUDA_loglikelihoodterm.h"
#include "CUDA_commonKernels.h"
#include "stdio.h"
#include "cuda.h"
#include "float.h"
#include "limits.h"

//#define DEBUG_VTKCUDA_ILLT

template void CUDA_ILLT_IncrementInformation<float>(float* labelData, float desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<double>(double* labelData, double desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<long>(long* labelData, long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<unsigned long>(unsigned long* labelData, unsigned long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<long long>(long long* labelData, long long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<unsigned long long>(unsigned long long* labelData, unsigned long long desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<int>(int* labelData, int desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<unsigned int>(unsigned int* labelData, unsigned int desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<short>(short* labelData, short desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<unsigned short>(unsigned short* labelData, unsigned short desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<char>(char* labelData, char desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<signed char>(signed char* labelData, signed char desiredValue, short* agreement, int size, cudaStream_t* stream);
template void CUDA_ILLT_IncrementInformation<unsigned char>(unsigned char* labelData, unsigned char desiredValue, short* agreement, int size, cudaStream_t* stream);

template< class T >
void CUDA_ILLT_IncrementInformation(T* labelData, T desiredValue, short* agreement, int size, cudaStream_t* stream){
    T* GPUBuffer = 0;

	cudaMalloc((void**) &GPUBuffer, sizeof(T)*size);
	cudaMemcpyAsync( GPUBuffer, labelData, sizeof(T)*size, cudaMemcpyHostToDevice, *stream );

	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_IncrementInformation: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	dim3 threads(NUMTHREADS,1,1);
	dim3 grid = GetGrid(size);
	IncrementBuffer<T><<<grid,threads,0,*stream>>>(GPUBuffer, desiredValue, agreement, size);
	cudaFree(GPUBuffer);

	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_IncrementInformation: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

void CUDA_ILLT_GetRelevantBuffers(short** agreement, int size, cudaStream_t* stream){
	cudaMalloc((void**) agreement, sizeof(short)*size);
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid = GetGrid(size);
	ZeroOutBuffer<short><<<grid,threads,0,*stream>>>(*agreement,size);

	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_GetRelevantBuffers: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

void CUDA_ILLT_CopyBackResult(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream){
	cudaThreadSynchronize();
	cudaMemcpy( CPUBuffer, GPUBuffer, sizeof(float)*size, cudaMemcpyDeviceToHost );
	cudaFree(GPUBuffer);

	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_CopyBackResult: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

void CUDA_ILLT_AllocateHistogram(float** histogramGPU, int size, cudaStream_t* stream){
	cudaMalloc((void**) histogramGPU, sizeof(float)*size);
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid = GetGrid(size);
	
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_AllocateHistogram: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

void CUDA_ILLT_ReturnBuffer(float* buffer){
	cudaFree(buffer);
}

template void CUDA_ILLT_CalculateHistogramAndTerms<double>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, double* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<long>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned long>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, unsigned long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<long long>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, long long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned long long>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, unsigned long long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<int>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, int* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned int>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, unsigned int* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<short>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, short* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned short>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, unsigned short* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<char>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, char* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<signed char>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, signed char* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned char>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, unsigned char* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<float>(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, float* image,	 short requiredAgreement, int imageSize, cudaStream_t* stream);


template void CUDA_ILLT_CalculateHistogramAndTerms2D<double>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, double* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<long>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<unsigned long>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, unsigned long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<long long>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, long long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<unsigned long long>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, unsigned long long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<int>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, int* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<unsigned int>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, unsigned int* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<short>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, short* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<unsigned short>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, unsigned short* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<char>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, char* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<signed char>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, signed char* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<unsigned char>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, unsigned char* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms2D<float>(float* outputBuffer, float* histogramGPU, int histsize, short* agreement, float* image,	 short requiredAgreement, int imageSize, cudaStream_t* stream);

template<class T>
__global__ void kern_PopulateWorkingUp(float* working, short* agreement, T* image, short requiredAgreement, int imageSize){
	int idx = CUDASTDOFFSET;
	float inputValue = (float) image[idx];
	short lAgreement = agreement[idx];
	float outputValue = (lAgreement < requiredAgreement) ? FLT_MIN: inputValue;
	if(idx < imageSize) working[idx] = outputValue;
}

template<class T>
__global__ void kern_PopulateWorkingDown(float* working, short* agreement, T* image, short requiredAgreement, int imageSize){
	int idx = CUDASTDOFFSET;
	float inputValue = (float) image[idx];
	short lAgreement = agreement[idx];
	float outputValue = (lAgreement < requiredAgreement) ? FLT_MAX: inputValue;
	if(idx < imageSize) working[idx] = outputValue;
}

__global__ void kern_PropogateUp(float* working, int span, int imageSize){
	int idx = CUDASTDOFFSET;
	float inputValue1 = working[idx];
	float inputValue2 = working[idx+span];
	float outputVal = (inputValue1 > inputValue2) ? inputValue1: inputValue2;
	if(idx+span < imageSize) working[idx] = outputVal;
}

__global__ void kern_PropogateDown(float* working, int span, int imageSize){
	int idx = CUDASTDOFFSET;
	float inputValue1 = working[idx];
	float inputValue2 = working[idx+span];
	float outputVal = (inputValue1 < inputValue2) ? inputValue1: inputValue2;
	if(idx+span < imageSize) working[idx] = outputVal;
}

template<class T>
__global__ void kern_PopulateHisto(float* histogramGPU, int histSize, short* agreement, T* image, short requiredAgreement, float imMin, float imMax, int imageSize){
	__shared__ float histogram[NUMTHREADS];
	int idx = threadIdx.x;
	imMin -= (imMax-imMin)*0.00625;
	imMax += (imMax-imMin)*0.00625;

	histogram[idx] = 1e-10f;
	__syncthreads();
	int repetitions = (imageSize-1) / blockDim.x + 1;
	int idxCurr = idx;
	for(int i = 0; i < repetitions; i++, idxCurr += blockDim.x){
		short localAgreement = agreement[idxCurr];
		float localValue = (float) image[idxCurr];
		int histInPos = (int) ( (float) (histSize-1) * ((localValue-imMin) / (imMax-imMin)) + 0.5f );
		int histPos = idx;
		for(int h = 0; h < histSize; h++){
			__syncthreads();
			histogram[histPos] += (idxCurr < imageSize && localAgreement >= requiredAgreement && histPos == histInPos) ? 1 : 0;
			histPos += (histPos < histSize-1) ? 1: -histPos;	
		}
	}
	__syncthreads();

	//normalize inefficiently
	if(idx==0){
		float sum = 0.0f;
		for(int h = 0; h < histSize; h++)
			sum += histogram[h];
		for(int h = 0; h < histSize; h++)
			histogram[h] /= sum;
	}
	
	__syncthreads();
	if( idx < histSize ) histogramGPU[idx] = histogram[idx];

}

template<class T>
__global__ void kern_PopulateHisto2D(float* histogramGPU, int histSize, short* agreement, T* image, short requiredAgreement, float imMin, float imMax, float sMin, float sMax, int imageSize){
	__shared__ float histogram[NUMTHREADS];
	int idx = threadIdx.x;
	imMin -= (imMax-imMin)*0.00625;
	imMax += (imMax-imMin)*0.00625;

	histogram[idx] = 1e-10f;
	__syncthreads();
	int repetitions = (imageSize-1) / blockDim.x + 1;
	int idxCurr = idx;
	for(int i = 0; i < repetitions; i++, idxCurr += blockDim.x){
		short localAgreement = agreement[idxCurr];
		float localValue1 = (float) image[2*idxCurr];
		float localValue2 = (float) image[2*idxCurr+1];
		int histInPos = (int) ( (float) (histSize-1) * ((localValue1-imMin) / (imMax-imMin)) + 0.5f );
		int histPos = idx;
		bool useIt = (idxCurr < imageSize && localAgreement >= requiredAgreement && localValue2 >= sMin && localValue2 < sMax);
		for(int h = 0; h < NUMTHREADS; h++){
			__syncthreads();
			histogram[histPos] += (useIt && histPos == histInPos) ? 1 : 0;
			histPos += (histPos < NUMTHREADS-1) ? 1: -histPos;	
		}
	}
	__syncthreads();
	histogramGPU[idx] = histogram[idx];

}

template<class T>
__global__ void kern_PopulateOutput(float* histogramGPU, int histSize, float* output, T* image, float imMin, float imMax, int imageSize){
	__shared__ float histogram[NUMTHREADS];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	imMin -= (imMax-imMin)*0.00625;
	imMax += (imMax-imMin)*0.00625;
	if( threadIdx.x < histSize ) histogram[threadIdx.x] = histogramGPU[threadIdx.x];
	__syncthreads();
	
	float localValue = (float) image[idx];
	int histPos = (int) ( (float) (histSize-1) * ((localValue-imMin) / (imMax-imMin)) + 0.5f );
	float histVal = (histPos < histSize && histPos >= 0) ? histogram[histPos] : 1e-10f;
	histVal = (histVal < 1e-10f) ? 1e-10f : histVal;
	histVal = log(histVal) / log(1e-10f);
	if(idx < imageSize) output[idx] = histVal;

}

template<class T>
__global__ void kern_PopulateOutput2D(float* histogramGPU, int histSize, float* output, T* image, float imMin, float imMax, float sMin, float sMax, int imageSize){
	__shared__ float histogram[NUMTHREADS];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	imMin -= (imMax-imMin)*0.00625;
	imMax += (imMax-imMin)*0.00625;
	if( threadIdx.x < NUMTHREADS ) histogram[threadIdx.x] = histogramGPU[threadIdx.x];
	__syncthreads();
	
	float localValue1 = (float) image[2*idx];
	float localValue2 = (float) image[2*idx+1];
	int histPos = (int) ( (float) (histSize-1) * ((localValue1-imMin) / (imMax-imMin)) + 0.5f );
	bool useIt = (localValue2 >= sMin && localValue2 < sMax);
	float histVal = (histPos < histSize && histPos >= 0) ? histogram[histPos] : 1e-10f;
	histVal = (histVal < 1e-10f) ? 1e-10f : histVal;
	histVal = log(histVal) / log(1e-10f);
	float oldHistVal = output[idx];
	histVal = useIt ? histVal: oldHistVal;
	if(idx < imageSize) output[idx] = histVal;

}


template< class T >
void CUDA_ILLT_CalculateHistogramAndTerms(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, T* image, short requiredAgreement, int imageSize, cudaStream_t* stream){
	
	histSize = (histSize < NUMTHREADS) ? histSize : NUMTHREADS;

	T* GPUInputBuffer = 0;
	float* GPUOutputBuffer = 0;
	float* GPUWorkingBuffer = 0;
	cudaMalloc((void**) &GPUInputBuffer, sizeof(T)*imageSize);
	cudaMalloc((void**) &GPUOutputBuffer, sizeof(float)*imageSize);
	cudaMalloc((void**) &GPUWorkingBuffer, sizeof(float)*imageSize);
	cudaMemcpyAsync( GPUInputBuffer, image, sizeof(T)*imageSize, cudaMemcpyHostToDevice, *stream );

	float imMax = 0;
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid = GetGrid(imageSize);
	kern_PopulateWorkingUp<T><<<grid,threads,0,*stream>>>(GPUWorkingBuffer, agreement, GPUInputBuffer, requiredAgreement, imageSize);
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_CalculateMinMax: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
	for(int t = (imageSize-1)/2+1; t > 0; t/=2){
		threads = dim3(NUMTHREADS,1,1);
		grid = GetGrid(t);
		kern_PropogateUp<<<grid,threads,0,*stream>>>(GPUWorkingBuffer, t, imageSize);

		#ifdef DEBUG_VTKCUDA_ILLT
			cudaThreadSynchronize();
			printf( "CUDA_ILLT_CalculateMinMax: " );
			printf( cudaGetErrorString( cudaGetLastError() ) );
			printf( "\n" );
		#endif
	}
	cudaMemcpyAsync( &imMax, GPUWorkingBuffer, sizeof(float), cudaMemcpyDeviceToHost, *stream );
	cudaThreadSynchronize();

	float imMin = 0;
	threads = dim3(NUMTHREADS,1,1);
	grid = GetGrid(imageSize);
	kern_PopulateWorkingDown<T><<<grid,threads,0,*stream>>>(GPUWorkingBuffer, agreement, GPUInputBuffer, requiredAgreement, imageSize);
	for(int t = (imageSize-1)/2+1; t > 0; t/=2){
		threads = dim3(NUMTHREADS,1,1);
		grid = GetGrid(t);
		kern_PropogateDown<<<grid,threads,0,*stream>>>(GPUWorkingBuffer, t, imageSize);

		
		#ifdef DEBUG_VTKCUDA_ILLT
			cudaThreadSynchronize();
			printf( "CUDA_ILLT_CalculateMinMax: " );
			printf( cudaGetErrorString( cudaGetLastError() ) );
			printf( "\n" );
		#endif
	}
	cudaMemcpyAsync( &imMin, GPUWorkingBuffer, sizeof(float), cudaMemcpyDeviceToHost, *stream );
	cudaThreadSynchronize();
	

	threads = dim3(NUMTHREADS,1,1);
	grid = dim3( 1, 1, 1);
	kern_PopulateHisto<T><<<grid,threads,0,*stream>>>(histogramGPU, histSize, agreement, GPUInputBuffer, requiredAgreement, imMax, imMin, imageSize);
	
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_CalculateHistogram: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	grid = GetGrid(imageSize);
	kern_PopulateOutput<T><<<grid,threads,0,*stream>>>(histogramGPU, histSize, GPUOutputBuffer, GPUInputBuffer, imMax, imMin, imageSize);

	cudaMemcpyAsync( outputBuffer, GPUOutputBuffer, sizeof(float)*imageSize, cudaMemcpyDeviceToHost, *stream );
	
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_CalculateTerms: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	cudaFree(GPUOutputBuffer);
	cudaFree(GPUInputBuffer);
	cudaFree(GPUWorkingBuffer);
	cudaFree(histogramGPU);
	cudaFree(agreement);
	
}

template< class T >
void CUDA_ILLT_CalculateHistogramAndTerms2D(float* outputBuffer, float* histogramGPU, int histSize, short* agreement, T* image, short requiredAgreement, int imageSize, cudaStream_t* stream){
	
	histSize = (histSize < NUMTHREADS) ? histSize : NUMTHREADS;

	T* GPUInputBuffer = 0;
	float* GPUOutputBuffer = 0;
	float* GPUWorkingBuffer = 0;
	cudaMalloc((void**) &GPUInputBuffer, 2*sizeof(T)*imageSize);
	cudaMalloc((void**) &GPUOutputBuffer, sizeof(float)*imageSize);
	cudaMalloc((void**) &GPUWorkingBuffer, sizeof(float)*imageSize);
	cudaMemcpyAsync( GPUInputBuffer, image, 2*sizeof(T)*imageSize, cudaMemcpyHostToDevice, *stream );

	float2 imMax = {0.0f, 0.0f};
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid = GetGrid(imageSize);
	kern_PopulateWorkingUp<T><<<grid,threads,0,*stream>>>(GPUWorkingBuffer, agreement, GPUInputBuffer, requiredAgreement, imageSize);
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_CalculateMinMax: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
	int t = 1; while(t/2<imageSize) t+=t;
	for( ; t > 1; t/=2){
		threads = dim3(NUMTHREADS,1,1);
		grid = GetGrid(t);
		kern_PropogateUp<<<grid,threads,0,*stream>>>(GPUWorkingBuffer, t, imageSize);

		#ifdef DEBUG_VTKCUDA_ILLT
			cudaThreadSynchronize();
			printf( "CUDA_ILLT_CalculateMinMax: " );
			printf( cudaGetErrorString( cudaGetLastError() ) );
			printf( "\n" );
		#endif
	}
	cudaMemcpyAsync( &imMax, GPUWorkingBuffer, sizeof(float2), cudaMemcpyDeviceToHost, *stream );
	cudaThreadSynchronize();

	float2 imMin = {0.0f, 0.0f};
	threads = dim3(NUMTHREADS,1,1);
	grid = GetGrid(imageSize);
	kern_PopulateWorkingDown<T><<<grid,threads,0,*stream>>>(GPUWorkingBuffer, agreement, GPUInputBuffer, requiredAgreement, imageSize);
	t = 1; while(t/2<imageSize) t+=t;
	for(; t > 1; t/=2){
		threads = dim3(NUMTHREADS,1,1);
		grid = GetGrid(t);
		kern_PropogateDown<<<grid,threads,0,*stream>>>(GPUWorkingBuffer, t, imageSize);

		#ifdef DEBUG_VTKCUDA_ILLT
			cudaThreadSynchronize();
			printf( "CUDA_ILLT_CalculateMinMax: " );
			printf( cudaGetErrorString( cudaGetLastError() ) );
			printf( "\n" );
		#endif
	}
	cudaMemcpyAsync( &imMin, GPUWorkingBuffer, sizeof(float2), cudaMemcpyDeviceToHost, *stream );
	cudaThreadSynchronize();
	
	//populate unnormalized histogram
	threads = dim3(NUMTHREADS,1,1);
	grid = dim3( 1, 1, 1);
	for(int comp = 0; comp < histSize; comp++){
		float secondMin = imMin.y + (float) comp * (imMax.y-imMin.y) / (float) histSize;
		float secondMax = (comp != histSize-1) ? imMin.y + (float) (comp+1) * (imMax.y-imMin.y) / (float) histSize : FLT_MAX;
		kern_PopulateHisto2D<T><<<grid,threads,0,*stream>>>(histogramGPU+comp*histSize, histSize, agreement, GPUInputBuffer, requiredAgreement, imMax.x, imMin.x, secondMin, secondMax, imageSize);
	}
	
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_PopulateHistogram: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	//normalize histogram
	threads = dim3(NUMTHREADS,1,1);
	grid = dim3( NUMTHREADS, 1, 1);
	float* dev_workingBuffer = 0;
	cudaMalloc( &dev_workingBuffer, histSize*histSize*sizeof(float) );
	CopyBuffers<<<grid, threads, 0, *stream>>>(dev_workingBuffer, histogramGPU, histSize*histSize);
	float sum = 1.0f;
	for(int j = histSize*histSize / 2; j >= histSize; j = j/2){
		dim3 tempGrid( j>histSize ? j/histSize : 1, 1, 1);
		SumOverLargeBuffer<<<tempGrid, threads, 0, *stream>>>(dev_workingBuffer,j,histSize*histSize);
	}
	SumData( histSize, histSize, 1, dev_workingBuffer, stream );
	cudaMemcpyAsync( &sum, dev_workingBuffer, sizeof(float), cudaMemcpyDeviceToHost, *stream );
	cudaStreamSynchronize(*stream);
	cudaFree(dev_workingBuffer);
	TranslateBuffer<<<grid,threads,0,*stream>>>(histogramGPU, 1.0f/sum, 0.0f, histSize*histSize);
	
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_NormalizeHistogram: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
	
	grid = GetGrid(imageSize);;
	for(int comp = 0; comp < histSize; comp++){
		float secondMin = imMin.y + (float) comp * (imMax.y-imMin.y) / (float) histSize;
		float secondMax = (comp != histSize-1) ? imMin.y + (float) (comp+1) * (imMax.y-imMin.y) / (float) histSize : FLT_MAX;
		kern_PopulateOutput2D<T><<<grid,threads,0,*stream>>>(histogramGPU+comp*histSize, histSize, GPUOutputBuffer, GPUInputBuffer, imMax.x, imMin.x, secondMin, secondMax, imageSize);
	}

	cudaMemcpyAsync( outputBuffer, GPUOutputBuffer, sizeof(float)*imageSize, cudaMemcpyDeviceToHost, *stream );
	
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_CalculateTerms: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif

	cudaFree(GPUOutputBuffer);
	cudaFree(GPUInputBuffer);
	cudaFree(GPUWorkingBuffer);
	cudaFree(histogramGPU);
	cudaFree(agreement);
	

}
