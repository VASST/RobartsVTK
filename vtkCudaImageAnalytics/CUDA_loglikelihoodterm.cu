#include "CUDA_loglikelihoodterm.h"
#include "stdio.h"
#include "cuda.h"
#include "float.h"
#include "limits.h"

#define NUMTHREADS 512

//#define DEBUG_VTKCUDA_ILLT

template<class T>
__global__ void kern_IncrementBuffer(T* labelBuffer, T desiredLabel, short* agreement, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	short newAgreement = agreement[idx];
	T labelValue = labelBuffer[idx];
	newAgreement += (labelValue == desiredLabel) ? 1 : 0;
	if( idx < size ) agreement[idx] = newAgreement;
}

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
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_IncrementBuffer<T><<<grid,threads,0,*stream>>>(GPUBuffer, desiredValue, agreement, size);
	cudaFree(GPUBuffer);

	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_IncrementInformation: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

template<class T>
__global__ void kern_ZeroOutBuffer(T* buffer, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx < size ) buffer[idx] = (T) 0;
}

void CUDA_ILLT_GetRelevantBuffers(short** agreement, float** output, int size, cudaStream_t* stream){
	cudaMalloc((void**) agreement, sizeof(short)*size);
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_ZeroOutBuffer<short><<<grid,threads,0,*stream>>>(*agreement,size);
	cudaMalloc((void**) output, sizeof(float)*size);

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
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	
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

template void CUDA_ILLT_CalculateHistogramAndTerms<double>(float* outputBuffer, float* histogramGPU, short* agreement, double* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<long>(float* outputBuffer, float* histogramGPU, short* agreement, long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned long>(float* outputBuffer, float* histogramGPU, short* agreement, unsigned long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<long long>(float* outputBuffer, float* histogramGPU, short* agreement, long long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned long long>(float* outputBuffer, float* histogramGPU, short* agreement, unsigned long long* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<int>(float* outputBuffer, float* histogramGPU, short* agreement, int* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned int>(float* outputBuffer, float* histogramGPU, short* agreement, unsigned int* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<short>(float* outputBuffer, float* histogramGPU, short* agreement, short* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned short>(float* outputBuffer, float* histogramGPU, short* agreement, unsigned short* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<char>(float* outputBuffer, float* histogramGPU, short* agreement, char* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<signed char>(float* outputBuffer, float* histogramGPU, short* agreement, signed char* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<unsigned char>(float* outputBuffer, float* histogramGPU, short* agreement, unsigned char* image,  short requiredAgreement, int imageSize, cudaStream_t* stream);
template void CUDA_ILLT_CalculateHistogramAndTerms<float>(float* outputBuffer, float* histogramGPU, short* agreement, float* image,	 short requiredAgreement, int imageSize, cudaStream_t* stream);

template<class T>
__global__ void kern_PopulateWorkingUp(float* working, short* agreement, T* image, short requiredAgreement, int imageSize){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float inputValue = (float) image[idx];
	short lAgreement = agreement[idx];
	float outputValue = (lAgreement < requiredAgreement) ? FLT_MIN: inputValue;
	if(idx < imageSize) working[idx] = outputValue;
}

template<class T>
__global__ void kern_PopulateWorkingDown(float* working, short* agreement, T* image, short requiredAgreement, int imageSize){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float inputValue = (float) image[idx];
	short lAgreement = agreement[idx];
	float outputValue = (lAgreement < requiredAgreement) ? FLT_MAX: inputValue;
	if(idx < imageSize) working[idx] = outputValue;
}

__global__ void kern_PropogateUp(float* working, int span, int imageSize){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float inputValue1 = working[idx];
	float inputValue2 = working[idx+span];
	float outputVal = (inputValue1 > inputValue2) ? inputValue1: inputValue2;
	if(idx+span < imageSize) working[idx] = outputVal;
}

__global__ void kern_PropogateDown(float* working, int span, int imageSize){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float inputValue1 = working[idx];
	float inputValue2 = working[idx+span];
	float outputVal = (inputValue1 < inputValue2) ? inputValue1: inputValue2;
	if(idx+span < imageSize) working[idx] = outputVal;
}

template<class T>
__global__ void kern_PopulateHisto(float* histogramGPU, short* agreement, T* image, short requiredAgreement, float imMin, float imMax, int imageSize){
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
		int histInPos = (int) ( (float) (NUMTHREADS-1) * ((localValue-imMin) / (imMax-imMin)) + 0.5f );
		int histPos = idx;
		for(int h = 0; h < NUMTHREADS; h++){
			__syncthreads();
			histogram[histPos] += (idxCurr < imageSize && localAgreement >= requiredAgreement && histPos == histInPos) ? 1 : 0;
			histPos += (histPos < NUMTHREADS-1) ? 1: -histPos;	
		}
	}
	__syncthreads();

	//normalize inefficiently
	if(idx==0){
		float sum = 0.0f;
		for(int h = 0; h < NUMTHREADS; h++)
			sum += histogram[h];
		for(int h = 0; h < NUMTHREADS; h++)
			histogram[h] /= sum;
	}
	
	__syncthreads();
	histogramGPU[idx] = histogram[idx];

}

template<class T>
__global__ void kern_PopulateOutput(float* histogramGPU, float* output, T* image, float imMin, float imMax, int imageSize){
	__shared__ float histogram[NUMTHREADS];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	imMin -= (imMax-imMin)*0.00625;
	imMax += (imMax-imMin)*0.00625;
	if( threadIdx.x < NUMTHREADS ) histogram[threadIdx.x] = histogramGPU[threadIdx.x];
	__syncthreads();
	
	float localValue = (float) image[idx];
	int histPos = (int) ( (float) (NUMTHREADS-1) * ((localValue-imMin) / (imMax-imMin)) + 0.5f );
	float histVal = (histPos < NUMTHREADS && histPos >= 0) ? histogram[histPos] : 1e-10f;
	histVal = (histVal < 1e-10f) ? 1e-10f : histVal;
	histVal = log(histVal) / log(1e-10f);
	if(idx < imageSize) output[idx] = histVal;

}

template< class T >
void CUDA_ILLT_CalculateHistogramAndTerms(float* outputBuffer, float* histogramGPU, short* agreement, T* image, short requiredAgreement, int imageSize, cudaStream_t* stream){
	
	T* GPUInputBuffer = 0;
	float* GPUOutputBuffer = 0;
	float* GPUWorkingBuffer = 0;
	cudaMalloc((void**) &GPUInputBuffer, sizeof(T)*imageSize);
	cudaMalloc((void**) &GPUOutputBuffer, sizeof(float)*imageSize);
	cudaMalloc((void**) &GPUWorkingBuffer, sizeof(float)*imageSize);
	cudaMemcpyAsync( GPUInputBuffer, image, sizeof(T)*imageSize, cudaMemcpyHostToDevice, *stream );

	float imMax = 0;
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (imageSize-1)/NUMTHREADS+1, 1, 1);
	kern_PopulateWorkingUp<T><<<grid,threads,0,*stream>>>(GPUWorkingBuffer, agreement, GPUInputBuffer, requiredAgreement, imageSize);
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_CalculateMinMax: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
	for(int t = (imageSize-1)/2+1; t > 0; t/=2){
		threads = dim3(NUMTHREADS,1,1);
		grid = dim3( (t-1)/NUMTHREADS+1, 1, 1);
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
	grid = dim3( (imageSize-1)/NUMTHREADS+1, 1, 1);
	kern_PopulateWorkingDown<T><<<grid,threads,0,*stream>>>(GPUWorkingBuffer, agreement, GPUInputBuffer, requiredAgreement, imageSize);
	for(int t = (imageSize-1)/2+1; t > 0; t/=2){
		threads = dim3(NUMTHREADS,1,1);
		grid = dim3( (t-1)/NUMTHREADS+1, 1, 1);
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
	kern_PopulateHisto<T><<<grid,threads,0,*stream>>>(histogramGPU, agreement, GPUInputBuffer, requiredAgreement, imMax, imMin, imageSize);
	
	#ifdef DEBUG_VTKCUDA_ILLT
		cudaThreadSynchronize();
		printf( "CUDA_ILLT_CalculateHistogram: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif


	grid = dim3( (imageSize-1)/NUMTHREADS+1, 1, 1);
	kern_PopulateOutput<T><<<grid,threads,0,*stream>>>(histogramGPU, GPUOutputBuffer, GPUInputBuffer, imMax, imMin, imageSize);

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
	

}
