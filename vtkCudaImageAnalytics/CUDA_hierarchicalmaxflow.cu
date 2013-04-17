#include "CUDA_hierarchicalmaxflow.h"
#include "stdio.h"
#include "cuda.h"

#define NUMTHREADS 512

//#define DEBUG_VTKCUDAHMF

void CUDA_GetGPUBuffers( int maxNumber, double maxPercent, float** buffer, int pad, int volSize, int* numberAcquired, double* percentAcquired ){

	size_t freeMemory, totalMemory;
	cudaError_t nErr = cudaSuccess;
	cudaMemGetInfo(&freeMemory, &totalMemory);

	int maxAllowed = (int) ( (((double) totalMemory * maxPercent) - (double)(2*pad)) / (double) (4 * volSize) );
	maxNumber = (maxNumber > maxAllowed) ? maxAllowed : maxNumber;
    //printf("===========================================================\n");
    //printf("Free/Total(kB): %f/%f\n", (float)freeMemory/1024.0f, (float)totalMemory/1024.0f);

	while( maxNumber > 0 ){
		nErr = cudaMalloc((void**) buffer, sizeof(float)*(maxNumber*volSize+2*pad));
		if( nErr == cudaSuccess ) break;
		maxNumber--; 
	}
	
	cudaMemGetInfo(&freeMemory, &totalMemory);
    //printf("===========================================================\n");
    //printf("Free/Total(kB): %f/%f\n", (float)freeMemory/1024.0f, (float)totalMemory/1024.0f);

	*numberAcquired = maxNumber;
	*percentAcquired = (double) sizeof(float)*(maxNumber*volSize+2*pad) / (double) totalMemory;

}

void CUDA_ReturnGPUBuffers(float* buffer){
	cudaFree(buffer);
}


void CUDA_CopyBufferToCPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream){
	cudaMemcpyAsync( CPUBuffer, GPUBuffer, sizeof(float)*size, cudaMemcpyDeviceToHost, *stream );
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_CopyBufferToCPU: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\t%p to %p ", GPUBuffer, CPUBuffer );
		printf( "\n" );
	#endif
}

void CUDA_CopyBufferToGPU(float* GPUBuffer, float* CPUBuffer, int size, cudaStream_t* stream){
	cudaMemcpyAsync( GPUBuffer, CPUBuffer, sizeof(float)*size, cudaMemcpyHostToDevice, *stream );
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_CopyBufferToGPU: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\t%p to %p ", CPUBuffer, GPUBuffer );
		printf( "\n" );
	#endif
}

__global__ void kern_ZeroOutBuffer(float* buffer, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx < size ) buffer[idx] = 0.0f;
}

void CUDA_zeroOutBuffer(float* GPUBuffer, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_ZeroOutBuffer<<<grid,threads,0,*stream>>>(GPUBuffer,size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_zeroOutBuffer: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_SetBufferValue(float* buffer, float value, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx < size ) buffer[idx] = value;
}

void CUDA_SetBufferToValue(float* GPUBuffer, float value, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_SetBufferValue<<<grid,threads,0,*stream>>>(GPUBuffer,value,size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_SetBufferToValue: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_DivideAndStoreBuffer(float* inBuffer, float* outBuffer, float number, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value = inBuffer[idx] * number;
	if( idx < size ) outBuffer[idx] = value;
}

void CUDA_divideAndStoreBuffer(float* inBuffer, float* outBuffer, float number, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_DivideAndStoreBuffer<<<grid,threads,0,*stream>>>(inBuffer,outBuffer,1.0f/number,size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_divideAndStoreBuffer: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_FindSinkPotentialAndStore(float* workingBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float iCC, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value = workingBuffer[idx] + incBuffer[idx] - divBuffer[idx] + labelBuffer[idx] * iCC;
	if( idx < size ) workingBuffer[idx] = value;
}

void CUDA_storeSinkFlowInBuffer(float* workingBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_FindSinkPotentialAndStore<<<grid,threads,0,*stream>>>(workingBuffer,incBuffer,divBuffer,labelBuffer,1.0f/CC,size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_storeSinkFlowInBuffer: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_FindSourcePotentialAndStore(float* workingBuffer, float* sinkBuffer, float* divBuffer, float* labelBuffer, float iCC, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value = workingBuffer[idx] + sinkBuffer[idx] + divBuffer[idx] - labelBuffer[idx] * iCC;
	if( idx < size ) workingBuffer[idx] = value;
}

void CUDA_storeSourceFlowInBuffer(float* workingBuffer, float* sinkBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_FindSourcePotentialAndStore<<<grid,threads,0,*stream>>>(workingBuffer,sinkBuffer,divBuffer,labelBuffer,1.0f/CC,size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_storeSourceFlowInBuffer: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_FindLeafSinkPotential(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float iCC, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value = incBuffer[idx] - divBuffer[idx] + labelBuffer[idx] * iCC;
	if( idx < size ) sinkBuffer[idx] = value;
}

void CUDA_updateLeafSinkFlow(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream ){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_FindLeafSinkPotential<<<grid,threads,0,*stream>>>(sinkBuffer,incBuffer,divBuffer,labelBuffer,1.0f/CC,size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_updateLeafSinkFlow: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_ApplyCapacity(float* sinkBuffer, float* capBuffer, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value = sinkBuffer[idx];
	float cap = capBuffer[idx];
	value = (value < 0.0f) ? 0.0f: value;
	value = (value > cap) ? cap: value;
	if( idx < size ) sinkBuffer[idx] = value;
}

void CUDA_constrainLeafSinkFlow(float* sinkBuffer, float* capBuffer, int size, cudaStream_t* stream ){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_ApplyCapacity<<<grid,threads,0,*stream>>>(sinkBuffer,capBuffer,size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_constrainLeafSinkFlow: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_UpdateLabel(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value = labelBuffer[idx] + CC*(incBuffer[idx] - divBuffer[idx] - sinkBuffer[idx]);
	value = saturate(value);
	if( idx < size ) labelBuffer[idx] = value;
}

void CUDA_updateLabel(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float CC, int size, cudaStream_t* stream ){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_UpdateLabel<<<grid,threads,0,*stream>>>(sinkBuffer,incBuffer,divBuffer,labelBuffer,CC,size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_updateLabel: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_CalcGradStep(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float stepSize, float iCC, int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value = stepSize*(sinkBuffer[idx] + divBuffer[idx] - incBuffer[idx] - labelBuffer[idx] * iCC);
	if( idx < size ) divBuffer[idx] = value;
}

void CUDA_flowGradientStep(float* sinkBuffer, float* incBuffer, float* divBuffer, float* labelBuffer, float stepSize, float CC, int size, cudaStream_t* stream ){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_CalcGradStep<<<grid,threads,0,*stream>>>(sinkBuffer,incBuffer,divBuffer,labelBuffer,stepSize,1.0f/CC,size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_flowGradientStep: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_DescentSpatialFlow(float* allowed, float* flowX, float* flowY, float* flowZ, const int2 dims, const int size){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int3 idxN;
	idxN.y = idx / dims.x;
	idxN.x = idx % dims.x;
	idxN.z = idxN.y / dims.y;
	idxN.y = idxN.y % dims.y;
	float currAllowed = allowed[idx];

	float xAllowed = allowed[idx-1];
	float yAllowed = allowed[idx-dims.x];
	float zAllowed = allowed[idx-dims.x*dims.y];
	
	float newFlowX = flowX[idx] - (currAllowed - xAllowed);
	if( idx < size ) flowX[idx] = idxN.x ? newFlowX : 0.0f;

	float newFlowY = flowY[idx] - (currAllowed - yAllowed);
	if( idx < size ) flowY[idx] = idxN.y ? newFlowY : 0.0f;

	float newFlowZ = flowZ[idx] - (currAllowed - zAllowed);
	if( idx < size ) flowZ[idx] = idxN.z ? newFlowZ : 0.0f;

}

void CUDA_applyStep(float* divBuffer, float* flowX, float* flowY, float* flowZ, int X, int Y, int Z, int size, cudaStream_t* stream ){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	int2 vDims = {X,Y};
	kern_DescentSpatialFlow<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, vDims, size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_applyStep: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_ComputeFlowMagVariSmooth(float* amount, float* flowX, float* flowY, float* flowZ, float* smooth, const float alpha, const int2 dims, const int size){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int2 idxN;
	idxN.y = idx / dims.x;
	idxN.x = idx % dims.x;
	idxN.y = idxN.y % dims.y;

	//compute flow in X
	float AmountUp = flowX[idx]; AmountUp *= AmountUp;
	float AmountDown = flowX[idx+1]; AmountDown = (idxN.x != dims.x-1 ? AmountDown*AmountDown : 0.0f);
	float FlowMag = AmountUp + AmountDown;
	
	//compute flow in Y
	AmountUp = flowY[idx]; AmountUp *= AmountUp;
	AmountDown = flowY[idx+dims.x]; AmountDown = (idxN.y != dims.y-1 ? AmountDown*AmountDown : 0.0f);
	FlowMag += AmountUp + AmountDown;
	
	//compute flow in Z
	AmountUp = flowZ[idx]; AmountUp *= AmountUp;
	AmountDown = flowZ[idx+dims.x*dims.y]; AmountDown = (idx+dims.x*dims.y < size ? AmountDown*AmountDown : 0.0f);
	FlowMag += AmountUp + AmountDown;

	//adjust to be proper
	FlowMag = sqrt( 0.5f * FlowMag );

	//find the constraint on the flow
	float smoothness = alpha * smooth[idx];

	//find the multiplier and output to buffer
	float multiplier = (FlowMag > smoothness) ? smoothness / FlowMag : 1.0f;
	if( idx < size ) amount[idx] = multiplier;

}

__global__ void kern_ComputeFlowMagConstSmooth(float* amount, float* flowX, float* flowY, float* flowZ, const float alpha, const int2 dims, const int size){
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int2 idxN;
	idxN.y = idx / dims.x;
	idxN.x = idx % dims.x;
	idxN.y = idxN.y % dims.y;
	
	//compute flow in X
	float AmountUp = flowX[idx]; AmountUp *= AmountUp;
	float AmountDown = flowX[idx+1]; AmountDown = (idxN.x != dims.x-1 ? AmountDown*AmountDown : 0.0f);
	float FlowMag = AmountUp + AmountDown;
	
	//compute flow in Y
	AmountUp = flowY[idx]; AmountUp *= AmountUp;
	AmountDown = flowY[idx+dims.x]; AmountDown = (idxN.y != dims.y-1 ? AmountDown*AmountDown : 0.0f);
	FlowMag += AmountUp + AmountDown;
	
	//compute flow in Z
	AmountUp = flowZ[idx]; AmountUp *= AmountUp;
	AmountDown = flowZ[idx+dims.x*dims.y]; AmountDown = (idx+dims.x*dims.y < size ? AmountDown*AmountDown : 0.0f);
	FlowMag += AmountUp + AmountDown;

	//adjust to be proper
	FlowMag = sqrt( 0.5f * FlowMag );

	//find the multiplier and output to buffer
	float multiplier = (FlowMag > alpha) ? alpha / FlowMag : 1.0f;
	if( idx < size ) amount[idx] = multiplier;

}

void CUDA_computeFlowMag(float* divBuffer, float* flowX, float* flowY, float* flowZ, float* smoothnessTerm, float smoothnessConstant, int X, int Y, int Z, int size, cudaStream_t* stream ){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	int2 vDims = {X,Y};
	if(smoothnessTerm)
		kern_ComputeFlowMagVariSmooth<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, smoothnessTerm, smoothnessConstant, vDims, size);
	else
		kern_ComputeFlowMagConstSmooth<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, smoothnessConstant, vDims, size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_computeFlowMag: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_Project(float* div, float* flowX, float* flowY, float* flowZ, const int2 dims, const int size){
		
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int3 idxN;
	idxN.y = idx / dims.x;
	idxN.x = idx % dims.x;
	idxN.z = idxN.y / dims.y;
	idxN.y = idxN.y % dims.y;
	
	float currAllowed = div[idx];
	float xAllowed = div[idx-1];
	float yAllowed = div[idx-dims.x];
	float zAllowed = div[idx-dims.x*dims.y];
	
	float newFlowX = flowX[idx] * 0.5f * (currAllowed + xAllowed);
	if( idx < size ) flowX[idx] = idxN.x ? newFlowX : 0.0f;

	float newFlowY = flowY[idx] * 0.5f * (currAllowed + yAllowed);
	if( idx < size ) flowY[idx] = idxN.y ? newFlowY : 0.0f;

	float newFlowZ = flowZ[idx] * 0.5f * (currAllowed + zAllowed);
	if( idx < size ) flowZ[idx] = idxN.z ? newFlowZ : 0.0f;

}

__global__ void kern_Divergence(float* div, float* flowX, float* flowY, float* flowZ, const int2 dims, const int size){
		
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int3 idxN;
	idxN.y = idx / dims.x;
	idxN.x = idx % dims.x;
	idxN.z = idxN.y / dims.y;
	idxN.y = idxN.y % dims.y;

	float xAllowed = flowX[idx+1];
	float yAllowed = flowY[idx+dims.x];
	float zAllowed = flowZ[idx+dims.x*dims.y];
	
	float divergence = flowX[idx]+flowY[idx]+flowZ[idx];
	divergence -= (idxN.x != dims.x-1) ? xAllowed : 0.0f;
	divergence -= (idxN.y != dims.y-1) ? yAllowed : 0.0f;
	divergence -= (idx < size-dims.x*dims.y) ? zAllowed : 0.0f;

	if( idx < size ) div[idx] = divergence;
}

void CUDA_projectOntoSet(float* divBuffer, float* flowX, float* flowY, float* flowZ, int X, int Y, int Z, int size, cudaStream_t* stream ){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	int2 vDims = {X,Y};
	kern_Project<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, vDims, size);
	kern_Divergence<<<grid,threads,0,*stream>>>(divBuffer, flowX, flowY, flowZ, vDims, size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_projectOntoSet: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_Copy(float* dst, float* src, const int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value = src[idx];
	if( idx < size ) dst[idx] = value;
}

void CUDA_CopyBuffer(float* dst, float* src, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_Copy<<<grid,threads,0,*stream>>>(dst, src, size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_CopyBuffer: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_Min(float* dst, float* src, const int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value1 = src[idx];
	float value2 = dst[idx];
	float minVal = (value1 < value2) ? value1 : value2;
	if( idx < size ) dst[idx] = minVal;
}

void CUDA_MinBuffer(float* dst, float* src, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_Min<<<grid,threads,0,*stream>>>(dst, src, size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_MinBuffer: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_Lbl(float* lbl, float* flo, float* cap, const int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value1 = cap[idx];
	float value2 = flo[idx];
	float minVal =  (value2 == value1) ? 1.0f : 0.0f;
	if( idx < size ) lbl[idx] = minVal;
}

void CUDA_LblBuffer(float* lbl, float* flo, float* cap, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_Lbl<<<grid,threads,0,*stream>>>(lbl, flo, cap, size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_LblBuffer: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_Sum(float* dst, float* src, const int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value1 = src[idx];
	float value2 = dst[idx];
	float minVal =  value1 + value2;
	if( idx < size ) dst[idx] = minVal;
}

void CUDA_SumBuffer(float* dst, float* src, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_Sum<<<grid,threads,0,*stream>>>(dst, src, size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_SumBuffer: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}

__global__ void kern_Div(float* dst, float* src, const int size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	float value1 = src[idx];
	float value2 = dst[idx];
	float minVal =  value2 / value1;
	if( idx < size ) dst[idx] = minVal;
}

void CUDA_DivBuffer(float* dst, float* src, int size, cudaStream_t* stream){
	dim3 threads(NUMTHREADS,1,1);
	dim3 grid( (size-1)/NUMTHREADS + 1, 1, 1);
	kern_Div<<<grid,threads,0,*stream>>>(dst, src, size);
	#ifdef DEBUG_VTKCUDAHMF
		cudaThreadSynchronize();
		printf( "CUDA_DivBuffer: " );
		printf( cudaGetErrorString( cudaGetLastError() ) );
		printf( "\n" );
	#endif
}