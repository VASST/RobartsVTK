#include "vtkImageMultiStatistics.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"
#include "vtkExecutive.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

//------------------------------------------------------------------------------
vtkImageMultiStatistics* vtkImageMultiStatistics::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkImageMultiStatistics");
  if(ret)
    {
    return (vtkImageMultiStatistics*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkImageMultiStatistics;
}

//----------------------------------------------------------------------------
vtkImageMultiStatistics::vtkImageMultiStatistics()
{
  //Initialize internal stucture
  this->NumberOfComponents = 0;
  this->AverageMagnitude = 0;
  this->Covariance = 0;
  this->JointEntropy = 0;
  this->Count = 0;
  this->NumberOfBins = 100;

  //configure the input ports
  this->SetNumberOfInputPorts(1);
}

vtkImageMultiStatistics::~vtkImageMultiStatistics()
{
	if( this->AverageMagnitude ) delete [] this->AverageMagnitude;
	if( this->Covariance ) delete [] this->Covariance;
	if( this->JointEntropy ) delete [] this->JointEntropy;
}

//------------------------------------------------------------
int vtkImageMultiStatistics::FillInputPortInformation(int i, vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  return this->Superclass::FillInputPortInformation(i,info);
}

void vtkImageMultiStatistics::SetInput(int idx, vtkImageData *input)
{
  // Ask the superclass to connect the input.
  this->SetNthInputConnection(0, idx, (input ? input->GetProducerPort() : 0));
}

vtkImageData *vtkImageMultiStatistics::GetInput(int idx)
{
  if (this->GetNumberOfInputConnections(0) <= idx)
    {
    return 0;
    }
  return vtkImageData::SafeDownCast(
    this->GetExecutive()->GetInputData(0, idx));
}

//----------------------------------------------------------------------------
double vtkImageMultiStatistics::GetAverageMagnitude(int component){
	this->Update(); 
	if( component < this->NumberOfComponents )
		return this->AverageMagnitude[component];
	else{
		vtkErrorMacro(<<"Cannot select component. Component not provided in input.");
		return 0.0;
	}
}

double vtkImageMultiStatistics::GetStandardDeviation(int component) {
	this->Update();
	if( component < this->NumberOfComponents )
		return sqrt(this->Covariance[component*this->NumberOfComponents+component]);
	else{
		vtkErrorMacro(<<"Cannot select component. Component not provided in input.");
		return 0.0;
	}
}

double vtkImageMultiStatistics::GetCovariance(int component1, int component2){
	this->Update();
	if( component1 < this->NumberOfComponents && component2 < this->NumberOfComponents )
		return this->Covariance[component1*this->NumberOfComponents+component2];
	else{
		vtkErrorMacro(<<"Cannot select components. At least one component is not provided in input.");
		return 0.0;
	}
}

double vtkImageMultiStatistics::GetSingleEntropy(int component){
	this->Update();
	if( component < this->NumberOfComponents )
		return sqrt(this->JointEntropy[component*this->NumberOfComponents+component]);
	else{
		vtkErrorMacro(<<"Cannot select component. Component not provided in input.");
		return 0.0;
	}
}

double vtkImageMultiStatistics::GetJointEntropy(int component1, int component2){
	this->Update();
	if( component1 < this->NumberOfComponents && component2 < this->NumberOfComponents )
		return this->JointEntropy[component1*this->NumberOfComponents+component2];
	else{
		vtkErrorMacro(<<"Cannot select components. At least one component is not provided in input.");
		return 0.0;
	}
}

long int vtkImageMultiStatistics::GetCount() {
	this->Update();
	return this->Count;
}

//----------------------------------------------------------------------------
void vtkImageMultiStatistics::SetEntropyResolution(int bins){
	if( bins < 1 ){
		vtkErrorMacro(<<"Invalid resolution.");
		return;
	}

	//if we have changed the resolution, mark as modified
	if( bins != this->NumberOfBins ){
		this->NumberOfBins = bins;
		this->Modified();
	}

}

int vtkImageMultiStatistics::GetEntropyResolution(){
	return this->NumberOfBins;
}

//----------------------------------------------------------------------------
template <class T, class S>
static void vtkImageMultiStatisticsExecuteWithMask(vtkImageMultiStatistics *self, 
					  T *inPtr,
					  vtkImageData *inData,
					  S *maskPtr,
					  vtkImageData *maskData,
					  long double **AverageMagnitude,
					  long double **Covariance,
					  long double **JointEntropy,
					  long int *Count, int N){
						  
	int inIdxX, inIdxY, inIdxZ;
	vtkIdType inIncX, inIncY, inIncZ;
	vtkIdType maskIncX, maskIncY, maskIncZ;
	int inIdxN, inIdxN2;
	int wholeInExt[6];
	int wholeMaskExt[6];
	
	T *curPtr;
	S *curMaskPtr;

	//use the output holders for temporary storage for statistical information
	long double* sum = *AverageMagnitude;
	long double* sum_squared = *Covariance;
	for(int i = 0; i < N; i++){
		sum[i] = 0.0;
		for( int j = 0; j < N; j++)
			sum_squared[i*N+j] = 0.0;
	}

	//Create temporary storage for information theoretic information
	int Resolution = self->GetEntropyResolution();
	long double* Maximum = new long double[N];
	long double* Minimum = new long double[N];
	for( int i = 0; i < N; i++ ){
		Minimum[i] = inData->GetPointData()->GetScalars()->GetRange(i)[0];
		Maximum[i] = inData->GetPointData()->GetScalars()->GetRange(i)[1];
	}
	unsigned int* DoubleHistogram = new unsigned int [(N * N - N) * Resolution * Resolution / 2];
	memset( DoubleHistogram, 0, sizeof(unsigned int) * (N * N - N) * Resolution * Resolution / 2 );
	unsigned int* SingleHistogram = new unsigned int [N * Resolution];
	memset( SingleHistogram, 0, sizeof(unsigned int) * N * Resolution );

	// Get increments to march through data 
	inData->GetWholeExtent(wholeInExt);
	inData->GetContinuousIncrements(wholeInExt, inIncX, inIncY, inIncZ);
	maskData->GetWholeExtent(wholeMaskExt);
	maskData->GetContinuousIncrements(wholeMaskExt, maskIncX, maskIncY, maskIncZ);
	*Count = 0;
	
	// Loop through input dataset once to gather stats
	curPtr = inPtr;
	curMaskPtr = maskPtr;
	for (inIdxZ = wholeInExt[4]; inIdxZ <= wholeInExt[5]; inIdxZ++) {
		for (inIdxY = wholeInExt[2]; !self->AbortExecute && inIdxY <= wholeInExt[3]; inIdxY++) {
			for (inIdxX = wholeInExt[0]; inIdxX <= wholeInExt[1]; inIdxX++) {
				
				//only process if we are in the non-zero part of the mask
				if( (double) *curMaskPtr != 0.0 ){
					int DimCount = 0;
					for( inIdxN = 0; inIdxN < N; inIdxN++ ){

						//do univariate 
						sum[inIdxN] += curPtr[inIdxN];
						sum_squared[inIdxN*N+inIdxN] += curPtr[inIdxN] * curPtr[inIdxN];
						int histIdx = (int)((double) Resolution * ((double) curPtr[inIdxN]-Minimum[inIdxN]) / (Maximum[inIdxN]-Minimum[inIdxN]));
						if( histIdx >= Resolution )
							histIdx = Resolution - 1;
						SingleHistogram[inIdxN*Resolution+histIdx]++;

						//do bivariate
						for( inIdxN2 = inIdxN+1; inIdxN2 < N; inIdxN2++ ){
							sum_squared[inIdxN*N+inIdxN2] += curPtr[inIdxN] * curPtr[inIdxN2];
							sum_squared[inIdxN+inIdxN2*N] += curPtr[inIdxN] * curPtr[inIdxN2];
							int hist2Idx = (int)((double) Resolution * ((double) curPtr[inIdxN2]-Minimum[inIdxN2]) / (Maximum[inIdxN2]-Minimum[inIdxN2]));
							if( hist2Idx >= Resolution ) hist2Idx = Resolution - 1;
							DoubleHistogram[DimCount*Resolution*Resolution + histIdx*Resolution + hist2Idx]++;
							DimCount++;
						}
					}
					(*Count)++;
				}


				curPtr += N;
				curMaskPtr++;
			}
			curPtr += inIncY;
			curMaskPtr += maskIncY;
		}
		curPtr += inIncZ;
		curMaskPtr += maskIncZ;
	}

	//compute the means
	for(int i = 0; i < N; i++){
		long double SafeSum = sum[i];
		(*AverageMagnitude)[i] = SafeSum / (double)*Count;
	}

	//compute the covariances
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			long double SafeSquareSum = sum_squared[i*N+j];
			(*Covariance)[i*N+j] = (SafeSquareSum / (long double)*Count) - (*AverageMagnitude)[i] * (*AverageMagnitude)[j];
		}
	}

	//compute the entropies
	int DimCount = 0;
	for(int i = 0; i < N; i++){

		//compute single entropy
		(*JointEntropy)[i*N+i] = 0.0;
		for(int r = 0; r < Resolution; r++){
			if( SingleHistogram[i*Resolution+r] )
				(*JointEntropy)[i*N+i] -=        ((long double) SingleHistogram[i*Resolution+r] / (long double) *Count)
											* log((long double) SingleHistogram[i*Resolution+r] / (long double) *Count) / log(2.0);
		}

		for(int j = i+1; j < N; j++){
			int totCount = 0;
			(*JointEntropy)[i*N+j] = 0.0;
			(*JointEntropy)[j*N+i] = 0.0;
			for(int r = 0; r < Resolution*Resolution; r++){
				if(	DoubleHistogram[DimCount*Resolution*Resolution+r] ){
					(*JointEntropy)[i*N+j] -=        ((long double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (long double) *Count)
												* log((long double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (long double) *Count) / log(2.0);
					(*JointEntropy)[j*N+i] -=        ((long double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (long double) *Count)
												* log((long double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (long double) *Count) / log(2.0);
					totCount += DoubleHistogram[DimCount*Resolution*Resolution+r];
				}
			}

			DimCount++;
		}
	}

	//release storage
	delete [] Maximum;
	delete [] Minimum;
	delete [] SingleHistogram;
	delete [] DoubleHistogram;

}

template <class T>
static void vtkImageMultiStatisticsExecuteWithMaskStart(vtkImageMultiStatistics *self, 
					  T *inPtr,
					  vtkImageData *inData,
					  vtkImageData *maskData,
					  long double **AverageMagnitude,
					  long double **Covariance,
					  long double **JointEntropy,
					  long int *Count, int N){

	void* maskPtr = maskData->GetScalarPointerForExtent( maskData->GetWholeExtent() );
	switch (maskData->GetScalarType()) {
		vtkTemplateMacro(vtkImageMultiStatisticsExecuteWithMask(
			self, inPtr, inData,
			(VTK_TT *) maskPtr, maskData,
			AverageMagnitude, Covariance, JointEntropy,
			Count, N));
		default:
			vtkErrorWithObjectMacro(self,<< "Update: Unknown ScalarType");
			return;
	}
}


template <class T>
static void vtkImageMultiStatisticsExecuteWithoutMask(vtkImageMultiStatistics *self, 
					  T *inPtr,
					  vtkImageData *inData,
					  long double **AverageMagnitude,
					  long double **Covariance,
					  long double **JointEntropy,
					  long int *Count, int N)
{
	int inIdxX, inIdxY, inIdxZ;
	vtkIdType inIncX, inIncY, inIncZ;
	int inIdxN, inIdxN2;
	int wholeInExt[6];
	T *curPtr;

	//use the output holders for temporary storage for statistical information
	long double* sum = *AverageMagnitude;
	long double* sum_squared = *Covariance;
	for(int i = 0; i < N; i++){
		sum[i] = 0.0;
		for( int j = 0; j < N; j++)
			sum_squared[i*N+j] = 0.0;
	}

	//Create temporary storage for information theoretic information
	int Resolution = self->GetEntropyResolution();
	long double* Maximum = new long double[N];
	long double* Minimum = new long double[N];
	for( int i = 0; i < N; i++ ){
		Minimum[i] = inData->GetPointData()->GetScalars()->GetRange(i)[0];
		Maximum[i] = inData->GetPointData()->GetScalars()->GetRange(i)[1];
	}
	unsigned int* DoubleHistogram = new unsigned int [(N * N - N) * Resolution * Resolution / 2];
	memset( DoubleHistogram, 0, sizeof(unsigned int) * (N * N - N) * Resolution * Resolution / 2 );
	unsigned int* SingleHistogram = new unsigned int [N * Resolution];
	memset( SingleHistogram, 0, sizeof(unsigned int) * N * Resolution );

	// Get increments to march through data 
	inData->GetWholeExtent(wholeInExt);
	inData->GetContinuousIncrements(wholeInExt, inIncX, inIncY, inIncZ);
	*Count = 0;
	
	// Loop through input dataset once to gather stats
	curPtr = inPtr;
	for (inIdxZ = wholeInExt[4]; inIdxZ <= wholeInExt[5]; inIdxZ++) {
		for (inIdxY = wholeInExt[2]; !self->AbortExecute && inIdxY <= wholeInExt[3]; inIdxY++) {
			for (inIdxX = wholeInExt[0]; inIdxX <= wholeInExt[1]; inIdxX++) {
				int DimCount = 0;
				for( inIdxN = 0; inIdxN < N; inIdxN++ ){

					//do univariate 
					sum[inIdxN] += curPtr[inIdxN];
					sum_squared[inIdxN*N+inIdxN] += curPtr[inIdxN] * curPtr[inIdxN];
					int histIdx = (int)((double) Resolution * ((double) curPtr[inIdxN]-Minimum[inIdxN]) / (Maximum[inIdxN]-Minimum[inIdxN]));
					if( histIdx >= Resolution )
						histIdx = Resolution - 1;
					SingleHistogram[inIdxN*Resolution+histIdx]++;

					//do bivariate
					for( inIdxN2 = inIdxN+1; inIdxN2 < N; inIdxN2++ ){
						sum_squared[inIdxN*N+inIdxN2] += curPtr[inIdxN] * curPtr[inIdxN2];
						sum_squared[inIdxN+inIdxN2*N] += curPtr[inIdxN] * curPtr[inIdxN2];
						int hist2Idx = (int)((double) Resolution * ((double) curPtr[inIdxN2]-Minimum[inIdxN2]) / (Maximum[inIdxN2]-Minimum[inIdxN2]));
						if( hist2Idx >= Resolution ) hist2Idx = Resolution - 1;
						DoubleHistogram[DimCount*Resolution*Resolution + histIdx*Resolution + hist2Idx]++;
						DimCount++;
					}
				}
				curPtr += N;
				(*Count)++;
			}
			curPtr += inIncY;
		}
		curPtr += inIncZ;
	}

	//compute the means
	for(int i = 0; i < N; i++){
		long double SafeSum = sum[i];
		(*AverageMagnitude)[i] = SafeSum / (double)*Count;
	}

	//compute the covariances
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			long double SafeSquareSum = sum_squared[i*N+j];
			(*Covariance)[i*N+j] = (SafeSquareSum / (long double)*Count) - (*AverageMagnitude)[i] * (*AverageMagnitude)[j];
		}
	}

	//compute the entropies
	int DimCount = 0;
	for(int i = 0; i < N; i++){

		//compute single entropy
		(*JointEntropy)[i*N+i] = 0.0;
		for(int r = 0; r < Resolution; r++){
			if( SingleHistogram[i*Resolution+r] )
				(*JointEntropy)[i*N+i] -=        ((long double) SingleHistogram[i*Resolution+r] / (long double) *Count)
											* log((long double) SingleHistogram[i*Resolution+r] / (long double) *Count) / log(2.0);
		}

		for(int j = i+1; j < N; j++){
			int totCount = 0;
			(*JointEntropy)[i*N+j] = 0.0;
			(*JointEntropy)[j*N+i] = 0.0;
			for(int r = 0; r < Resolution*Resolution; r++){
				if(	DoubleHistogram[DimCount*Resolution*Resolution+r] ){
					(*JointEntropy)[i*N+j] -=        ((long double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (long double) *Count)
												* log((long double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (long double) *Count) / log(2.0);
					(*JointEntropy)[j*N+i] -=        ((long double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (long double) *Count)
												* log((long double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (long double) *Count) / log(2.0);
					totCount += DoubleHistogram[DimCount*Resolution*Resolution+r];
				}
			}

			DimCount++;
		}
	}

	//release storage
	delete [] Maximum;
	delete [] Minimum;
	delete [] SingleHistogram;
	delete [] DoubleHistogram;

}



// Description:
// Make sure input is available then call the templated execute method to
// deal with the particular data type.
void vtkImageMultiStatistics::Update() {

	vtkImageData *input = this->GetInput(0);
	void *inPtr;
	int wholeInExt[6];

	// make sure input is available
	if (!input) {
		vtkErrorMacro(<< "No input...can't execute!");
		return;
	}

	input->Update();
	input->GetWholeExtent(wholeInExt);
	inPtr = input->GetScalarPointerForExtent(wholeInExt);

	// get the number of input components
	int oldNumberOfComponents = this->NumberOfComponents;
	this->NumberOfComponents = input->GetNumberOfScalarComponents();
	if (this->NumberOfComponents < 1){
		vtkErrorMacro(<< "Input must have at least 1 scalar component");
		return;
	}
  
	//fetch and check the mask data
	vtkImageData *mask = this->GetInput(1);
	if( mask ){
		int MaskExtent[6];
		mask->GetWholeExtent( MaskExtent );
		if( MaskExtent[0] != wholeInExt[0] || MaskExtent[1] != wholeInExt[1] || MaskExtent[2] != wholeInExt[2] ||
			MaskExtent[3] != wholeInExt[3] || MaskExtent[4] != wholeInExt[4] || MaskExtent[5] != wholeInExt[5] ){
			vtkErrorMacro(<< "Mask is not the same extent as the input.");
			mask = 0;
		}
		if( mask && mask->GetNumberOfScalarComponents() != 1 ){
			vtkErrorMacro(<< "Mask can only have 1 scalar component.");
			mask = 0;
		}
	}

	//update the size of the information holders
	if( this->NumberOfComponents != oldNumberOfComponents ){
		if( this->AverageMagnitude ) delete [] this->AverageMagnitude;
		this->AverageMagnitude = new long double[this->NumberOfComponents];
		if( this->Covariance ) delete [] this->Covariance;
		this->Covariance = new long double[this->NumberOfComponents*this->NumberOfComponents];
		if( this->JointEntropy ) delete [] this->JointEntropy;
		this->JointEntropy = new long double[this->NumberOfComponents*this->NumberOfComponents];
	}

	if (input->GetMTime() > this->ExecuteTime || this->GetMTime() > this->ExecuteTime ){
		this->InvokeEvent(vtkCommand::StartEvent, NULL);

		// reset Abort flag
		this->AbortExecute = 0;
		this->Progress = 0.0;

		//if there is no mask, run the regular version
		if( !mask )
			switch (input->GetScalarType()) {
				vtkTemplateMacro(vtkImageMultiStatisticsExecuteWithoutMask(
					this, (VTK_TT *) (inPtr), input,
					&(this->AverageMagnitude),
					&(this->Covariance),
					&(this->JointEntropy),
					&(this->Count), this->NumberOfComponents));
				default:
					vtkErrorMacro(<< "Update: Unknown ScalarType");
					return;
			}
		else
			switch (input->GetScalarType()) {
				vtkTemplateMacro(vtkImageMultiStatisticsExecuteWithMaskStart(
					this, (VTK_TT *) (inPtr), input, mask,
					&(this->AverageMagnitude),
					&(this->Covariance),
					&(this->JointEntropy),
					&(this->Count), this->NumberOfComponents));
				default:
					vtkErrorMacro(<< "Update: Unknown ScalarType");
					return;
			}

		this->ExecuteTime.Modified();
		if (!this->AbortExecute) {
			this->UpdateProgress(1.0);
		}
		this->InvokeEvent(vtkCommand::EndEvent, NULL);
	}
	if (input->ShouldIReleaseData()) {
		input->ReleaseData();
	}
}





void vtkImageMultiStatistics::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkProcessObject::PrintSelf(os,indent);

  if (!this->GetInput())
    {
    return;
    }
  
  os << indent << "Components: " << this->NumberOfComponents << "\n";
  os << indent << "Count: " << this->Count << "\n";
  for( int i = 0; i < this->NumberOfComponents; i++)
	  os << indent << "AverageMagnitude (" << i << "): " << this->AverageMagnitude[i] << "\n";
  
  for( int i = 0; i < this->NumberOfComponents; i++)
	for( int j = 0; j < this->NumberOfComponents; j++)
	  os << indent << "Covariance: (" << i << "," << j << "): " << this->Covariance[i*this->NumberOfComponents+j] << "\n";

  for( int i = 0; i < this->NumberOfComponents; i++)
	  os << indent << "StandardDeviation: (" << i << "): " << sqrt(this->Covariance[i*this->NumberOfComponents+i]) << "\n";
  
  for( int i = 0; i < this->NumberOfComponents; i++)
	for( int j = 0; j < this->NumberOfComponents; j++)
	  os << indent << "Entropy: (" << i << "," << j << "): " << this->JointEntropy[i*this->NumberOfComponents+j] << "\n";

}

