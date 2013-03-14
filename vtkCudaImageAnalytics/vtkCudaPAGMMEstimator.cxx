#include "vtkCudaPAGMMEstimator.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

vtkStandardNewMacro(vtkCudaPAGMMEstimator);

vtkCudaPAGMMEstimator::vtkCudaPAGMMEstimator(){
	//configure the IO ports
	this->SetNumberOfInputPorts(3);
	this->SetNumberOfInputConnections(0,1);
	this->SetNumberOfInputConnections(1,1);
	this->SetNumberOfInputConnections(2,1);
	this->SetNumberOfOutputPorts(2);

	//initialize the weights to 1
	this->WeightNormalization = true;
	for(int i = 0; i < MAX_DIMENSIONALITY; i++){
		this->UnnormalizedWeights[i] = 1.0f;
		this->info.Weights[i] = 1.0f;
	}

	//initialize conservativeness and scale
	this->Q = 0.01;
	this->Scale = 1.0;
}

vtkCudaPAGMMEstimator::~vtkCudaPAGMMEstimator(){
}

//------------------------------------------------------------
//Commands for vtkCudaObject compatibility

void vtkCudaPAGMMEstimator::Reinitialize(int withData){
	//TODO
}

void vtkCudaPAGMMEstimator::Deinitialize(int withData){
}


//----------------------------------------------------------------------------

void vtkCudaPAGMMEstimator::SetWeight(int index, double weight){
	if( index >= 0 && index < MAX_DIMENSIONALITY && weight >= 0.0 )
		this->UnnormalizedWeights[index] = weight;
}

void vtkCudaPAGMMEstimator::SetWeights(const double* weights){
	for(int i = 0; i < MAX_DIMENSIONALITY; i++)
		try{
			this->UnnormalizedWeights[i] = weights[i];
		}catch(...){
			this->UnnormalizedWeights[i] = 1.0;
		}
}

double vtkCudaPAGMMEstimator::GetWeight(int index){
	if( index >= 0 && index < MAX_DIMENSIONALITY )
		return this->UnnormalizedWeights[index];
	return 0.0;
}

double* vtkCudaPAGMMEstimator::GetWeights(){
	return this->UnnormalizedWeights;
}

void vtkCudaPAGMMEstimator::SetWeightNormalization(bool set){
	this->WeightNormalization = set;
}

bool vtkCudaPAGMMEstimator::GetWeightNormalization(){
	return this->WeightNormalization;
}
//------------------------------------------------------------

void vtkCudaPAGMMEstimator::SetConservativeness(double q){
	if( q != this->Q && q >= 0.0 && q <= 1.0 ){
		this->Q = q;
		this->Modified();
	}
}

double vtkCudaPAGMMEstimator::GetConservativeness(){
	return this->Q;
}

void vtkCudaPAGMMEstimator::SetScale(double s){
	if( s != this->Scale && s >= 0.0 ){
		this->Scale = s;
		this->Modified();
	}
}

double vtkCudaPAGMMEstimator::GetScale(){
	return this->Scale;
}

//------------------------------------------------------------

int vtkCudaPAGMMEstimator::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* seededImageInfo = (inputVector[2])->GetInformationObject(0);
	vtkImageData* seededImage = vtkImageData::SafeDownCast(seededImageInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkInformation* outputDataInfo = outputVector->GetInformationObject(0);
	vtkInformation* outputGMMInfo = outputVector->GetInformationObject(1);
	vtkImageData* outDataImage = vtkImageData::SafeDownCast(outputDataInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* outGMMImage = vtkImageData::SafeDownCast(outputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkDataObject::SetPointDataActiveScalarInfo(outputDataInfo, VTK_FLOAT, seededImage->GetScalarRange()[1] );
    vtkDataObject::SetPointDataActiveScalarInfo(outputGMMInfo,  VTK_FLOAT, seededImage->GetScalarRange()[1] );
	return 1;
}

int vtkCudaPAGMMEstimator::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* inputDataInfo = (inputVector[0])->GetInformationObject(0);
	vtkImageData* inputDataImage = vtkImageData::SafeDownCast(inputDataInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkInformation* inputGMMInfo = (inputVector[1])->GetInformationObject(0);
	vtkImageData* inputGMMImage = vtkImageData::SafeDownCast(inputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));
	
	vtkInformation* outputDataInfo = outputVector->GetInformationObject(0);
	vtkImageData* outDataImage = vtkImageData::SafeDownCast(outputDataInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkInformation* outputGMMInfo = outputVector->GetInformationObject(1);
	vtkImageData* outGMMImage = vtkImageData::SafeDownCast(outputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));

	outputGMMInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),inputGMMImage->GetExtent(),6);
	outputGMMInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inputGMMImage->GetExtent(),6);
	outputDataInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),inputDataImage->GetExtent(),6);
	outputDataInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inputDataImage->GetExtent(),6);

	return 1;
}

int vtkCudaPAGMMEstimator::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){
	//collect input data information							
	vtkInformation* inputDataInfo = (inputVector[0])->GetInformationObject(0);
	vtkInformation* inputGMMInfo = (inputVector[1])->GetInformationObject(0);
	vtkInformation* seededDataInfo = (inputVector[2])->GetInformationObject(0);
	vtkImageData* inputDataImage = vtkImageData::SafeDownCast(inputDataInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* inputGMMImage = vtkImageData::SafeDownCast(inputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* seededDataImage = vtkImageData::SafeDownCast(seededDataInfo->Get(vtkDataObject::DATA_OBJECT()));

	//get output data information containers
	vtkInformation* outputDataInfo = outputVector->GetInformationObject(0);
	vtkInformation* outputGMMInfo = outputVector->GetInformationObject(1);
	vtkImageData* outputDataImage = vtkImageData::SafeDownCast(outputDataInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* outputGMMImage = vtkImageData::SafeDownCast(outputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));
	
	//figure out the extent of the output
	this->info.NumberOfDimensions = inputDataImage->GetNumberOfScalarComponents();
	this->info.NumberOfLabels = seededDataImage->GetScalarRange()[1];
    int updateExtent[6];
    outputDataInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), updateExtent);
	outputDataImage->SetScalarTypeToFloat();
	outputDataImage->SetNumberOfScalarComponents( this->info.NumberOfLabels );
	outputDataImage->SetExtent(updateExtent);
	outputDataImage->AllocateScalars();
    outputGMMInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), updateExtent);
	outputGMMImage->SetScalarTypeToFloat();
	outputGMMImage->SetNumberOfScalarComponents( this->info.NumberOfLabels );
	outputGMMImage->SetExtent(updateExtent);
	outputGMMImage->AllocateScalars();
	
	//get volume information for containers
	outputDataImage->GetDimensions( this->info.VolumeSize );
	outputGMMImage->GetDimensions( this->info.GMMSize );

	//get range for weight normalization
	double* Range = new double[2*(this->info.NumberOfDimensions)];
	for(int i = 0; i < this->info.NumberOfDimensions; i++)
		inputDataImage->GetPointData()->GetScalars()->GetRange(Range+2*i,i);

	//update weights
	if( this->WeightNormalization )
		for(int i = 0; i < this->info.NumberOfDimensions; i++)
			this->info.Weights[i] = this->UnnormalizedWeights[i] / ((Range[2*i+1] - Range[2*i] > 0.0) ? (Range[2*i+1] - Range[2*i]) : 1.0);
	else
		for(int i = 0; i < this->info.NumberOfDimensions; i++)
			this->info.Weights[i] = this->UnnormalizedWeights[i];


	//calculate P according tot he Naive model
	int N = this->info.GMMSize[0]*this->info.GMMSize[1];
	float P = (Q > 0.0) ? this->Q / (1.0 - pow(1.0-this->Q,N)) : 1.0 / ((double)N);

	//run algorithm on CUDA
	this->ReserveGPU();
	CUDAalgo_applyPAGMMModel( (float*) inputDataImage->GetScalarPointer(), (float*) inputGMMImage->GetScalarPointer(),
							  (float*) outputDataImage->GetScalarPointer(), (float*) outputGMMImage->GetScalarPointer(),
							  (char*) seededDataImage->GetScalarPointer(), this->info, P, this->Q, this->Scale, this->GetStream() );

	return 1;
}