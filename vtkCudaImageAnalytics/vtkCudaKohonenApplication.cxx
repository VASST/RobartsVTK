#include "vtkCudaKohonenApplication.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

vtkStandardNewMacro(vtkCudaKohonenApplication);

vtkCudaKohonenApplication::vtkCudaKohonenApplication(){
	//configure the input ports
	this->SetNumberOfInputPorts(2);
	this->SetNumberOfInputConnections(0,1);
	this->SetNumberOfInputConnections(1,1);

	//initialize the weights to 1
	this->WeightNormalization = true;
	for(int i = 0; i < MAX_DIMENSIONALITY; i++){
		this->UnnormalizedWeights[i] = 1.0f;
		this->info.Weights[i] = 1.0f;
	}
}

vtkCudaKohonenApplication::~vtkCudaKohonenApplication(){
}

//------------------------------------------------------------
//Commands for vtkCudaObject compatibility

void vtkCudaKohonenApplication::Reinitialize(int withData){
	//TODO
}

void vtkCudaKohonenApplication::Deinitialize(int withData){
}


//----------------------------------------------------------------------------

void vtkCudaKohonenApplication::SetWeight(int index, double weight){
	if( index >= 0 && index < MAX_DIMENSIONALITY && weight >= 0.0 )
		this->UnnormalizedWeights[index] = weight;
}

void vtkCudaKohonenApplication::SetWeights(const double* weights){
	for(int i = 0; i < MAX_DIMENSIONALITY; i++)
		try{
			this->UnnormalizedWeights[i] = weights[i];
		}catch(...){
			this->UnnormalizedWeights[i] = 1.0;
		}
}

double vtkCudaKohonenApplication::GetWeight(int index){
	if( index >= 0 && index < MAX_DIMENSIONALITY )
		return this->UnnormalizedWeights[index];
	return 0.0;
}

double* vtkCudaKohonenApplication::GetWeights(){
	return this->UnnormalizedWeights;
}

void vtkCudaKohonenApplication::SetWeightNormalization(bool set){
	this->WeightNormalization = set;
}

bool vtkCudaKohonenApplication::GetWeightNormalization(){
	return this->WeightNormalization;
}

//------------------------------------------------------------
int vtkCudaKohonenApplication::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkDataObject::SetPointDataActiveScalarInfo(outputInfo, VTK_SHORT, 2);
	return 1;
}

int vtkCudaKohonenApplication::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* kohonenInfo = (inputVector[1])->GetInformationObject(0);
	vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkImageData* kohonenData = vtkImageData::SafeDownCast(kohonenInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
	
	kohonenInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),kohonenData->GetExtent(),6);
	kohonenInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),kohonenData->GetExtent(),6);
	outputInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),inData->GetExtent(),6);
	outputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inData->GetExtent(),6);

	return 1;
}

int vtkCudaKohonenApplication::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){

	vtkInformation* kohonenInfo = (inputVector[1])->GetInformationObject(0);
	vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkImageData* kohonenData = vtkImageData::SafeDownCast(kohonenInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
	
	//figure out the extent of the output
    int updateExtent[6];
    outputInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), updateExtent);
	outData->SetScalarTypeToFloat();
	outData->SetNumberOfScalarComponents(2);
	outData->SetExtent(updateExtent);
	outData->SetWholeExtent(updateExtent);
	outData->AllocateScalars();
	
	//update information container
	this->info.NumberOfDimensions = inData->GetNumberOfScalarComponents();
	inData->GetDimensions( this->info.VolumeSize );
	kohonenData->GetDimensions( this->info.KohonenMapSize );
	
	//update weights
	if( this->WeightNormalization ){
		for(int i = 0; i < this->info.NumberOfDimensions; i++){
			double Range[2];
			inData->GetPointData()->GetScalars()->GetRange(Range,i);
			this->info.Weights[i] = this->UnnormalizedWeights[i] / (this->info.NumberOfDimensions*((Range[1] - Range[0] > 0.0) ? (Range[1] - Range[0]) : 1.0));
		}
	}else{
		for(int i = 0; i < this->info.NumberOfDimensions; i++){
			this->info.Weights[i] = this->UnnormalizedWeights[i] / this->info.NumberOfDimensions;
		}
	}

	//pass it over to the GPU
	this->ReserveGPU();
	CUDAalgo_applyKohonenMap( (float*) inData->GetScalarPointer(), (float*) kohonenData->GetScalarPointer(),
							  (short*) outData->GetScalarPointer(), this->info, this->GetStream() );

	return 1;
}