#include "vtkCudaKSOMLikelihood.h"
#include "CUDA_KSOMlikelihood.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

vtkStandardNewMacro(vtkCudaKSOMLikelihood);

vtkCudaKSOMLikelihood::vtkCudaKSOMLikelihood(){
	//configure the IO ports
	this->SetNumberOfInputPorts(3);
	this->SetNumberOfInputConnections(0,1);
	this->SetNumberOfInputConnections(1,1);
	this->SetNumberOfInputConnections(2,1);
	this->SetNumberOfOutputPorts(1);

	//initialize conservativeness and scale
	this->Scale = 1.0;
}

vtkCudaKSOMLikelihood::~vtkCudaKSOMLikelihood(){
}

//------------------------------------------------------------
//Commands for vtkCudaObject compatibility

void vtkCudaKSOMLikelihood::Reinitialize(int withData){
	//TODO
}

void vtkCudaKSOMLikelihood::Deinitialize(int withData){
}


//----------------------------------------------------------------------------

void vtkCudaKSOMLikelihood::SetScale(double s){
	if( s != this->Scale && s >= 0.0 ){
		this->Scale = s;
		this->Modified();
	}
}

double vtkCudaKSOMLikelihood::GetScale(){
	return this->Scale;
}

//------------------------------------------------------------

int vtkCudaKSOMLikelihood::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* seededImageInfo = (inputVector[2])->GetInformationObject(0);
	vtkImageData* seededImage = vtkImageData::SafeDownCast(seededImageInfo->Get(vtkDataObject::DATA_OBJECT()));

	vtkInformation* outputGMMInfo = outputVector->GetInformationObject(0);
	vtkImageData* outGMMImage = vtkImageData::SafeDownCast(outputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));

    vtkDataObject::SetPointDataActiveScalarInfo(outputGMMInfo,  VTK_FLOAT, seededImage->GetScalarRange()[1] );
	return 1;
}

int vtkCudaKSOMLikelihood::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* inputDataInfo = (inputVector[0])->GetInformationObject(0);
	vtkImageData* inputDataImage = vtkImageData::SafeDownCast(inputDataInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkInformation* inputGMMInfo = (inputVector[1])->GetInformationObject(0);
	vtkImageData* inputGMMImage = vtkImageData::SafeDownCast(inputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));
	
	vtkInformation* outputGMMInfo = outputVector->GetInformationObject(0);
	vtkImageData* outGMMImage = vtkImageData::SafeDownCast(outputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));

	outputGMMInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),inputGMMImage->GetExtent(),6);
	outputGMMInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inputGMMImage->GetExtent(),6);

	return 1;
}

int vtkCudaKSOMLikelihood::RequestData(vtkInformation *request, 
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
	vtkInformation* outputGMMInfo = outputVector->GetInformationObject(0);
	vtkImageData* outputGMMImage = vtkImageData::SafeDownCast(outputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));
	
	//figure out the extent of the output
	this->info.NumberOfDimensions = inputDataImage->GetNumberOfScalarComponents();
	this->info.NumberOfLabels = seededDataImage->GetScalarRange()[1];
    int updateExtent[6];
    outputGMMInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), updateExtent);
	outputGMMImage->SetScalarTypeToFloat();
	outputGMMImage->SetNumberOfScalarComponents( this->info.NumberOfLabels );
	outputGMMImage->SetExtent(updateExtent);
	outputGMMImage->AllocateScalars();
	
	//get volume information for containers
	inputDataImage->GetDimensions( this->info.VolumeSize );
	outputGMMImage->GetDimensions( this->info.GMMSize );

	//get range for weight normalization
	double* Range = new double[2*(this->info.NumberOfDimensions)];
	for(int i = 0; i < this->info.NumberOfDimensions; i++)
		inputDataImage->GetPointData()->GetScalars()->GetRange(Range+2*i,i);

	//calculate P according tot he Naive model
	int N = this->info.GMMSize[0]*this->info.GMMSize[1];

	//run algorithm on CUDA
	this->ReserveGPU();
	CUDAalgo_applyKSOMLLModel( (float*) inputDataImage->GetScalarPointer(), (float*) inputGMMImage->GetScalarPointer(),
							  (float*) outputGMMImage->GetScalarPointer(),
							  (char*) seededDataImage->GetScalarPointer(), this->info, this->Scale, this->GetStream() );

	return 1;
}