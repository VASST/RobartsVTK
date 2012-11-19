#include "vtkCudaKohonenReprojector.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

vtkStandardNewMacro(vtkCudaKohonenReprojector);

vtkCudaKohonenReprojector::vtkCudaKohonenReprojector(){
	//configure the input ports
	this->SetNumberOfInputPorts(2);
	this->SetNumberOfInputConnections(0,1);
	this->SetNumberOfInputConnections(1,1);
}

vtkCudaKohonenReprojector::~vtkCudaKohonenReprojector(){
}

//------------------------------------------------------------
//Commands for vtkCudaObject compatibility

void vtkCudaKohonenReprojector::Reinitialize(int withData){
	//TODO
}

void vtkCudaKohonenReprojector::Deinitialize(int withData){
}


//------------------------------------------------------------
int vtkCudaKohonenReprojector::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkInformation* kohonenInfo = (inputVector[1])->GetInformationObject(0);
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* kohonenData = vtkImageData::SafeDownCast(kohonenInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkDataObject::SetPointDataActiveScalarInfo(outputInfo, VTK_FLOAT, kohonenData->GetNumberOfScalarComponents());
	return 1;
}

int vtkCudaKohonenReprojector::RequestUpdateExtent(
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

int vtkCudaKohonenReprojector::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){

	vtkInformation* kohonenInfo = (inputVector[1])->GetInformationObject(0);
	vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkImageData* kohonenData = vtkImageData::SafeDownCast(kohonenInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));

	//update information container
	this->info.NumberOfDimensions = kohonenData->GetNumberOfScalarComponents();
	inData->GetDimensions( this->info.VolumeSize );
	kohonenData->GetDimensions( this->info.KohonenMapSize );

	//figure out the extent of the output
    int updateExtent[6];
    outputInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), updateExtent);
	outData->SetScalarTypeToFloat();
	outData->SetNumberOfScalarComponents(this->info.NumberOfDimensions);
	outData->SetExtent(updateExtent);
	outData->SetWholeExtent(updateExtent);
	outData->AllocateScalars();
	
	//sanity check on the number of input dimensions
	if( inData->GetNumberOfScalarComponents() != 2 ){
		vtkErrorMacro("Input data needs to have two scalar dimensions.");
		return 0;
	}

	//pass it over to the GPU
	this->ReserveGPU();
	CUDAalgo_reprojectKohonenMap( (float*) inData->GetScalarPointer(), (float*) kohonenData->GetScalarPointer(),
							  (float*) outData->GetScalarPointer(), this->info, this->GetStream() );

	return 1;
}