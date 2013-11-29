#include "vtkCudaKSOMProbability.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

vtkStandardNewMacro(vtkCudaKSOMProbability);

vtkCudaKSOMProbability::vtkCudaKSOMProbability(){
	//configure the input ports
	this->SetNumberOfInputPorts(3);
	this->SetNumberOfInputConnections(0,1);
	this->SetNumberOfInputConnections(1,1);
	this->SetNumberOfOutputPorts(1);

	//initialize the scale to 1
	this->Scale = 1.0;
	this->Entropy = false;
}

vtkCudaKSOMProbability::~vtkCudaKSOMProbability(){
}

void vtkCudaKSOMProbability::SetImageInput(vtkImageData* in){
	this->SetInput(0,in);
}

void vtkCudaKSOMProbability::SetMapInput(vtkImageData* in){
	this->SetInput(1,in);
}

void vtkCudaKSOMProbability::SetProbabilityInput(vtkImageData* in, int index){
	this->SetNthInputConnection( 2, index,  in->GetProducerPort() );
	this->SetNumberOfOutputPorts( std::max( this->GetNumberOfOutputPorts(), index+1 ) );
}

int vtkCudaKSOMProbability::FillInputPortInformation(int i, vtkInformation* info){
	if( i == 2 ){
		info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
		info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
	}else{
		info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 0);
		info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 0);
	}
	info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
	return this->Superclass::FillInputPortInformation(i,info);
}

//------------------------------------------------------------
//Commands for vtkCudaObject compatibility

void vtkCudaKSOMProbability::Reinitialize(int withData){
	//TODO
}

void vtkCudaKSOMProbability::Deinitialize(int withData){
}

//------------------------------------------------------------
int vtkCudaKSOMProbability::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkDataObject::SetPointDataActiveScalarInfo(outputInfo, VTK_FLOAT, 2);
	return 1;
}

int vtkCudaKSOMProbability::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
	vtkImageData* inputData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
	inputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inputData->GetExtent(),6);

	inputInfo = (inputVector[1])->GetInformationObject(0);
	inputData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
	inputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inputData->GetExtent(),6);

    if( this->GetNumberOfInputConnections(2) ){
        inputInfo = (inputVector[2])->GetInformationObject(0);
        inputData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
        inputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inputData->GetExtent(),6);
    }

	return 1;
}

int vtkCudaKSOMProbability::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){
								
	vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
	vtkInformation* kohonenInfo = (inputVector[1])->GetInformationObject(0);
	vtkImageData* kohonenData = vtkImageData::SafeDownCast(kohonenInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
		
    //get the probability maps
	float** probabilityBuffers = new float* [this->GetNumberOfOutputPorts()];
	if( this->GetNumberOfInputConnections(2) > 0 )
		for(int i = 0; i < this->GetNumberOfOutputPorts(); i++){
			vtkInformation* probabilityInfo = (inputVector[3])->GetInformationObject(i);
			vtkImageData* probabilityData = vtkImageData::SafeDownCast(probabilityInfo->Get(vtkDataObject::DATA_OBJECT()));
			probabilityBuffers[i] = (float*) probabilityData->GetScalarPointer();
		}

	//figure out the extent of the output
	float** outputBuffers = new float* [this->GetNumberOfOutputPorts()];
	for(int i = 0; i < this->GetNumberOfOutputPorts(); i++){
		vtkInformation* outputInfo = outputVector->GetInformationObject(i);
		vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
		outData->ShallowCopy(inData);
		outData->SetExtent(inData->GetExtent());
		outData->SetWholeExtent(inData->GetExtent());
		outData->SetSpacing(inData->GetSpacing());
		outData->SetOrigin(inData->GetOrigin());
		outData->SetScalarTypeToFloat();
		outData->SetNumberOfScalarComponents(1);
		outData->AllocateScalars();
		outputBuffers[i] = (float*) outData->GetScalarPointer();
	}

	//update information container
	this->info.NumberOfLabels = this->GetNumberOfOutputPorts() ? this->GetNumberOfOutputPorts(): 1;
	this->info.NumberOfDimensions = inData->GetNumberOfScalarComponents();
	inData->GetDimensions( this->info.VolumeSize );
	kohonenData->GetDimensions( this->info.KohonenMapSize );
	
	//update scale
	this->info.Scale = 1.0 / (this->Scale*this->Scale);

	//pass it over to the GPU
	this->ReserveGPU();
	CUDAalgo_applyProbabilityMaps( (float*) inData->GetScalarPointer(), (float*) kohonenData->GetScalarPointer(),
									probabilityBuffers, outputBuffers, this->GetNumberOfInputConnections(3) > 0, this->Entropy, this->info, this->GetStream() );

	
	delete outputBuffers;
	delete probabilityBuffers;
	return 1;
}
