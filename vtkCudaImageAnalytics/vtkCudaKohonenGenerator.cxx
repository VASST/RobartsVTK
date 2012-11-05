#include "vtkCudaKohonenGenerator.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"

vtkStandardNewMacro(vtkCudaKohonenGenerator);

vtkCudaKohonenGenerator::vtkCudaKohonenGenerator(){
	this->outExt[0] = 0;
	this->outExt[1] = 0;
	this->outExt[2] = 0;
	this->outExt[3] = 0;
	this->outExt[4] = 0;
	this->outExt[5] = 0;
	this->alphaInit = 1.0f;
	this->alphaDecay = 0.99f;
	this->widthInit = 1.0f;
	this->widthDecay = 0.99f;
	this->numIterations = 100;
	
	this->info.KohonenMapSize[0] = 256;
	this->info.KohonenMapSize[1] = 256;
	this->info.KohonenMapSize[2] = 1;
	for(int i = 0; i < 16; i++){
		this->info.Weights[i] = 1.0f;
	}
	this->info.BatchSize = 100;
	this->info.MaxEpochs = 100;
	this->info.flags = 0;

	//configure the input ports
	this->SetNumberOfInputPorts(1);
}

vtkCudaKohonenGenerator::~vtkCudaKohonenGenerator(){
}

//------------------------------------------------------------
//Commands for vtkCudaObject compatibility

void vtkCudaKohonenGenerator::Reinitialize(int withData){
	//TODO
}

void vtkCudaKohonenGenerator::Deinitialize(int withData){
}

//------------------------------------------------------------
//Accessors and mutators

void vtkCudaKohonenGenerator::SetAlphaInitial(double alphaInitial){
	if(alphaInitial >= 0.0 && alphaInitial <= 1.0)
		this->alphaInit = alphaInitial;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetAlphaDecay(double alphaDecay){
	if(alphaDecay >= 0.0 && alphaDecay <= 1.0)
		this->alphaDecay = alphaDecay;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetWidthInitial(double widthInitial){
	if(widthInitial >= 0.0 && widthInitial <= 1.0)
		this->widthInit = widthInitial;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetWidthDecay(double widthDecay){
	if(widthDecay >= 0.0 && widthDecay <= 1.0)
		this->widthDecay = widthDecay;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetKohonenMapSize(int SizeX, int SizeY){
	if(SizeX < 1 || SizeY < 1) return;
	
	this->info.KohonenMapSize[0] = SizeX;
	this->info.KohonenMapSize[1] = SizeY;
}
//------------------------------------------------------------

void vtkCudaKohonenGenerator::SetInput(int idx, vtkDataObject *input)
{
  // Ask the superclass to connect the input.
  this->SetNthInputConnection(0, idx, (input ? input->GetProducerPort() : 0));
}

//----------------------------------------------------------------------------
vtkDataObject *vtkCudaKohonenGenerator::GetInput(int idx)
{
  if (this->GetNumberOfInputConnections(0) <= idx)
    {
    return 0;
    }
  return vtkImageData::SafeDownCast(
    this->GetExecutive()->GetInputData(0, idx));
}
//----------------------------------------------------------------------------
int vtkCudaKohonenGenerator::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkDataObject::SetPointDataActiveScalarInfo(outputInfo, VTK_FLOAT, inData->GetNumberOfScalarComponents());
	return 1;
}

int vtkCudaKohonenGenerator::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** vtkNotUsed(inputVector),
  vtkInformationVector* outputVector)
{
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
	int outExt[6];
	outExt[0] = outExt[2] = outExt[4] = 0;
	outExt[1] = this->info.KohonenMapSize[0]-1;
	outExt[3] = this->info.KohonenMapSize[1]-1;
	outExt[5] = this->info.KohonenMapSize[2]-1;
    outputInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),outExt,6);
	outputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),outExt,6);
	return 1;
}

int vtkCudaKohonenGenerator::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){
								
	vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
	vtkInformation* maskInfo = (inputVector[0])->GetInformationObject(1);
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* maskData = (maskInfo) ? vtkImageData::SafeDownCast(maskInfo->Get(vtkDataObject::DATA_OBJECT())) : 0;
	
    int updateExtent[6];
    outputInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), updateExtent);
	outData->SetScalarTypeToFloat();
	outData->SetNumberOfScalarComponents(inData->GetNumberOfScalarComponents());
	outData->SetExtent(updateExtent);
	outData->SetWholeExtent(updateExtent);
	outData->AllocateScalars();

	//update information container
	this->info.NumberOfDimensions = inData->GetNumberOfScalarComponents();
	inData->GetDimensions( this->info.VolumeSize );
	this->info.BatchSize = (this->info.VolumeSize[0]*this->info.VolumeSize[0] + this->info.VolumeSize[1]*this->info.VolumeSize[1] + this->info.VolumeSize[2]*this->info.VolumeSize[2])/15.0;

	//pass information to CUDA
	this->ReserveGPU();
	CUDAalgo_generateKohonenMap( (float*) inData->GetScalarPointer(), (float*) outData->GetScalarPointer(), (maskData) ? (char*) maskData->GetScalarPointer() : 0, this->info,
		this->alphaInit, this->alphaDecay, this->widthInit*sqrt((double)(this->info.KohonenMapSize[0]*this->info.KohonenMapSize[1])),
		this->widthDecay, this->GetStream() );

	return 1;
}