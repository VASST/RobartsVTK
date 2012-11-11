#include "vtkCudaKohonenGenerator.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

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
	this->BatchPercent = 1.0/15.0;
	
	this->info.KohonenMapSize[0] = 256;
	this->info.KohonenMapSize[1] = 256;
	this->info.KohonenMapSize[2] = 1;
	this->WeightNormalization = true;
	for(int i = 0; i < MAX_DIMENSIONALITY; i++){
		this->UnnormalizedWeights[i] = 1.0f;
		this->info.Weights[i] = 1.0f;
	}
	this->info.MaxEpochs = 1000;
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

void vtkCudaKohonenGenerator::SetWeight(int index, double weight){
	if( index >= 0 && index < MAX_DIMENSIONALITY && weight >= 0.0 )
		this->UnnormalizedWeights[index] = weight;
}

void vtkCudaKohonenGenerator::SetWeights(const double* weights){
	for(int i = 0; i < MAX_DIMENSIONALITY; i++)
		try{
			this->UnnormalizedWeights[i] = weights[i];
		}catch(...){
			this->UnnormalizedWeights[i] = 1.0;
		}
}

double vtkCudaKohonenGenerator::GetWeight(int index){
	if( index >= 0 && index < MAX_DIMENSIONALITY )
		return this->UnnormalizedWeights[index];
	return 0.0;
}

double* vtkCudaKohonenGenerator::GetWeights(){
	return this->UnnormalizedWeights;
}

void vtkCudaKohonenGenerator::SetWeightNormalization(bool set){
	this->WeightNormalization = set;
}

bool vtkCudaKohonenGenerator::GetWeightNormalization(){
	return this->WeightNormalization;
}

void vtkCudaKohonenGenerator::SetNumberOfIterations(int number){
	if( number > 0 )
		this->info.MaxEpochs = number;
}

int vtkCudaKohonenGenerator::GetNumberOfIterations(){
	return this->info.MaxEpochs;
}

void vtkCudaKohonenGenerator::SetBatchSize(double fraction){
	if( fraction >= 0.0 )
		this->BatchPercent = fraction;
}

double vtkCudaKohonenGenerator::GetBatchSize(){
	return this->BatchPercent;
}


//------------------------------------------------------------
int vtkCudaKohonenGenerator::FillInputPortInformation(int i, vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  return this->Superclass::FillInputPortInformation(i,info);
}

void vtkCudaKohonenGenerator::SetInput(int idx, vtkDataObject *input)
{
  // Ask the superclass to connect the input.
  this->SetNthInputConnection(0, idx, (input ? input->GetProducerPort() : 0));
}

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
	this->info.BatchSize = (this->info.VolumeSize[0]*this->info.VolumeSize[0] + this->info.VolumeSize[1]*this->info.VolumeSize[1] + this->info.VolumeSize[2]*this->info.VolumeSize[2])*this->BatchPercent;

	//get range
	double* Range = new double[2*(this->info.NumberOfDimensions)];
	for(int i = 0; i < this->info.NumberOfDimensions; i++){
		inData->GetPointData()->GetScalars()->GetRange(Range+2*i,i);
	}

	//update weights
	if( this->WeightNormalization ){
		for(int i = 0; i < this->info.NumberOfDimensions; i++){
			this->info.Weights[i] = this->UnnormalizedWeights[i] / ((Range[2*i+1] - Range[2*i] > 0.0) ? (Range[2*i+1] - Range[2*i]) : 1.0);
		}
	}else{
		for(int i = 0; i < this->info.NumberOfDimensions; i++){
			this->info.Weights[i] = this->UnnormalizedWeights[i];
		}
	}

	//pass information to CUDA
	this->ReserveGPU();
	CUDAalgo_generateKohonenMap( (float*) inData->GetScalarPointer(), (float*) outData->GetScalarPointer(), (maskData) ? (char*) maskData->GetScalarPointer() : 0, Range, this->info, 
		this->alphaInit, this->alphaDecay, this->widthInit*sqrt((double)(this->info.KohonenMapSize[0]*this->info.KohonenMapSize[1])),
		this->widthDecay, this->GetStream() );
	
	//clean up temporaries
	delete Range;

	return 1;
}