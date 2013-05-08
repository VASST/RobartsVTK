#include "vtkCudaKohonenGenerator.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "float.h"

vtkStandardNewMacro(vtkCudaKohonenGenerator);

vtkCudaKohonenGenerator::vtkCudaKohonenGenerator(){
	this->outExt[0] = 0;
	this->outExt[1] = 0;
	this->outExt[2] = 0;
	this->outExt[3] = 0;
	this->outExt[4] = 0;
	this->outExt[5] = 0;

	this->MeansAlphaInit = 1.0f;
	this->MeansAlphaDecay = 0.99f;
	this->MeansAlphaBaseline = 0.0f;
	this->MeansAlphaProlong = 0.0f;
	
	this->MeansWidthInit = 1.0f;
	this->MeansWidthDecay = 0.99f;
	this->MeansWidthBaseline = 0.0f;
	this->MeansWidthProlong = 0.0f;
	
	this->VarsAlphaInit = 1.0f;
	this->VarsAlphaDecay = 0.99f;
	this->VarsAlphaBaseline = 0.0f;
	this->VarsAlphaProlong = 0.0f;
	
	this->VarsWidthInit = 1.0f;
	this->VarsWidthDecay = 0.99f;
	this->VarsWidthBaseline = 0.0f;
	this->VarsWidthProlong = 0.0f;

	this->BatchPercent = 1.0/15.0;
	this->UseAllVoxels = false;
	this->UseMask = false;
	
	this->info.KohonenMapSize[0] = 256;
	this->info.KohonenMapSize[1] = 256;
	this->info.KohonenMapSize[2] = 1;
	this->WeightNormalization = true;
	for(int i = 0; i < MAX_DIMENSIONALITY; i++){
		this->UnnormalizedWeights[i] = 1.0f;
		this->info.Weights[i] = 1.0f;
	}
	this->MaxEpochs = 1000;
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

void vtkCudaKohonenGenerator::SetMeansAlphaInitial(double alphaInitial){
	if(alphaInitial >= 0.0 && alphaInitial <= 1.0)
		this->MeansAlphaInit = alphaInitial;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetMeansAlphaBaseline(double alphaBaseline){
	if(alphaBaseline >= 0.0 && alphaBaseline <= 1.0)
		this->MeansAlphaBaseline = alphaBaseline;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetMeansAlphaProlong(double alphaProlong){
	if(alphaProlong >= 0.0 )
		this->MeansAlphaProlong = alphaProlong;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetMeansAlphaDecay(double alphaDecay){
	if(alphaDecay >= 0.0 && alphaDecay <= 1.0)
		this->MeansAlphaDecay = alphaDecay;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetVarsAlphaInitial(double alphaInitial){
	if(alphaInitial >= 0.0 && alphaInitial <= 1.0)
		this->VarsAlphaInit = alphaInitial;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetVarsAlphaBaseline(double alphaBaseline){
	if(alphaBaseline >= 0.0 && alphaBaseline <= 1.0)
		this->VarsAlphaBaseline = alphaBaseline;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetVarsAlphaProlong(double alphaProlong){
	if(alphaProlong >= 0.0 )
		this->VarsAlphaProlong = alphaProlong;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetVarsAlphaDecay(double alphaDecay){
	if(alphaDecay >= 0.0 && alphaDecay <= 1.0)
		this->VarsAlphaDecay = alphaDecay;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetMeansWidthInitial(double widthInitial){
	if(widthInitial >= 0.0 && widthInitial <= 1.0)
		this->MeansWidthInit = widthInitial;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetMeansWidthProlong(double widthProlong){
	if(widthProlong >= 0.0 )
		this->MeansWidthProlong = widthProlong;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetMeansWidthDecay(double widthDecay){
	if(widthDecay >= 0.0 && widthDecay <= 1.0)
		this->MeansWidthDecay = widthDecay;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetMeansWidthBaseline(double widthBaseline){
	if(widthBaseline >= 0.0 && widthBaseline <= 1.0)
		this->MeansWidthBaseline = widthBaseline;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetVarsWidthInitial(double widthInitial){
	if(widthInitial >= 0.0 && widthInitial <= 1.0)
		this->VarsWidthInit = widthInitial;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetVarsWidthProlong(double widthProlong){
	if(widthProlong >= 0.0 )
		this->VarsWidthProlong = widthProlong;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetVarsWidthDecay(double widthDecay){
	if(widthDecay >= 0.0 && widthDecay <= 1.0)
		this->VarsWidthDecay = widthDecay;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetVarsWidthBaseline(double widthBaseline){
	if(widthBaseline >= 0.0 && widthBaseline <= 1.0)
		this->VarsWidthBaseline = widthBaseline;
	this->Modified();
}

void vtkCudaKohonenGenerator::SetKohonenMapSize(int SizeX, int SizeY){
	if(SizeX < 1 || SizeY < 1) return;
	
	this->info.KohonenMapSize[0] = SizeX;
	this->info.KohonenMapSize[1] = SizeY;
}

bool vtkCudaKohonenGenerator::GetUseAllVoxelsFlag(){
	return this->UseAllVoxels;
}

void vtkCudaKohonenGenerator::SetUseAllVoxelsFlag(bool t){
	if( t != this->UseAllVoxels ){
		this->UseAllVoxels = t;
		this->Modified();
	}
}

bool vtkCudaKohonenGenerator::GetUseMaskFlag(){
	return this->UseMask;
}

void vtkCudaKohonenGenerator::SetUseMaskFlag(bool t){
	if( t != this->UseMask ){
		this->UseMask = t;
		this->Modified();
	}
}

//------------------------------------------------------------

void vtkCudaKohonenGenerator::SetWeight(int index, double weight){
	if( index >= 0 && index < MAX_DIMENSIONALITY && weight >= 0.0 ){
		this->UnnormalizedWeights[index] = weight;
		this->Modified();
	}
}

void vtkCudaKohonenGenerator::SetWeights(const double* weights){
	for(int i = 0; i < MAX_DIMENSIONALITY; i++)
		try{
			this->UnnormalizedWeights[i] = weights[i];
		}catch(...){
			this->UnnormalizedWeights[i] = 1.0;
		}
	this->Modified();
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
	if( set != this->WeightNormalization ){
		this->WeightNormalization = set;
		this->Modified();
	}
}

bool vtkCudaKohonenGenerator::GetWeightNormalization(){
	return this->WeightNormalization;
}

void vtkCudaKohonenGenerator::SetNumberOfIterations(int number){
	if( number >= 0 && this->MaxEpochs != number ){
		this->MaxEpochs = number;
		this->Modified();
	}
}

int vtkCudaKohonenGenerator::GetNumberOfIterations(){
	return this->MaxEpochs;
}

void vtkCudaKohonenGenerator::SetBatchSize(double fraction){
	if( fraction >= 0.0 && this->BatchPercent != fraction ){
		this->BatchPercent = fraction;
		this->Modified();
	}
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
	
	//get general information
	int NumPictures = (inputVector[0])->GetNumberOfInformationObjects() / (this->UseMask ? 2 : 1);
	if( NumPictures < 1 ){
		vtkErrorMacro(<<"No pictures to train on.");
		return -1;
	}
	vtkInformation* outputInfo = outputVector->GetInformationObject(0);
	vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
	
	//make sure that the number of components is constant and the input type is FLOAT, and collect volume sizes
	vtkImageData* inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData* maskData = 0;
	int* VolumeSize = new int[ 3*NumPictures ];
	int SumDiagonal = 0;
	this->info.NumberOfDimensions = inData->GetNumberOfScalarComponents();
	for(int p = 0; p < NumPictures; p++){
		
		if( this->UseMask ){
			inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p)->Get(vtkDataObject::DATA_OBJECT()));
			maskData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p+1)->Get(vtkDataObject::DATA_OBJECT()));
		}else{
			inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(p)->Get(vtkDataObject::DATA_OBJECT()));
		}

		inData->GetDimensions( &(VolumeSize[3*p]) );
		SumDiagonal += VolumeSize[3*p]*VolumeSize[3*p];
		SumDiagonal += VolumeSize[3*p+1]*VolumeSize[3*p+1];
		SumDiagonal += VolumeSize[3*p+2]*VolumeSize[3*p+2];

		if( inData->GetNumberOfScalarComponents() != this->info.NumberOfDimensions ){
			vtkErrorMacro(<<"Data objects need to have a consistant number of components");
			delete VolumeSize;
			return -1;
		}
		if( inData->GetScalarType() != VTK_FLOAT ){
			vtkErrorMacro(<<"Data objects need to be of type float");
			delete VolumeSize;
			return -1;
		}
		if( this->UseMask && maskData->GetScalarType() != VTK_CHAR &&
							 maskData->GetScalarType() != VTK_SIGNED_CHAR &&
							 maskData->GetScalarType() != VTK_UNSIGNED_CHAR ){
			std::cout << maskData->GetScalarType() << std::endl;
			vtkErrorMacro(<<"Mask objects need to be of type char");
			delete VolumeSize;
			return -1;
		}
	}

    int updateExtent[6];
    outputInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), updateExtent);
	outData->SetScalarTypeToFloat();
	outData->SetNumberOfScalarComponents(2*inData->GetNumberOfScalarComponents());
	outData->SetExtent(updateExtent);
	outData->SetWholeExtent(updateExtent);
	outData->AllocateScalars();

	//update information container
	int BatchSize = (this->UseAllVoxels) ? -1 : SumDiagonal * this->BatchPercent;

	//get range
	double* Range = new double[2*(this->info.NumberOfDimensions)];
	for(int i = 0; i < this->info.NumberOfDimensions; i++){
		Range[2*i] = DBL_MAX; Range[2*i+1] = DBL_MIN;
		for(int p = 0; p < NumPictures; p++){
			double tempRange[2];
			if( this->UseMask ){
				inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p)->Get(vtkDataObject::DATA_OBJECT()));
			}else{
				inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(p)->Get(vtkDataObject::DATA_OBJECT()));
			}
			inData->GetPointData()->GetScalars()->GetRange(tempRange,i);
			Range[2*i] = std::min( tempRange[0], Range[2*i] );
			Range[2*i+1] = std::max( tempRange[1], Range[2*i+1] );
		}
	}

	//get scalar pointers
	float** inputDataPtr = new float* [NumPictures];
	char** maskDataPtr = (this->UseMask) ? new char*[NumPictures]: 0;
	for(int p = 0; p < NumPictures; p++){
		if( this->UseMask ){
			inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p)->Get(vtkDataObject::DATA_OBJECT()));
			inputDataPtr[p] = (float*) inData->GetScalarPointer();
			maskData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p+1)->Get(vtkDataObject::DATA_OBJECT()));
			maskDataPtr[p] = (char*) maskData->GetScalarPointer();
		}else{
			inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(p)->Get(vtkDataObject::DATA_OBJECT()));
			inputDataPtr[p] = (float*) inData->GetScalarPointer();
		}
	}


	//update weights
	if( this->WeightNormalization ){
		for(int i = 0; i < this->info.NumberOfDimensions; i++){
			this->info.Weights[i] = this->UnnormalizedWeights[i] / ((Range[2*i+1] - Range[2*i] > 0.0) ? (Range[2*i+1] - Range[2*i]) : 1.0);
			this->info.Weights[i] *= this->info.Weights[i];
		}
	}else{
		for(int i = 0; i < this->info.NumberOfDimensions; i++){
			this->info.Weights[i] = this->UnnormalizedWeights[i]*this->UnnormalizedWeights[i];
		}
	}

	//pass information to CUDA
	this->ReserveGPU();
	CUDAalgo_generateKohonenMap( inputDataPtr, (float*) outData->GetScalarPointer(), maskDataPtr,
		Range, VolumeSize, NumPictures, this->info, MaxEpochs, BatchSize, 
		(1+exp((this->MeansAlphaDecay - 1.0)*this->MeansAlphaProlong))*(this->MeansAlphaInit-this->MeansAlphaBaseline), this->MeansAlphaBaseline,
		1.0 - this->MeansAlphaDecay, (this->MeansAlphaDecay - 1.0)*this->MeansAlphaProlong,
		(1+exp((this->MeansWidthDecay - 1.0)*this->MeansWidthProlong))*(this->MeansWidthInit-this->MeansWidthBaseline), this->MeansWidthBaseline,
		1.0 - this->MeansWidthDecay, (this->MeansWidthDecay - 1.0)*this->MeansWidthProlong,
		(1+exp((this->VarsAlphaDecay - 1.0)*this->VarsAlphaProlong))*(this->VarsAlphaInit-this->VarsAlphaBaseline), this->VarsAlphaBaseline,
		1.0 - this->VarsAlphaDecay, (this->VarsAlphaDecay - 1.0)*this->VarsAlphaProlong,
		(1+exp((this->VarsWidthDecay - 1.0)*this->VarsWidthProlong))*(this->VarsWidthInit-this->VarsWidthBaseline), this->VarsWidthBaseline,
		1.0 - this->VarsWidthDecay, (this->VarsWidthDecay - 1.0)*this->VarsWidthProlong,
		this->GetStream() );
	
	//clean up temporaries
	delete Range;
	delete VolumeSize;
	delete inputDataPtr;
	if( this->UseMask ) delete maskDataPtr;

	return 1;
}
