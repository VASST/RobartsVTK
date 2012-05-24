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
	this->alphaDecay = 0.75f;
	this->widthInit = 1.0f;
	this->widthDecay = 0.75f;
	this->widthDecay = 0.75f;
	this->numIterations = 100;
}

vtkCudaKohonenGenerator::~vtkCudaKohonenGenerator(){

}

//------------------------------------------------------------
//Commands for vtkCudaObject compatibility

void vtkCudaKohonenGenerator::Reinitialize(int withData){
	//TODO
}

void vtkCudaKohonenGenerator::Deinitialize(int withData){
	this->ReserveGPU();
	CUDAsetup_loadNDImage( this->GetStream() );
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

void vtkCudaKohonenGenerator::UsePositionDataOn(){
	this->info.flags = this->info.flags | 1 ;
}

void vtkCudaKohonenGenerator::UsePositionDataOff(){
	this->info.flags = this->info.flags | 1 ;
}

//------------------------------------------------------------

// The execute method created by the subclass.
void vtkCudaKohonenGenerator::ThreadedRequestData(
  vtkInformation* vtkNotUsed( request ),
  vtkInformationVector** vtkNotUsed( inputVector ),
  vtkInformationVector* vtkNotUsed( outputVector ),
  vtkImageData ***inData, 
  vtkImageData **outData,
  int extent[6], 
  int threadId)
{
  this->ThreadedExecute(inData[0][0], outData[0], extent, threadId);
}

// The execute method created by the subclass.
void vtkCudaKohonenGenerator::ThreadedExecute(
  vtkImageData * inData, 
  vtkImageData * outData,
  int extent[6], 
  int vtkNotUsed(threadId))
{

	//update information container
	this->info.numberOfDimensions = inData->GetNumberOfScalarComponents();
	inData->GetDimensions( this->info.VolumeSize );
	outData->SetDimensions( this->info.OutputResolution );
	outData->AllocateScalars();
	double spacing[3];
	inData->GetSpacing(spacing);
	this->info.spacing[0] = (float) (1.0 / spacing[0]);
	this->info.spacing[1] = (float) (1.0 / spacing[1]);
	this->info.spacing[2] = (float) (1.0 / spacing[2]);

	//pass information to CUDA
	this->ReserveGPU();
	CUDAsetup_loadNDImage( (float*) inData->GetScalarPointer(), this->info, this->GetStream() );
	CUDAalgo_generateKohonenMap( (float*) outData->GetScalarPointer(), this->info, this->GetStream() );
}