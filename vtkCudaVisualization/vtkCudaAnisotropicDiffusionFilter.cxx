#include "vtkCudaAnisotropicDiffusionFilter.h"

#include "CUDA_anisotropicDiffusionAlgo.h"

//VTK compatibility
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkCudaAnisotropicDiffusionFilter);

vtkCudaAnisotropicDiffusionFilter::vtkCudaAnisotropicDiffusionFilter(){
	this->output = vtkImageData::New();
	this->sigma = 0.0f;
	this->lambda = 0.5;
	this->numIterations = 0;
	this->caster = vtkImageCast::New();

}

vtkCudaAnisotropicDiffusionFilter::~vtkCudaAnisotropicDiffusionFilter(){
	this->output->Delete();
	this->caster->Delete();
}

void vtkCudaAnisotropicDiffusionFilter::SetInput(vtkImageData* image){
	this->caster->SetInput(image);
	this->caster->SetOutputScalarTypeToFloat();
	this->caster->Update();
}

vtkImageData* vtkCudaAnisotropicDiffusionFilter::GetOutput(){
	return this->output;
}
void vtkCudaAnisotropicDiffusionFilter::SetLambda(float lambda){
	if(lambda > 1.0f || lambda < 0.0f) return;
	this->lambda = lambda;
}

float vtkCudaAnisotropicDiffusionFilter::GetLambda(){
	return this->lambda;
}

void vtkCudaAnisotropicDiffusionFilter::SetSigma(float sigma){
	if(sigma >= 0.0f) this->sigma = sigma;
}
	
float vtkCudaAnisotropicDiffusionFilter::GetSigma(){
	return this->sigma;
}
	
void vtkCudaAnisotropicDiffusionFilter::SetNumberOfIterations(unsigned int numIterations){
	this->numIterations = numIterations;
}

unsigned int vtkCudaAnisotropicDiffusionFilter::GetNumberOfIterations(){
	return this->numIterations;
}

void vtkCudaAnisotropicDiffusionFilter::Update(){

	//if we have no input, just return
	if(this->caster->GetInput() == 0) return;

	//update the input
	this->caster->Update();

	//update the buffer for the output
	this->output->SetDimensions( this->caster->GetOutput()->GetDimensions() );
	this->output->SetSpacing( this->caster->GetOutput()->GetSpacing() );
	this->output->SetOrigin( this->caster->GetOutput()->GetOrigin() );
	this->output->SetScalarTypeToFloat();
	this->output->SetNumberOfScalarComponents(1);
	this->output->AllocateScalars();
	this->output->Update();

	//run the actual algorith
	CUDAanisotropicDiffusionAlgo_doFilter(  (float*) this->caster->GetOutput()->GetScalarPointer(),
											(float*) this->output->GetScalarPointer(),
											this->output->GetDimensions(),
											this->lambda, this->sigma, this->numIterations);

}