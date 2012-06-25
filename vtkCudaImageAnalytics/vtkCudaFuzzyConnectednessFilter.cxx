#include "vtkCudaFuzzyConnectednessFilter.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkCudaFuzzyConnectednessFilter);

int vtkCudaFuzzyConnectednessFilter::RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector){

	// get the info objects
	vtkImageData *inData = vtkImageData::SafeDownCast(this->GetInput());
	vtkImageData *outData = this->GetOutput();
	if( !inData || !outData ) return -1;

	//load the CUDA information struct

	//run algorithm

	return 1;
}

vtkCudaFuzzyConnectednessFilter::vtkCudaFuzzyConnectednessFilter(){
	this->SNorm = 0;
	this->TNorm = 0;

	this->Information = new Fuzzy_Connectedness_Information();
	this->Information->snorm = 0;
	this->Information->tnorm = 0;
	this->Information->connectedness = 8;
	this->Information->VolumeSize[0] = this->Information->VolumeSize[1] = this->Information->VolumeSize[2] = 0;
	this->Information->spacing[0] = this->Information->spacing[1] = this->Information->spacing[2] = 0.0f;
}

vtkCudaFuzzyConnectednessFilter::~vtkCudaFuzzyConnectednessFilter(){
	delete this->Information;
}
