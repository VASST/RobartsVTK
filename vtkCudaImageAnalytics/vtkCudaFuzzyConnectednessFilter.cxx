#include "vtkCudaFuzzyConnectednessFilter.h"
#include "vtkObjectFactory.h"
#include "vtkSetGet.h"

vtkStandardNewMacro(vtkCudaFuzzyConnectednessFilter);

int vtkCudaFuzzyConnectednessFilter::RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector){

	// get the info objects
	vtkImageData *inData = vtkImageData::SafeDownCast(this->GetInput(0));
	vtkImageData *seedData = vtkImageData::SafeDownCast(this->GetInput(1));
	vtkImageData *outData = this->GetOutput();
	if( !inData || !seedData || !outData ) return -1;

	//make sure that there is only 1 component and it is not a double
	if( inData->GetNumberOfScalarComponents() != 1 ||
		seedData->GetNumberOfScalarComponents() != 1 ||
		inData->GetScalarType() == VTK_DOUBLE ||
		seedData->GetScalarType() != VTK_FLOAT ){
      vtkErrorMacro(<< "Execute: Invalid number of components or input type");
      return -1;
	}

	//make sure the seed image and the actual image are the same size
	int* dimIn = inData->GetDimensions();
	int* dimSeed = seedData->GetDimensions();
	if( dimIn[0] != dimSeed[0] || dimIn[1] != dimSeed[1] || dimIn[2] != dimSeed[2] ){
      vtkErrorMacro(<< "Execute: Seed image not the same size as the input image");
      return -1;
	}

	//scale the output image appropriately
	outData->SetScalarTypeToFloat();
	outData->SetNumberOfScalarComponents(1);
	outData->SetExtent( inData->GetExtent() );
	outData->SetSpacing( inData->GetSpacing() );
	outData->SetOrigin( inData->GetOrigin() );
	outData->AllocateScalars();
	
	//load the CUDA information struct
	this->Information->snorm = this->SNorm;
	this->Information->tnorm = this->TNorm;
	this->Information->VolumeSize.x = inData->GetExtent()[1] - inData->GetExtent()[0] + 1;
	this->Information->VolumeSize.y = inData->GetExtent()[3] - inData->GetExtent()[2] + 1;
	this->Information->VolumeSize.z = inData->GetExtent()[5] - inData->GetExtent()[4] + 1;
	this->Information->Spacing.x = inData->GetSpacing()[0];
	this->Information->Spacing.y = inData->GetSpacing()[1];
	this->Information->Spacing.z = inData->GetSpacing()[2];
	this->Information->distanceWeight = this->DistanceWeight;
	this->Information->gradientWeight = this->GradientWeight;

	//run algorithm
	this->ReserveGPU();
	switch (inData->GetScalarType())
    {
    vtkTemplateMacro(
		CUDAalgo_calculateConnectedness((float*) outData->GetScalarPointer(),
										(float*) seedData->GetScalarPointer(),
										1000,
										inData->GetScalarPointer(), 1,
										*(this->Information), this->GetStream() )
	);
    default:
      vtkErrorMacro(<< "Execute: Unknown input ScalarType");
      return -1;
    }

	return 1;
}

vtkCudaFuzzyConnectednessFilter::vtkCudaFuzzyConnectednessFilter(){
	this->SNorm = 0;
	this->TNorm = 0;

	this->Information = new Fuzzy_Connectedness_Information();
	this->Information->snorm = 0;
	this->Information->tnorm = 0;
	this->Information->VolumeSize.x = this->Information->VolumeSize.y = this->Information->VolumeSize.z = 0;
	this->Information->Spacing.x = this->Information->Spacing.y = this->Information->Spacing.z = 1.0f;
}

vtkCudaFuzzyConnectednessFilter::~vtkCudaFuzzyConnectednessFilter(){
	delete this->Information;
}
