#include "vtkCudaFuzzyConnectednessFilter.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkSetGet.h"
#include <vtkVersion.h>

vtkStandardNewMacro(vtkCudaFuzzyConnectednessFilter);

int vtkCudaFuzzyConnectednessFilter::RequestData(vtkInformation* request,
    vtkInformationVector** inputVector,
    vtkInformationVector* outputVector)
{
  // get the info objects
  vtkImageData *seedData = vtkImageData::SafeDownCast(this->GetInput(0));
  vtkImageData *affData = vtkImageData::SafeDownCast(this->GetInput(1));
  vtkImageData *outData = this->GetOutput();
  if( !affData || !seedData || !outData )
  {
    return -1;
  }

  //make sure that there is only 1 component and it is not a double
  if( seedData->GetScalarType() != VTK_FLOAT ||
      affData->GetScalarType() != VTK_FLOAT ||
      affData->GetNumberOfScalarComponents() != 3 )
  {
    vtkErrorMacro( "Execute: Input data is not in FLOAT form or the affinity does not have 3 components");
    return -1;
  }

  //make sure the seed image and the actual image are the same size
  int* dimAff = affData->GetDimensions();
  int* dimSeed = seedData->GetDimensions();
  if( dimAff[0] != dimSeed[0] || dimAff[1] != dimSeed[1] || dimAff[2] != dimSeed[2] )
  {
    vtkErrorMacro( "Execute: Seed image not the same size as the affinity image");
    return -1;
  }

  //scale the output image appropriately
#if (VTK_MAJOR_VERSION < 6)
  outData->SetScalarTypeToFloat();
  outData->SetNumberOfScalarComponents(seedData->GetNumberOfScalarComponents());
  outData->SetExtent( seedData->GetExtent() );
  outData->SetWholeExtent( seedData->GetExtent() );
  outData->SetSpacing( seedData->GetSpacing() );
  outData->SetOrigin( seedData->GetOrigin() );
  outData->AllocateScalars();
#else
  outData->SetExtent( seedData->GetExtent() );
  outData->SetSpacing( seedData->GetSpacing() );
  outData->SetOrigin( seedData->GetOrigin() );
  outData->AllocateScalars(VTK_FLOAT, seedData->GetNumberOfScalarComponents());
#endif

  //load the CUDA information struct
  this->Information->snorm = this->SNorm;
  this->Information->tnorm = this->TNorm;
  this->Information->VolumeSize.x = seedData->GetExtent()[1] - seedData->GetExtent()[0] + 1;
  this->Information->VolumeSize.y = seedData->GetExtent()[3] - seedData->GetExtent()[2] + 1;
  this->Information->VolumeSize.z = seedData->GetExtent()[5] - seedData->GetExtent()[4] + 1;
  this->Information->NumObjects = seedData->GetNumberOfScalarComponents();
  this->Information->Spacing.x = seedData->GetSpacing()[0];
  this->Information->Spacing.y = seedData->GetSpacing()[1];
  this->Information->Spacing.z = seedData->GetSpacing()[2];

  //figure out a good number of iterations (exact number required for SNORM=0, lower bound for rest)
  int numIts = this->Information->VolumeSize.x*
               this->Information->VolumeSize.y*
               this->Information->VolumeSize.z;

  //run algorithm
  this->ReserveGPU();
  CUDAalgo_calculateConnectedness((float*) outData->GetScalarPointer(),
                                  (float*) seedData->GetScalarPointer(),
                                  (float*) affData->GetScalarPointer(),
                                  numIts, *(this->Information), this->GetStream() );
  return 1;
}

void vtkCudaFuzzyConnectednessFilter::Reinitialize(int withData = 0)
{
  //no long-term data stored and no helper classes, so no body for this method
}

void vtkCudaFuzzyConnectednessFilter::Deinitialize(int withData = 0)
{
  //no long-term data stored and no helper classes, so no body for this method
}

vtkCudaFuzzyConnectednessFilter::vtkCudaFuzzyConnectednessFilter()
{
  this->SNorm = 0;
  this->TNorm = 0;

  this->Information = new Fuzzy_Connectedness_Information();
  this->Information->snorm = 0;
  this->Information->tnorm = 0;
  this->Information->VolumeSize.x = this->Information->VolumeSize.y = this->Information->VolumeSize.z = 0;
  this->Information->Spacing.x = this->Information->Spacing.y = this->Information->Spacing.z = 1.0f;
}

vtkCudaFuzzyConnectednessFilter::~vtkCudaFuzzyConnectednessFilter()
{
  delete this->Information;
}