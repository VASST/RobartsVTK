#include "vtkAlgorithmOutput.h"
#include "vtkCudaPAGMMEstimator.h"
#include "vtkDataArray.h"
#include "vtkImageCast.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTransform.h"
#include <vtkVersion.h>

vtkStandardNewMacro(vtkCudaPAGMMEstimator);

vtkCudaPAGMMEstimator::vtkCudaPAGMMEstimator()
{
  //configure the IO ports
  this->SetNumberOfInputPorts(3);
  this->SetNumberOfInputConnections(0,1);
  this->SetNumberOfInputConnections(1,1);
  this->SetNumberOfInputConnections(2,1);
  this->SetNumberOfOutputPorts(1);

  //initialize conservativeness and scale
  this->Q = 0.01;
  this->Scale = 1.0;
}

vtkCudaPAGMMEstimator::~vtkCudaPAGMMEstimator()
{
}

//------------------------------------------------------------
//Commands for CudaObject compatibility

void vtkCudaPAGMMEstimator::Reinitialize(int withData)
{
  //TODO
}

void vtkCudaPAGMMEstimator::Deinitialize(int withData)
{
}


//----------------------------------------------------------------------------

void vtkCudaPAGMMEstimator::SetConservativeness(double q)
{
  if( q != this->Q && q >= 0.0 && q <= 1.0 )
  {
    this->Q = q;
    this->Modified();
  }
}

double vtkCudaPAGMMEstimator::GetConservativeness()
{
  return this->Q;
}

void vtkCudaPAGMMEstimator::SetScale(double s)
{
  if( s != this->Scale && s >= 0.0 )
  {
    this->Scale = s;
    this->Modified();
  }
}

double vtkCudaPAGMMEstimator::GetScale()
{
  return this->Scale;
}

//------------------------------------------------------------

int vtkCudaPAGMMEstimator::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* seededImageInfo = (inputVector[2])->GetInformationObject(0);
  vtkImageData* seededImage = vtkImageData::SafeDownCast(seededImageInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkInformation* outputGMMInfo = outputVector->GetInformationObject(0);
  vtkImageData* outGMMImage = vtkImageData::SafeDownCast(outputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkDataObject::SetPointDataActiveScalarInfo(outputGMMInfo,  VTK_FLOAT, seededImage->GetScalarRange()[1] );
  return 1;
}

int vtkCudaPAGMMEstimator::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* inputDataInfo = (inputVector[0])->GetInformationObject(0);
  vtkImageData* inputDataImage = vtkImageData::SafeDownCast(inputDataInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkInformation* inputGMMInfo = (inputVector[1])->GetInformationObject(0);
  vtkImageData* inputGMMImage = vtkImageData::SafeDownCast(inputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));

  inputGMMInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inputGMMImage->GetExtent(),6);
  inputDataInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inputDataImage->GetExtent(),6);

  return 1;
}

int vtkCudaPAGMMEstimator::RequestData(vtkInformation *request,
                                       vtkInformationVector **inputVector,
                                       vtkInformationVector *outputVector)
{
  //collect input data information
  vtkInformation* inputDataInfo = (inputVector[0])->GetInformationObject(0);
  vtkInformation* inputGMMInfo = (inputVector[1])->GetInformationObject(0);
  vtkInformation* seededDataInfo = (inputVector[2])->GetInformationObject(0);
  vtkImageData* inputDataImage = vtkImageData::SafeDownCast(inputDataInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* inputGMMImage = vtkImageData::SafeDownCast(inputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* seededDataImage = vtkImageData::SafeDownCast(seededDataInfo->Get(vtkDataObject::DATA_OBJECT()));

  //get output data information containers
  vtkInformation* outputGMMInfo = outputVector->GetInformationObject(0);
  vtkImageData* outputGMMImage = vtkImageData::SafeDownCast(outputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));

  //figure out the extent of the output
  this->Info.NumberOfDimensions = inputDataImage->GetNumberOfScalarComponents();
  this->Info.NumberOfLabels = seededDataImage->GetScalarRange()[1];
#if (VTK_MAJOR_VERSION < 6)
  outputGMMImage->SetScalarTypeToFloat();
  outputGMMImage->SetNumberOfScalarComponents( this->Info.NumberOfLabels );
  outputGMMImage->SetExtent(inputGMMImage->GetExtent());
  outputGMMImage->SetWholeExtent(inputGMMImage->GetExtent());
  outputGMMImage->AllocateScalars();
#else
  outputGMMImage->SetExtent(inputGMMImage->GetExtent());
  outputGMMImage->AllocateScalars(VTK_FLOAT, this->Info.NumberOfLabels);
#endif

  //get volume information for containers
  inputDataImage->GetDimensions( this->Info.VolumeSize );
  outputGMMImage->GetDimensions( this->Info.GMMSize );

  //get range for weight normalization
  double* Range = new double[2*(this->Info.NumberOfDimensions)];
  for(int i = 0; i < this->Info.NumberOfDimensions; i++)
  {
    inputDataImage->GetPointData()->GetScalars()->GetRange(Range+2*i,i);
  }

  //calculate P according tot he Naive model
  int N = this->Info.GMMSize[0]*this->Info.GMMSize[1];
  float P = (Q > 0.0) ? this->Q / (1.0 - pow(1.0-this->Q,N)) : 1.0 / ((double)N);

  //run algorithm on CUDA
  this->ReserveGPU();
  CUDAalgo_applyPAGMMModel( (float*) inputDataImage->GetScalarPointer(), (float*) inputGMMImage->GetScalarPointer(),
                            (float*) outputGMMImage->GetScalarPointer(),
                            (char*) seededDataImage->GetScalarPointer(), this->Info, P, this->Q, this->Scale, this->GetStream() );

  return 1;
}