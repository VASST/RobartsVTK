#include "CUDA_KSOMlikelihood.h"
#include "vtkAlgorithmOutput.h"
#include "vtkCudaKSOMLikelihood.h"
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

vtkStandardNewMacro(vtkCudaKSOMLikelihood);

vtkCudaKSOMLikelihood::vtkCudaKSOMLikelihood()
{
  //configure the IO ports
  this->SetNumberOfInputPorts(3);
  this->SetNumberOfInputConnections(0, 1);
  this->SetNumberOfInputConnections(1, 1);
  this->SetNumberOfInputConnections(2, 1);
  this->SetNumberOfOutputPorts(1);

  //initialize conservativeness and scale
  this->Scale = 1.0;
}

vtkCudaKSOMLikelihood::~vtkCudaKSOMLikelihood()
{
}

//------------------------------------------------------------
//Commands for CudaObject compatibility

void vtkCudaKSOMLikelihood::Reinitialize(bool withData /*= false*/)
{
  //TODO
}

void vtkCudaKSOMLikelihood::Deinitialize(bool withData /*= false*/)
{
}

//----------------------------------------------------------------------------

void vtkCudaKSOMLikelihood::SetScale(double s)
{
  if (s != this->Scale && s >= 0.0)
  {
    this->Scale = s;
    this->Modified();
  }
}

double vtkCudaKSOMLikelihood::GetScale()
{
  return this->Scale;
}

//------------------------------------------------------------

int vtkCudaKSOMLikelihood::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* seededImageInfo = (inputVector[2])->GetInformationObject(0);
  vtkImageData* seededImage = vtkImageData::SafeDownCast(seededImageInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkInformation* outputGMMInfo = outputVector->GetInformationObject(0);
  vtkImageData* outGMMImage = vtkImageData::SafeDownCast(outputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkDataObject::SetPointDataActiveScalarInfo(outputGMMInfo,  VTK_FLOAT, seededImage->GetScalarRange()[1]);
  return 1;
}

int vtkCudaKSOMLikelihood::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* inputDataInfo = (inputVector[0])->GetInformationObject(0);
  vtkImageData* inputDataImage = vtkImageData::SafeDownCast(inputDataInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkInformation* inputGMMInfo = (inputVector[1])->GetInformationObject(0);
  vtkImageData* inputGMMImage = vtkImageData::SafeDownCast(inputGMMInfo->Get(vtkDataObject::DATA_OBJECT()));

  inputGMMInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), inputGMMImage->GetExtent(), 6);
  inputDataInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), inputDataImage->GetExtent(), 6);

  return 1;
}

int vtkCudaKSOMLikelihood::RequestData(vtkInformation* request,
                                       vtkInformationVector** inputVector,
                                       vtkInformationVector* outputVector)
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
  outputGMMImage->SetExtent(inputGMMImage->GetExtent());
  outputGMMImage->AllocateScalars(VTK_FLOAT, this->Info.NumberOfLabels);

  //get volume information for containers
  inputDataImage->GetDimensions(this->Info.VolumeSize);
  outputGMMImage->GetDimensions(this->Info.GMMSize);

  //get range for weight normalization
  double* Range = new double[2 * (this->Info.NumberOfDimensions)];
  for (int i = 0; i < this->Info.NumberOfDimensions; i++)
  {
    inputDataImage->GetPointData()->GetScalars()->GetRange(Range + 2 * i, i);
  }

  //calculate P according tot he Naive model
  int N = this->Info.GMMSize[0] * this->Info.GMMSize[1];

  //run algorithm on CUDA
  this->ReserveGPU();
  CUDAalgo_applyKSOMLLModel((float*) inputDataImage->GetScalarPointer(), (float*) inputGMMImage->GetScalarPointer(),
                            (float*) outputGMMImage->GetScalarPointer(),
                            (char*) seededDataImage->GetScalarPointer(), this->Info, this->Scale, this->GetStream());

  return 1;
}
