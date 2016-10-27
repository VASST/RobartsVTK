#include "vtkAlgorithmOutput.h"
#include "vtkCudaKohonenApplication.h"
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

vtkStandardNewMacro(vtkCudaKohonenApplication);

vtkCudaKohonenApplication::vtkCudaKohonenApplication()
{
  //configure the input ports
  this->SetNumberOfInputPorts(2);
  this->SetNumberOfInputConnections(0, 1);
  this->SetNumberOfInputConnections(1, 1);

  //initialize the scale to 1
  this->Scale = 1.0;
}

vtkCudaKohonenApplication::~vtkCudaKohonenApplication()
{
}

//------------------------------------------------------------
//Commands for CudaObject compatibility

void vtkCudaKohonenApplication::Reinitialize(bool withData /*= false*/)
{
  //TODO
}

void vtkCudaKohonenApplication::Deinitialize(bool withData /*= false*/)
{
}


//------------------------------------------------------------
int vtkCudaKohonenApplication::FillInputPortInformation(int i, vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 0);
  info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 0);
  return this->Superclass::FillInputPortInformation(i, info);
}
//----------------------------------------------------------------------------

void vtkCudaKohonenApplication::SetScale(double s)
{
  if (this->Scale != s && s >= 0.0)
  {
    this->Scale = s;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
double vtkCudaKohonenApplication::GetScale()
{
  return this->Scale;
}

void vtkCudaKohonenApplication::SetDataInputData(vtkImageData* d)
{
  this->SetInputData(0, d);
}

void vtkCudaKohonenApplication::SetMapInputData(vtkImageData* d)
{
  this->SetInputData(1, d);
}
void vtkCudaKohonenApplication::SetDataInputConnection(vtkAlgorithmOutput* d)
{
  this->vtkImageAlgorithm::SetInputConnection(0, d);
}

void vtkCudaKohonenApplication::SetMapInputConnection(vtkAlgorithmOutput* d)
{
  this->vtkImageAlgorithm::SetInputConnection(1, d);
}

vtkImageData* vtkCudaKohonenApplication::GetDataInput()
{
  return (vtkImageData*) this->GetInput(0);
}

vtkImageData* vtkCudaKohonenApplication::GetMapInput()
{
  return (vtkImageData*) this->GetInput(1);
}
//------------------------------------------------------------
int vtkCudaKohonenApplication::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkDataObject::SetPointDataActiveScalarInfo(outputInfo, VTK_FLOAT, 2);
  return 1;
}

int vtkCudaKohonenApplication::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* kohonenInfo = (inputVector[1])->GetInformationObject(0);
  vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* kohonenData = vtkImageData::SafeDownCast(kohonenInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));

  kohonenInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), kohonenData->GetExtent(), 6);
  inputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), inData->GetExtent(), 6);

  return 1;
}

int vtkCudaKohonenApplication::RequestData(vtkInformation* request,
    vtkInformationVector** inputVector,
    vtkInformationVector* outputVector)
{

  vtkInformation* kohonenInfo = (inputVector[1])->GetInformationObject(0);
  vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* kohonenData = vtkImageData::SafeDownCast(kohonenInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));

  outData->ShallowCopy(inData);
  outData->SetExtent(inData->GetExtent());
  outData->AllocateScalars(VTK_FLOAT, 2);

  //update information container
  this->Info.NumberOfDimensions = inData->GetNumberOfScalarComponents();
  inData->GetDimensions(this->Info.VolumeSize);
  kohonenData->GetDimensions(this->Info.KohonenMapSize);

  //update scale
  this->Info.Scale = this->Scale;

  //pass it over to the GPU
  this->ReserveGPU();
  CUDAalgo_applyKohonenMap((float*) inData->GetScalarPointer(),
                           (float*) kohonenData->GetScalarPointer(),
                           (float*) outData->GetScalarPointer(), this->Info, this->GetStream());

  return 1;
}