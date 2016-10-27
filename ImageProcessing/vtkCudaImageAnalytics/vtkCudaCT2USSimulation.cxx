#include "vtkCudaCT2USSimulation.h"
#include "vtkImageCast.h"
#include "vtkImageData.h"
#include "vtkObjectFactory.h"
#include "vtkTransform.h"
#include <vtkVersion.h>

vtkStandardNewMacro(vtkCudaCT2USSimulation);

void vtkCudaCT2USSimulation::Reinitialize(bool withData /*= false*/)
{
  if (withData && this->Caster->GetInput())
  {
    this->SetInput((vtkImageData*) this->Caster->GetInput());
  }
}

void vtkCudaCT2USSimulation::Deinitialize(bool withData /*= false*/)
{
  if (this->Caster->GetInput())
  {
    this->ReserveGPU();
    CUDAsetup_unloadCTImage(this->GetStream());
  }
}

void vtkCudaCT2USSimulation::SetInput(vtkImageData* inData, int i)
{
  //if 0 is our identifier, we are adding the CT
  if (i == 0)
  {
    this->SetInput(inData);

    //if 1 is the identifier, we are adding a base ultrasound
  }
  else if (i == 1 && inData != 0 && inData->GetScalarType() == VTK_UNSIGNED_CHAR)
  {
    this->Information.optimalParam = true;
    if (this->InputUltrasound == inData)
    {
      return;
    }
    if (this->InputUltrasound)
    {
      this->InputUltrasound->UnRegister(this);
    }
    this->InputUltrasound = inData;
    this->InputUltrasound->Register(this);

    //load the input volume into the CUDA kernel
    this->ReserveGPU();
    CUDAsetup_loadUSImage((unsigned char*) this->InputUltrasound->GetScalarPointer(), this->InputUltrasound->GetDimensions(),
                          this->GetStream());

    //if we pass a null image, treat as if we don't want to compute cross-correlation
  }
  else if (i == 1 && inData == 0)
  {
    this->Information.optimalParam = false;
    if (this->InputUltrasound)
    {
      this->InputUltrasound->UnRegister(this);
    }
    this->InputUltrasound = 0;
  }
}

void vtkCudaCT2USSimulation::SetInput(vtkImageData* i)
{
  //load the input to a texture
  this->Caster->SetInputDataObject(i);
  this->Caster->Update();

  //get the volume size from the input
  double inputOrigin[3];
  double spacing[3];
  int inputExtent[6];
  int dims[3];
  i->GetOrigin(inputOrigin);
  i->GetExtent(inputExtent);
  i->GetSpacing(spacing);
  i->GetDimensions(dims);

  //set the volume dimensions
  this->Information.VolumeSize.x = dims[0];
  this->Information.VolumeSize.y = dims[1];
  this->Information.VolumeSize.z = dims[2];

  //get the spacing information from the input
  this->Information.spacing.x = 0.5 / spacing[0];
  this->Information.spacing.y = 0.5 / spacing[1];
  this->Information.spacing.z = 0.5 / spacing[2];

  // Compute the origin of the extent the volume origin is at voxel (0,0,0)
  // but we want to consider (0,0,0) in voxels to be at
  // (inputExtent[0], inputExtent[2], inputExtent[4]).
  double extentOrigin[3];
  extentOrigin[0] = inputOrigin[0] + inputExtent[0] * spacing[0];
  extentOrigin[1] = inputOrigin[1] + inputExtent[2] * spacing[1];
  extentOrigin[2] = inputOrigin[2] + inputExtent[4] * spacing[2];

  // Create a transform that will account for the scaling and translation of
  // the scalar data. The is the volume to voxels matrix.
  vtkTransform* VoxelsTransform = vtkTransform::New();
  VoxelsTransform->Identity();
  VoxelsTransform->Translate(extentOrigin[0], extentOrigin[1], extentOrigin[2]);
  VoxelsTransform->Scale(spacing[0], spacing[1], spacing[2]);

  // Now we actually have the world to voxels matrix - copy it out
  vtkMatrix4x4* WorldToVoxelsMatrix = vtkMatrix4x4::New();
  WorldToVoxelsMatrix->DeepCopy(VoxelsTransform->GetMatrix());
  WorldToVoxelsMatrix->Invert();

  //output the CT location information to the information holder
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      this->Information.WorldToVolume[i * 4 + j] = WorldToVoxelsMatrix->GetElement(i, j);
    }
  }
  VoxelsTransform->Delete();
  WorldToVoxelsMatrix->Delete();

  //load the input volume into the CUDA kernel
  this->ReserveGPU();
  CUDAsetup_loadCTImage((float*) this->Caster->GetOutput()->GetScalarPointer(), this->Information,
                        this->GetStream());

}

void vtkCudaCT2USSimulation::SetTransform(vtkTransform* t)
{
  this->UsTransform = t;
  this->Modified();
}

#include "vtkTimerLog.h"

void vtkCudaCT2USSimulation::Update()
{

  //if we are missing either input or transform, do not update
  if (!this->Caster->GetInput() || !this->UsTransform)
  {
    return;
  }

  vtkTimerLog* timer = vtkTimerLog::New();
  timer->StartTimer();

  //output the ultrasound location information to the information holder
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      this->Information.UltraSoundToWorld[i * 4 + j] = this->UsTransform->GetMatrix()->GetElement(i, j);
    }
  }

  //run the algorithm
  this->ReserveGPU();
  CUDAalgo_simulateUltraSound((float*) this->DensOutput->GetScalarPointer(),
                              (float*) this->TransOutput->GetScalarPointer(),
                              (float*) this->ReflOutput->GetScalarPointer(),
                              (unsigned char*) this->UsOutput->GetScalarPointer(),
                              this->Information, this->GetStream());
  this->CallSyncThreads();

  timer->StopTimer();
  //std::cout << timer->GetElapsedTime() << std::endl;
  timer->Delete();

}

vtkImageData* vtkCudaCT2USSimulation::GetOutput()
{
  return this->UsOutput;
}

vtkImageData* vtkCudaCT2USSimulation::GetOutput(int i)
{
  switch (i)
  {
  case 0:
    return this->UsOutput;
  case 1:
    return this->TransOutput;
  case 2:
    return this->ReflOutput;
  case 3:
    return this->DensOutput;
  default:
    return 0;
  }
}

void vtkCudaCT2USSimulation::SetOutputResolution(int x, int y, int z)
{

  //if we are 2D, treat us as such (make sure z is still depth)
  if (z == 1)
  {
    this->Information.Resolution.x = x;
    this->Information.Resolution.y = z;
    this->Information.Resolution.z = y;
  }
  else
  {
    this->Information.Resolution.x = x;
    this->Information.Resolution.y = y;
    this->Information.Resolution.z = z;
  }

  //create new output image buffers
  if (!this->DensOutput)
  {
    this->DensOutput = vtkImageData::New();
  }
  this->DensOutput->SetExtent(0, x - 1,
                              0, y - 1,
                              0, z - 1);
  this->DensOutput->SetOrigin(0, 0, 0);
  this->DensOutput->SetSpacing(1.0, 1.0, 1.0);
  this->Update();
  this->DensOutput->AllocateScalars(VTK_FLOAT, 1);

  if (!this->TransOutput)
  {
    this->TransOutput = vtkImageData::New();
  }
  this->TransOutput->SetExtent(0, x - 1,
                               0, y - 1,
                               0, z - 1);
  this->TransOutput->SetOrigin(0, 0, 0);
  this->TransOutput->SetSpacing(1.0, 1.0, 1.0);
  this->Update();
  this->TransOutput->AllocateScalars(VTK_FLOAT, 1);

  if (!this->ReflOutput)
  {
    this->ReflOutput = vtkImageData::New();
  }
  this->ReflOutput->SetExtent(0, x - 1,
                              0, y - 1,
                              0, z - 1);
  this->ReflOutput->SetOrigin(0, 0, 0);
  this->ReflOutput->SetSpacing(1.0, 1.0, 1.0);
  this->Update();
  this->ReflOutput->AllocateScalars(VTK_FLOAT, 1);

  //create a new simulated image
  if (!this->UsOutput)
  {
    this->UsOutput = vtkImageData::New();
  }
  this->UsOutput->SetExtent(0, x - 1,
                            0, y - 1,
                            0, z - 1);
  this->UsOutput->SetOrigin(0, 0, 0);
  this->UsOutput->SetSpacing(1.0, 1.0, 1.0);
  this->Update();
  this->UsOutput->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
}

void vtkCudaCT2USSimulation::SetLogarithmicScaleFactor(float factor)
{
  this->Information.a = factor;
}

void vtkCudaCT2USSimulation::SetTotalReflectionThreshold(float threshold)
{
  this->Information.reflectionThreshold = threshold;
}

void vtkCudaCT2USSimulation::SetLinearCombinationAlpha(float a)
{
  this->Information.alpha = a;
}

void vtkCudaCT2USSimulation::SetLinearCombinationBeta(float b)
{
  this->Information.beta = b;
}

void vtkCudaCT2USSimulation::SetLinearCombinationBias(float bias)
{
  this->Information.bias = bias;
}

void vtkCudaCT2USSimulation::SetProbeWidth(float xWidth, float yWidth)
{
  this->Information.probeWidth.x = xWidth;
  this->Information.probeWidth.y = yWidth;
}

void vtkCudaCT2USSimulation::SetFanAngle(float xAngle, float yAngle)
{
  this->Information.fanAngle.x = xAngle * 3.1415926 / 180.0;
  this->Information.fanAngle.y = yAngle * 3.1415926 / 180.0;
}

void vtkCudaCT2USSimulation::SetNearClippingDepth(float depth)
{
  this->Information.StartDepth = depth;
}

void vtkCudaCT2USSimulation::SetFarClippingDepth(float depth)
{
  this->Information.EndDepth = depth;
}

void vtkCudaCT2USSimulation::SetDensityScaleModel(float scale, float offset)
{
  this->Information.hounsfieldScale = scale;
  this->Information.hounsfieldOffset = offset;
}

float vtkCudaCT2USSimulation::GetCrossCorrelation()
{
  if (this->AutoGenerateLinearCombination)
  {
    return this->Information.crossCorrelation;
  }
  return -1.0f;
}

vtkCudaCT2USSimulation::vtkCudaCT2USSimulation()
{
  this->UsOutput = 0;
  this->DensOutput = 0;
  this->TransOutput = 0;
  this->ReflOutput = 0;
  this->UsTransform = 0;
  this->Information.a = 1.0f;
  this->Alpha = 0.5f;
  this->Beta = 0.5f;
  this->Bias = 0.0f;
  this->Information.StartDepth = 0.0f;
  this->Information.EndDepth = 0.0f;
  this->Information.VolumeSize.x = this->Information.VolumeSize.y = this->Information.VolumeSize.z = 0;
  this->Information.Resolution.x = this->Information.Resolution.y = this->Information.Resolution.z = 0;
  this->Information.probeWidth.x = 0.0f;
  this->Information.probeWidth.y = 0.0f;
  this->Information.fanAngle.x = 0.0f;
  this->Information.fanAngle.y = 0.0f;
  this->Information.reflectionThreshold = 1000000.0;
  this->Information.hounsfieldScale = 1.0;
  this->Information.hounsfieldOffset = -1024.0;
  this->Information.optimalParam = false;

  this->Caster = vtkImageCast::New();
  this->Caster->SetOutputScalarTypeToFloat();
  this->Caster->ClampOverflowOn();

  this->AutoGenerateLinearCombination = false;
}

vtkCudaCT2USSimulation::~vtkCudaCT2USSimulation()
{
  if (this->UsOutput)
  {
    this->UsOutput->Delete();
    this->TransOutput->Delete();
    this->ReflOutput->Delete();
    this->DensOutput->Delete();
  }
  this->Caster->Delete();
}

float vtkCudaCT2USSimulation::GetLinearCombinationAlpha()
{
  return this->Information.alpha;
}
float vtkCudaCT2USSimulation::GetLinearCombinationBeta()
{
  return this->Information.beta;
}

float vtkCudaCT2USSimulation::GetLinearCombinationBias()
{
  return this->Information.bias;
}