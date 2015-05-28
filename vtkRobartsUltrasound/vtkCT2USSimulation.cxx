#include "vtkCT2USSimulation.h"
#include "vtkObjectFactory.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include <vtkVersion.h>

struct vtkCT2USSimulationInformation {

  // The resolution of the rStartering screen.
  int Resolution[3];
  int VolumeSize[3];
  double Spacing[3];

  //the world to volume transformation
  double WorldToVolume[16];

  //the pose and structure of the US
  double UltraSoundToWorld[16];
  double probeWidth[2];
  double fanAngle[2];
  double StartDepth;
  double EndDepth;
  
  //input scaling to Hounsfield units
  double HounsfieldScale;
  double HounsfieldOffset;

  //output scaling parameters
  double a;
  double alpha;
  double beta;
  double bias;

  //threshold value for total reflection (in Hounsfield Units)
  double ReflectionThreshold;

};

vtkStandardNewMacro(vtkCT2USSimulation);

void vtkCT2USSimulation::SetTransform( vtkTransform * t ){
  if( this->usTransform == t ) return;
  if( this->usTransform ) this->usTransform->UnRegister(this);
  this->usTransform = t;
  if( this->usTransform ) this->usTransform->Register(this);
  this->Modified();
}

#include "vtkTimerLog.h"

void vtkCT2USSimulation::ThreadedExecute(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads){

  //if we are a valid ray, simulate the ultrasound
  int idInUse = threadId;
  while( idInUse < this->CT2USInformation->Resolution[0]*this->CT2USInformation->Resolution[1] ){

    //find the index in the image
    int index[2];
    index[0] = idInUse % this->CT2USInformation->Resolution[0];
    index[1] = idInUse / this->CT2USInformation->Resolution[0];

    //find the normalized indices
    double normIndex[2];
    normIndex[0] = (double) (index[0]+index[0]) / (double) this->CT2USInformation->Resolution[0] - 1.0;
    normIndex[1] = (double) (index[1]+index[1]) / (double) this->CT2USInformation->Resolution[1] - 1.0;
  
    //find the vector for the ray through the volume
    double rayStart[3];
    double rayInc[3];
    this->FindVectors( normIndex, rayStart, rayInc );

    //TODO sample along the ray
    this->SampleAlongRay( index, rayStart, rayInc, this->CT2USInformation->Resolution[2],
      this->CT2USInformation->Resolution[0], this->CT2USInformation->Resolution[1],
      (char*) outData->GetScalarPointer() );

    //move to the next possible ray
    idInUse += numThreads;
  }
}

void vtkCT2USSimulation::SampleAlongRay(const int index[2], double rayStart[2], const double rayInc[2], const int numStepsToTake,
                    const int xResolution, const int yResolution, char* const outputUltrasound){

  //collect parameters (makes code a little cleaner)
  const int* volumeSize = this->CT2USInformation->VolumeSize;
  const double* spacing = this->CT2USInformation->Spacing;
  int actIndex = 3*(index[0] + index[1] * xResolution);
  int indexInc = 3*xResolution*yResolution;
  bool isValid = (index[1] < yResolution);
  const double& threshold = this->CT2USInformation->ReflectionThreshold;
  const double densitySlope = 0.5*0.00025;
  const double densityIntercept = 512.0*0.00025;
  const double& HounsFieldScale = this->CT2USInformation->HounsfieldScale;
  const double& HounsFieldOffset = this->CT2USInformation->HounsfieldOffset;
  const double& alpha = this->CT2USInformation->alpha;
  const double& beta = this->CT2USInformation->beta;
  const double& bias = this->CT2USInformation->bias;

  double directionMag = sqrt( rayInc[0]*rayInc[0] + rayInc[1]*rayInc[1] + rayInc[2]*rayInc[2] );
  double worldDirectionMag = 2.0 * sqrt( rayInc[0]*rayInc[0]/(spacing[0]*spacing[0]) +
                       rayInc[1]*rayInc[1]/(spacing[1]*spacing[1]) +
                       rayInc[2]*rayInc[2]/(spacing[2]*spacing[2]) );

  //set up running accumulators
  double transmission = 1.0f;

  //set up output scaling parameters
  double multiplier = this->CT2USInformation->a;
  double divisor = 1.0 / log(1.0+multiplier);

  for(unsigned int numStepsTaken = 0; numStepsTaken < numStepsToTake; numStepsTaken++){

    //create default values for the sample point
    double density = 0.0f;
    double transmissionLost = 1.0f;
    double pointReflection = 0.0f;

    double attenuation = 0.0f;

    if(!(rayStart[0] < 0.0f || rayStart[1] < 0.0f || rayStart[2] < 0.0f ||
       rayStart[0] > (double)(volumeSize[0] - 1) ||
       rayStart[1] > (double)(volumeSize[1] - 1) ||
       rayStart[2] > (double)(volumeSize[2] - 1) )){

      //get the attenuation and gradient of the attenuation in Hounsfield units
      double CTVal;
      double gradient[3];
      this->GetCTValue(rayStart,CTVal,gradient);
      attenuation = HounsFieldScale* CTVal + HounsFieldOffset;
      gradient[0] *= HounsFieldScale;
      gradient[1] *= HounsFieldScale;
      gradient[2] *= HounsFieldScale;
      double gradMagSquared = gradient[0]*gradient[0] + gradient[1]*gradient[1] + gradient[2]*gradient[2];
      double gradMag = sqrt( gradMagSquared );

      //calculate the reflection, density and transmission at this sample point
      transmissionLost = 0.0;
      if( gradMag < threshold ){
        transmissionLost = 1.0f - gradMagSquared * worldDirectionMag / (4.0 * attenuation * attenuation);
        if( transmissionLost < 0.0 ) transmissionLost = 0.0;
        if( transmissionLost > 1.0 ) transmissionLost = 1.0;
      }
      pointReflection  = transmission * -(rayInc[0]*gradient[0] + rayInc[1]*gradient[1] + rayInc[2]*gradient[2]) * gradMag / ( 4.0 * attenuation * attenuation * directionMag );
      density          = (transmission > 0.0f) ? densitySlope * attenuation + densityIntercept : 0.0;

    }

    //scale the point reflection
    pointReflection = log( 1 + multiplier * pointReflection ) * divisor;
    if( pointReflection < 0.0 )
      pointReflection = 0.0;
    else
      if( pointReflection > 1.0 )
        pointReflection = 1.0;
    
    //create the output image
    double outputPixel = 255.0*(alpha*density+beta*pointReflection+bias);
    if( outputPixel < 0.0 ) outputPixel = 0.0;
    if( outputPixel > 255.0 ) outputPixel = 255.0;
    outputUltrasound[actIndex+0] = outputPixel;
    outputUltrasound[actIndex+1] = outputPixel;
    outputUltrasound[actIndex+2] = outputPixel;

    //update the running values
    transmission *= transmissionLost;

    //update the sampling location
    actIndex += indexInc;
    rayStart[0] += rayInc[0];
    rayStart[1] += rayInc[1];
    rayStart[2] += rayInc[2];

  }


}

void vtkCT2USSimulation::FindVectors(const double normIndex[2], double rayStart[2], double rayInc[2]){

  //find the US coordinates of this particular beam's Start point
  double usStart[3];
  usStart[0] = tan( this->CT2USInformation->fanAngle[0] * normIndex[0] );
  usStart[1] = tan( this->CT2USInformation->fanAngle[1] * normIndex[1] );
  usStart[2] = sqrt( this->CT2USInformation->StartDepth * this->CT2USInformation->StartDepth / 
              ( 1.0 + usStart[0]*usStart[0] + usStart[1]*usStart[1]) );
  usStart[0] = 0.5 * this->CT2USInformation->probeWidth[0]*normIndex[0] + usStart[0]*usStart[2];
  usStart[1] = 0.5 * this->CT2USInformation->probeWidth[1]*normIndex[1] + usStart[1]*usStart[2];

  //find the Start vector in world coordinates
  double worldStart[4];
  worldStart[0] = this->CT2USInformation->UltraSoundToWorld[ 0] * usStart[0] + this->CT2USInformation->UltraSoundToWorld[ 1] * usStart[1] + this->CT2USInformation->UltraSoundToWorld[ 2] * usStart[2] + this->CT2USInformation->UltraSoundToWorld[ 3];
  worldStart[1] = this->CT2USInformation->UltraSoundToWorld[ 4] * usStart[0] + this->CT2USInformation->UltraSoundToWorld[ 5] * usStart[1] + this->CT2USInformation->UltraSoundToWorld[ 6] * usStart[2] + this->CT2USInformation->UltraSoundToWorld[ 7];
  worldStart[2] = this->CT2USInformation->UltraSoundToWorld[ 8] * usStart[0] + this->CT2USInformation->UltraSoundToWorld[ 9] * usStart[1] + this->CT2USInformation->UltraSoundToWorld[10] * usStart[2] + this->CT2USInformation->UltraSoundToWorld[11];
  worldStart[3] = this->CT2USInformation->UltraSoundToWorld[12] * usStart[0] + this->CT2USInformation->UltraSoundToWorld[13] * usStart[1] + this->CT2USInformation->UltraSoundToWorld[14] * usStart[2] + this->CT2USInformation->UltraSoundToWorld[15];
  worldStart[0] /= worldStart[3]; 
  worldStart[1] /= worldStart[3]; 
  worldStart[2] /= worldStart[3];

  //transform the Start into volume co-ordinates
  rayStart[0]   = this->CT2USInformation->WorldToVolume[ 0]*worldStart[0] + this->CT2USInformation->WorldToVolume[ 1]*worldStart[1] + this->CT2USInformation->WorldToVolume[ 2]*worldStart[2] + this->CT2USInformation->WorldToVolume[ 3];
  rayStart[1]   = this->CT2USInformation->WorldToVolume[ 4]*worldStart[0] + this->CT2USInformation->WorldToVolume[ 5]*worldStart[1] + this->CT2USInformation->WorldToVolume[ 6]*worldStart[2] + this->CT2USInformation->WorldToVolume[ 7];
  rayStart[2]   = this->CT2USInformation->WorldToVolume[ 8]*worldStart[0] + this->CT2USInformation->WorldToVolume[ 9]*worldStart[1] + this->CT2USInformation->WorldToVolume[10]*worldStart[2] + this->CT2USInformation->WorldToVolume[11];
  worldStart[3] = this->CT2USInformation->WorldToVolume[12]*worldStart[0] + this->CT2USInformation->WorldToVolume[13]*worldStart[1] + this->CT2USInformation->WorldToVolume[14]*worldStart[2] + this->CT2USInformation->WorldToVolume[15];
  rayStart[0] /= worldStart[3];
  rayStart[1] /= worldStart[3];
  rayStart[2] /= worldStart[3];
  
  //find the US coordinates of this particular beam's Start point
  double usEnd[3];
  usEnd[0] = tan( this->CT2USInformation->fanAngle[0] * normIndex[0] );
  usEnd[1] = tan( this->CT2USInformation->fanAngle[1] * normIndex[1] );
  usEnd[2] = sqrt( (this->CT2USInformation->EndDepth * this->CT2USInformation->EndDepth) / 
            ( 1.0 + usEnd[0]*usEnd[0] + usEnd[1]*usEnd[1]) );
  usEnd[0] = 0.5 * this->CT2USInformation->probeWidth[0] * normIndex[0] + usEnd[0]*usEnd[2];
  usEnd[1] = 0.5 * this->CT2USInformation->probeWidth[1] * normIndex[1] + usEnd[1]*usEnd[2];
  
  //find the End vector in world coordinates
  double worldEnd[4];
  worldEnd[0] = this->CT2USInformation->UltraSoundToWorld[ 0] * usEnd[0] + this->CT2USInformation->UltraSoundToWorld[ 1] * usEnd[1] + this->CT2USInformation->UltraSoundToWorld[ 2] * usEnd[2] + this->CT2USInformation->UltraSoundToWorld[ 3];
  worldEnd[1] = this->CT2USInformation->UltraSoundToWorld[ 4] * usEnd[0] + this->CT2USInformation->UltraSoundToWorld[ 5] * usEnd[1] + this->CT2USInformation->UltraSoundToWorld[ 6] * usEnd[2] + this->CT2USInformation->UltraSoundToWorld[ 7];
  worldEnd[2] = this->CT2USInformation->UltraSoundToWorld[ 8] * usEnd[0] + this->CT2USInformation->UltraSoundToWorld[ 9] * usEnd[1] + this->CT2USInformation->UltraSoundToWorld[10] * usEnd[2] + this->CT2USInformation->UltraSoundToWorld[11];
  worldEnd[3] = this->CT2USInformation->UltraSoundToWorld[12] * usEnd[0] + this->CT2USInformation->UltraSoundToWorld[13] * usEnd[1] + this->CT2USInformation->UltraSoundToWorld[14] * usEnd[2] + this->CT2USInformation->UltraSoundToWorld[15];
  worldEnd[0] /= worldEnd[3]; 
  worldEnd[1] /= worldEnd[3]; 
  worldEnd[2] /= worldEnd[3];

  //transform the End into volume co-ordinates
  double rayEnd[3];
  rayEnd[0]   = this->CT2USInformation->WorldToVolume[ 0]*worldEnd[0] + this->CT2USInformation->WorldToVolume[ 1]*worldEnd[1] + this->CT2USInformation->WorldToVolume[ 2]*worldEnd[2] + this->CT2USInformation->WorldToVolume[ 3];
  rayEnd[1]   = this->CT2USInformation->WorldToVolume[ 4]*worldEnd[0] + this->CT2USInformation->WorldToVolume[ 5]*worldEnd[1] + this->CT2USInformation->WorldToVolume[ 6]*worldEnd[2] + this->CT2USInformation->WorldToVolume[ 7];
  rayEnd[2]   = this->CT2USInformation->WorldToVolume[ 8]*worldEnd[0] + this->CT2USInformation->WorldToVolume[ 9]*worldEnd[1] + this->CT2USInformation->WorldToVolume[10]*worldEnd[2] + this->CT2USInformation->WorldToVolume[11];
  worldEnd[3] = this->CT2USInformation->WorldToVolume[12]*worldEnd[0] + this->CT2USInformation->WorldToVolume[13]*worldEnd[1] + this->CT2USInformation->WorldToVolume[14]*worldEnd[2] + this->CT2USInformation->WorldToVolume[15];
  rayEnd[0] /= worldEnd[3];
  rayEnd[1] /= worldEnd[3];
  rayEnd[2] /= worldEnd[3];

  //calculate the increment vector
  rayInc[0] = (rayStart[0]-rayEnd[0]) / this->CT2USInformation->Resolution[2];
  rayInc[1] = (rayStart[1]-rayEnd[1]) / this->CT2USInformation->Resolution[2];
  rayInc[2] = (rayStart[2]-rayEnd[2]) / this->CT2USInformation->Resolution[2];

}

void vtkCT2USSimulation::SetOutputResolution(int x, int y, int z){

  //if we are 2D, treat us as such (make sure z is still depth)
  if( z == 1){
    this->CT2USInformation->Resolution[0] = x;
    this->CT2USInformation->Resolution[1] = z;
    this->CT2USInformation->Resolution[2] = y;
  }else{
    this->CT2USInformation->Resolution[0] = x;
    this->CT2USInformation->Resolution[1] = y;
    this->CT2USInformation->Resolution[2] = z;
  }
  
}

struct vtkCT2USSimulationThreadStruct
{
  vtkCT2USSimulation *Filter;
  vtkImageData   *inData;
  vtkImageData   *outData;
};

VTK_THREAD_RETURN_TYPE vtkCT2USSimulationThreadedExecute( void *arg )
{
  vtkCT2USSimulationThreadStruct *str;
  int threadId, threadCount;
  
  threadId = static_cast<vtkMultiThreader::ThreadInfo *>(arg)->ThreadID;
  threadCount = static_cast<vtkMultiThreader::ThreadInfo *>(arg)->NumberOfThreads;
  
  str = static_cast<vtkCT2USSimulationThreadStruct *>
  (static_cast<vtkMultiThreader::ThreadInfo *>(arg)->UserData);

  str->Filter->ThreadedExecute(str->inData, str->outData, threadId, threadCount);

  return VTK_THREAD_RETURN_VALUE;
}

int vtkCT2USSimulation::RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector){

  // get the info objects
  vtkImageData *inData = vtkImageData::SafeDownCast(this->GetInput());
  vtkImageData *outData = this->GetOutput();
  if( !inData || !outData || !this->usTransform ) return -1;

  // get the output extent and reallocate the output buffer if necessary
  int extent[6];
  outData->GetExtent(extent);
  bool reallocateScalars = !(outData->GetNumberOfScalarComponents() == 3);
  if( this->CT2USInformation->Resolution[1] != 1 ){
    for (int idx = 0; idx < 3; ++idx){
      if(extent[2*idx] != 0 || extent[2*idx+1] != this->CT2USInformation->Resolution[idx]-1){
        extent[2*idx] = 0;
        extent[2*idx+1] = this->CT2USInformation->Resolution[idx]-1;
        reallocateScalars = true;
      }
    }
  }else{
    if(extent[0] != 0 || extent[1] != this->CT2USInformation->Resolution[0]-1){
      extent[0] = 0;
      extent[1] = this->CT2USInformation->Resolution[0]-1;
      reallocateScalars = true;
    }
    if(extent[2] != 0 || extent[3] != this->CT2USInformation->Resolution[2]-1){
      extent[2] = 0;
      extent[3] = this->CT2USInformation->Resolution[2]-1;
      reallocateScalars = true;
    }
    if(extent[4] != 0 || extent[5] != this->CT2USInformation->Resolution[1]-1){
      extent[4] = 0;
      extent[5] = this->CT2USInformation->Resolution[1]-1;
      reallocateScalars = true;
    }
  }
  if(reallocateScalars){
    outData->SetExtent(extent);

#if (VTK_MAJOR_VERSION <= 5)
	//outData->SetWholeExtent(extent);
	outData->SetScalarTypeToUnsignedChar();
	outData->SetNumberOfScalarComponents(3);
	outData->AllocateScalars();
#else
	outData->AllocateScalars(VTK_UNSIGNED_CHAR, 3);
#endif

  }

  //create an interpolating function out of the volume
  this->Interpolator->SetVolume(inData);

  //output the ultrasound location information to the information holder
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      this->CT2USInformation->UltraSoundToWorld[i*4+j] = this->usTransform->GetMatrix()->GetElement(i,j);
    }
  }

  //get the volume size from the input
  double inputOrigin[3];
  double spacing[3];
  int inputExtent[6];

  inData->GetExtent(inputExtent);
  inData->GetOrigin(inputOrigin);
  inData->GetSpacing(spacing);

  //set the volume dimensions
  this->CT2USInformation->VolumeSize[0] = inputExtent[1]-inputExtent[0]+1;
  this->CT2USInformation->VolumeSize[1] = inputExtent[3]-inputExtent[2]+1;
  this->CT2USInformation->VolumeSize[2] = inputExtent[5]-inputExtent[4]+1;

  //get the spacing information from the input
  this->CT2USInformation->Spacing[0] = 0.5 / spacing[0];
  this->CT2USInformation->Spacing[1] = 0.5 / spacing[1];
  this->CT2USInformation->Spacing[2] = 0.5 / spacing[2];

  // Compute the origin of the extent the volume origin is at voxel (0,0,0)
  // but we want to consider (0,0,0) in voxels to be at
  // (inputExtent[0], inputExtent[2], inputExtent[4]).
  double extentOrigin[3];
  extentOrigin[0] = inputOrigin[0] + inputExtent[0]*spacing[0];
  extentOrigin[1] = inputOrigin[1] + inputExtent[2]*spacing[1];
  extentOrigin[2] = inputOrigin[2] + inputExtent[4]*spacing[2];

  // Create a transform that will account for the scaling and translation of
  // the scalar data. The is the volume to voxels matrix.
  this->VoxelsTransform->Identity();
  this->VoxelsTransform->Translate( extentOrigin[0], extentOrigin[1], extentOrigin[2] );
  this->VoxelsTransform->Scale( spacing[0], spacing[1], spacing[2] );
  this->Interpolator->SetTransform(this->VoxelsTransform);
  
  // Now we actually have the world to voxels matrix - copy it out
  this->WorldToVoxelsMatrix->DeepCopy( VoxelsTransform->GetMatrix() );
  this->WorldToVoxelsMatrix->Invert();

  //output the CT location information to the information holder
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      this->CT2USInformation->WorldToVolume[i*4+j] = WorldToVoxelsMatrix->GetElement(i,j);
    }
  }
  
  //set up the threader
  vtkCT2USSimulationThreadStruct str;
  str.Filter = this;
  str.inData = inData;
  str.outData = outData;
  this->Threader->SetNumberOfThreads(this->NumberOfThreads);
  this->Threader->SetSingleMethod(vtkCT2USSimulationThreadedExecute, &str);  

  // always shut off debugging to avoid threading problems with GetMacros
  int debug = this->Debug;
  this->Debug = 0;
  this->Threader->SingleMethodExecute();
  this->Debug = debug;

  return 1;
}

void vtkCT2USSimulation::GetCTValue(double i[3], double& f, double g[3]){
  f = this->Interpolator->FunctionValue(i);
  this->Interpolator->FunctionGradient(i,g);
}

void vtkCT2USSimulation::SetLogarithmicScaleFactor(double factor){
  this->CT2USInformation->a = factor;
}

void vtkCT2USSimulation::SetTotalReflectionThreshold(double threshold){
  this->CT2USInformation->ReflectionThreshold = threshold;
}

void vtkCT2USSimulation::SetLinearCombinationAlpha(double a){
  this->CT2USInformation->alpha = a;
}

void vtkCT2USSimulation::SetLinearCombinationBeta(double b){
  this->CT2USInformation->beta = b;
}

void vtkCT2USSimulation::SetLinearCombinationBias(double bias){
  this->CT2USInformation->bias = bias;
}

void vtkCT2USSimulation::SetProbeWidth(double xWidth, double yWidth){
  this->CT2USInformation->probeWidth[0] = xWidth;
  this->CT2USInformation->probeWidth[1] = yWidth;
}

void vtkCT2USSimulation::SetFanAngle(double xAngle, double yAngle){
  this->CT2USInformation->fanAngle[0] = xAngle * 3.1415926 / 180.0;
  this->CT2USInformation->fanAngle[1] = yAngle * 3.1415926 / 180.0;
}

void vtkCT2USSimulation::SetNearClippingDepth(double depth){
  this->CT2USInformation->StartDepth = depth;
}

void vtkCT2USSimulation::SetFarClippingDepth(double depth){
  this->CT2USInformation->EndDepth = depth;
}

void vtkCT2USSimulation::SetDensityScaleModel(double scale, double offset){
  this->CT2USInformation->HounsfieldScale = scale;
  this->CT2USInformation->HounsfieldOffset = offset;
}

vtkCT2USSimulation::vtkCT2USSimulation(){
  this->usTransform = 0;
  this->CT2USInformation = new vtkCT2USSimulationInformation();
  this->CT2USInformation->a = 1.0;
  this->CT2USInformation->alpha = 0.5;
  this->CT2USInformation->beta = 0.5;
  this->CT2USInformation->bias = 0.0;
  this->CT2USInformation->StartDepth = 0.0;
  this->CT2USInformation->EndDepth = 0.0;
  this->CT2USInformation->VolumeSize[0] = this->CT2USInformation->VolumeSize[1] = this->CT2USInformation->VolumeSize[2] = 0;
  this->CT2USInformation->Resolution[0] = this->CT2USInformation->Resolution[1] = this->CT2USInformation->Resolution[2] = 0;
  this->CT2USInformation->probeWidth[0] = 0.0;
  this->CT2USInformation->probeWidth[1] = 0.0;
  this->CT2USInformation->fanAngle[0] = 0.0;
  this->CT2USInformation->fanAngle[1] = 0.0;
  this->CT2USInformation->ReflectionThreshold = 1000000.0;
  this->CT2USInformation->HounsfieldScale = 1.0;
  this->CT2USInformation->HounsfieldOffset = -1024.0;

  this->WorldToVoxelsMatrix = vtkMatrix4x4::New();
  this->VoxelsTransform = vtkTransform::New();

  this->Threader = vtkMultiThreader::New();
  this->NumberOfThreads = this->Threader->GetNumberOfThreads();

  this->Interpolator = vtkImplicitVolume::New();
}

vtkCT2USSimulation::~vtkCT2USSimulation(){
  
  //clean up the temporary transforms
  VoxelsTransform->Delete();
  WorldToVoxelsMatrix->Delete();

  if( this->usTransform ) this->usTransform->UnRegister(this);
  delete this->CT2USInformation;
  this->Threader->Delete();
  this->Interpolator->Delete();
}
