#include "vtkCudaVoxelClassifier.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"

#include "vector_types.h"

#include <vtkVersion.h> // For VTK_MAJOR_VERSION

vtkStandardNewMacro(vtkCudaVoxelClassifier);

vtkCudaVoxelClassifier::vtkCudaVoxelClassifier(){

  //configure the input ports
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfInputConnections(0,1);

  //initialize additional inputs to 0
  this->ClippingPlanes = 0;
  this->KeyholePlanes = 0;
  this->PrimaryFunction = 0;
  this->KeyholeFunction = 0;

  this->TextureSize = 512;

}

vtkCudaVoxelClassifier::~vtkCudaVoxelClassifier(){
  if(this->PrimaryFunction) this->PrimaryFunction->UnRegister(this);
  if(this->KeyholeFunction) this->KeyholeFunction->UnRegister(this);
  if(this->ClippingPlanes) this->ClippingPlanes->UnRegister(this);
  if(this->KeyholePlanes) this->KeyholePlanes->UnRegister(this);
}

//------------------------------------------------------------
//Commands for vtkCudaObject compatibility

void vtkCudaVoxelClassifier::Reinitialize(int withData){
  //TODO
}

void vtkCudaVoxelClassifier::Deinitialize(int withData){
}


//------------------------------------------------------------
int vtkCudaVoxelClassifier::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkDataObject::SetPointDataActiveScalarInfo(outputInfo, VTK_SHORT, 1);
  return 1;
}

int vtkCudaVoxelClassifier::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
  vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
  
  inputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inData->GetExtent(),6);

  return 1;
}

int vtkCudaVoxelClassifier::RequestData(vtkInformation *request, 
              vtkInformationVector **inputVector, 
              vtkInformationVector *outputVector){

  vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
  
  //exit and throw error message if something is wrong with input configuration
  if( !inData ){
    vtkErrorMacro(<<"This filter requires an input image.");
    return -1;
  }
  if( inData->GetScalarType() != VTK_FLOAT ){
    vtkErrorMacro(<<"The input must be of type float.");
    return -1;
  }
  if( !(this->PrimaryFunction) ){
    vtkErrorMacro(<<"There must be a primary transfer function.");
    return -1;
  }

  //figure out the extent of the output
#if (VTK_MAJOR_VERSION <= 5)
  outData->SetScalarTypeToShort();
  outData->SetNumberOfScalarComponents(1);
  outData->SetExtent( inData->GetExtent() );
  outData->SetWholeExtent( inData->GetExtent() );
  outData->SetOrigin( inData->GetOrigin() );
  outData->SetSpacing( inData->GetSpacing() );
  outData->AllocateScalars();
#else
  outData->SetExtent( inData->GetExtent() );
  outData->SetOrigin( inData->GetOrigin() );
  outData->SetSpacing( inData->GetSpacing() );
  outData->AllocateScalars(VTK_SHORT, 1);
#endif
  
  //update planes
  this->ComputeMatrices( inData );
  this->FigurePlanes( this->ClippingPlanes, this->ClassifierInfo.ClippingPlanes,
            &(this->ClassifierInfo.NumberOfClippingPlanes) );
  this->FigurePlanes( this->KeyholePlanes, this->ClassifierInfo.KeyholePlanes,
            &(this->ClassifierInfo.NumberOfKeyholePlanes) );

  //get the range of the input
  double scalarRange[4];
  inData->GetPointData()->GetScalars()->GetRange(scalarRange,0);
  inData->GetPointData()->GetScalars()->GetRange(scalarRange+2,1);
  double functionRange[] = {  this->PrimaryFunction->getMinIntensity(), this->PrimaryFunction->getMaxIntensity(), 
                this->PrimaryFunction->getMinGradient(), this->PrimaryFunction->getMaxGradient() };
  if( this->KeyholeFunction ){
    double kfunctionRange[] = {  this->KeyholeFunction->getMinIntensity(), this->KeyholeFunction->getMaxIntensity(), 
                  this->KeyholeFunction->getMinGradient(), this->KeyholeFunction->getMaxGradient() };
    functionRange[0] = (kfunctionRange[0] < functionRange[0] ) ? kfunctionRange[0] : functionRange[0];
    functionRange[1] = (kfunctionRange[1] > functionRange[1] ) ? kfunctionRange[1] : functionRange[1];
    functionRange[2] = (kfunctionRange[2] < functionRange[2] ) ? kfunctionRange[2] : functionRange[2];
    functionRange[3] = (kfunctionRange[3] > functionRange[3] ) ? kfunctionRange[3] : functionRange[3];
  }
  double minIntensity1 = (scalarRange[0] > functionRange[0] ) ? scalarRange[0] : functionRange[0];
  double maxIntensity1 = (scalarRange[1] < functionRange[1] ) ? scalarRange[1] : functionRange[1];
  double minIntensity2 = (scalarRange[2] > functionRange[2] ) ? scalarRange[2] : functionRange[2];
  double maxIntensity2 = (scalarRange[3] < functionRange[3] ) ? scalarRange[3] : functionRange[3];
  
  //update information container
  this->ClassifierInfo.Intensity1Low = minIntensity1;
  this->ClassifierInfo.Intensity1Multiplier = 1.0 / ( maxIntensity1 - minIntensity1 );
  this->ClassifierInfo.Intensity2Low = minIntensity2;
  this->ClassifierInfo.Intensity2Multiplier = 1.0 / ( maxIntensity2 - minIntensity2 );
  inData->GetDimensions( this->ClassifierInfo.VolumeSize );
  this->ClassifierInfo.TextureSize = this->TextureSize;

  //update transfer functions
  short* PrimaryTexture = new short[this->TextureSize*this->TextureSize];
  memset( PrimaryTexture, 0, sizeof(short)*this->TextureSize*this->TextureSize );
  this->PrimaryFunction->GetClassifyTable( PrimaryTexture,this->TextureSize,this->TextureSize,minIntensity1,
                       maxIntensity1, 0, minIntensity2, maxIntensity2, 0, 0 );
  short* KeyholeTexture = new short[this->TextureSize*this->TextureSize];
  memset( KeyholeTexture, 0, sizeof(short)*this->TextureSize*this->TextureSize );
  if( this->KeyholeFunction && this->ClassifierInfo.NumberOfKeyholePlanes > 0 ){
    this->KeyholeFunction->GetClassifyTable( KeyholeTexture,this->TextureSize,this->TextureSize,minIntensity1,
                         maxIntensity1, 0, minIntensity2, maxIntensity2, 0, 0 );
  }

  //pass it over to the GPU
  this->ReserveGPU();
  CUDAalgo_classifyVoxels( (float*) inData->GetScalarPointer(), PrimaryTexture, KeyholeTexture, this->TextureSize,
               (short*) outData->GetScalarPointer(), this->ClassifierInfo, this->GetStream() );

  //deallocate temperaries
  delete[] PrimaryTexture;
  delete[] KeyholeTexture;

  return 1;
}
//------------------------------------------------------------

void vtkCudaVoxelClassifier::SetFunction(vtkCuda2DTransferFunction* func){
  if( this->PrimaryFunction != func ){
    if(this->PrimaryFunction) this->PrimaryFunction->UnRegister(this);
    this->PrimaryFunction = func;
    func->Register(this);
    this->Modified();
  }
}
  
vtkCuda2DTransferFunction* vtkCudaVoxelClassifier::GetFunction(){
  return this->PrimaryFunction;
}

void vtkCudaVoxelClassifier::SetKeyholeFunction(vtkCuda2DTransferFunction* func){
  if( this->KeyholeFunction != func ){
    if(this->KeyholeFunction) this->KeyholeFunction->UnRegister(this);
    this->KeyholeFunction = func;
    func->Register(this);
    this->Modified();
  }
}

vtkCuda2DTransferFunction* vtkCudaVoxelClassifier::GetKeyholeFunction(){
  return this->KeyholeFunction;
}

//------------------------------------------------------------

vtkCxxSetObjectMacro(vtkCudaVoxelClassifier,KeyholePlanes,vtkPlaneCollection);

void vtkCudaVoxelClassifier::AddKeyholePlane(vtkPlane *plane){
  if (this->KeyholePlanes == NULL){
    this->KeyholePlanes = vtkPlaneCollection::New();
    this->KeyholePlanes->Register(this);
    this->KeyholePlanes->Delete();
  }

  this->KeyholePlanes->AddItem(plane);
  this->Modified();
}

void vtkCudaVoxelClassifier::RemoveKeyholePlane(vtkPlane *plane){
  if (this->KeyholePlanes == NULL) vtkErrorMacro(<< "Cannot remove Keyhole plane: mapper has none");
  this->KeyholePlanes->RemoveItem(plane);
  this->Modified();
}

void vtkCudaVoxelClassifier::RemoveAllKeyholePlanes(){
  if ( this->KeyholePlanes ) this->KeyholePlanes->RemoveAllItems();
}

void vtkCudaVoxelClassifier::SetKeyholePlanes(vtkPlanes *planes){
  vtkPlane *plane;
  if (!planes) return;

  int numPlanes = planes->GetNumberOfPlanes();

  this->RemoveAllKeyholePlanes();
  for (int i=0; i<numPlanes && i<6; i++){
    plane = vtkPlane::New();
    planes->GetPlane(i, plane);
    this->AddKeyholePlane(plane);
    plane->Delete();
  }
}
//------------------------------------------------------------

vtkCxxSetObjectMacro(vtkCudaVoxelClassifier,ClippingPlanes,vtkPlaneCollection);

void vtkCudaVoxelClassifier::AddClippingPlane(vtkPlane *plane){
  if (this->ClippingPlanes == NULL){
    this->ClippingPlanes = vtkPlaneCollection::New();
    this->ClippingPlanes->Register(this);
    this->ClippingPlanes->Delete();
  }

  this->ClippingPlanes->AddItem(plane);
  this->Modified();
}

void vtkCudaVoxelClassifier::RemoveClippingPlane(vtkPlane *plane){
  if (this->ClippingPlanes == NULL) vtkErrorMacro(<< "Cannot remove Clipping plane: mapper has none");
  this->ClippingPlanes->RemoveItem(plane);
  this->Modified();
}

void vtkCudaVoxelClassifier::RemoveAllClippingPlanes(){
  if ( this->ClippingPlanes ) this->ClippingPlanes->RemoveAllItems();
}

void vtkCudaVoxelClassifier::SetClippingPlanes(vtkPlanes *planes){
  vtkPlane *plane;
  if (!planes) return;

  int numPlanes = planes->GetNumberOfPlanes();

  this->RemoveAllClippingPlanes();
  for (int i=0; i<numPlanes && i<6; i++){
    plane = vtkPlane::New();
    planes->GetPlane(i, plane);
    this->AddClippingPlane(plane);
    plane->Delete();
  }
}

//-----------------------------------------------------------------------
void vtkCudaVoxelClassifier::FigurePlanes(vtkPlaneCollection* planes, float* planesArray, int* numberOfPlanes){

  //figure out the number of planes
  *numberOfPlanes = 0;
  if(planes) *numberOfPlanes = planes->GetNumberOfItems();
  
  //if we don't have a good number of planes, act as if we have none
  if( *numberOfPlanes != 6 ){
    *numberOfPlanes = 0;
    return;
  }

  double worldNormal[3];
  double worldOrigin[3];
  double volumeOrigin[4];

  //load the planes into the local buffer and then into the CUDA buffer, providing the required pointer at the end
  for(int i = 0; i < *numberOfPlanes; i++){
    vtkPlane* onePlane = planes->GetItem(i);
    
    onePlane->GetNormal(worldNormal);
    onePlane->GetOrigin(worldOrigin);

    planesArray[4*i] = worldNormal[0]*VoxelsToWorldMatrix[0]  + worldNormal[1]*VoxelsToWorldMatrix[4]  + worldNormal[2]*VoxelsToWorldMatrix[8];
    planesArray[4*i+1] = worldNormal[0]*VoxelsToWorldMatrix[1]  + worldNormal[1]*VoxelsToWorldMatrix[5]  + worldNormal[2]*VoxelsToWorldMatrix[9];
    planesArray[4*i+2] = worldNormal[0]*VoxelsToWorldMatrix[2]  + worldNormal[1]*VoxelsToWorldMatrix[6]  + worldNormal[2]*VoxelsToWorldMatrix[10];

    volumeOrigin[0] = worldOrigin[0]*WorldToVoxelsMatrix[0]  + worldOrigin[1]*WorldToVoxelsMatrix[1]  + worldOrigin[2]*WorldToVoxelsMatrix[2]  + WorldToVoxelsMatrix[3];
    volumeOrigin[1] = worldOrigin[0]*WorldToVoxelsMatrix[4]  + worldOrigin[1]*WorldToVoxelsMatrix[5]  + worldOrigin[2]*WorldToVoxelsMatrix[6]  + WorldToVoxelsMatrix[7];
    volumeOrigin[2] = worldOrigin[0]*WorldToVoxelsMatrix[8]  + worldOrigin[1]*WorldToVoxelsMatrix[9]  + worldOrigin[2]*WorldToVoxelsMatrix[10] + WorldToVoxelsMatrix[11];
    volumeOrigin[3] = worldOrigin[0]*WorldToVoxelsMatrix[12] + worldOrigin[1]*WorldToVoxelsMatrix[13] + worldOrigin[2]*WorldToVoxelsMatrix[14] + WorldToVoxelsMatrix[15];
    if ( volumeOrigin[3] != 1.0 ) { volumeOrigin[0] /= volumeOrigin[3]; volumeOrigin[1] /= volumeOrigin[3]; volumeOrigin[2] /= volumeOrigin[3]; }

    planesArray[4*i+3] = -(planesArray[4*i]*volumeOrigin[0] + planesArray[4*i+1]*volumeOrigin[1] + planesArray[4*i+2]*volumeOrigin[2]);
  }

}

void vtkCudaVoxelClassifier::ComputeMatrices(vtkImageData* inputData)
{
  //get the input origin, spacing and extents
  vtkImageData* img = inputData;
  double inputOrigin[3];
  double inputSpacing[3];
  int inputExtent[6];
  img->GetOrigin(inputOrigin);
  img->GetSpacing(inputSpacing);
  img->GetExtent(inputExtent);

  // Compute the origin of the extent the volume origin is at voxel (0,0,0)
  // but we want to consider (0,0,0) in voxels to be at
  // (inputExtent[0], inputExtent[2], inputExtent[4]).
  double extentOrigin[3];
  extentOrigin[0] = inputOrigin[0] + inputExtent[0]*inputSpacing[0];
  extentOrigin[1] = inputOrigin[1] + inputExtent[2]*inputSpacing[1];
  extentOrigin[2] = inputOrigin[2] + inputExtent[4]*inputSpacing[2];
    
  // Create a transform that will account for the scaling and translation of
  // the scalar data. The is the volume to voxels matrix.
  vtkTransform* VoxelsTransform = vtkTransform::New();
  VoxelsTransform->Identity();
  VoxelsTransform->Translate( extentOrigin[0], extentOrigin[1], extentOrigin[2] );
  VoxelsTransform->Scale( inputSpacing[0], inputSpacing[1], inputSpacing[2] );

  // Now concatenate the volume's matrix with this scalar data matrix (sending the result off as the voxels to world matrix)
  this->SetVoxelsToWorldMatrix( VoxelsTransform->GetMatrix() );

  // Invert the transform (sending the result off as the world to voxels matrix)
  vtkMatrix4x4* WorldToVoxelsMatrix = vtkMatrix4x4::New();
  WorldToVoxelsMatrix->DeepCopy( VoxelsTransform->GetMatrix() );
  WorldToVoxelsMatrix->Invert();
  this->SetWorldToVoxelsMatrix(WorldToVoxelsMatrix);

  VoxelsTransform->Delete();

}

void vtkCudaVoxelClassifier::SetWorldToVoxelsMatrix(vtkMatrix4x4* matrix){
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      this->WorldToVoxelsMatrix[i*4+j] = matrix->GetElement(i,j);
    }
  }
}

void vtkCudaVoxelClassifier::SetVoxelsToWorldMatrix(vtkMatrix4x4* matrix){
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      this->VoxelsToWorldMatrix[i*4+j] = matrix->GetElement(i,j);
    }
  }
}
