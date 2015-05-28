#include "vtkImageSeedFilter.h"

vtkImageSeedFilter::vtkImageSeedFilter(){
}

vtkImageSeedFilter::~vtkImageSeedFilter(){
  this->Clear();
}

void vtkImageSeedFilter::AddPointInVoxelCoordinates(double point[3], int component){
  if(component < 1 ){
    vtkErrorMacro(<<"Invalid component number");
    return;
  }

  this->pointsInVoxelX.push_back( point[0] );
  this->pointsInVoxelY.push_back( point[1] );
  this->pointsInVoxelZ.push_back( point[2] );
  this->pointsInVoxelW.push_back( component );
}

void vtkImageSeedFilter::AddPointInVolumeCoordinates(double point[3], int component){
  if(component < 1 ){
    vtkErrorMacro(<<"Invalid component number");
    return;
  }

  this->pointsInVolumeX.push_back( point[0] );
  this->pointsInVolumeY.push_back( point[1] );
  this->pointsInVolumeZ.push_back( point[2] );
  this->pointsInVolumeW.push_back( component );
}

void vtkImageSeedFilter::Clear(){
  this->pointsInVolumeW.clear();
  this->pointsInVolumeX.clear();
  this->pointsInVolumeY.clear();
  this->pointsInVolumeZ.clear();
  this->pointsInVoxelW.clear();
  this->pointsInVoxelX.clear();
  this->pointsInVoxelY.clear();
  this->pointsInVoxelZ.clear();
}

int vtkImageSeedFilter::RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector){

  // get the info objects
  vtkImageData *inData = vtkImageData::SafeDownCast(this->GetInput());
  vtkImageData *outData = this->GetOutput();
  if( !inData || !outData ) return -1;

  //size the out object to be the same as the input one
  int* extent = inData->GetExtent();
  int* dims = inData->GetDimensions();
  double* spacing = inData->GetSpacing();
  double* origin = inData->GetOrigin();
  outData->SetExtent( extent );
  outData->SetSpacing( inData->GetSpacing() );
  outData->SetOrigin( inData->GetOrigin() );
  outData->AllocateScalars(VTK_FLOAT, this->NumberOfComponents);
  float* outDataPtr = (float*) outData->GetScalarPointer();

  //cycle through input points in voxel co-ordinates
  std::vector<double>::iterator itX = this->pointsInVoxelX.begin();
  std::vector<double>::iterator itY = this->pointsInVoxelY.begin();
  std::vector<double>::iterator itZ = this->pointsInVoxelZ.begin();
  std::vector<int>::iterator itW = this->pointsInVoxelW.begin();
  int N = this->pointsInVoxelX.size();
  for(int i = 0; i < N; i++){

    //check if the component is in the volume
    bool in = true;
    in = in && ( *itW < this->NumberOfComponents );
    in = in && ( *itX >= extent[0] );
    in = in && ( *itX <= extent[1] );
    in = in && ( *itY >= extent[2] );
    in = in && ( *itY <= extent[3] );
    in = in && ( *itZ >= extent[4] );
    in = in && ( *itZ <= extent[5] );

    //if it is, add it
    int index = this->NumberOfComponents * ( ((int) (*itX+0.5)-extent[0]) + dims[0]*
                         ( ((int) (*itY+0.5)-extent[2]) + dims[1]*
                         ( ((int) (*itZ+0.5)-extent[2]) ) ) ) + *itW;
    outDataPtr[index] = 1.0f;

    //advance to the next point
    itX++; itY++; itZ++; itW++;
  }

  
  //cycle through input points in volume co-ordinates
  itX = this->pointsInVolumeX.begin();
  itY = this->pointsInVolumeY.begin();
  itZ = this->pointsInVolumeZ.begin();
  itW = this->pointsInVolumeW.begin();
  N = this->pointsInVolumeX.size();
  for(int i = 0; i < N; i++){

    //convert first to voxel co-ordinates
    double voxel[3];
    voxel[0] = (*itX - origin[0]) / spacing[0];
    voxel[1] = (*itX - origin[1]) / spacing[1];
    voxel[2] = (*itX - origin[2]) / spacing[2];

    //check if the component is in the volume
    bool in = true;
    in = in && ( *itW < this->NumberOfComponents );
    in = in && ( voxel[0] >= extent[0] );
    in = in && ( voxel[1] <= extent[1] );
    in = in && ( voxel[2] >= extent[2] );
    in = in && ( voxel[3] <= extent[3] );
    in = in && ( voxel[4] >= extent[4] );
    in = in && ( voxel[5] <= extent[5] );

    //if it is, add it
    int index = this->NumberOfComponents * ( ((int) (voxel[0]+0.5)-extent[0]) + dims[0]*
                         ( ((int) (voxel[1]+0.5)-extent[2]) + dims[1]*
                         ( ((int) (voxel[2]+0.5)-extent[2]) ) ) ) + *itW;
    outDataPtr[index] = 1.0f;

    //advance to the next point
    itX++; itY++; itZ++; itW++;
  }

  return 1;
}
