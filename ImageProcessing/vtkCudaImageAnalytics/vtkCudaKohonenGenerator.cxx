#include "vtkCudaKohonenGenerator.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "float.h"

#include "vtkPCAStatistics.h"
#include "vtkTable.h"
#include "vtkDoubleArray.h"
#include "vtkMath.h"

#include <vtkVersion.h> // For VTK_MAJOR_VERSION

vtkStandardNewMacro(vtkCudaKohonenGenerator);

vtkCudaKohonenGenerator::vtkCudaKohonenGenerator(){
  this->outExt[0] = 0;
  this->outExt[1] = 0;
  this->outExt[2] = 0;
  this->outExt[3] = 0;
  this->outExt[4] = 0;
  this->outExt[5] = 0;
  
  this->MeansAlphaSchedule = vtkPiecewiseFunction::New();
  this->MeansWidthSchedule = vtkPiecewiseFunction::New();
  this->VarsAlphaSchedule = vtkPiecewiseFunction::New();
  this->VarsWidthSchedule = vtkPiecewiseFunction::New();
  this->WeightsAlphaSchedule = vtkPiecewiseFunction::New();
  this->WeightsWidthSchedule = vtkPiecewiseFunction::New();

  this->BatchPercent = 1.0/15.0;
  this->UseAllVoxels = false;
  this->UseMask = false;
  
  this->info.KohonenMapSize[0] = 256;
  this->info.KohonenMapSize[1] = 256;
  this->info.KohonenMapSize[2] = 1;
  this->MaxEpochs = 1000;
  this->info.flags = 0;

  //configure the input ports
  this->SetNumberOfInputPorts(1);

}

vtkCudaKohonenGenerator::~vtkCudaKohonenGenerator(){
  if(this->MeansAlphaSchedule) this->MeansAlphaSchedule->Delete();
  if(this->MeansWidthSchedule) this->MeansWidthSchedule->Delete();
  if(this->VarsAlphaSchedule) this->VarsAlphaSchedule->Delete();
  if(this->VarsWidthSchedule) this->VarsWidthSchedule->Delete();
  if(this->WeightsAlphaSchedule) this->WeightsAlphaSchedule->Delete();
  if(this->WeightsWidthSchedule) this->WeightsWidthSchedule->Delete();
}

//
// Jacobi iteration for the solution of eigenvectors/eigenvalues of a nxn
// real symmetric matrix. Square nxn matrix a; size of matrix in n;
// output eigenvalues in w; and output eigenvectors in v. Resulting
// eigenvalues/vectors are sorted in decreasing order; eigenvectors are
// normalized.
#define VTK_ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
        a[k][l]=h+s*(g-h*tau)
#define VTK_MAX_ROTATIONS 20
template<class T>
int vtkJacobiN(T **a, int n, T *w, T **v)
{
  int i, j, k, iq, ip, numPos;
  T tresh, theta, tau, t, sm, s, h, g, c, tmp;
  T bspace[4], zspace[4];
  T *b = bspace;
  T *z = zspace;

  // only allocate memory if the matrix is large
  if (n > 4)
    {
    b = new T[n];
    z = new T[n]; 
    }

  // initialize
  for (ip=0; ip<n; ip++) 
    {
    for (iq=0; iq<n; iq++)
      {
      v[ip][iq] = 0.0;
      }
    v[ip][ip] = 1.0;
    }
  for (ip=0; ip<n; ip++) 
    {
    b[ip] = w[ip] = a[ip][ip];
    z[ip] = 0.0;
    }

  // begin rotation sequence
  for (i=0; i<VTK_MAX_ROTATIONS; i++) 
    {
    sm = 0.0;
    for (ip=0; ip<n-1; ip++) 
      {
      for (iq=ip+1; iq<n; iq++)
        {
        sm += fabs(a[ip][iq]);
        }
      }
    if (sm == 0.0)
      {
      break;
      }

    if (i < 3)                                // first 3 sweeps
      {
      tresh = 0.2*sm/(n*n);
      }
    else
      {
      tresh = 0.0;
      }

    for (ip=0; ip<n-1; ip++) 
      {
      for (iq=ip+1; iq<n; iq++) 
        {
        g = 100.0*fabs(a[ip][iq]);

        // after 4 sweeps
        if (i > 3 && (fabs(w[ip])+g) == fabs(w[ip])
        && (fabs(w[iq])+g) == fabs(w[iq]))
          {
          a[ip][iq] = 0.0;
          }
        else if (fabs(a[ip][iq]) > tresh) 
          {
          h = w[iq] - w[ip];
          if ( (fabs(h)+g) == fabs(h))
            {
            t = (a[ip][iq]) / h;
            }
          else 
            {
            theta = 0.5*h / (a[ip][iq]);
            t = 1.0 / (fabs(theta)+sqrt(1.0+theta*theta));
            if (theta < 0.0)
              {
              t = -t;
              }
            }
          c = 1.0 / sqrt(1+t*t);
          s = t*c;
          tau = s/(1.0+c);
          h = t*a[ip][iq];
          z[ip] -= h;
          z[iq] += h;
          w[ip] -= h;
          w[iq] += h;
          a[ip][iq]=0.0;

          // ip already shifted left by 1 unit
          for (j = 0;j <= ip-1;j++) 
            {
            VTK_ROTATE(a,j,ip,j,iq);
            }
          // ip and iq already shifted left by 1 unit
          for (j = ip+1;j <= iq-1;j++) 
            {
            VTK_ROTATE(a,ip,j,j,iq);
            }
          // iq already shifted left by 1 unit
          for (j=iq+1; j<n; j++) 
            {
            VTK_ROTATE(a,ip,j,iq,j);
            }
          for (j=0; j<n; j++) 
            {
            VTK_ROTATE(v,j,ip,j,iq);
            }
          }
        }
      }

    for (ip=0; ip<n; ip++) 
      {
      b[ip] += z[ip];
      w[ip] = b[ip];
      z[ip] = 0.0;
      }
    }

  //// this is NEVER called
  if ( i >= VTK_MAX_ROTATIONS )
    {
    vtkGenericWarningMacro(
       "vtkMath::Jacobi: Error extracting eigenfunctions");
    return 0;
    }

  // sort eigenfunctions                 these changes do not affect accuracy 
  for (j=0; j<n-1; j++)                  // boundary incorrect
    {
    k = j;
    tmp = w[k];
    for (i=j+1; i<n; i++)                // boundary incorrect, shifted already
      {
      if (w[i] >= tmp)                   // why exchage if same?
        {
        k = i;
        tmp = w[k];
        }
      }
    if (k != j) 
      {
      w[k] = w[j];
      w[j] = tmp;
      for (i=0; i<n; i++) 
        {
        tmp = v[i][j];
        v[i][j] = v[i][k];
        v[i][k] = tmp;
        }
      }
    }
  // insure eigenvector consistency (i.e., Jacobi can compute vectors that
  // are negative of one another (.707,.707,0) and (-.707,-.707,0). This can
  // reek havoc in hyperstreamline/other stuff. We will select the most
  // positive eigenvector.
  int ceil_half_n = (n >> 1) + (n & 1);
  for (j=0; j<n; j++)
    {
    for (numPos=0, i=0; i<n; i++)
      {
      if ( v[i][j] >= 0.0 )
        {
        numPos++;
        }
      }
//    if ( numPos < ceil(double(n)/double(2.0)) )
    if ( numPos < ceil_half_n)
      {
      for(i=0; i<n; i++)
        {
        v[i][j] *= -1.0;
        }
      }
    }

  if (n > 4)
    {
    delete [] b;
    delete [] z;
    }
  return 1;
}
#undef VTK_ROTATE
#undef VTK_MAX_ROTATIONS

//------------------------------------------------------------
//Commands for CudaObject compatibility

void vtkCudaKohonenGenerator::Reinitialize(int withData){
  //TODO
}

void vtkCudaKohonenGenerator::Deinitialize(int withData){
}

//------------------------------------------------------------
//Accessors and mutators

void vtkCudaKohonenGenerator::SetKohonenMapSize(int SizeX, int SizeY){
  if(SizeX < 1 || SizeY < 1) return;
  
  this->info.KohonenMapSize[0] = SizeX;
  this->info.KohonenMapSize[1] = SizeY;
}

bool vtkCudaKohonenGenerator::GetUseAllVoxelsFlag(){
  return this->UseAllVoxels;
}

void vtkCudaKohonenGenerator::SetUseAllVoxelsFlag(bool t){
  if( t != this->UseAllVoxels ){
    this->UseAllVoxels = t;
    this->Modified();
  }
}

bool vtkCudaKohonenGenerator::GetUseMaskFlag(){
  return this->UseMask;
}

void vtkCudaKohonenGenerator::SetUseMaskFlag(bool t){
  if( t != this->UseMask ){
    this->UseMask = t;
    this->Modified();
  }
}

//------------------------------------------------------------

void vtkCudaKohonenGenerator::SetNumberOfIterations(int number){
  if( number >= 0 && this->MaxEpochs != number ){
    this->MaxEpochs = number;
    this->Modified();
  }
}

int vtkCudaKohonenGenerator::GetNumberOfIterations(){
  return this->MaxEpochs;
}

void vtkCudaKohonenGenerator::SetBatchSize(double fraction){
  if( fraction >= 0.0 && this->BatchPercent != fraction ){
    this->BatchPercent = fraction;
    this->Modified();
  }
}

double vtkCudaKohonenGenerator::GetBatchSize(){
  return this->BatchPercent;
}


//------------------------------------------------------------
int vtkCudaKohonenGenerator::FillInputPortInformation(int i, vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  return this->Superclass::FillInputPortInformation(i,info);
}
#if (VTK_MAJOR_VERSION < 6)
void vtkCudaKohonenGenerator::SetInput(int idx, vtkDataObject *input)
{
  // Ask the superclass to connect the input.
  this->SetNthInputConnection(0, idx, (input ? input->GetProducerPort() : 0));
}
#else
void vtkCudaKohonenGenerator::SetInputConnection(int idx, vtkAlgorithmOutput *input)
{
  // Ask the superclass to connect the input.
  this->SetNthInputConnection(0, idx, (input ? input : 0));
}
#endif

vtkDataObject *vtkCudaKohonenGenerator::GetInput(int idx)
{
  if (this->GetNumberOfInputConnections(0) <= idx)
    {
    return 0;
    }
  return vtkImageData::SafeDownCast(
    this->GetExecutive()->GetInputData(0, idx));
}

//----------------------------------------------------------------------------

int vtkCudaKohonenGenerator::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkDataObject::SetPointDataActiveScalarInfo(outputInfo, VTK_FLOAT, inData->GetNumberOfScalarComponents());
  return 1;
}

int vtkCudaKohonenGenerator::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  for(int i = 0; i < inputVector[0]->GetNumberOfInformationObjects(); i++){
    vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(i);
    vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
    inputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inData->GetExtent(),6);
  }
  return 1;
}

int vtkCudaKohonenGenerator::RequestData(vtkInformation *request, 
              vtkInformationVector **inputVector, 
              vtkInformationVector *outputVector){
  
  //get general information
  int NumPictures = (inputVector[0])->GetNumberOfInformationObjects() / (this->UseMask ? 2 : 1);
  if( NumPictures < 1 ){
    vtkErrorMacro(<<"No pictures to train on.");
    return -1;
  }
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
  
  //make sure that the number of components is constant and the input type is FLOAT, and collect volume sizes
  vtkImageData* inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* maskData = 0;
  int* VolumeSize = new int[ 3*NumPictures ];
  int SumSamples = 0;
  this->info.NumberOfDimensions = inData->GetNumberOfScalarComponents();
  for(int p = 0; p < NumPictures; p++){
    
    if( this->UseMask ){
      inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p)->Get(vtkDataObject::DATA_OBJECT()));
      maskData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p+1)->Get(vtkDataObject::DATA_OBJECT()));
    }else{
      inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(p)->Get(vtkDataObject::DATA_OBJECT()));
    }

    inData->GetDimensions( &(VolumeSize[3*p]) );
    int CurrentVolumeSize = VolumeSize[3*p]*VolumeSize[3*p+1]*VolumeSize[3*p+2];
    if( this->UseMask ){
      char* MaskPtr = (char*) maskData->GetScalarPointer();
      for(int i = 0; i < CurrentVolumeSize; MaskPtr++, i++)
                if(*MaskPtr) SumSamples++;
    }else{
      SumSamples += CurrentVolumeSize;
    }

    if( inData->GetNumberOfScalarComponents() != this->info.NumberOfDimensions ){
      vtkErrorMacro(<<"Data objects need to have a consistant number of components");
      delete VolumeSize;
      return -1;
    }
    if( inData->GetScalarType() != VTK_FLOAT ){
      vtkErrorMacro(<<"Data objects need to be of type float");
      delete VolumeSize;
      return -1;
    }
    if( this->UseMask && maskData->GetScalarType() != VTK_CHAR &&
               maskData->GetScalarType() != VTK_SIGNED_CHAR &&
               maskData->GetScalarType() != VTK_UNSIGNED_CHAR ){
      std::cout << maskData->GetScalarType() << std::endl;
      vtkErrorMacro(<<"Mask objects need to be of type char");
      delete VolumeSize;
      return -1;
    }
  }

  int outputExtent[6] = {0, this->info.KohonenMapSize[0]-1, 0, this->info.KohonenMapSize[1]-1, 0, 0};
#if (VTK_MAJOR_VERSION < 6)
  outData->SetScalarTypeToFloat();
  outData->SetNumberOfScalarComponents(2*inData->GetNumberOfScalarComponents()+1);
  outData->SetExtent(outputExtent);
  outData->SetWholeExtent(outputExtent);
  outData->AllocateScalars();
#else
  outData->SetExtent(outputExtent);
  outData->AllocateScalars(VTK_FLOAT, 2*inData->GetNumberOfScalarComponents()+1);
#endif

  //update information container
  int BatchSize = (this->UseAllVoxels) ? -1 : SumSamples * this->BatchPercent;

  //get range
  double* Range = new double[2*(this->info.NumberOfDimensions)];
  for(int i = 0; i < this->info.NumberOfDimensions; i++){
    Range[2*i] = DBL_MAX; Range[2*i+1] = DBL_MIN;
    for(int p = 0; p < NumPictures; p++){
      double tempRange[2];
      if( this->UseMask ){
        inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p)->Get(vtkDataObject::DATA_OBJECT()));
      }else{
        inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(p)->Get(vtkDataObject::DATA_OBJECT()));
      }
      inData->GetPointData()->GetScalars()->GetRange(tempRange,i);
      Range[2*i] = std::min( tempRange[0], Range[2*i] );
      Range[2*i+1] = std::max( tempRange[1], Range[2*i+1] );
    }
  }

  //get scalar pointers
  float** inputDataPtr = new float* [NumPictures];
  char** maskDataPtr = (this->UseMask) ? new char*[NumPictures]: 0;
  int* FullVolumeSize = new int[NumPictures];
  int* SampleSize = new int[NumPictures];
  for(int p = 0; p < NumPictures; p++){
    if( this->UseMask ){
      inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p)->Get(vtkDataObject::DATA_OBJECT()));
      inputDataPtr[p] = (float*) inData->GetScalarPointer();
      maskData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(2*p+1)->Get(vtkDataObject::DATA_OBJECT()));
      maskDataPtr[p] = (char*) maskData->GetScalarPointer();
      FullVolumeSize[p] = (inData->GetExtent()[1]-inData->GetExtent()[0]+1)*
                (inData->GetExtent()[3]-inData->GetExtent()[2]+1)*
                (inData->GetExtent()[5]-inData->GetExtent()[4]+1);
      SampleSize[p] = 0;
      for(int x = 0; x < FullVolumeSize[p]; x++)
        SampleSize[p] += (maskDataPtr[p])[x] ? 1 : 0;

    }else{
      inData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(p)->Get(vtkDataObject::DATA_OBJECT()));
      inputDataPtr[p] = (float*) inData->GetScalarPointer();
      FullVolumeSize[p] = (inData->GetExtent()[1]-inData->GetExtent()[0]+1)*
                (inData->GetExtent()[3]-inData->GetExtent()[2]+1)*
                (inData->GetExtent()[5]-inData->GetExtent()[4]+1);
      SampleSize[p] = FullVolumeSize[p];
    }
  }

  //find means
  int N = info.NumberOfDimensions;
  double* Means = new double[N];
  for(int n = 0; n < N; n++)
    Means[n] = 0.0;
  int TotalVolumeSize = 0;
  for(int p = 0; p < NumPictures; p++){
    TotalVolumeSize += SampleSize[p];
    for(int x = 0; x < FullVolumeSize[p]; x++)
      for(int n = 0; n < N; n++)
        Means[n] += (!this->UseMask || (maskDataPtr[p])[x] > 0 ) ? (inputDataPtr[p])[x*N+n] : 0;
  }
  for(int n = 0; n < N; n++)
    Means[n] /= (double) TotalVolumeSize;

  //find covariances
  double* Covariance = new double[N*N];
  for(int n = 0; n < N*N; n++)
    Covariance[n] = 0.0;
  for(int p = 0; p < NumPictures; p++)
    for(int x = 0; x < FullVolumeSize[p]; x++)
      for(int n1 = 0; n1 < N; n1++) for(int n2 = 0; n2 < N; n2++)
        Covariance[n1*N+n2] += (!this->UseMask || (maskDataPtr[p])[x] > 0 ) ?
          ((inputDataPtr[p])[x*N+n1]-Means[n1]) * ((inputDataPtr[p])[x*N+n2]-Means[n2]) : 0;
  for(int n = 0; n < N*N; n++)
    Covariance[n] /= (double) TotalVolumeSize;

  //find primary and secondary eigenvectors
  double* Eigenvalues = new double[N];
  double* Eigenvectors = new double[N*N];
  double** EigenvectorsDual = new double*[N];
  for(int n = 0; n < N; n++) EigenvectorsDual[n] = &(Eigenvectors[n*N]);
  double** CovarianceDual = new double*[N];
  for(int n = 0; n < N; n++) CovarianceDual[n] = &(Covariance[n*N]);
  vtkJacobiN<double>(CovarianceDual,N,Eigenvalues,EigenvectorsDual);
  delete EigenvectorsDual;
  delete CovarianceDual;
  
  double* Eig1 = new double[N];
  double* Eig2 = new double[N];
  for(int n = 0; n < N; n++)
    Eig1[n] = sqrt(Eigenvalues[0])*Eigenvectors[n*N];
  for(int n = 0; n < N; n++)
    Eig2[n] = sqrt(Eigenvalues[1])*Eigenvectors[n*N+1];

  //create information holders
  int KMapSize[3];
  float* device_KohonenMap = 0;
  float* device_tempSpace = 0;
  float* device_DistanceBuffer = 0;
  short2* device_IndexBuffer = 0;
  float* device_WeightBuffer = 0;

  //get epsilon to prevent NaNs
  double RegularizationPercentage = 0.25;
  this->info.epsilon = RegularizationPercentage / (double)(this->info.KohonenMapSize[0]*this->info.KohonenMapSize[1]);

  //pass information to CUDA
  this->ReserveGPU();
  CUDAalgo_KSOMInitialize( Means, Covariance, Eig1, Eig2, this->info, KMapSize,
                &device_KohonenMap, &device_tempSpace,
                &device_DistanceBuffer, &device_IndexBuffer, &device_WeightBuffer,
                this->MeansWidthSchedule->GetValue(0.0),
                this->VarsWidthSchedule->GetValue(0.0),
                this->WeightsWidthSchedule->GetValue(0.0),
                this->GetStream() );
  delete Covariance;
  delete Means;
  delete Eigenvectors;
  delete Eigenvalues;
  delete Eig1;
  delete Eig2;

  for(int epoch = 0; epoch < this->MaxEpochs; epoch++)
    CUDAalgo_KSOMIteration( inputDataPtr,  maskDataPtr, epoch, KMapSize,
                &device_KohonenMap, &device_tempSpace,
                &device_DistanceBuffer, &device_IndexBuffer, &device_WeightBuffer,
                VolumeSize, NumPictures, this->info, BatchSize,
                this->MeansAlphaSchedule->GetValue(epoch), this->MeansWidthSchedule->GetValue(epoch),
                this->VarsAlphaSchedule->GetValue(epoch), this->VarsWidthSchedule->GetValue(epoch),
                this->WeightsAlphaSchedule->GetValue(epoch), this->WeightsWidthSchedule->GetValue(epoch),
                this->GetStream() );
  CUDAalgo_KSOMOffLoad( (float*) outData->GetScalarPointer(), &device_KohonenMap, &device_tempSpace,
              &device_DistanceBuffer, &device_IndexBuffer, &device_WeightBuffer,
              this->info, this->GetStream() );
  
  //clean up temporaries
  delete Range;
  delete VolumeSize;
  delete FullVolumeSize;
  delete SampleSize;
  delete inputDataPtr;
  if( this->UseMask ) delete maskDataPtr;

  return 1;
}
