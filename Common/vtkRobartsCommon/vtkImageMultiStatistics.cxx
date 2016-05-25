#include "vtkImageMultiStatistics.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"
#include "vtkExecutive.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkMath.h"
#include <vtkStreamingDemandDrivenPipeline.h>

//------------------------------------------------------------------------------
vtkImageMultiStatistics* vtkImageMultiStatistics::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkImageMultiStatistics");
  if(ret)
  {
    return (vtkImageMultiStatistics*)ret;
  }
  // If the factory was unable to create the object, then create it here.
  return new vtkImageMultiStatistics;
}

//----------------------------------------------------------------------------
vtkImageMultiStatistics::vtkImageMultiStatistics()
{
  //Initialize internal structure
  this->NumberOfComponents = 0;
  this->AverageMagnitude = 0;
  this->MeanSquared = 0;
  this->Covariance = 0;
  this->JointEntropy = 0;
  this->PCAAxisVectors = 0;
  this->PCAVariance = 0;
  this->TotalEntropy = 0.0;
  this->Count = 0;
  this->NumberOfBins = 100;

  //configure the input ports
  this->SetNumberOfInputPorts(1);
}

vtkImageMultiStatistics::~vtkImageMultiStatistics()
{
  if( this->AverageMagnitude ) delete [] this->AverageMagnitude;
  if( this->MeanSquared ) delete [] this->MeanSquared;
  if( this->PCAVariance ) delete [] this->PCAVariance;

  if( this->Covariance ){
    for( int i = 0; i < this->NumberOfComponents; i++)
      delete [] this->Covariance[i];
    delete [] this->Covariance;
  }

  if( this->JointEntropy ){
    for( int i = 0; i < this->NumberOfComponents; i++)
      delete [] this->JointEntropy[i];
    delete [] this->JointEntropy;
  }

  if( this->PCAAxisVectors ){
    for( int i = 0; i < this->NumberOfComponents; i++)
      delete [] this->PCAAxisVectors[i];
    delete [] this->PCAAxisVectors;
  }
}

//------------------------------------------------------------
int vtkImageMultiStatistics::FillInputPortInformation(int i, vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  return this->Superclass::FillInputPortInformation(i,info);
}

void vtkImageMultiStatistics::SetInputData(int port, vtkImageData *input)
{
  // Ask the superclass to connect the input.
  this->SetInputDataObject(port, input);
}

void vtkImageMultiStatistics::SetInputConnection(int port, vtkAlgorithmOutput *input)
{
  // Ask the superclass to connect the input.
  this->vtkAlgorithm::SetInputConnection(port, input);
}

//----------------------------------------------------------------------------
void vtkImageMultiStatistics::SetInputConnection(vtkAlgorithmOutput *input)
{
  SetInputConnection(0,input);
}

vtkImageData *vtkImageMultiStatistics::GetInput(int idx)
{
  if (this->GetNumberOfInputConnections(0) <= idx)
  {
    return 0;
  }
  return vtkImageData::SafeDownCast(
    this->GetExecutive()->GetInputData(0, idx));
}

//----------------------------------------------------------------------------
vtkImageData * vtkImageMultiStatistics::GetInput()
{
  return GetInput(0);
}

//----------------------------------------------------------------------------
double vtkImageMultiStatistics::GetAverageMagnitude(int component){
  if( component < this->NumberOfComponents ){
    this->Update(); 
    return this->AverageMagnitude[component];
  }else{
    vtkErrorMacro("Cannot select component. Component not provided in input.");
    return 0.0;
  }
}

double vtkImageMultiStatistics::GetMeanSquared(int component){
  if( component < this->NumberOfComponents ){
    this->Update(); 
    return this->MeanSquared[component];
  }else{
    vtkErrorMacro("Cannot select component. Component not provided in input.");
    return 0.0;
  }
}

double vtkImageMultiStatistics::GetStandardDeviation(int component) {
  if( component < this->NumberOfComponents ){
    this->Update();
    return sqrt(this->Covariance[component][component]);
  }else{
    vtkErrorMacro("Cannot select component. Component not provided in input.");
    return 0.0;
  }
}

double vtkImageMultiStatistics::GetCovariance(int component1, int component2)
{
  if( component1 < this->NumberOfComponents && component2 < this->NumberOfComponents )
  {
    this->Update();
    return this->Covariance[component1][component2];
  }
  else
  {
    vtkErrorMacro("Cannot select components. At least one component is not provided in input.");
    return 0.0;
  }
}

double vtkImageMultiStatistics::GetPCAWeight(int significance, int component)
{
  if( significance < this->NumberOfComponents && component < this->NumberOfComponents )
  {
    this->Update();
    return this->PCAAxisVectors[component][significance];
  }
  else
  {
    vtkErrorMacro("Cannot select components. At least one component is not provided in input.");
    return 0.0;
  }
}

double vtkImageMultiStatistics::GetPCAVariance(int significance)
{
  if( significance < this->NumberOfComponents )
  {
    return this->PCAVariance[significance];
    this->Update();
  }
  else
  {
    vtkErrorMacro("Cannot select component. Component not provided in input.");
    return 0.0;
  }
}

double vtkImageMultiStatistics::GetSingleEntropy(int component)
{
  if( component < this->NumberOfComponents ){
    return sqrt(this->JointEntropy[component][component]);
    this->Update();
  }
  else
  {
    vtkErrorMacro("Cannot select component. Component not provided in input.");
    return 0.0;
  }
}

double vtkImageMultiStatistics::GetJointEntropy(int component1, int component2)
{
  if( component1 < this->NumberOfComponents && component2 < this->NumberOfComponents )
  {
    this->Update();
    return this->JointEntropy[component1][component2];
  }
  else
  {
    vtkErrorMacro("Cannot select components. At least one component is not provided in input.");
    return 0.0;
  }
}

double vtkImageMultiStatistics::GetTotalEntropy()
{
  this->Update();
  return this->TotalEntropy;
}

long int vtkImageMultiStatistics::GetCount() 
{
  this->Update();
  return this->Count;
}

//----------------------------------------------------------------------------
void vtkImageMultiStatistics::SetEntropyResolution(int bins)
{
  if( bins < 1 ){
    vtkErrorMacro("Invalid resolution.");
    return;
  }

  //if we have changed the resolution, mark as modified
  if( bins != this->NumberOfBins ){
    this->NumberOfBins = bins;
    this->Modified();
  }
}

int vtkImageMultiStatistics::GetEntropyResolution()
{
  return this->NumberOfBins;
}

//----------------------------------------------------------------------------
#define MAX_COMPONENTS 20
struct vtkImageMultiStatisticsKDNode 
{

  int Index[MAX_COMPONENTS];
  int Partition;
  int Value;
  int N;

  vtkImageMultiStatisticsKDNode* Parent;
  vtkImageMultiStatisticsKDNode* LeftChild;
  vtkImageMultiStatisticsKDNode* RightChild;

  vtkImageMultiStatisticsKDNode (int N, vtkImageMultiStatisticsKDNode* parent) 
  {
    this->N = N;
    //this->Index = new int[N];
    this->LeftChild = 0;
    this->RightChild = 0;
    this->Parent = parent;
    this->Partition = parent ? (parent->Partition + 1 ) % N : 0;
    this->Value = 0;
  }

  ~vtkImageMultiStatisticsKDNode () 
  {
    //delete [] Index;
    if( LeftChild ) delete LeftChild;
    if( RightChild ) delete RightChild;
  }

  long double GetEntropy( int NumPoints )
  {
    //get entropy central to this bin
    long double LocalEntropy = -((long double)this->Value / (long double) NumPoints) *
      log( (long double)this->Value / (long double) NumPoints ) / log(2.0);

    //get entropy from child bins
    if( this->LeftChild )
      LocalEntropy += this->LeftChild->GetEntropy(NumPoints);
    if( this->RightChild )
      LocalEntropy += this->RightChild->GetEntropy(NumPoints);

    //return to parent
    return LocalEntropy;
  }

  void AddPoint( int* Index )
  {

    //see if it belongs to this node
    bool ThisNode = true;
    for( int i = 0; i < this->N; i++){
      if( this->Index[i] != Index[i] ){
        ThisNode = false;
        break;
      }
    }
    if( ThisNode ){
      this->Value++;
      return;
    }else if(this->Value == 0){
      for( int i = 0; i < this->N; i++)
        this->Index[i] = Index[i];
      this->Value = 1;
      return;
    }

    //see if it belongs to the left child
    if( Index[this->Partition] < this->Index[this->Partition] )
    {
      if( this->LeftChild ){
        this->LeftChild->AddPoint( Index );
      }else{
        this->LeftChild = new vtkImageMultiStatisticsKDNode(this->N,this);
        for( int i = 0; i < this->N; i++)
          this->LeftChild->Index[i] = Index[i];
        this->LeftChild->Value = 1;
      }
      return;
    }

    //else it belongs to the right child
    if( this->RightChild ){
      this->RightChild->AddPoint( Index );
    }else{
      this->RightChild = new vtkImageMultiStatisticsKDNode(this->N,this);
      for( int i = 0; i < this->N; i++)
        this->RightChild->Index[i] = Index[i];
      this->RightChild->Value = 1;
    }

  }

};


//----------------------------------------------------------------------------
template <class T, class S>
static void vtkImageMultiStatisticsExecuteWithMask(vtkImageMultiStatistics *self, 
                                                   T *inPtr,
                                                   vtkImageData *inData,
                                                   S *maskPtr,
                                                   vtkImageData *maskData,
                                                   double *AverageMagnitude,
                                                   double *MeanSquared,
                                                   double **Covariance,
                                                   double **JointEntropy,
                                                   long int *Count,
                                                   double *WholeEntropy, int N)
{

  int inIdxX, inIdxY, inIdxZ;
  vtkIdType inIncX, inIncY, inIncZ;
  vtkIdType maskIncX, maskIncY, maskIncZ;
  int inIdxN, inIdxN2;
  int wholeInExt[6];
  int wholeMaskExt[6];

  T *curPtr;
  S *curMaskPtr;

  vtkImageMultiStatisticsKDNode* MultiHist = new vtkImageMultiStatisticsKDNode(N,0);

  //use the output holders for temporary storage for statistical information
  double* sum = AverageMagnitude;
  long double** sum_squared = new long double* [N];
  for(int i = 0; i < N; i++){
    sum[i] = 0.0;
    sum_squared[i] = new long double [N];
    for( int j = 0; j < N; j++)
      sum_squared[i][j] = 0.0;
  }

  //Create temporary storage for information theoretic information
  int* HistIndices = new int[N];
  int Resolution = self->GetEntropyResolution();
  double* Maximum = new double[N];
  double* Minimum = new double[N];
  for( int i = 0; i < N; i++ ){
    Minimum[i] = inData->GetPointData()->GetScalars()->GetRange(i)[0];
    Maximum[i] = inData->GetPointData()->GetScalars()->GetRange(i)[1];
  }
  unsigned int* DoubleHistogram = new unsigned int [(N * N - N) * Resolution * Resolution / 2];
  memset( DoubleHistogram, 0, sizeof(unsigned int) * (N * N - N) * Resolution * Resolution / 2 );
  unsigned int* SingleHistogram = new unsigned int [N * Resolution];
  memset( SingleHistogram, 0, sizeof(unsigned int) * N * Resolution );

  // Get increments to march through data 
  inData->GetExtent(wholeInExt);
  maskData->GetExtent(wholeMaskExt);
  inData->GetContinuousIncrements(wholeInExt, inIncX, inIncY, inIncZ);
  maskData->GetContinuousIncrements(wholeMaskExt, maskIncX, maskIncY, maskIncZ);

  *Count = 0;

  // Loop through input dataset once to gather stats
  curPtr = inPtr;
  curMaskPtr = maskPtr;
  for (inIdxZ = wholeInExt[4]; inIdxZ <= wholeInExt[5]; inIdxZ++) {
    for (inIdxY = wholeInExt[2]; !self->AbortExecute && inIdxY <= wholeInExt[3]; inIdxY++) {
      for (inIdxX = wholeInExt[0]; inIdxX <= wholeInExt[1]; inIdxX++) {

        //only process if we are in the non-zero part of the mask
        if( (double) *curMaskPtr != 0.0 ){

          //calculate the histogram indices and add point to histogram
          for( inIdxN = 0; inIdxN < N; inIdxN++ ){
            int histIdx = (int)((double) Resolution * ((double) curPtr[inIdxN]-Minimum[inIdxN]) / (Maximum[inIdxN]-Minimum[inIdxN]));
            if( Maximum[inIdxN] <= Minimum[inIdxN] ) histIdx = 0;
            if( histIdx >= Resolution )
              histIdx = Resolution - 1;
            HistIndices[inIdxN] = histIdx;
          }
          MultiHist->AddPoint( HistIndices );

          int DimCount = 0;
          for( inIdxN = 0; inIdxN < N; inIdxN++ ){

            //do univariate 
            sum[inIdxN] += curPtr[inIdxN];
            sum_squared[inIdxN][inIdxN] += curPtr[inIdxN] * curPtr[inIdxN];
            SingleHistogram[inIdxN*Resolution+HistIndices[inIdxN]]++;

            //do bivariate
            for( inIdxN2 = inIdxN+1; inIdxN2 < N; inIdxN2++ ){
              sum_squared[inIdxN][inIdxN2] += curPtr[inIdxN] * curPtr[inIdxN2];
              sum_squared[inIdxN2][inIdxN]  += curPtr[inIdxN] * curPtr[inIdxN2];
              DoubleHistogram[DimCount*Resolution*Resolution + HistIndices[inIdxN]*Resolution + HistIndices[inIdxN2]]++;
              DimCount++;
            }
          }
          (*Count)++;
        }


        curPtr += N;
        curMaskPtr++;
      }
      curPtr += inIncY;
      curMaskPtr += maskIncY;
    }
    curPtr += inIncZ;
    curMaskPtr += maskIncZ;
  }

  //compute the means
  for(int i = 0; i < N; i++){
    double SafeSum = sum[i];
    AverageMagnitude[i] = SafeSum / (double)*Count;
    MeanSquared[i] = sum_squared[i][i] / (double)*Count;
  }

  //compute the covariances
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      Covariance[i][j] = (sum_squared[i][j] / (long double)*Count) - AverageMagnitude[i] * AverageMagnitude[j];
    }
  }

  //compute the entropies
  int DimCount = 0;
  for(int i = 0; i < N; i++){

    //compute single entropy
    JointEntropy[i][i] = 0.0;
    for(int r = 0; r < Resolution; r++){
      if( SingleHistogram[i*Resolution+r] )
        JointEntropy[i][i] -=        ((double) SingleHistogram[i*Resolution+r] / (double) *Count)
        * log((double) SingleHistogram[i*Resolution+r] / (double) *Count) / log(2.0);
    }

    for(int j = i+1; j < N; j++){
      int totCount = 0;
      JointEntropy[i][j] = 0.0;
      for(int r = 0; r < Resolution*Resolution; r++){
        if(  DoubleHistogram[DimCount*Resolution*Resolution+r] ){
          JointEntropy[i][j] -=        ((double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (double) *Count)
            * log((double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (double) *Count) / log(2.0);
          totCount += DoubleHistogram[DimCount*Resolution*Resolution+r];
        }
      }
      JointEntropy[j][i] = JointEntropy[i][j];
      DimCount++;
    }
  }
  *WholeEntropy = MultiHist->GetEntropy( *Count );

  //release storage
  delete [] Maximum;
  delete [] Minimum;
  delete [] SingleHistogram;
  delete [] DoubleHistogram;
  delete [] HistIndices;
  for(int i = 0; i < N; i++)
    delete [] sum_squared[i];
  delete [] sum_squared;

  //asynchronously delete the huge histogram somehow... //TODO
  delete MultiHist;

}

template <class T>
static void vtkImageMultiStatisticsExecuteWithMaskStart(vtkImageMultiStatistics *self, 
                                                        T *inPtr,
                                                        vtkImageData *inData,
                                                        vtkImageData *maskData,
                                                        double *AverageMagnitude,
                                                        double *MeanSquared,
                                                        double **Covariance,
                                                        double **JointEntropy,
                                                        long int *Count,
                                                        double *WholeEntropy, int N)
{
  void* maskPtr = maskData->GetScalarPointerForExtent( maskData->GetExtent() );

  switch (maskData->GetScalarType()) {
    vtkTemplateMacro(vtkImageMultiStatisticsExecuteWithMask(
      self, inPtr, inData,
      (VTK_TT *) maskPtr, maskData,
      AverageMagnitude, MeanSquared, Covariance, JointEntropy,
      Count, WholeEntropy, N));
  default:
    vtkErrorWithObjectMacro(self,<< "Update: Unknown ScalarType");
    return;
  }
}


template <class T>
static void vtkImageMultiStatisticsExecuteWithoutMask(vtkImageMultiStatistics *self, 
                                                      T *inPtr,
                                                      vtkImageData *inData,
                                                      double *AverageMagnitude,
                                                      double* MeanSquared,
                                                      double **Covariance,
                                                      double **JointEntropy,
                                                      long int *Count,
                                                      double *WholeEntropy, int N)
{
  int inIdxX, inIdxY, inIdxZ;
  vtkIdType inIncX, inIncY, inIncZ;
  int inIdxN, inIdxN2;
  int wholeInExt[6];
  T *curPtr;

  vtkImageMultiStatisticsKDNode* MultiHist = new vtkImageMultiStatisticsKDNode(N,0);

  //use the output holders for temporary storage for statistical information
  double* sum = AverageMagnitude;
  long double** sum_squared = new long double* [N];
  for(int i = 0; i < N; i++){
    sum[i] = 0.0;
    sum_squared[i] = new long double [N];
    for( int j = 0; j < N; j++)
      sum_squared[i][j] = 0.0;
  }

  //Create temporary storage for information theoretic information
  int* HistIndices = new int[N];
  int Resolution = self->GetEntropyResolution();
  double* Maximum = new double[N];
  double* Minimum = new double[N];
  for( int i = 0; i < N; i++ ){
    Minimum[i] = inData->GetPointData()->GetScalars()->GetRange(i)[0];
    Maximum[i] = inData->GetPointData()->GetScalars()->GetRange(i)[1];
  }
  unsigned int* DoubleHistogram = new unsigned int [(N * N - N) * Resolution * Resolution / 2];
  memset( DoubleHistogram, 0, sizeof(unsigned int) * (N * N - N) * Resolution * Resolution / 2 );
  unsigned int* SingleHistogram = new unsigned int [N * Resolution];
  memset( SingleHistogram, 0, sizeof(unsigned int) * N * Resolution );

  // Get increments to march through data
  inData->GetExtent(wholeInExt);
  inData->GetContinuousIncrements(wholeInExt, inIncX, inIncY, inIncZ);
  *Count = 0;

  // Loop through input dataset once to gather stats
  curPtr = inPtr;
  for (inIdxZ = wholeInExt[4]; inIdxZ <= wholeInExt[5]; inIdxZ++) {
    for (inIdxY = wholeInExt[2]; !self->AbortExecute && inIdxY <= wholeInExt[3]; inIdxY++) {
      for (inIdxX = wholeInExt[0]; inIdxX <= wholeInExt[1]; inIdxX++) {

        //calculate the histogram indices and add point to histogram
        for( inIdxN = 0; inIdxN < N; inIdxN++ ){
          int histIdx = (int)((double) Resolution * ((double) curPtr[inIdxN]-Minimum[inIdxN]) / (Maximum[inIdxN]-Minimum[inIdxN]));
          if( Maximum[inIdxN] == Minimum[inIdxN] ) histIdx = 0;
          if( histIdx >= Resolution )
            histIdx = Resolution - 1;
          HistIndices[inIdxN] = histIdx;
        }
        MultiHist->AddPoint( HistIndices );

        int DimCount = 0;
        for( inIdxN = 0; inIdxN < N; inIdxN++ ){

          //do univariate 
          sum[inIdxN] += curPtr[inIdxN];
          sum_squared[inIdxN][inIdxN] += curPtr[inIdxN] * curPtr[inIdxN];
          SingleHistogram[inIdxN*Resolution+HistIndices[inIdxN]]++;

          //do bivariate
          for( inIdxN2 = inIdxN+1; inIdxN2 < N; inIdxN2++ ){
            sum_squared[inIdxN][inIdxN2] += curPtr[inIdxN] * curPtr[inIdxN2];
            sum_squared[inIdxN2][inIdxN]  += curPtr[inIdxN] * curPtr[inIdxN2];
            DoubleHistogram[DimCount*Resolution*Resolution + HistIndices[inIdxN]*Resolution + HistIndices[inIdxN2]]++;
            DimCount++;
          }
        }
        (*Count)++;
        curPtr += N;
      }
      curPtr += inIncY;
    }
    curPtr += inIncZ;
  }

  //compute the means
  for(int i = 0; i < N; i++){
    double SafeSum = sum[i];
    AverageMagnitude[i] = SafeSum / (double)*Count;
    MeanSquared[i] = sum_squared[i][i] / (double)*Count;
  }

  //compute the covariances
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      Covariance[i][j] = (sum_squared[i][j] / (long double)*Count) - AverageMagnitude[i] * AverageMagnitude[j];

  //compute the entropies
  int DimCount = 0;
  for(int i = 0; i < N; i++){

    //compute single entropy
    JointEntropy[i][i] = 0.0;
    for(int r = 0; r < Resolution; r++){
      if( SingleHistogram[i*Resolution+r] )
        JointEntropy[i][i] -=        ((double) SingleHistogram[i*Resolution+r] / (double) *Count)
        * log((double) SingleHistogram[i*Resolution+r] / (double) *Count) / log(2.0);
    }

    for(int j = i+1; j < N; j++){
      int totCount = 0;
      JointEntropy[i][j] = 0.0;
      for(int r = 0; r < Resolution*Resolution; r++){
        if(  DoubleHistogram[DimCount*Resolution*Resolution+r] ){
          JointEntropy[i][j] -=        ((double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (double) *Count)
            * log((double) DoubleHistogram[DimCount*Resolution*Resolution+r] / (double) *Count) / log(2.0);
          totCount += DoubleHistogram[DimCount*Resolution*Resolution+r];
        }
      }
      JointEntropy[j][i] = JointEntropy[i][j];
      DimCount++;
    }
  }
  *WholeEntropy = MultiHist->GetEntropy( *Count );

  //release storage
  delete [] Maximum;
  delete [] Minimum;
  delete [] SingleHistogram;
  delete [] DoubleHistogram;
  delete [] HistIndices;
  for(int i = 0; i < N; i++)
    delete [] sum_squared[i];
  delete [] sum_squared;

  //asynchronously delete the huge histogram somehow... //TODO
  delete MultiHist;

}

// Description:
// Make sure input is available then call the templated execute method to
// deal with the particular data type.
void vtkImageMultiStatistics::Update() {

  vtkImageData *input = this->GetInput(0);
  void *inPtr;
  int wholeInExt[6];

  // make sure input is available
  if (!input) {
    vtkErrorMacro( "No input...can't execute!");
    return;
  }
  this->Update();
  this->UpdateInformation();
  this->GetOutputInformation(0)->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), wholeInExt);
  inPtr = input->GetScalarPointerForExtent(wholeInExt);

  // get the number of input components
  int oldNumberOfComponents = this->NumberOfComponents;
  this->NumberOfComponents = input->GetNumberOfScalarComponents();
  if (this->NumberOfComponents < 1){
    vtkErrorMacro( "Input must have at least 1 scalar component");
    return;
  }

  //fetch and check the mask data
  vtkImageData *mask = this->GetInput(1);
  if( mask ){
    int MaskExtent[6];
    this->UpdateInformation();
    this->GetOutputInformation(1)->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), MaskExtent);

    if( MaskExtent[0] != wholeInExt[0] || MaskExtent[1] != wholeInExt[1] || MaskExtent[2] != wholeInExt[2] ||
      MaskExtent[3] != wholeInExt[3] || MaskExtent[4] != wholeInExt[4] || MaskExtent[5] != wholeInExt[5] ){
        vtkErrorMacro( "Mask is not the same extent as the input.");
        mask = 0;
    }
    if( mask && mask->GetNumberOfScalarComponents() != 1 ){
      vtkErrorMacro( "Mask can only have 1 scalar component.");
      mask = 0;
    }
  }

  //update the size of the information holders
  if( this->NumberOfComponents != oldNumberOfComponents ){
    if( this->AverageMagnitude ) delete [] this->AverageMagnitude;
    this->AverageMagnitude = new double[this->NumberOfComponents];

    if( this->MeanSquared ) delete [] this->MeanSquared;
    this->MeanSquared = new double[this->NumberOfComponents];

    if( this->Covariance ){
      for( int i = 0; i < this->NumberOfComponents; i++)
        delete [] this->Covariance[i];
      delete [] this->Covariance;
    }
    this->Covariance = new double*[this->NumberOfComponents];
    for(int i = 0; i < this->NumberOfComponents; i++)
      this->Covariance[i] = new double[this->NumberOfComponents];

    if( this->JointEntropy ){
      for( int i = 0; i < this->NumberOfComponents; i++)
        delete [] this->JointEntropy[i];
      delete [] this->JointEntropy;
    }
    this->JointEntropy = new double*[this->NumberOfComponents];
    for(int i = 0; i < this->NumberOfComponents; i++)
      this->JointEntropy[i] = new double[this->NumberOfComponents];

    if( this->PCAVariance ) delete [] this->PCAVariance;
    this->PCAVariance = new double[this->NumberOfComponents];

    if( this->PCAAxisVectors ){
      for( int i = 0; i < this->NumberOfComponents; i++)
        delete [] this->PCAAxisVectors[i];
      delete [] this->PCAAxisVectors;
    }
    this->PCAAxisVectors = new double*[this->NumberOfComponents];
    for(int i = 0; i < this->NumberOfComponents; i++)
      this->PCAAxisVectors[i] = new double[this->NumberOfComponents];
  }


  if (input->GetMTime() > this->ExecuteTime || this->GetMTime() > this->ExecuteTime ){
    this->InvokeEvent(vtkCommand::StartEvent, NULL);

    // reset Abort flag
    this->AbortExecute = 0;
    this->Progress = 0.0;

    //if there is no mask, run the regular version
    if( !mask )
      switch (input->GetScalarType()) {
        vtkTemplateMacro(vtkImageMultiStatisticsExecuteWithoutMask(
          this, (VTK_TT *) (inPtr), input,
          this->AverageMagnitude, this->MeanSquared,
          this->Covariance,
          this->JointEntropy,
          &(this->Count), &(this->TotalEntropy), this->NumberOfComponents));
      default:
        vtkErrorMacro( "Update: Unknown ScalarType");
        return;
    }
    else
      switch (input->GetScalarType()) {
        vtkTemplateMacro(vtkImageMultiStatisticsExecuteWithMaskStart(
          this, (VTK_TT *) (inPtr), input, mask,
          this->AverageMagnitude,  this->MeanSquared,
          this->Covariance,
          this->JointEntropy,
          &(this->Count), &(this->TotalEntropy), this->NumberOfComponents));
      default:
        vtkErrorMacro( "Update: Unknown ScalarType");
        return;
    }

    this->ExecuteTime.Modified();
    if (!this->AbortExecute) {
      this->UpdateProgress(1.0);
    }
    this->InvokeEvent(vtkCommand::EndEvent, NULL);
  }

  //update PCA results
  double** temporaryCovariance = new double* [this->NumberOfComponents];
  for(int i = 0; i < this->NumberOfComponents; i++){
    temporaryCovariance[i] = new double[this->NumberOfComponents];
    for(int j = 0; j < this->NumberOfComponents; j++){
      temporaryCovariance[i][j] = Covariance[i][j];
    }
  }
  vtkMath::JacobiN(temporaryCovariance, this->NumberOfComponents, this->PCAVariance, this->PCAAxisVectors );
  for(int i = 0; i < this->NumberOfComponents; i++)
    delete [] temporaryCovariance[i];
  delete [] temporaryCovariance;

}

void vtkImageMultiStatistics::PrintSelf(ostream& os, vtkIndent indent)
{
  Superclass::PrintSelf(os,indent);

  if (!this->GetInput())
  {
    return;
  }

  os << indent << "Components: " << this->NumberOfComponents << "\n";
  os << indent << "Count: " << this->Count << "\n";
  for( int i = 0; i < this->NumberOfComponents; i++)
    os << indent << "AverageMagnitude (" << i << "): " << this->AverageMagnitude[i] << "\n";

  for( int i = 0; i < this->NumberOfComponents; i++)
    for( int j = 0; j < this->NumberOfComponents; j++)
      os << indent << "Covariance: (" << i << "," << j << "): " << this->Covariance[i][j] << "\n";

  for( int i = 0; i < this->NumberOfComponents; i++)
    os << indent << "StandardDeviation: (" << i << "): " << sqrt(this->Covariance[i][i]) << "\n";

  for( int i = 0; i < this->NumberOfComponents; i++)
    os << indent << "MeanSquared (" << i << "): " << this->MeanSquared[i] << "\n";

  os << indent << "Total Entropy: " << this->TotalEntropy << "\n";

  for( int i = 0; i < this->NumberOfComponents; i++)
    for( int j = 0; j < this->NumberOfComponents; j++)
      os << indent << "Joint Entropy: (" << i << "," << j << "): " << this->JointEntropy[i][j] << "\n";

  for( int i = 0; i < this->NumberOfComponents; i++)
    os << indent << "PCA Variance: (" << i << "): " << this->PCAVariance[i] << "\n";

  for( int i = 0; i < this->NumberOfComponents; i++)
    for( int j = 0; j < this->NumberOfComponents; j++)
      os << indent << "PCA Vector: (" << i << "," << j << "): " << this->PCAAxisVectors[j][i] << "\n";

}

