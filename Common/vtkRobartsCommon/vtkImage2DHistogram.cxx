#include "vtkDataArray.h"
#include "vtkImage2DHistogram.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkMultiThreader.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

namespace
{
  struct vtkImage2DHistogramThreadStruct
  {
    vtkImage2DHistogram *Filter;
    vtkImageData   *inData;
    vtkImageData   *outData;
  };
}

vtkStandardNewMacro(vtkImage2DHistogram);

vtkImage2DHistogram::vtkImage2DHistogram()
{
  this->Resolution[0] = 100;
  this->Resolution[1] = 100;
  this->Threader = vtkMultiThreader::New();
  this->NumberOfThreads = 10;
}

vtkImage2DHistogram::~vtkImage2DHistogram()
{
  this->Threader->Delete();
}

void vtkImage2DHistogram::ThreadedExecute(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads)
{
  //cast the call down to handle the input data differences properly
  switch (inData->GetScalarType())
  {
    vtkTemplateMacro(
      ThreadedExecuteCasted<VTK_TT>(inData, outData, threadId, numThreads));
    default:
      if(threadId == 0) vtkErrorMacro( "Execute: Unknown input ScalarType");
      return;
  }

  //configure the input ports
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfInputConnections(0,1);
}

template< class T >
void vtkImage2DHistogram::ThreadedExecuteCasted(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads)
{
  //find the total volume size
  int totalVolumeSize = inData->GetDimensions()[0] * inData->GetDimensions()[1] * inData->GetDimensions()[2];
  int* outPtr = (int*) outData->GetScalarPointer();
  T* inPtr = (T*) inData->GetScalarPointer();
  int numIComp = inData->GetNumberOfScalarComponents();

  double* OutSpacing = outData->GetSpacing();
  double* OutMinimum = outData->GetOrigin();

  //clear all the bins to 0
  memset( (void*) outPtr, 0, this->Resolution[0]*this->Resolution[1]*sizeof(int) );

  //iterate over all the pixels we're responsible for and fill the histogram bins
  int idInUse = threadId;
  while( idInUse < totalVolumeSize )
  {
    //find the index of the appropriate bin
    T value1 = inPtr[idInUse*2];
    T value2 = inPtr[idInUse*2+1];
    int bin1 = ((double) value1 - OutMinimum[0]) / OutSpacing[0];
    int bin2 = ((double) value2 - OutMinimum[1]) / OutSpacing[1];

    //increment that bin
    outPtr[bin1 + this->Resolution[0]*bin2] ++;

    //move to the next pixel we're responsible for
    idInUse += numThreads;
  }
}

VTK_THREAD_RETURN_TYPE vtkImage2DHistogramThreadedExecute( void *arg )
{
  vtkImage2DHistogramThreadStruct *str;
  int threadId, threadCount;

  threadId = static_cast<vtkMultiThreader::ThreadInfo *>(arg)->ThreadID;
  threadCount = static_cast<vtkMultiThreader::ThreadInfo *>(arg)->NumberOfThreads;

  str = static_cast<vtkImage2DHistogramThreadStruct *>
  (static_cast<vtkMultiThreader::ThreadInfo *>(arg)->UserData);

  str->Filter->ThreadedExecute(str->inData, str->outData, threadId, threadCount);

  return VTK_THREAD_RETURN_VALUE;
}

int vtkImage2DHistogram::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkDataObject::SetPointDataActiveScalarInfo(outputInfo, VTK_INT, 1);
  return 1;
}

int vtkImage2DHistogram::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));

  int OutputExtent[6];
  OutputExtent[0] = OutputExtent[2] = OutputExtent[4] = OutputExtent[5] = 0;
  OutputExtent[1] = this->Resolution[0];
  OutputExtent[2] = this->Resolution[1];

  outputInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),OutputExtent,6);
  outputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),OutputExtent,6);

  return 1;
}

int vtkImage2DHistogram::RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector){

  vtkInformation* inputInfo = (inputVector[0])->GetInformationObject(0);
  vtkInformation* outputInfo = outputVector->GetInformationObject(0);
  vtkImageData* inData = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkImageData* outData = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));

  if( inData->GetNumberOfScalarComponents() != 2 )
  {
    vtkErrorMacro("Input must have 2 scalar components.");
    return -1;
  }

  //figure out the extent of the output
    int updateExtent[6];
    outputInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), updateExtent);
  outData->SetExtent(updateExtent);
#if (VTK_MAJOR_VERSION < 6)
  outData->SetNumberOfScalarComponents( 1 );
  outData->SetScalarType( VTK_INT );
  outData->AllocateScalars();
#else
  outData->AllocateScalars(VTK_INT, 1);
#endif

  //set all the spacing and origin parameters
  double* Range1 = inData->GetPointData()->GetScalars()->GetRange(0);
  double* Range2 = inData->GetPointData()->GetScalars()->GetRange(1);
  outData->SetSpacing( (Range1[1]-Range1[0]) / (Resolution[0]-1), (Range2[1]-Range2[0]) / (Resolution[1]-1),0 );
  outData->SetOrigin( Range1[0], Range2[0], 0 );

  //set up the threader
  vtkImage2DHistogramThreadStruct str;
  str.Filter = this;
  str.inData = inData;
  str.outData = outData;
  this->Threader->SetNumberOfThreads(this->NumberOfThreads);
  this->Threader->SetSingleMethod(vtkImage2DHistogramThreadedExecute, &str);

  // always shut off debugging to avoid threading problems with GetMacros
  int debug = this->Debug;
  this->Debug = 0;
  this->Threader->SingleMethodExecute();
  this->Debug = debug;

  return 1;
}

void vtkImage2DHistogram::SetResolution( int res[2] ){
  if( res[0] > 0 && res[1] > 0 ){
    this->Resolution[0] = res[0];
    this->Resolution[1] = res[1];
  }
}
