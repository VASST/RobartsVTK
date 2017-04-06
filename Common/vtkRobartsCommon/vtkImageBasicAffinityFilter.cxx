/*=========================================================================

Robarts Visualization Toolkit

Copyright (c) 2016 Virtual Augmentation and Simulation for Surgery and Therapy, Robarts Research Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

=========================================================================*/

#include "vtkImageBasicAffinityFilter.h"
#include "vtkObjectFactory.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkStreamingDemandDrivenPipeline.h"

//----------------------------------------------------------------------------

vtkStandardNewMacro(vtkImageBasicAffinityFilter);

//----------------------------------------------------------------------------
void vtkImageBasicAffinityFilter::ThreadedExecute(vtkImageData* inData, vtkImageData* outData, int threadId, int numThreads)
{
  // cast the call down to handle the input data differences properly
  switch (inData->GetScalarType())
  {
      vtkTemplateMacro(
        ThreadedExecuteCasted<VTK_TT>(inData, outData, threadId, numThreads));
    default:
      if (threadId == 0) { vtkErrorMacro("Execute: Unknown input ScalarType"); }
      return;
  }
}

//----------------------------------------------------------------------------
template< class T >
void vtkImageBasicAffinityFilter::ThreadedExecuteCasted(vtkImageData* inData, vtkImageData* outData, int threadId, int numThreads)
{
  //find the total volume size
  int totalVolumeSize = inData->GetDimensions()[0] * inData->GetDimensions()[1] * inData->GetDimensions()[2];
  float* outPtr = (float*) outData->GetScalarPointer();
  T* inPtr = (T*) inData->GetScalarPointer();
  int numIComp = inData->GetNumberOfScalarComponents();

  //iterate over all the pixels we're responsible for and generate the affinity images
  int idInUse = threadId;
  while (idInUse < totalVolumeSize)
  {
    // find our co-ordinates in voxel space
    int coordinates[3];
    coordinates[0] = idInUse % inData->GetDimensions()[0] + inData->GetExtent()[0];
    coordinates[2] = idInUse / inData->GetDimensions()[0];
    coordinates[1] = coordinates[2] % inData->GetDimensions()[1] + inData->GetExtent()[2];
    coordinates[2] = coordinates[2] / inData->GetDimensions()[1] + inData->GetExtent()[4];

    // if we are not the last in the X direction, update the X affinity
    float xAffinity = 0.0f;
    if (coordinates[0] != inData->GetExtent()[1])
    {
      float dataDifference = 0;
      for (int i = 0; i < numIComp; i++)
      {
        float singleDiff = (float) inPtr[numIComp * idInUse + i] - (float) inPtr[numIComp * (idInUse + 1) + i];
        dataDifference += singleDiff * singleDiff;
      }
      xAffinity = exp(-1 * (this->DistanceWeight * inData->GetSpacing()[0] * inData->GetSpacing()[0] +
                            this->IntensityWeight * dataDifference));
    }
    ((float*)outData->GetScalarPointer())[3 * idInUse] = xAffinity;

    //if we are not the last in the Y direction, update the Y affinity
    float yAffinity = 0.0f;
    if (coordinates[1] != inData->GetExtent()[3])
    {
      float dataDifference = 0;
      for (int i = 0; i < numIComp; i++)
      {
        float singleDiff = (float) inPtr[numIComp * idInUse + i] - (float) inPtr[numIComp * (idInUse + inData->GetDimensions()[0]) + i];
        dataDifference += singleDiff * singleDiff;
      }
      yAffinity = exp(-1 * (this->DistanceWeight * inData->GetSpacing()[1] * inData->GetSpacing()[1] +
                            this->IntensityWeight * dataDifference));
    }
    ((float*)outData->GetScalarPointer())[3 * idInUse + 1] = yAffinity;

    //if we are not the last in the Z direction, update the Z affinity
    float zAffinity = 0.0f;
    if (coordinates[2] != inData->GetExtent()[5])
    {
      float dataDifference = 0;
      for (int i = 0; i < numIComp; i++)
      {
        float singleDiff = (float) inPtr[numIComp * idInUse + i] - (float) inPtr[numIComp * (idInUse + inData->GetDimensions()[0] * inData->GetDimensions()[1]) + i];
        dataDifference += singleDiff * singleDiff;
      }
      zAffinity = exp(-1 * (this->DistanceWeight * inData->GetSpacing()[2] * inData->GetSpacing()[2] +
                            this->IntensityWeight * dataDifference));
    }
    ((float*)outData->GetScalarPointer())[3 * idInUse + 2] = zAffinity;

    //move to the next pixel we're responsible for
    idInUse += numThreads;
  }
}

//----------------------------------------------------------------------------
struct vtkImageBasicAffinityFilterThreadStruct
{
  vtkImageBasicAffinityFilter* Filter;
  vtkImageData*   inData;
  vtkImageData*   outData;
};

//----------------------------------------------------------------------------
VTK_THREAD_RETURN_TYPE vtkImageBasicAffinityFilterThreadedExecute(void* arg)
{
  vtkImageBasicAffinityFilterThreadStruct* str;
  int threadId, threadCount;

  threadId = static_cast<vtkMultiThreader::ThreadInfo*>(arg)->ThreadID;
  threadCount = static_cast<vtkMultiThreader::ThreadInfo*>(arg)->NumberOfThreads;

  str = static_cast<vtkImageBasicAffinityFilterThreadStruct*>
        (static_cast<vtkMultiThreader::ThreadInfo*>(arg)->UserData);

  str->Filter->ThreadedExecute(str->inData, str->outData, threadId, threadCount);

  return VTK_THREAD_RETURN_VALUE;
}

//----------------------------------------------------------------------------
int vtkImageBasicAffinityFilter::RequestData(vtkInformation* request, vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  // get the info objects
  vtkImageData* inData = vtkImageData::SafeDownCast(this->GetInput());
  vtkImageData* outData = this->GetOutput();
  if (!inData || !outData) { return -1; }

  // get the output extent and reallocate the output buffer if necessary
  int* extent = inData->GetExtent();
  bool reallocateScalars = (outData->GetExtent()[0] != inData->GetExtent()[0]) ||
                           (outData->GetExtent()[1] != inData->GetExtent()[1]) ||
                           (outData->GetExtent()[2] != inData->GetExtent()[2]) ||
                           (outData->GetExtent()[3] != inData->GetExtent()[3]) ||
                           (outData->GetExtent()[4] != inData->GetExtent()[4]) ||
                           (outData->GetExtent()[5] != inData->GetExtent()[5]) ;
  if (reallocateScalars)
  {
    outData->SetExtent(extent);
    outData->AllocateScalars(VTK_FLOAT, 3);
  }

  //set all the spacing and origin parameters
  outData->SetSpacing(inData->GetSpacing());
  outData->SetOrigin(inData->GetOrigin());

  //set up the threader
  vtkImageBasicAffinityFilterThreadStruct str;
  str.Filter = this;
  str.inData = inData;
  str.outData = outData;
  this->Threader->SetNumberOfThreads(this->NumberOfThreads);
  this->Threader->SetSingleMethod(vtkImageBasicAffinityFilterThreadedExecute, &str);

  // always shut off debugging to avoid threading problems with GetMacros
  bool debug = this->Debug;
  this->Debug = 0;
  this->Threader->SingleMethodExecute();
  this->Debug = debug;

  return 1;
}

//----------------------------------------------------------------------------
vtkImageBasicAffinityFilter::vtkImageBasicAffinityFilter()
{
  this->DistanceWeight = 0;
  this->IntensityWeight = 0;
  this->Threader = vtkMultiThreader::New();
  this->NumberOfThreads = 10;
}

//----------------------------------------------------------------------------
vtkImageBasicAffinityFilter::~vtkImageBasicAffinityFilter()
{
  this->Threader->Delete();
}