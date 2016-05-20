/*=========================================================================

Program:   Visualization Toolkit
Module:    vtkImageLogLikelihood.cxx

Copyright (c) Martin Rajchl, Robarts Research Institute

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaImageLogLikelihood.cxx
*
*  @brief Implementation file with definitions for the CUDA accelerated log likelihood data term. This
*      generates entropy data terms based on the histogram of a set of provided seeds.
*
*  @author Martin Rajchl (Dr. Peters' Lab (VASST) at Robarts Research Institute)
*  
*  @note August 27th 2013 - Documentation first compiled. (jshbaxter)
*
*/

#include "vtkImageLogLikelihood.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <math.h>
#include <float.h>


vtkStandardNewMacro(vtkImageLogLikelihood);

//----------------------------------------------------------------------------
vtkImageLogLikelihood::vtkImageLogLikelihood()
{
  this->NormalizeDataTerm = 0;
  this->LabelID = 1.0;
  this->HistogramResolution = 1.0f;
  this->RequiredAgreement = 0.8;
  this->SetNumberOfInputPorts(2);
  this->SetNumberOfThreads(1);
}

vtkImageLogLikelihood::~vtkImageLogLikelihood(){

}

//----------------------------------------------------------------------------
// The output extent is the intersection.
int vtkImageLogLikelihood::RequestInformation (
  vtkInformation * vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  // get the info objects
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_FLOAT, 1);

  int numLabelMaps = 0;
  for(int i = 0; i < inputVector[1]->GetNumberOfInformationObjects(); i++)
    numLabelMaps++;

  int ext[6], ext2[6], idx;

  inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext);

  // two input take intersection

  if (!numLabelMaps)
  {
    vtkErrorMacro( "At least one label map must be specified.");
    return 1;
  }

  for(int i = 0; i < inputVector[1]->GetNumberOfInformationObjects(); i++){
    vtkInformation *inInfo2 = inputVector[1]->GetInformationObject(i);
    inInfo2->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext2);
    for (idx = 0; idx < 3; ++idx)
    {
      if (ext2[idx*2] > ext[idx*2])
      {
        ext[idx*2] = ext2[idx*2];
      }
      if (ext2[idx*2+1] < ext[idx*2+1])
      {
        ext[idx*2+1] = ext2[idx*2+1];
      }
    }
  }

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext,6);

  return 1;
}

//----------------------------------------------------------------------------

template <class T>
void vtkImageLogLikelihoodExecute2(vtkImageLogLikelihood *self,
                                   vtkImageData *in1Data, T *in1Ptr,
                                   vtkImageData **in2Data,
                                   vtkImageData *outData, float *outPtr,
                                   int outExt[6], int numLabels, int lblType, int id)
{
  //get some scalar pointer to appease the compiler
  void* scalarPtr = 0;
  for(int i = 0; i < numLabels; i++)
    if( in2Data[i] ){
      scalarPtr = in2Data[i]->GetScalarPointer();
      break;
    }

    //move down another type
    switch (lblType)
    {
      vtkTemplateMacro(
        vtkImageLogLikelihoodExecute(self,in1Data,
        (T*) in1Ptr,
        in2Data,
        static_cast<VTK_TT *>(scalarPtr),
        outData,
        outPtr, outExt,
        numLabels, id));
    default:
      vtkErrorWithObjectMacro(self, << "Execute: Unknown ScalarType");
      return;
    }

}
//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
// Handles the two input operations
template <class T, class TT>
void vtkImageLogLikelihoodExecute(vtkImageLogLikelihood *self,
                                  vtkImageData *in1Data, T *in1Ptr,
                                  vtkImageData **in2Data, TT *in2Ptr,
                                  vtkImageData *outData, float *outPtr,
                                  int outExt[6], int numLabels, int id)
{

  int dimens = in1Data->GetNumberOfScalarComponents();
  T* inputBuffer = (T*) in1Data->GetScalarPointer();
  int volumeSize = in1Data->GetDimensions()[0]*
    in1Data->GetDimensions()[1]*
    in1Data->GetDimensions()[2];

  float* outBuffer =  (float*)outData->GetScalarPointer();

  //find the actual number of non-null labels
  int actualNumLabels = 0;
  for(int label = 0; label < numLabels; label++ )
  {
    if( in2Data[label] ) actualNumLabels++;
  }

  float minVal[2] = {FLT_MAX, FLT_MAX};
  float maxVal[2] = {FLT_MIN, FLT_MIN};

  // calculate sample size
  int szSample = 0;
  for(int idx = 0; idx < volumeSize; idx++ )
  {
    int agree = 0;
    for(int label = 0; label < numLabels; label++ )
    {
      if( in2Data[label] )
        if( (int) ((TT*)in2Data[label]->GetScalarPointer())[idx] == self->GetLabelID() )
          agree++;
    }
    if( (double) agree >= self->GetRequiredAgreement()*(double)(actualNumLabels) )
    {
      szSample++;
      if(dimens == 1)
      {
        minVal[0] = (minVal[0] < inputBuffer[idx]) ? minVal[0] : inputBuffer[idx];
        maxVal[0] = (maxVal[0] > inputBuffer[idx]) ? maxVal[0] : inputBuffer[idx];
      }
      else
      {
        minVal[0] = (minVal[0] < inputBuffer[2*idx  ]) ? minVal[0] : inputBuffer[2*idx  ];
        maxVal[0] = (maxVal[0] > inputBuffer[2*idx  ]) ? maxVal[0] : inputBuffer[2*idx  ];
        minVal[1] = (minVal[1] < inputBuffer[2*idx+1]) ? minVal[1] : inputBuffer[2*idx+1];
        maxVal[1] = (maxVal[1] > inputBuffer[2*idx+1]) ? maxVal[1] : inputBuffer[2*idx+1];
      }
    }
  }

  if(szSample == 0)
  {
    std::fill_n( outBuffer, volumeSize, 1.0 );
    return;
  }

  // calculate histogram and ML data term
  float resolution = self->GetHistogramResolution();
  int szHist[2] = {(int)(maxVal[0]-minVal[0]/resolution)+1,
    (int)(maxVal[1]-minVal[1]/resolution)+1};
  int szHistTot = szHist[0]*szHist[1];
  float *hist = new float[szHistTot];
  std::fill_n(hist, szHistTot, 0.0f);

  // Fill histogram bins
  for(int idx = 0; idx < volumeSize; idx++ ){
    int agree = 0;
    for(int label = 0; label < numLabels; label++ ){
      if( in2Data[label] )
        if( (int) ((TT*)in2Data[label]->GetScalarPointer())[idx] == self->GetLabelID() )
          agree++;
    }
    if( (double) agree >= self->GetRequiredAgreement()*(double)(actualNumLabels) ){
      if(dimens == 1){
        hist[(int)((inputBuffer[idx]-minVal[0]) / resolution)]++;
      }else{
        hist[(int)((inputBuffer[2*idx]-minVal[0]) / resolution)*szHist[1] + (int)((inputBuffer[2*idx+1]-minVal[1]) / resolution)]++;
      }
    }
  }

  // Normalize histogram
  for (int i = 0; i < szHistTot; i++)
    hist[i] = hist[i]/ (float)szSample + 1.0e-10;

  // Calculate log likelihood dataterm
  for (int idx = 0; idx < volumeSize; idx++){
    if(dimens == 1)
      outBuffer[idx] =( inputBuffer[idx] < minVal[0] ||  (int)((inputBuffer[idx]-minVal[0]) / resolution) >= szHist[0]) ?
      -log(1.0e-10) : (-log(hist[ (int)((inputBuffer[idx]-minVal[0]) / resolution) ]));
    else
      outBuffer[idx] = ( inputBuffer[2*idx  ] < minVal[0] ||  (int)((inputBuffer[2*idx  ]-minVal[0]) / resolution) >= szHist[0] ) ? -log(1.0e-10) : (
      ( inputBuffer[2*idx+1] < minVal[1] ||  (int)((inputBuffer[2*idx+1]-minVal[1]) / resolution) >= szHist[1] ) ? -log(1.0e-10) :
      (-log(hist[ (int)((inputBuffer[2*idx]-minVal[0]) / resolution)*szHist[1] + (int)((inputBuffer[2*idx+1]-minVal[1]) / resolution) ])) );

  }

  // Normalize costs to [0,1]
  if( self->GetNormalizeDataTerm() )
    for (int idx = 0; idx < volumeSize; idx++)
      outBuffer[idx] = (outBuffer[idx] / -log(1.0e-10) );

  delete[] hist;

  return;
}

//----------------------------------------------------------------------------
// This method is passed an input and output image, and executes the filter
// algorithm to fill the output from the inputs.
// It just executes a switch statement to call the correct function for
// the image data types.
void vtkImageLogLikelihood::ThreadedRequestData(
  vtkInformation * vtkNotUsed( request ),
  vtkInformationVector ** inputVector,
  vtkInformationVector * outputVector,
  vtkImageData ***inData,
  vtkImageData **outData,
  int outExt[6], int id)
{
  void *inPtr1;
  void *outPtr;

  inPtr1 = inData[0][0]->GetScalarPointerForExtent(outExt);
  outPtr = outData[0]->GetScalarPointerForExtent(outExt);
  outData[0]->SetScalarType(VTK_FLOAT, outputVector->GetInformationObject(0));

  int numLabelMaps = 0;
  for(int i = 0; i < inputVector[1]->GetNumberOfInformationObjects(); i++)
  {
    if( inData[1][i] )
    {
      numLabelMaps++;
    }
  }

  if (numLabelMaps == 0) 
  {
    vtkErrorMacro("At least one label map is required.");
    return;
  }

  // this filter expects the output data-type to be float.
  if (outData[0]->GetScalarType() != VTK_FLOAT)
  {
    vtkErrorMacro("Output data type must be float.");
    return;
  }

  // this filter expects the label map to be of type char
  int LabelType = -1;
  for(int i = 0; i < inputVector[1]->GetNumberOfInformationObjects(); i++)
  {
    if( !inData[1][i] )
      continue;
    if( LabelType == -1 )
      LabelType = inData[1][i]->GetScalarType();

    if (inData[1][i]->GetScalarType() != LabelType) 
    {
      vtkErrorMacro( "Label maps must be of same type." );
      return;
    }
    if ( inData[1][i]->GetNumberOfScalarComponents() != 1 ) 
    {
      vtkErrorMacro( "Label map can only have 1 component." );
      return;
    }
  }

  // this filter expects that inputs that have the same number of components
  if (inData[0][0]->GetNumberOfScalarComponents() != 1 && inData[0][0]->GetNumberOfScalarComponents() != 2)
  {
    vtkErrorMacro( "Execute: Image can only have one or two components.");
    return;
  }

  switch (inData[0][0]->GetScalarType())
  {
    vtkTemplateMacro(
      vtkImageLogLikelihoodExecute2(this,inData[0][0],
      static_cast<VTK_TT *>(inPtr1),
      inData[1],
      outData[0],
      static_cast<float *>(outPtr), outExt,
      inputVector[1]->GetNumberOfInformationObjects(),
      LabelType,id));
  default:
    vtkErrorMacro( "Execute: Unknown ScalarType");
    return;
  }
}

//----------------------------------------------------------------------------
int vtkImageLogLikelihood::FillInputPortInformation(
  int port, vtkInformation* info)
{
  if( port == 1 ) info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(),1);
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  return 1;
}

//----------------------------------------------------------------------------
void vtkImageLogLikelihood::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

}

//----------------------------------------------------------------------------
void vtkImageLogLikelihood::SetInputDataImage(vtkDataObject *in)
{
  this->SetInputDataObject(0,in);
}

//----------------------------------------------------------------------------
void vtkImageLogLikelihood::SetInputConnection(vtkAlgorithmOutput *in)
{
  this->vtkThreadedImageAlgorithm::SetInputConnection(0, in);
}

//----------------------------------------------------------------------------
void vtkImageLogLikelihood::SetInputLabelMapConnection(vtkAlgorithmOutput *in, int number)
{
  if(number >= 0)
  {
    this->SetNthInputConnection(1,number,in);
  }
}

//----------------------------------------------------------------------------
void vtkImageLogLikelihood::SetInputLabelMapData(vtkDataObject *in, int number)
{
  if(number >= 0)
  {
    this->SetInputDataInternal(number, in);
  }
}

//----------------------------------------------------------------------------
void vtkImageLogLikelihood::SetNormalizeDataTermOn()
{
  this->NormalizeDataTerm = 1;
}

//----------------------------------------------------------------------------
void vtkImageLogLikelihood::SetNormalizeDataTermOff()
{
  this->NormalizeDataTerm = 0;
}

//----------------------------------------------------------------------------
int vtkImageLogLikelihood::GetNormalizeDataTerm()
{
  return (this->NormalizeDataTerm);
}
