/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImageVote.cxx

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkImageVote.cxx
 *
 *  @brief Implementation file with definitions for the CPU-based voting operation. This module
 *      Takes a probabilistic or weighted image, and replaces each voxel with a label corresponding
 *      to the input image with the highest value at that location. ( argmax{} operation )
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#include "vtkDataArray.h"
#include "vtkImageVote.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTreeDFSIterator.h"
#include "vtkTrivialProducer.h"

#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#define SQR(X) X*X

vtkStandardNewMacro(vtkImageVote);

vtkImageVote::vtkImageVote(){
  
  //configure the IO ports
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(1);

  //set up the input mapping structure
  this->InputPortMapping.clear();
  this->BackwardsInputPortMapping.clear();
  this->FirstUnusedPort = 0;

  this->OutputDataType = VTK_SHORT;
    this->SetNumberOfThreads(1);

  this->Lock = vtkBarrierLock::New();
  this->Lock->SetRepeatable(true);

}

vtkImageVote::~vtkImageVote(){
  this->InputPortMapping.clear();
  this->BackwardsInputPortMapping.clear();
}

//------------------------------------------------------------

int vtkImageVote::FillInputPortInformation(int i, vtkInformation* info){
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  return this->Superclass::FillInputPortInformation(i,info);
}

void vtkImageVote::SetInput(int idx, vtkDataObject *input)
{
  //we are adding/switching an input, so no need to resort list
  if( input != NULL ){
  
    //if their is no pair in the mapping, create one
    if( this->InputPortMapping.find(idx) == this->InputPortMapping.end() ){
      int portNumber = this->FirstUnusedPort;
      this->FirstUnusedPort++;
      this->InputPortMapping.insert(std::pair<vtkIdType,int>(idx,portNumber));
      this->BackwardsInputPortMapping.insert(std::pair<vtkIdType,int>(portNumber,idx));
    }
#if (VTK_MAJOR_VERSION < 6)
    this->SetNthInputConnection(0, this->InputPortMapping.find(idx)->second, input->GetProducerPort() );
#else
    vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
    tp->SetInputDataObject(input);
    this->SetNthInputConnection(0, this->InputPortMapping.find(idx)->second, tp->GetOutputPort() );
#endif

  }else{
    //if their is no pair in the mapping, just exit, nothing to do
    if( this->InputPortMapping.find(idx) == this->InputPortMapping.end() ) return;

    int portNumber = this->InputPortMapping.find(idx)->second;
    this->InputPortMapping.erase(this->InputPortMapping.find(idx));
    this->BackwardsInputPortMapping.erase(this->BackwardsInputPortMapping.find(portNumber));

    //if we are the last input, no need to reshuffle
    if(portNumber == this->FirstUnusedPort - 1){
      this->SetNthInputConnection(0, portNumber,  0);
    
    //if we are not, move the last input into this spot
    }else{
      vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedPort - 1));
#if (VTK_MAJOR_VERSION < 6)
      this->SetNthInputConnection(0, portNumber, swappedInput->GetProducerPort() );
#else
      vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
      tp->SetInputDataObject(swappedInput);
      this->SetNthInputConnection(0, portNumber, tp->GetOutputPort() );
#endif
      this->SetNthInputConnection(0, this->FirstUnusedPort - 1, 0 );

      //correct the mappings
      vtkIdType swappedId = this->BackwardsInputPortMapping.find(this->FirstUnusedPort - 1)->second;
      this->InputPortMapping.erase(this->InputPortMapping.find(swappedId));
      this->BackwardsInputPortMapping.erase(this->BackwardsInputPortMapping.find(this->FirstUnusedPort - 1));
      this->InputPortMapping.insert(std::pair<vtkIdType,int>(swappedId,portNumber) );
      this->BackwardsInputPortMapping.insert(std::pair<int,vtkIdType>(portNumber,swappedId) );

    }

    //decrement the number of unused ports
    this->FirstUnusedPort--;

  }
}

vtkDataObject *vtkImageVote::GetInput(int idx)
{
  if( this->InputPortMapping.find(idx) == this->InputPortMapping.end() )
    return 0;
  return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputPortMapping.find(idx)->second));
}

//----------------------------------------------------------------------------

int vtkImageVote::CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int* NumLabels, int* DataType, int* NumComponents){
  
  *DataType = -1;
  Extent[0] = -1;
  *NumComponents = -1;

  //make sure that every image is the correct size and same datatype
  for(unsigned int inputPortNumber = 0; inputPortNumber < this->InputPortMapping.size(); inputPortNumber++){
    
    //verify extent
    if( Extent[0] == -1 ){
      vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
      CurrImage->GetExtent(Extent);
    }else{
      int CurrExtent[6];
      vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
      CurrImage->GetExtent(CurrExtent);
      if( CurrExtent[0] != Extent[0] || CurrExtent[1] != Extent[1] || CurrExtent[2] != Extent[2] ||
        CurrExtent[3] != Extent[3] || CurrExtent[4] != Extent[4] || CurrExtent[5] != Extent[5] ){
        vtkErrorMacro(<<"Inconsistant object extent.");
        return -1;
      }
    }

    //verify data type
    if( *DataType == -1 ){
      vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
      *DataType = CurrImage->GetScalarType();
    }else{
      vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
      if( *DataType != CurrImage->GetScalarType() ){
        vtkErrorMacro(<<"Inconsistant object data type.");
        return -1;
      }
    }

    //verify number of components
    if( *NumComponents == -1 ){
      vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
      *NumComponents = CurrImage->GetNumberOfScalarComponents();
    }else{
      vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
      if( *NumComponents != CurrImage->GetNumberOfScalarComponents() ){
        vtkErrorMacro(<<"Inconsistant object data type.");
        return -1;
      }
    }
  }

  return 0;
}

int vtkImageVote::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  //check input for consistency
  int Extent[6]; int NumLabels; int DataType; int NumComponents;
  int result = CheckInputConsistancy( inputVector, Extent, &NumLabels, &DataType, &NumComponents );
  if( result || NumLabels == 0 ) return -1;        

  return 1;
}

int vtkImageVote::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  //check input for consistency
  int Extent[6]; int NumLabels; int DataType; int NumComponents;
  int result = CheckInputConsistancy( inputVector, Extent, &NumLabels, &DataType, &NumComponents );
  if( result || NumLabels == 0 ) return -1;        

  //set up the extents
  vtkInformation *outputInfo = outputVector->GetInformationObject(0);
  vtkImageData *outputBuffer = vtkImageData::SafeDownCast(outputInfo->Get(vtkDataObject::DATA_OBJECT()));
  outputInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),Extent,6);
  outputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),Extent,6);

  return 1;
}


template<class T>
void vtkImageVoteExecute(vtkImageVote *self,
        vtkImageData **inData,
        vtkImageData *outData,
        int* outExt, T* unUsed, int threadId ){

  T** inPtr = new T* [self->GetNumberOfInputConnections(0)];
  for(int i = 0; i < self->GetNumberOfInputConnections(0); i++ )
    inPtr[i] = (T*) inData[i]->GetScalarPointer();

  void* outPtr = outData->GetScalarPointer();

  switch (outData->GetScalarType()) {
    vtkTemplateMacro(vtkImageVoteExecute(self, inData, inPtr, outData, static_cast<VTK_TT *>(outPtr), outExt, threadId ));
    default:
      vtkGenericWarningMacro("Execute: Unknown output ScalarType");
      return;
    }

  delete[] inPtr;
}

template <class IT, class OT>
void vtkImageVoteExecute(vtkImageVote *self,
        vtkImageData **inData, IT **inPtr,
        vtkImageData *outData, OT *outPtr,
        int* outExt, int threadId ){
  
  int NumThreads = self->GetNumberOfThreads();
  int VolumeSize = (outExt[1]-outExt[0]+1)*(outExt[3]-outExt[2]+1)*(outExt[5]-outExt[4]+1);
  for(int idx = threadId*VolumeSize/NumThreads; idx < (threadId+1)*VolumeSize/NumThreads; idx++){
    OT maxIdentifier = -1;
    IT maxValue = 0;
    for( int iv = 0; iv < self->GetNumberOfInputConnections(0); iv++ )
      if( inPtr[iv] && (maxIdentifier == -1 || inPtr[iv][idx] > maxValue ) ) {
        maxIdentifier = self->GetMappedTerm<OT>(iv);
        maxValue = inPtr[iv][idx];
      }
    outPtr[idx] = (maxIdentifier == -1) ? (OT) 0: maxIdentifier;
  }

}

void vtkImageVote::ThreadedRequestData(vtkInformation *request,
                                     vtkInformationVector **inputVector,
                                     vtkInformationVector *outputVector,
                                     vtkImageData ***inData,
                                     vtkImageData **outData,
                                     int extent[6], int threadId){

  //check input for consistency
  int Extent[6]; int NumLabels; int DataType; int NumComponents;
  int result = CheckInputConsistancy( inputVector, Extent, &NumLabels, &DataType, &NumComponents );
  if( result || NumLabels == 0 ) return;        
  
  //allocate output image (using short)
  if( threadId == 0 ){
    (*outData)->SetExtent(Extent);
#if (VTK_MAJOR_VERSION < 6 )
    (*outData)->SetScalarType(this->OutputDataType);
    (*outData)->AllocateScalars();
#else
    (*outData)->AllocateScalars(this->OutputDataType, NumComponents);
#endif
  }

  this->Lock->Initialize(this->GetNumberOfThreads());
  this->Lock->Enter();

  //call typed method
  switch (DataType) {
    vtkTemplateMacro(vtkImageVoteExecute(this, *inData, *outData, Extent, static_cast<VTK_TT *>(0), threadId ));
    default:
      vtkGenericWarningMacro("Execute: Unknown output ScalarType");
      return;
    }

}
