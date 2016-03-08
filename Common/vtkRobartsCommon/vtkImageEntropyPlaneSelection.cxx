/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImageEntropyPlaneSelection.cxx

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkImageEntropyPlaneSelection.cxx
 *
 *  @brief Implementation file with definitions for a class to select orthogonal, axis-aligned
 *      planes for plane selection purposes based on the entropy of the probabilistic
 *      segmentation.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  
 *  @note September 5th 2013 - Documentation first compiled. (jshbaxter)
 *
 */

#include "vtkImageData.h"
#include "vtkImageEntropyPlaneSelection.h"
#include "vtkObjectFactory.h"
#include "vtkSmartPointer.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTrivialProducer.h"
#include "vtkVersionMacros.h"

vtkStandardNewMacro(vtkImageEntropyPlaneSelection);

vtkImageEntropyPlaneSelection::vtkImageEntropyPlaneSelection(){
  
  //configure the IO ports
  this->SetNumberOfInputPorts(1);
  this->SetNumberOfOutputPorts(0);
  this->InputDataPortMapping.clear();
  this->BackwardsInputDataPortMapping.clear();
  this->FirstUnusedDataPort = 0;

  EntropyInX = 0;
  EntropyInY = 0;
  EntropyInZ = 0;
  
  Extent[0] = Extent[2] = Extent[4] = -1;
  Extent[1] = Extent[3] = Extent[5] = 0;

}


vtkImageEntropyPlaneSelection::~vtkImageEntropyPlaneSelection(){
  if( EntropyInX ) delete EntropyInX;
  if( EntropyInY ) delete EntropyInY;
  if( EntropyInZ ) delete EntropyInZ;
}
//------------------------------------------------------------

int vtkImageEntropyPlaneSelection::FillInputPortInformation(int i, vtkInformation* info){
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  return this->Superclass::FillInputPortInformation(i,info);
}

void vtkImageEntropyPlaneSelection::SetInput(int idx, vtkDataObject *input)
{
  //we are adding/switching an input, so no need to resort list
  if( input != NULL ){
  
    //if their is no pair in the mapping, create one
    if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() ){
      int portNumber = this->FirstUnusedDataPort;
      this->FirstUnusedDataPort++;
      this->InputDataPortMapping[idx] = portNumber;
      this->BackwardsInputDataPortMapping[portNumber] = idx;
    }
#if (VTK_MAJOR_VERSION < 6)
    this->SetNthInputConnection(0, this->InputDataPortMapping[idx], input->GetProducerPort() );
#else
    vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
    tp->SetInputDataObject(input);
    this->SetNthInputConnection(0, this->InputDataPortMapping[idx], tp->GetOutputPort() );
#endif

  }else{
    //if there is no pair in the mapping, just exit, nothing to do
    if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() ) return;

    int portNumber = this->InputDataPortMapping[idx];
    this->InputDataPortMapping.erase(this->InputDataPortMapping.find(idx));
    this->BackwardsInputDataPortMapping.erase(this->BackwardsInputDataPortMapping.find(portNumber));

    //if we are the last input, no need to reshuffle
    if(portNumber == this->FirstUnusedDataPort - 1){
      this->SetNthInputConnection(0, portNumber,  0);
    
    //if we are not, move the last input into this spot
    }else{
      vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedDataPort - 1));
#if (VTK_MAJOR_VERSION < 6)
      this->SetNthInputConnection(0, portNumber, swappedInput->GetProducerPort() );
#else
      vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
      tp->SetInputDataObject(swappedInput);
      this->SetNthInputConnection(0, portNumber, tp->GetOutputPort() );
#endif
      this->SetNthInputConnection(0, this->FirstUnusedDataPort - 1, 0 );

      //correct the mappings
      vtkIdType swappedId = this->BackwardsInputDataPortMapping[this->FirstUnusedDataPort - 1];
      //this->InputDataPortMapping.erase(this->InputDataPortMapping.find(swappedId));
      this->BackwardsInputDataPortMapping.erase(this->BackwardsInputDataPortMapping.find(this->FirstUnusedDataPort - 1));
      this->InputDataPortMapping[swappedId] = portNumber;
      this->BackwardsInputDataPortMapping[portNumber] = swappedId;

    }

    //decrement the number of unused ports
    this->FirstUnusedDataPort--;

  }
}

vtkDataObject *vtkImageEntropyPlaneSelection::GetInput(int idx)
{
  if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
    return 0;
  return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputDataPortMapping[idx]));
}

//------------------------------------------------------------

inline double GetFromArray(double* values, int index, int reference){
  return values[index-reference];
}

double vtkImageEntropyPlaneSelection::GetEntropyInX(int slice){
  if( slice < Extent[0] || slice > Extent[1] || !EntropyInX ) return -1.0;
  return GetFromArray(EntropyInX,slice,Extent[0]);
}

double vtkImageEntropyPlaneSelection::GetEntropyInY(int slice){
  if( slice < Extent[2] || slice > Extent[3] || !EntropyInY ) return -1.0;
  return GetFromArray(EntropyInY,slice,Extent[2]);
}

double vtkImageEntropyPlaneSelection::GetEntropyInZ(int slice){
  if( slice < Extent[4] || slice > Extent[5] || !EntropyInZ ) return -1.0;
  return GetFromArray(EntropyInZ,slice,Extent[4]);
}

int vtkImageEntropyPlaneSelection::GetSliceInX(){
  if( !EntropyInX ) return -1;
  double maxEntropy = 0.0;
  int slice = Extent[0];
  for(int i = Extent[0]; i <= Extent[1]; i++){
    if( GetFromArray(EntropyInX,i,Extent[0]) > maxEntropy ){
      maxEntropy = GetFromArray(EntropyInX,i,Extent[0]);
      slice = i;
    }
  }
  return slice;
}

int vtkImageEntropyPlaneSelection::GetSliceInY(){
  if( !EntropyInY ) return -1;
  double maxEntropy = 0.0;
  int slice = Extent[2];
  for(int i = Extent[2]; i <= Extent[3]; i++){
    if( GetFromArray(EntropyInY,i,Extent[2]) > maxEntropy ){
      maxEntropy = GetFromArray(EntropyInY,i,Extent[2]);
      slice = i;
    }
  }
  return slice;
}

int vtkImageEntropyPlaneSelection::GetSliceInZ(){
  if( !EntropyInZ ) return -1;
  double maxEntropy = 0.0;
  int slice = Extent[4];
  for(int i = Extent[4]; i <= Extent[5]; i++){
    if( GetFromArray(EntropyInZ,i,Extent[4]) > maxEntropy ){
      maxEntropy = GetFromArray(EntropyInZ,i,Extent[4]);
      slice = i;
    }
  }
  return slice;
}

//------------------------------------------------------------

int vtkImageEntropyPlaneSelection::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  return 1;
}

int vtkImageEntropyPlaneSelection::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{

  //set up the extents
  for(int j = 0; j < this->GetNumberOfInputConnections(0); j++){
    vtkInformation *inputInfo = inputVector[0]->GetInformationObject(j);
    vtkImageData *inputBuffer = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
    inputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),inputBuffer->GetExtent(),6);
  }

  return 1;
}

int vtkImageEntropyPlaneSelection::RequestData(vtkInformation *request, 
              vtkInformationVector **inputVector, 
              vtkInformationVector *outputVector){
                
  //get extent
  vtkInformation *inputInfo = (inputVector[0])->GetInformationObject(0);
  vtkImageData *inputBuffer = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
  inputBuffer->GetExtent(Extent);

  //create the buffers for the accumulated entropy
  int VolumeSize = (Extent[1]-Extent[0]+1)*(Extent[3]-Extent[2]+1)*(Extent[5]-Extent[4]+1);
  this->EntropyInX = new double[Extent[1]-Extent[0]+1];
  this->EntropyInY = new double[Extent[3]-Extent[2]+1];
  this->EntropyInZ = new double[Extent[5]-Extent[4]+1];
  for(int i = 0; i < Extent[1]-Extent[0]+1; i++)
    this->EntropyInX[i] = 0.0;
  for(int i = 0; i < Extent[3]-Extent[2]+1; i++)
    this->EntropyInY[i] = 0.0;
  for(int i = 0; i < Extent[5]-Extent[4]+1; i++)
    this->EntropyInZ[i] = 0.0;

  //iterate through the images
  int idx = 0;
  for(int z = 0; z < Extent[5]-Extent[4]+1; z++)
  for(int y = 0; y < Extent[3]-Extent[2]+1; y++)
  for(int x = 0; x < Extent[1]-Extent[0]+1; x++, idx++){
    double accumulator = 0.0;
    for(int i = 0; i < this->GetNumberOfInputConnections(0); i++)
      accumulator += ((float*)vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(i)->Get(vtkDataObject::DATA_OBJECT()))->GetScalarPointer())[idx];
    for(int i = 0; i < this->GetNumberOfInputConnections(0); i++){
      float value = ((float*)vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(i)->Get(vtkDataObject::DATA_OBJECT()))->GetScalarPointer())[idx];
      float entropy = (value > 0.0f) ? (double) -value / accumulator * log( (double) value / accumulator ) : 0.0;
      if( accumulator == 0.0 ) entropy = log( (double) this->GetNumberOfInputConnections(0) );
      this->EntropyInX[x] += entropy;
      this->EntropyInY[y] += entropy;
      this->EntropyInZ[z] += entropy;
    }
  }

  return 1;
}

int vtkImageEntropyPlaneSelection::RequestDataObject(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector ,
  vtkInformationVector* outputVector){

  vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
  if (!inInfo)
    return 0;
  vtkImageData *input = vtkImageData::SafeDownCast(inInfo->Get(vtkImageData::DATA_OBJECT()));
 
  if (input)
    return 1;
  return 0;
}