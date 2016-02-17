#include "vtkCudaHierarchicalMaxFlowDecomposition.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkTreeDFSIterator.h"
#include "vtkTrivialProducer.h"
#include "vtkSmartPointer.h"

#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include <set>
#include <list>

#include "CUDA_hierarchicalmaxflowdecomp.h"

#define SQR(X) X*X

vtkStandardNewMacro(vtkCudaHierarchicalMaxFlowDecomposition);

vtkCudaHierarchicalMaxFlowDecomposition::vtkCudaHierarchicalMaxFlowDecomposition(){

  //configure the IO ports
  this->SetNumberOfInputPorts(3);
  this->SetNumberOfOutputPorts(0);

  //set up the input mapping structure
  this->InputDataPortMapping.clear();
  this->BackwardsInputDataPortMapping.clear();
  this->FirstUnusedDataPort = 0;
  this->InputSmoothnessPortMapping.clear();
  this->BackwardsInputSmoothnessPortMapping.clear();
  this->FirstUnusedSmoothnessPort = 0;
  this->InputLabelPortMapping.clear();
  this->BackwardsInputLabelPortMapping.clear();
  this->FirstUnusedLabelPort = 0;

  //set the other values to defaults
  this->Hierarchy = 0;
  this->LeafMap.clear();
  this->BranchMap.clear();

  F0 = 0.0;
  F = 0;

}

vtkCudaHierarchicalMaxFlowDecomposition::~vtkCudaHierarchicalMaxFlowDecomposition(){
  if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
  this->InputDataPortMapping.clear();
  this->BackwardsInputDataPortMapping.clear();
  this->InputSmoothnessPortMapping.clear();
  this->BackwardsInputSmoothnessPortMapping.clear();
  this->InputLabelPortMapping.clear();
  this->BackwardsInputLabelPortMapping.clear();
  this->LeafMap.clear();
  this->BranchMap.clear();

  if( this->F )
    delete this->F;
}

void vtkCudaHierarchicalMaxFlowDecomposition::Reinitialize(int withData = 0){
  //no long-term data stored and no helper classes, so no body for this method
}

void vtkCudaHierarchicalMaxFlowDecomposition::Deinitialize(int withData = 0){
  //no long-term data stored and no helper classes, so no body for this method
}
//------------------------------------------------------------

void vtkCudaHierarchicalMaxFlowDecomposition::SetHierarchy(vtkTree* graph){
  if( graph != this->Hierarchy ){
    if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
    this->Hierarchy = graph;
    if( this->Hierarchy ) this->Hierarchy->Register(this);
    this->Modified();

    //update output mapping
    this->LeafMap.clear();
    vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
    iterator->SetTree(this->Hierarchy);
    iterator->SetStartVertex(this->Hierarchy->GetRoot());
    while( iterator->HasNext() ){
      vtkIdType node = iterator->Next();
      if( this->Hierarchy->IsLeaf(node) )
        this->LeafMap[node] = (int) this->LeafMap.size();
    }
    iterator->Delete();

    //update number of output ports
    this->SetNumberOfOutputPorts((int) this->LeafMap.size());

  }
}

vtkTree* vtkCudaHierarchicalMaxFlowDecomposition::GetHierarchy(){
  return this->Hierarchy;
}

//------------------------------------------------------------

int vtkCudaHierarchicalMaxFlowDecomposition::FillInputPortInformation(int i, vtkInformation* info){
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  return this->Superclass::FillInputPortInformation(i,info);
}

void vtkCudaHierarchicalMaxFlowDecomposition::SetDataInput(int idx, vtkDataObject *input)
{
  //we are adding/switching an input, so no need to resort list
  if( input != NULL )
  {
    //if their is no pair in the mapping, create one
    if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
    {
      int portNumber = this->FirstUnusedDataPort;
      this->FirstUnusedDataPort++;
      this->InputDataPortMapping[idx] = portNumber;
      this->BackwardsInputDataPortMapping[portNumber] = idx;
    }
#if (VTK_MAJOR_VERSION < 6)
    this->SetNthInputConnection(2, this->InputDataPortMapping[idx], input->GetProducerPort() );
#else
    vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
    tp->SetInputDataObject(input);
    this->SetNthInputConnection(2, this->InputDataPortMapping[idx], tp->GetOutputPort() );
#endif
  }
  else
  {
    //if there is no pair in the mapping, just exit, nothing to do
    if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() ) 
      return;

    int portNumber = this->InputDataPortMapping[idx];
    this->InputDataPortMapping.erase(this->InputDataPortMapping.find(idx));
    this->BackwardsInputDataPortMapping.erase(this->BackwardsInputDataPortMapping.find(portNumber));

    //if we are the last input, no need to reshuffle
    if(portNumber == this->FirstUnusedDataPort - 1)
    {
      this->SetNthInputConnection(0, portNumber,  0);
    }
    else
    {
      //if we are not, move the last input into this spot
      vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedDataPort - 1));
#if (VTK_MAJOR_VERSION < 6)
      this->SetNthInputConnection(2, this->InputSmoothnessPortMapping[idx], swappedInput->GetProducerPort() );
#else
      vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
      tp->SetInputDataObject(swappedInput);
      this->SetNthInputConnection(2, this->InputSmoothnessPortMapping[idx], tp->GetOutputPort() );
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

void vtkCudaHierarchicalMaxFlowDecomposition::SetSmoothnessInput(int idx, vtkDataObject *input)
{
  //we are adding/switching an input, so no need to resort list
  if( input != NULL ){

    //if their is no pair in the mapping, create one
    if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() ){
      int portNumber = this->FirstUnusedSmoothnessPort;
      this->FirstUnusedSmoothnessPort++;
      this->InputSmoothnessPortMapping[idx] = portNumber;
      this->BackwardsInputSmoothnessPortMapping[portNumber] = idx;
    }
#if (VTK_MAJOR_VERSION < 6)
    this->SetNthInputConnection(2, this->InputSmoothnessPortMapping[idx], input->GetProducerPort() );
#else
    vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
    tp->SetInputDataObject(input);
    this->SetNthInputConnection(2, this->InputSmoothnessPortMapping[idx], tp->GetOutputPort() );
#endif

  }else{
    //if there is no pair in the mapping, just exit, nothing to do
    if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() ) return;

    int portNumber = this->InputSmoothnessPortMapping[idx];
    this->InputSmoothnessPortMapping.erase(this->InputSmoothnessPortMapping.find(idx));
    this->BackwardsInputSmoothnessPortMapping.erase(this->BackwardsInputSmoothnessPortMapping.find(portNumber));

    //if we are the last input, no need to reshuffle
    if(portNumber == this->FirstUnusedSmoothnessPort - 1){
      this->SetNthInputConnection(2, portNumber,  0);

      //if we are not, move the last input into this spot
    }else{
      vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedSmoothnessPort - 1));
#if (VTK_MAJOR_VERSION < 6)
      this->SetNthInputConnection(2, portNumber, swappedInput->GetProducerPort() );
#else
      vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
      tp->SetInputDataObject(swappedInput);
      this->SetNthInputConnection(2, portNumber, tp->GetOutputPort() );
#endif
      this->SetNthInputConnection(2, this->FirstUnusedSmoothnessPort - 1, 0 );

      //correct the mappings
      vtkIdType swappedId = this->BackwardsInputSmoothnessPortMapping[this->FirstUnusedSmoothnessPort - 1];
      //this->InputSmoothnessPortMapping.erase(this->InputSmoothnessPortMapping.find(swappedId));
      this->BackwardsInputSmoothnessPortMapping.erase(this->BackwardsInputSmoothnessPortMapping.find(this->FirstUnusedSmoothnessPort - 1));
      this->InputSmoothnessPortMapping[swappedId] = portNumber;
      this->BackwardsInputSmoothnessPortMapping[portNumber] = swappedId;

    }

    //decrement the number of unused ports
    this->FirstUnusedSmoothnessPort--;

  }
}

vtkDataObject *vtkCudaHierarchicalMaxFlowDecomposition::GetDataInput(int idx)
{
  if( this->InputDataPortMapping.find(idx) == this->InputDataPortMapping.end() )
    return 0;
  return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputDataPortMapping[idx]));
}

vtkDataObject *vtkCudaHierarchicalMaxFlowDecomposition::GetSmoothnessInput(int idx)
{
  if( this->InputSmoothnessPortMapping.find(idx) == this->InputSmoothnessPortMapping.end() )
    return 0;
  return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(2, this->InputSmoothnessPortMapping[idx]));
}

void vtkCudaHierarchicalMaxFlowDecomposition::SetLabelInput(int idx, vtkDataObject *input)
{
  //we are adding/switching an input, so no need to resort list
  if( input != NULL )
  {
    //if their is no pair in the mapping, create one
    if( this->InputLabelPortMapping.find(idx) == this->InputLabelPortMapping.end() )
    {
      int portNumber = this->FirstUnusedLabelPort;
      this->FirstUnusedLabelPort++;
      this->InputLabelPortMapping[idx] = portNumber;
      this->BackwardsInputLabelPortMapping[portNumber] = idx;
    }
#if (VTK_MAJOR_VERSION < 6 )
    this->SetNthInputConnection(1, this->InputLabelPortMapping[idx], input->GetProducerPort() );
#else
    vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
    tp->SetInputDataObject(input);
    this->SetNthInputConnection(1, this->InputLabelPortMapping[idx], tp->GetOutputPort() );
#endif
  }
  else
  {
    //if their is no pair in the mapping, just exit, nothing to do
    if( this->InputLabelPortMapping.find(idx) == this->InputLabelPortMapping.end() )
      return;

    int portNumber = this->InputLabelPortMapping[idx];
    this->InputLabelPortMapping.erase(this->InputLabelPortMapping.find(idx));
    this->BackwardsInputLabelPortMapping.erase(this->BackwardsInputLabelPortMapping.find(portNumber));

    //if we are the last input, no need to reshuffle
    if(portNumber == this->FirstUnusedLabelPort - 1){
      this->SetNthInputConnection(1, portNumber,  0);

      //if we are not, move the last input into this spot
    }
    else
    {
      vtkImageData* swappedInput = vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->FirstUnusedLabelPort - 1));
#if (VTK_MAJOR_VERSION < 6)
      this->SetNthInputConnection(1, portNumber, swappedInput->GetProducerPort() );
#else
      vtkSmartPointer<vtkTrivialProducer> tp = vtkSmartPointer<vtkTrivialProducer>::New();
      tp->SetInputDataObject(swappedInput);
      this->SetNthInputConnection(1, portNumber, tp->GetOutputPort() );
#endif
      this->SetNthInputConnection(1, this->FirstUnusedLabelPort - 1, 0 );

      //correct the mappings
      vtkIdType swappedId = this->BackwardsInputLabelPortMapping[this->FirstUnusedLabelPort - 1];
      //this->InputLabelPortMapping.erase(this->InputLabelPortMapping.find(swappedId));
      //this->BackwardsInputLabelPortMapping.erase(this->BackwardsInputLabelPortMapping.find(this->FirstUnusedLabelPort - 1));
      this->InputLabelPortMapping[swappedId] = portNumber;
      this->BackwardsInputLabelPortMapping[portNumber] = swappedId;

    }

    //decrement the number of unused ports
    this->FirstUnusedLabelPort--;
  }
}

vtkDataObject *vtkCudaHierarchicalMaxFlowDecomposition::GetLabelInput(int idx)
{
  if( this->InputLabelPortMapping.find(idx) == this->InputLabelPortMapping.end() )
    return 0;
  return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(1, this->InputLabelPortMapping[idx]));
}

double vtkCudaHierarchicalMaxFlowDecomposition::GetF(vtkIdType n){
  if( !this->Hierarchy ){
    vtkErrorMacro(<<"Hierarchy must be provided.");
    return 0.0;
  }
  if( n == this->Hierarchy->GetRoot() ){
    vtkErrorMacro(<<"Smoothness term not found.");
    return 0.0;
  }
  if( this->BranchMap.find(n) == this->BranchMap.end() && this->LeafMap.find(n) == this->LeafMap.end() ){
    vtkErrorMacro(<<"Smoothness term not found.");
    return 0.0;
  }
  this->Update();
  if( this->Hierarchy->IsLeaf(n) )
    return this->F[this->LeafMap[n] + (int) this->BranchMap.size()];
  return this->F[this->BranchMap[n]];
}

double vtkCudaHierarchicalMaxFlowDecomposition::GetF0(){
  if( !this->Hierarchy ){
    vtkErrorMacro(<<"Hierarchy must be provided.");
    return 0.0;
  }
  this->Update();
  return this->F0;
}

//----------------------------------------------------------------------------

int vtkCudaHierarchicalMaxFlowDecomposition::CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int& NumNodes, int& NumLeaves, int& NumEdges ){

  //check to make sure that the hierarchy is specified
  if( !this->Hierarchy ){
    vtkErrorMacro(<<"Hierarchy must be provided.");
    return -1;
  }

  //check to make sure that there is an image associated with each leaf node
  NumLeaves = 0;
  NumNodes = 0;
  Extent[0] = -1;
  vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
  iterator->SetTree(this->Hierarchy);
  iterator->SetStartVertex(this->Hierarchy->GetRoot());
  while( iterator->HasNext() ){
    vtkIdType node = iterator->Next();
    NumNodes++;


    //make sure all leaf nodes have a data term
    if( this->Hierarchy->IsLeaf(node) ){
      NumLeaves++;

      if( this->InputDataPortMapping.find(node) == this->InputDataPortMapping.end() ){
        vtkErrorMacro(<<"Missing data prior for leaf node.");
        return -1;
      }
      if( this->InputLabelPortMapping.find(node) == this->InputLabelPortMapping.end() ){
        vtkErrorMacro(<<"Missing label map for leaf node.");
        return -1;
      }

      int inputPortNumber = this->InputDataPortMapping[node];

      if( !(inputVector[0])->GetInformationObject(inputPortNumber) && (inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ){
        vtkErrorMacro(<<"Missing data prior for leaf node.");
        return -1;
      }

      int inputLabelPortNumber = this->InputLabelPortMapping[node];

      if( !(inputVector[1])->GetInformationObject(inputPortNumber) && (inputVector[1])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ){
        vtkErrorMacro(<<"Missing label map for leaf node.");
        return -1;
      }


    }

    //check validity of data terms
    if( this->InputDataPortMapping.find(node) != this->InputDataPortMapping.end() ){
      int inputPortNumber = this->InputDataPortMapping[node];
      if( (inputVector[0])->GetInformationObject(inputPortNumber) &&
        (inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ){

          //check for right scalar type
          vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
          if( CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1 ){
            vtkErrorMacro(<<"Data type must be FLOAT and only have one component.");
            return -1;
          }
          if( CurrImage->GetScalarRange()[0] < 0.0 ){
            vtkErrorMacro(<<"Data prior must be non-negative.");
            return -1;
          }

          //check to make sure that the sizes are consistant
          if( Extent[0] == -1 ){
            CurrImage->GetExtent(Extent);
          }else{
            int CurrExtent[6];
            CurrImage->GetExtent(CurrExtent);
            if( CurrExtent[0] != Extent[0] || CurrExtent[1] != Extent[1] || CurrExtent[2] != Extent[2] ||
              CurrExtent[3] != Extent[3] || CurrExtent[4] != Extent[4] || CurrExtent[5] != Extent[5] ){
                vtkErrorMacro(<<"Inconsistant object extent.");
                return -1;
            }
          }

      }
    }

    //check validity of label terms
    if( this->InputLabelPortMapping.find(node) != this->InputLabelPortMapping.end() ){
      int inputPortNumber = this->InputLabelPortMapping[node];
      if( (inputVector[1])->GetInformationObject(inputPortNumber) &&
        (inputVector[1])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ){

          //check for right scalar type
          vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
          if( CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1 ){
            vtkErrorMacro(<<"Label type must be FLOAT and only have one component.");
            return -1;
          }
          if( CurrImage->GetScalarRange()[0] < 0.0 ){
            vtkErrorMacro(<<"Label map must be non-negative.");
            return -1;
          }

          //check to make sure that the sizes are consistant
          if( Extent[0] == -1 ){
            CurrImage->GetExtent(Extent);
          }else{
            int CurrExtent[6];
            CurrImage->GetExtent(CurrExtent);
            if( CurrExtent[0] != Extent[0] || CurrExtent[1] != Extent[1] || CurrExtent[2] != Extent[2] ||
              CurrExtent[3] != Extent[3] || CurrExtent[4] != Extent[4] || CurrExtent[5] != Extent[5] ){
                vtkErrorMacro(<<"Inconsistant object extent.");
                return -1;
            }
          }
      }
    }

    if( this->InputSmoothnessPortMapping.find(node) != this->InputSmoothnessPortMapping.end() ){
      int inputPortNumber = this->InputSmoothnessPortMapping[node];
      if( (inputVector[2])->GetInformationObject(inputPortNumber) &&
        (inputVector[2])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()) ){

          //check for right scalar type
          vtkImageData* CurrImage = vtkImageData::SafeDownCast((inputVector[2])->GetInformationObject(inputPortNumber)->Get(vtkDataObject::DATA_OBJECT()));
          if( CurrImage->GetScalarType() != VTK_FLOAT || CurrImage->GetNumberOfScalarComponents() != 1 ){
            vtkErrorMacro(<<"Smoothness type must be FLOAT and only have one component.");
            return -1;
          }
          if( CurrImage->GetScalarRange()[0] < 0.0 ){
            vtkErrorMacro(<<"Smoothness prior must be non-negative.");
            return -1;
          }

          //check to make sure that the sizes are consistant
          if( Extent[0] == -1 ){
            CurrImage->GetExtent(Extent);
          }else{
            int CurrExtent[6];
            CurrImage->GetExtent(CurrExtent);
            if( CurrExtent[0] != Extent[0] || CurrExtent[1] != Extent[1] || CurrExtent[2] != Extent[2] ||
              CurrExtent[3] != Extent[3] || CurrExtent[4] != Extent[4] || CurrExtent[5] != Extent[5] ){
                vtkErrorMacro(<<"Inconsistant object extent.");
                return -1;
            }
          }
      }
    }

  }
  iterator->Delete();

  NumEdges = NumNodes - 1;

  return 0;
}

int vtkCudaHierarchicalMaxFlowDecomposition::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  //check input for consistancy
  int Extent[6]; int NumNodes; int NumLeaves; int NumEdges;
  int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
  if( result || NumNodes == 0 ) return -1;

  return 1;
}

int vtkCudaHierarchicalMaxFlowDecomposition::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
  //check input for consistancy
  int Extent[6]; int NumNodes; int NumLeaves; int NumEdges;
  int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
  if( result || NumNodes == 0 ) return -1;

  //set up the extents
  for(int i = 0; i < 2; i++ ){
    for(int j = 0; j < this->GetNumberOfInputConnections(i); j++){
      vtkInformation *inputInfo = outputVector->GetInformationObject(i);
      vtkImageData *inputBuffer = vtkImageData::SafeDownCast(inputInfo->Get(vtkDataObject::DATA_OBJECT()));
      inputInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),Extent,6);
    }
  }

  return 1;
}

int vtkCudaHierarchicalMaxFlowDecomposition::RequestData(vtkInformation *request, 
                                                         vtkInformationVector **inputVector, 
                                                         vtkInformationVector *outputVector){

                                                           //check input consistancy
                                                           int Extent[6]; int NumNodes; int NumLeaves; int NumEdges;
                                                           int result = CheckInputConsistancy( inputVector, Extent, NumNodes, NumLeaves, NumEdges );
                                                           if( result || NumNodes == 0 ) return -1;
                                                           int NumBranches = NumNodes - NumLeaves - 1;

                                                           if( this->Debug )
                                                             vtkDebugMacro(<< "Starting input data preparation." );

                                                           //set the number of output ports
                                                           outputVector->SetNumberOfInformationObjects(NumLeaves);
                                                           this->SetNumberOfOutputPorts(NumLeaves);

                                                           //find the size of the volume
                                                           VX = (Extent[1] - Extent[0] + 1);
                                                           VY = (Extent[3] - Extent[2] + 1);
                                                           VZ = (Extent[5] - Extent[4] + 1);
                                                           VolumeSize = VX * VY * VZ;

                                                           //create relevant node identifier to buffer mappings
                                                           this->BranchMap.clear();
                                                           vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
                                                           iterator->SetTree(this->Hierarchy);
                                                           iterator->SetStartVertex(this->Hierarchy->GetRoot());
                                                           while( iterator->HasNext() ){
                                                             vtkIdType node = iterator->Next();
                                                             if( node != this->Hierarchy->GetRoot() && !this->Hierarchy->IsLeaf(node) )
                                                               BranchMap[node] = (int) this->BranchMap.size();
                                                           }
                                                           iterator->Delete();

                                                           //get the data term buffers and figure out F0
                                                           this->F0 = 0.0;
                                                           iterator = vtkTreeDFSIterator::New();
                                                           iterator->SetTree(this->Hierarchy);
                                                           while( iterator->HasNext() ){
                                                             vtkIdType node = iterator->Next();
                                                             if( this->Hierarchy->IsLeaf(node) ){
                                                               vtkImageData* CurrData = vtkImageData::SafeDownCast((inputVector[0])->GetInformationObject(this->InputDataPortMapping[node])->Get(vtkDataObject::DATA_OBJECT()));
                                                               vtkImageData* CurrLabel = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(this->InputLabelPortMapping[node])->Get(vtkDataObject::DATA_OBJECT()));

                                                               //TODO accumulate into F0 term
                                                               this->F0 +=  CUDA_GHMFD_DataTermForLabel((float*) CurrData->GetScalarPointer(), (float*) CurrLabel->GetScalarPointer(), this->VolumeSize, this->GetStream());
                                                             }
                                                           }
                                                           iterator->Delete();

                                                           //get the smoothness term buffers
                                                           this->leafSmoothnessTermBuffers = new float* [NumLeaves];
                                                           this->branchSmoothnessTermBuffers = new float* [NumBranches];
                                                           iterator = vtkTreeDFSIterator::New();
                                                           iterator->SetTree(this->Hierarchy);
                                                           while( iterator->HasNext() ){
                                                             vtkIdType node = iterator->Next();
                                                             if( node == this->Hierarchy->GetRoot() ) continue;

                                                             vtkImageData* CurrImage = 0;
                                                             if( this->InputSmoothnessPortMapping.find(node) != this->InputSmoothnessPortMapping.end() ){
                                                               int inputNumber = this->InputSmoothnessPortMapping[node];
                                                               CurrImage = vtkImageData::SafeDownCast((inputVector[2])->GetInformationObject(inputNumber)->Get(vtkDataObject::DATA_OBJECT()));
                                                             }
                                                             if( this->Hierarchy->IsLeaf(node) )
                                                               leafSmoothnessTermBuffers[this->LeafMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
                                                             else
                                                               branchSmoothnessTermBuffers[this->BranchMap[node]] = CurrImage ? (float*) CurrImage->GetScalarPointer() : 0;
                                                           }
                                                           iterator->Delete();

                                                           //clear container for holding the results
                                                           if( this->F )
                                                             delete this->F;
                                                           this->F = new double[NumBranches+NumLeaves];
                                                           for(int i = 0; i < NumBranches+NumLeaves; i++ ) this->F[i] = 0.0;

                                                           //Figure out smoothness Fi's
                                                           this->devGradientBuffer = CUDA_GHMFD_GetBuffer( VolumeSize, this->GetStream() );
                                                           this->BranchLabelBuffer.clear();
                                                           FigureOutSmoothness( this->Hierarchy->GetRoot(), inputVector );
                                                           this->BranchLabelBuffer.clear();
                                                           CUDA_GHMFD_ReturnBuffer( this->devGradientBuffer );

                                                           //deallocate temporary variables
                                                           delete[] leafSmoothnessTermBuffers;
                                                           delete[] branchSmoothnessTermBuffers;

                                                           return 1;
}

int vtkCudaHierarchicalMaxFlowDecomposition::RequestDataObject(
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

void vtkCudaHierarchicalMaxFlowDecomposition::FigureOutSmoothness(vtkIdType currNode, vtkInformationVector **inputVector){

  //pass smoothness terms onto children
  int NumKids = this->Hierarchy->GetNumberOfChildren(currNode);
  vtkIdType Parent = this->Hierarchy->GetParent(currNode);
  for(int kid = 0; kid < NumKids; kid++)
    FigureOutSmoothness( this->Hierarchy->GetChild(currNode,kid), inputVector );

  //if we are the source node, we contribute nothing to smoothness
  if( currNode == this->Hierarchy->GetRoot() ) return;

  //create parent label if not already made
  if( Parent != this->Hierarchy->GetRoot() && !this->BranchLabelBuffer[Parent] )
    this->BranchLabelBuffer[Parent] = CUDA_GHMFD_GetBuffer( this->VolumeSize, this->GetStream() );

  //if we are a child, we have no need for label summation
  if(NumKids == 0){

    //calculate value and add labelling to parent's buffer
    vtkImageData* CurrLabel = vtkImageData::SafeDownCast((inputVector[1])->GetInformationObject(this->InputLabelPortMapping[currNode])->Get(vtkDataObject::DATA_OBJECT()));
    int SmoothnessTermUsed = this->LeafMap[currNode] + (int) this->BranchMap.size();
    if( this->leafSmoothnessTermBuffers[this->LeafMap[currNode]] )
      this->F[SmoothnessTermUsed] = CUDA_GHMFD_LeafSmoothnessForLabel(this->leafSmoothnessTermBuffers[this->LeafMap[currNode]], (float*) CurrLabel->GetScalarPointer(),
      this->VX, this->VY, this->VZ, this->VolumeSize, this->BranchLabelBuffer[Parent], this->devGradientBuffer, this->GetStream());
    else
      this->F[SmoothnessTermUsed] = CUDA_GHMFD_LeafNoSmoothnessForLabel( (float*) CurrLabel->GetScalarPointer(),
      this->VX, this->VY, this->VZ, this->VolumeSize, this->BranchLabelBuffer[Parent], this->devGradientBuffer, this->GetStream());

    //if we are a branch, we may need to sum and store child labels then compute terms
  }else{

    //calculate value and add labelling to parent's buffer
    int SmoothnessTermUsed = this->BranchMap[currNode];
    if( this->branchSmoothnessTermBuffers[this->BranchMap[currNode]] )
      this->F[SmoothnessTermUsed] = CUDA_GHMFD_BranchSmoothnessForLabel(this->branchSmoothnessTermBuffers[this->BranchMap[currNode]], this->BranchLabelBuffer[currNode],
      this->VX, this->VY, this->VZ, this->VolumeSize, this->BranchLabelBuffer[Parent], this->devGradientBuffer, this->GetStream());
    else
      this->F[SmoothnessTermUsed] = CUDA_GHMFD_BranchNoSmoothnessForLabel( this->BranchLabelBuffer[currNode],
      this->VX, this->VY, this->VZ, this->VolumeSize, this->BranchLabelBuffer[Parent], this->devGradientBuffer, this->GetStream());

    //return own label buffer
    CUDA_GHMFD_ReturnBuffer( this->BranchLabelBuffer[currNode] );
    this->BranchLabelBuffer.erase( currNode );
  }


}