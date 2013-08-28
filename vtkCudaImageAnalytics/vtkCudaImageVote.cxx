/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaImageVote.cxx

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaImageVote.cxx
 *
 *  @brief Implementation file with definitions for the CUDA accelerated voting operation. This module
 *			Takes a probabilistic or weighted image, and replaces each voxel with a label corresponding
 *			to the input image with the highest value at that location. ( argmax{} operation )
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *	
 *	@note August 27th 2013 - Documentation first compiled.
 *
 *  @note This is the base class for GPU accelerated max-flow segmentors in vtkCudaImageAnalytics
 *
 */

#include "vtkCudaImageVote.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkTreeDFSIterator.h"

#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include "CUDA_imagevote.h"

#define SQR(X) X*X

vtkStandardNewMacro(vtkCudaImageVote);

vtkCudaImageVote::vtkCudaImageVote(){
	
	//configure the IO ports
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);

	//set up the input mapping structure
	this->InputPortMapping.clear();
	this->BackwardsInputPortMapping.clear();
	this->FirstUnusedPort = 0;

	this->OutputDataType = VTK_SHORT;

}

vtkCudaImageVote::~vtkCudaImageVote(){
	this->InputPortMapping.clear();
	this->BackwardsInputPortMapping.clear();
}

//------------------------------------------------------------

int vtkCudaImageVote::FillInputPortInformation(int i, vtkInformation* info){
	info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
	info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
	info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
	return this->Superclass::FillInputPortInformation(i,info);
}

void vtkCudaImageVote::SetInput(int idx, vtkDataObject *input)
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
		this->SetNthInputConnection(0, this->InputPortMapping.find(idx)->second, input->GetProducerPort() );

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
			this->SetNthInputConnection(0, portNumber, swappedInput->GetProducerPort() );
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

vtkDataObject *vtkCudaImageVote::GetInput(int idx)
{
	if( this->InputPortMapping.find(idx) == this->InputPortMapping.end() )
		return 0;
	return vtkImageData::SafeDownCast( this->GetExecutive()->GetInputData(0, this->InputPortMapping.find(idx)->second));
}

//----------------------------------------------------------------------------

int vtkCudaImageVote::CheckInputConsistancy( vtkInformationVector** inputVector, int* Extent, int* NumLabels, int* DataType){
	
	*DataType = -1;
	Extent[0] = -1;
	*NumLabels = (int) this->InputPortMapping.size();

	//make sure that every image is the correct size and same datatype
	for(int inputPortNumber = 0; inputPortNumber < this->InputPortMapping.size(); inputPortNumber++){
		
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

	}

	return 0;
}

int vtkCudaImageVote::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	//check input for consistancy
	int Extent[6]; int NumLabels; int DataType;
	int result = CheckInputConsistancy( inputVector, Extent, &NumLabels, &DataType );
	if( result || NumLabels == 0 ) return -1;				

	return 1;
}

int vtkCudaImageVote::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	//check input for consistancy
	int Extent[6]; int NumLabels; int DataType;
	int result = CheckInputConsistancy( inputVector, Extent, &NumLabels, &DataType );
	if( result || NumLabels == 0 ) return -1;				

	//set up the extents
	for(int inputPortNumber = 0; inputPortNumber < this->InputPortMapping.size(); inputPortNumber++)
		(inputVector[0])->GetInformationObject(inputPortNumber)->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),Extent,6);

	return 1;
}


template<class T>
void vtkCudaImageVoteExecute(vtkCudaImageVote *self,
				vtkImageData **inData,
				vtkImageData *outData,
				T* unUsed){

	T** inPtr = new T* [self->GetNumberOfInputConnections(0)];
	for(int i = 0; i < self->GetNumberOfInputConnections(0); i++ )
		inPtr[i] = (T*) inData[i]->GetScalarPointer();

	void* outPtr = outData->GetScalarPointer();

	switch (outData->GetScalarType()) {
		vtkTemplateMacro(vtkCudaImageVoteExecute(self, inData, inPtr, outData, static_cast<VTK_TT *>(outPtr) ));
		default:
		  vtkGenericWarningMacro("Execute: Unknown output ScalarType");
		  return;
    }

	delete[] inPtr;
}

template <typename IT, typename OT>
void vtkCudaImageVoteExecute(vtkCudaImageVote *self,
				vtkImageData **inData, IT **inPtr,
				vtkImageData *outData, OT *outPtr ){
	int outExt[6];
	outData->GetExtent(outExt);
	int VolumeSize = (outExt[1]-outExt[0]+1)*(outExt[3]-outExt[2]+1)*(outExt[5]-outExt[4]+1);
	int inputNumber = self->GetNumberOfInputConnections(0);

	//compute the map
	OT* map = new OT[inputNumber];
	for( int iv = 0; iv < inputNumber; iv++ )
		map[iv] = self->GetMappedTerm<OT>(iv);

	//perform voting
	CUDA_CIV_COMPUTE( inPtr, inputNumber, outPtr, map, VolumeSize, self->GetStream() );
	
	//deallocate the temporary storage with the map
	delete[] map;
}

int vtkCudaImageVote::RequestData(vtkInformation *request,
                                     vtkInformationVector **inputVector,
                                     vtkInformationVector *outputVector){

	//check input for consistancy
	int Extent[6]; int NumLabels; int DataType;
	int result = CheckInputConsistancy( inputVector, Extent, &NumLabels, &DataType );
	if( result || NumLabels == 0 ) return -1;				
	
	vtkImageData* outData = vtkImageData::SafeDownCast(outputVector->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData** inData =  new vtkImageData* [NumLabels];
	for(int i = 0; i < NumLabels; i++)
		inData[i] = vtkImageData::SafeDownCast(inputVector[0]->GetInformationObject(i)->Get(vtkDataObject::DATA_OBJECT()));

	//allocate output image (using short)
	outData->SetScalarType(this->OutputDataType);
	outData->SetExtent(Extent);
	outData->SetWholeExtent(Extent);
	outData->AllocateScalars();

	//call typed method
	switch (DataType) {
		vtkTemplateMacro(vtkCudaImageVoteExecute(this, inData, outData, static_cast<VTK_TT *>(0) ));
		default:
		  vtkGenericWarningMacro("Execute: Unknown output ScalarType");
		  return -1;
    }

	delete[] inData;
	return 1;

}
