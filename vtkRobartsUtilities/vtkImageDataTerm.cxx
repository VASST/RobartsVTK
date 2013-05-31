/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageDataTerm.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkImageDataTerm.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <math.h>

vtkCxxRevisionMacro(vtkImageDataTerm, "$Revision: 1.56 $");
vtkStandardNewMacro(vtkImageDataTerm);

//----------------------------------------------------------------------------
vtkImageDataTerm::vtkImageDataTerm()
{
  this->Operation = VTK_CONSTANT;
  this->ConstantK1 = 1.0;
  this->ConstantK2 = 1.0;
  this->ConstantC1 = 0.0;
  this->ConstantC2 = 0.0;
  this->Entropy = false;
  this->Normalize = true;
  this->SetNumberOfInputPorts(2);
}

//----------------------------------------------------------------------------
// The output extent is the intersection.
int vtkImageDataTerm::RequestInformation (
  vtkInformation * vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *outputVector)
{
  // get the info objects
  vtkInformation* outInfo = outputVector->GetInformationObject(0);
  vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation *inInfo2 = inputVector[1]->GetInformationObject(0);

  int ext[6], ext2[6], idx;

  inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext);

  // two input take intersection
  if (this->Operation == VTK_LOGISTIC || this->Operation == VTK_GAUSSIAN) {
    if (inInfo2){
      inInfo2->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext2);
      for (idx = 0; idx < 3; ++idx) {
        if (ext2[idx*2] > ext[idx*2]) ext[idx*2] = ext2[idx*2];
        if (ext2[idx*2+1] < ext[idx*2+1]) ext[idx*2+1] = ext2[idx*2+1]; 
      }
    }
  }

  outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext,6);

  return 1;
}


//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
// Handles the one input operations
template <class T>
void vtkImageDataTermExecute1(vtkImageDataTerm *self,
                                 vtkImageData *in1Data, T *in1Ptr,
                                 vtkImageData *outData, T *outPtr,
                                 int outExt[6], int id)
{
  int idxR, idxY, idxZ;
  int maxY, maxZ;
  vtkIdType inIncX, inIncY, inIncZ;
  vtkIdType outIncX, outIncY, outIncZ;
  int rowLength;
  unsigned long count = 0;
  unsigned long target;
  int op = self->GetOperation();
  T* origOutPtr = outPtr;

  // find the region to loop over
  rowLength = (outExt[1] - outExt[0]+1)*in1Data->GetNumberOfScalarComponents();
  maxY = outExt[3] - outExt[2];
  maxZ = outExt[5] - outExt[4];
  target = static_cast<unsigned long>((maxZ+1)*(maxY+1)/50.0);
  target++;

  // Get increments to march through data
  in1Data->GetContinuousIncrements(outExt, inIncX, inIncY, inIncZ);
  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);

  // Get constants from the main class
  bool entropy = self->GetEntropyUsed();
  double constantk1 = self->GetConstantK1();
  double constantc1 = ((op == VTK_CONSTANT || op == VTK_SPIKE) && entropy) ? -log(self->GetConstantC1()) : self->GetConstantC1();
  double constantc2 = ((op == VTK_CONSTANT || op == VTK_SPIKE) && entropy) ? -log(self->GetConstantC2()) : self->GetConstantC1();

  // Loop through output pixels
	switch (op){
	case VTK_CONSTANT:
		for(idxZ = 0; idxZ <= maxZ; idxZ++){
			for (idxY = 0; idxY <= maxY; idxY++){
				if( !id ){
					if (!(count%target)) self->UpdateProgress(count/(50.0*target));
					count++;
				}
				for (idxR = 0; idxR < rowLength; idxR++){
					*outPtr = constantc1;
					outPtr++;
					in1Ptr++;
				}
				outPtr += outIncY;
				in1Ptr += inIncY;
			}
			outPtr += outIncZ;
			in1Ptr += inIncZ;
		}
		break;
		
	case VTK_LOGISTIC:
		for(idxZ = 0; idxZ <= maxZ; idxZ++){
			for (idxY = 0; idxY <= maxY; idxY++){
				if( !id ){
					if (!(count%target)) self->UpdateProgress(count/(50.0*target));
					count++;
				}
				for (idxR = 0; idxR < rowLength; idxR++){
					*outPtr = (T)( entropy ? 
						log(1.0 + exp(-(constantk1*((double)*in1Ptr - constantc1))) ) :
						1.0 / (1.0 + exp(-(constantk1*((double)*in1Ptr - constantc1))) ) );
					outPtr++;
					in1Ptr++;
				}
				outPtr += outIncY;
				in1Ptr += inIncY;
			}
			outPtr += outIncZ;
			in1Ptr += inIncZ;
		}
		break;

	case VTK_GAUSSIAN:
		for(idxZ = 0; idxZ <= maxZ; idxZ++){
			for (idxY = 0; idxY <= maxY; idxY++){
				if( !id ){
					if (!(count%target)) self->UpdateProgress(count/(50.0*target));
					count++;
				}
				for (idxR = 0; idxR < rowLength; idxR++){
					*outPtr = (T)( entropy ? 
						0.5*(((double)*in1Ptr - constantc1)/constantk1) * (((double)*in1Ptr - constantc1)/constantk1) :
						exp(-0.5 * (((double)*in1Ptr - constantc1)/constantk1) * (((double)*in1Ptr - constantc1)/constantk1)));
					outPtr++;
					in1Ptr++;
				}
				outPtr += outIncY;
				in1Ptr += inIncY;
			}
			outPtr += outIncZ;
			in1Ptr += inIncZ;
		}
		break;

	case VTK_SPIKE:
		for(idxZ = 0; idxZ <= maxZ; idxZ++){
			for (idxY = 0; idxY <= maxY; idxY++){
				if( !id ){
					if (!(count%target)) self->UpdateProgress(count/(50.0*target));
					count++;
				}
				for (idxR = 0; idxR < rowLength; idxR++){
					*outPtr = (*in1Ptr == constantk1) ? constantc1: constantc2 ;
					outPtr++;
					in1Ptr++;
				}
				outPtr += outIncY;
				in1Ptr += inIncY;
			}
			outPtr += outIncZ;
			in1Ptr += inIncZ;
		}
		break;
	}
	
	//if we are not normalizing, we can leave now
	if( !self->GetNormalize() ) return;

  // Get normalization constants
  double normOffset = 0.0;
  double normMultiply = 1.0 / constantc1;
  double Range[2]; in1Data->GetScalarRange(Range); 
  if( op == VTK_GAUSSIAN && entropy){
	  double valAtMax = 0.5*((Range[1] - constantc1)/constantk1) * ((Range[1] - constantc1)/constantk1);
	  double valAtMin = 0.5*((Range[0] - constantc1)/constantk1) * ((Range[0] - constantc1)/constantk1);
	  double maxValue = (valAtMax >= valAtMin) ? valAtMax : valAtMin;
	  double minValue = (valAtMax <  valAtMin) ? valAtMax : valAtMin;
	  if( constantc1 > Range[0] && constantc1 < Range[1] ) minValue = 0.0f;
	  normOffset = -minValue;
	  normMultiply = 1.0 / (maxValue-minValue);
  }else if( op == VTK_GAUSSIAN && !entropy){
	  double Range[2]; in1Data->GetScalarRange(Range); 
	  double valAtMax = exp(-0.5 * ((Range[1] - constantc1)/constantk1) * ((Range[1] - constantc1)/constantk1));
	  double valAtMin = exp(-0.5 * ((Range[0] - constantc1)/constantk1) * ((Range[0] - constantc1)/constantk1));
	  double minValue = exp( -((valAtMax >= valAtMin) ? valAtMax : valAtMin) );
	  double maxValue = exp( -((valAtMax <  valAtMin) ? valAtMax : valAtMin) );
	  if( constantc1 > Range[0] && constantc1 < Range[1] ) maxValue = 1.0f;
	  normOffset = -minValue;
	  normMultiply = 1.0 / (maxValue-minValue);
  }else if( op == VTK_LOGISTIC && entropy){	  double Range[2]; in1Data->GetScalarRange(Range); 
	  double valAtMax = log(1.0 + exp(-(constantk1*(Range[1] - constantc1))) );
	  double valAtMin = log(1.0 + exp(-(constantk1*(Range[0] - constantc1))) );
	  double maxValue = (valAtMax >= valAtMin) ? valAtMax : valAtMin;
	  double minValue = (valAtMax <  valAtMin) ? valAtMax : valAtMin;
	  normOffset = -minValue;
	  normMultiply = 1.0 / (maxValue-minValue);
  }else if( op == VTK_LOGISTIC && !entropy){
	  double valAtMax = 1.0 / (1.0 + exp(-(constantk1*(Range[1] - constantc1))));
	  double valAtMin = 1.0 / (1.0 + exp(-(constantk1*(Range[0] - constantc1))));
	  double maxValue = (valAtMax >= valAtMin) ? valAtMax : valAtMin;
	  double minValue = (valAtMax <  valAtMin) ? valAtMax : valAtMin;
	  normOffset = -minValue;
	  normMultiply = 1.0 / (maxValue-minValue);
  }

  //apply the normalization
  outPtr = origOutPtr;
  for(idxZ = 0; idxZ <= maxZ; idxZ++){
	  for (idxY = 0; idxY <= maxY; idxY++){
			if( !id ){
				if (!(count%target)) self->UpdateProgress(count/(50.0*target));
				count++;
			}
			for (idxR = 0; idxR < rowLength; idxR++){
				*outPtr = (T)( ((double)(*outPtr) + normOffset) * normMultiply );
				outPtr++;
			}
			outPtr += outIncY;
		}
		outPtr += outIncZ;
  }

}



//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
// Handles the two input operations
template <class T>
void vtkImageDataTermExecute2(vtkImageDataTerm *self,
                                 vtkImageData *in1Data, T *in1Ptr,
                                 vtkImageData *in2Data, T *in2Ptr,
                                 vtkImageData *outData, T *outPtr,
                                 int outExt[6], int id)
{
  int idxR, idxY, idxZ;
  int maxY, maxZ;
  vtkIdType inIncX, inIncY, inIncZ;
  vtkIdType in2IncX, in2IncY, in2IncZ;
  vtkIdType outIncX, outIncY, outIncZ;
  int rowLength;
  unsigned long count = 0;
  unsigned long target;
  int op = self->GetOperation();
  
  // Get constants from the main class
  bool entropy = self->GetEntropyUsed();
  double constantk1 = self->GetConstantK1();
  double constantc1 = (op == VTK_CONSTANT && entropy) ? -log(self->GetConstantC1()) : self->GetConstantC1();
  double constantk2 = self->GetConstantK2();
  double constantc2 = (op == VTK_CONSTANT && entropy) ? -log(self->GetConstantC2()) : self->GetConstantC2();


  // find the region to loop over
  rowLength = (outExt[1] - outExt[0]+1)*in1Data->GetNumberOfScalarComponents();
  maxY = outExt[3] - outExt[2];
  maxZ = outExt[5] - outExt[4];
  target = static_cast<unsigned long>((maxZ+1)*(maxY+1)/50.0);
  target++;

  // Get increments to march through data
  in1Data->GetContinuousIncrements(outExt, inIncX, inIncY, inIncZ);
  in2Data->GetContinuousIncrements(outExt, in2IncX, in2IncY, in2IncZ);
  outData->GetContinuousIncrements(outExt, outIncX, outIncY, outIncZ);

    // Loop through output pixels
	switch (op){
	case VTK_CONSTANT:
		for(idxZ = 0; idxZ <= maxZ; idxZ++){
			for (idxY = 0; idxY <= maxY; idxY++){
				if( !id ){
					if (!(count%target)) self->UpdateProgress(count/(50.0*target));
					count++;
				}
				for (idxR = 0; idxR < rowLength; idxR++){
					*outPtr = constantc1;
					outPtr++;
					in1Ptr++;
					in2Ptr++;
				}
				outPtr += outIncY;
				in1Ptr += inIncY;
				in2Ptr += in2IncY;
			}
			outPtr += outIncZ;
			in1Ptr += inIncZ;
			in2Ptr += in2IncZ;
		}
		break;
		
	case VTK_LOGISTIC:
		for(idxZ = 0; idxZ <= maxZ; idxZ++){
			for (idxY = 0; idxY <= maxY; idxY++){
				if( !id ){
					if (!(count%target)) self->UpdateProgress(count/(50.0*target));
					count++;
				}
				for (idxR = 0; idxR < rowLength; idxR++){
					*outPtr = (T)( entropy ? 
						log((1.0 + exp(-(constantk1*((double)*in1Ptr - constantc1)))) *
						    (1.0 + exp(-(constantk2*((double)*in2Ptr - constantc2)))) ) :
						1.0 / ((1.0 + exp(-(constantk1*((double)*in1Ptr - constantc1)))) *
							   (1.0 + exp(-(constantk2*((double)*in2Ptr - constantc2)))) ));
					outPtr++;
					in1Ptr++;
					in2Ptr++;
				}
				outPtr += outIncY;
				in1Ptr += inIncY;
				in2Ptr += in2IncY;
			}
			outPtr += outIncZ;
			in1Ptr += inIncZ;
			in2Ptr += in2IncZ;
		}
		break;

	case VTK_GAUSSIAN:
		for(idxZ = 0; idxZ <= maxZ; idxZ++){
			for (idxY = 0; idxY <= maxY; idxY++){
				if( !id ){
					if (!(count%target)) self->UpdateProgress(count/(50.0*target));
					count++;
				}
				for (idxR = 0; idxR < rowLength; idxR++){
					*outPtr = (T)( entropy ? 
						0.5 * (((double)*in1Ptr - constantc1)/constantk1) * (((double)*in1Ptr - constantc1)/constantk1) +
						0.5 * (((double)*in2Ptr - constantc2)/constantk2) * (((double)*in2Ptr - constantc2)/constantk2) :
						exp(-0.5 * (((double)*in1Ptr - constantc1)/constantk1) * (((double)*in1Ptr - constantc1)/constantk1) -
							 0.5 * (((double)*in2Ptr  - constantc2)/constantk2 * (((double)*in2Ptr - constantc2)/constantk2) )) );
					outPtr++;
					in1Ptr++;
					in2Ptr++;
				}
				outPtr += outIncY;
				in1Ptr += inIncY;
				in2Ptr += in2IncY;
			}
			outPtr += outIncZ;
			in1Ptr += inIncZ;
			in2Ptr += in2IncZ;
		}
		break;
	}

}


//----------------------------------------------------------------------------
// This method is passed a input and output datas, and executes the filter
// algorithm to fill the output from the inputs.
// It just executes a switch statement to call the correct function for
// the datas data types.
void vtkImageDataTerm::ThreadedRequestData(
  vtkInformation * vtkNotUsed( request ),
  vtkInformationVector ** vtkNotUsed( inputVector ),
  vtkInformationVector * vtkNotUsed( outputVector ),
  vtkImageData ***inData,
  vtkImageData **outData,
  int outExt[6], int id)
{
  void *inPtr1;
  void *outPtr;

  inPtr1 = inData[0][0]->GetScalarPointerForExtent(outExt);
  outPtr = outData[0]->GetScalarPointerForExtent(outExt);

  

  if (inData[1] && inData[1][0])
    {
	void *inPtr2;
	inPtr2 = inData[1][0]->GetScalarPointerForExtent(outExt);

    // this filter expects that input is the same type as output.
    if (inData[0][0]->GetScalarType() != outData[0]->GetScalarType())
      {
      vtkErrorMacro(<< "Execute: input1 ScalarType, "
                    <<  inData[0][0]->GetScalarType()
                    << ", must match output ScalarType "
                    << outData[0]->GetScalarType());
      return;
      }

    if (inData[1][0]->GetScalarType() != outData[0]->GetScalarType())
      {
      vtkErrorMacro(<< "Execute: input2 ScalarType, "
                    << inData[1][0]->GetScalarType()
                    << ", must match output ScalarType "
                  << outData[0]->GetScalarType());
      return;
      }

    // this filter expects that inputs that have the same number of components
    if (inData[0][0]->GetNumberOfScalarComponents() !=
        inData[1][0]->GetNumberOfScalarComponents())
      {
      vtkErrorMacro(<< "Execute: input1 NumberOfScalarComponents, "
                    << inData[0][0]->GetNumberOfScalarComponents()
                    << ", must match out input2 NumberOfScalarComponents "
                    << inData[1][0]->GetNumberOfScalarComponents());
      return;
      }

    switch (inData[0][0]->GetScalarType())
      {
      vtkTemplateMacro(
        vtkImageDataTermExecute2(this,inData[0][0],
                                    static_cast<VTK_TT *>(inPtr1),
                                    inData[1][0],
                                    static_cast<VTK_TT *>(inPtr2),
                                    outData[0],
                                    static_cast<VTK_TT *>(outPtr), outExt,
                                    id));
      default:
        vtkErrorMacro(<< "Execute: Unknown ScalarType");
        return;
      }
    }
  else
    {
    // this filter expects that input is the same type as output.
    if (inData[0][0]->GetScalarType() != outData[0]->GetScalarType())
      {
      vtkErrorMacro(<< "Execute: input ScalarType, "
        << inData[0][0]->GetScalarType()
        << ", must match out ScalarType " << outData[0]->GetScalarType());
      return;
      }

    switch (inData[0][0]->GetScalarType())
      {
      vtkTemplateMacro(
        vtkImageDataTermExecute1(this, inData[0][0],
                                    static_cast<VTK_TT *>(inPtr1),
                                    outData[0], static_cast<VTK_TT *>(outPtr),
                                    outExt, id));
      default:
        vtkErrorMacro(<< "Execute: Unknown ScalarType");
        return;
      }
    }
}

//----------------------------------------------------------------------------
int vtkImageDataTerm::FillInputPortInformation(
  int port, vtkInformation* info)
{
  if (port == 1)
    {
    info->Set(vtkAlgorithm::INPUT_IS_OPTIONAL(), 1);
    }
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
  return 1;
}

//----------------------------------------------------------------------------
void vtkImageDataTerm::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);

  os << indent << "Operation: " << this->Operation << "\n";
  os << indent << "ConstantK1: " << this->ConstantK1 << "\n";
  os << indent << "ConstantK2: " << this->ConstantK2 << "\n";
  os << indent << "ConstantC1: " << this->ConstantC1 << "\n";
  os << indent << "ConstantC2: " << this->ConstantC2 << "\n";
  os << indent << "ExpressAsEntropy: " << (this->Entropy ? "On" : "Off") << "\n";
}

