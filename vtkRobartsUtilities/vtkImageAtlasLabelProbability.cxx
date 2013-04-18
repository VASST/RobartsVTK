#include "vtkImageAtlasLabelProbability.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <math.h>
#include <float.h>


vtkStandardNewMacro(vtkImageAtlasLabelProbability);

//----------------------------------------------------------------------------
vtkImageAtlasLabelProbability::vtkImageAtlasLabelProbability()
{
    this->NormalizeDataTerm = 0;
    this->LabelID = 1.0;
	this->Entropy = false;
    this->SetNumberOfInputPorts(0);
    this->SetNumberOfThreads(10);
	this->MaxValueToGive = 100.0;
}

vtkImageAtlasLabelProbability::~vtkImageAtlasLabelProbability(){

}

//----------------------------------------------------------------------------
// The output extent is the intersection.
int vtkImageAtlasLabelProbability::RequestInformation (
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
	
	for(int i = 0; i < inputVector[0]->GetNumberOfInformationObjects(); i++){
		vtkInformation *inInfo2 = inputVector[0]->GetInformationObject(i);
		inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext2);
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
//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
// Handles the two input operations
template <class T>
void vtkImageAtlasLabelProbabilityExecute(vtkImageAtlasLabelProbability *self,
                                  vtkImageData **inData, T *inPtr,
                                  vtkImageData *outData, float *outPtr,
                                  int outExt[6], int numLabels, int id)
{

    T* inputBuffer = (T*) inData[0]->GetScalarPointer();
    int volumeSize = inData[0]->GetDimensions()[0]*
                     inData[0]->GetDimensions()[1]*
                     inData[0]->GetDimensions()[2];

    float* outBuffer =  (float*)outData->GetScalarPointer();
    //std::fill_n(outBuffer, volumeSize , 0.0f);

	//find the actual number of non-null labels
	int actualNumLabels = 0;
	for(int label = 0; label < numLabels; label++ )
		if( inData[label] ) actualNumLabels++;

    
	for(int idx = (id*volumeSize) / self->GetNumberOfThreads(); idx < ((id+1)*volumeSize) / self->GetNumberOfThreads(); idx++ ){
	
		//find the number of agreed on pixels
		int agree = 0;
		for(int label = 0; label < numLabels; label++ ){
			if( inData[label] )
				if( (int) ((T*)inData[label]->GetScalarPointer())[idx] == self->GetLabelID() )
					agree++;
		}

		//there is no agreement, assign the maximum value
		if( agree == 0 ){
			if( self->GetEntropy() && self->GetNormalizeDataTerm() == 0 )
				outBuffer[idx] = 1.0f;
			else if( self->GetEntropy() )
				self->GetMaxValueToGive();
			else
				outBuffer[idx] = 0.0f;
		}else{
			if( self->GetEntropy() && self->GetNormalizeDataTerm() == 0 )
				outBuffer[idx] = (log((float)actualNumLabels) - log((float)agree) < self->GetMaxValueToGive()) ?
				  (log((float)actualNumLabels) - log((float)agree)) / self->GetMaxValueToGive() :
				  1.0f ;
			else if( self->GetEntropy() )
				outBuffer[idx] = log((float)actualNumLabels) - log((float)agree);
			else
				outBuffer[idx] = (float) agree / (float) actualNumLabels;
		}

    }

    
    return;
}


//----------------------------------------------------------------------------
// This method is passed a input and output datas, and executes the filter
// algorithm to fill the output from the inputs.
// It just executes a switch statement to call the correct function for
// the datas data types.
void vtkImageAtlasLabelProbability::ThreadedRequestData(
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
	outData[0]->SetScalarTypeToFloat();
	
	int numLabelMaps = 0;
	for(int i = 0; i < inputVector[1]->GetNumberOfInformationObjects(); i++)
		if(inData[1][i]) numLabelMaps++;

    if (numLabelMaps == 0) {
        vtkErrorMacro(<< "At least one label map is required." );
        return;
    }

    // this filter expects the output datatype to be float.
    if (outData[0]->GetScalarType() != VTK_FLOAT)
    {
        vtkErrorMacro(<< "Output data type must be float." );
        return;
    }

	// this filter expects the label map to be of type char
	int LabelType = -1;
	for(int i = 0; i < inputVector[1]->GetNumberOfInformationObjects(); i++){
		if( !inData[1][i] ) continue;
		if( LabelType == -1 ) LabelType = inData[1][i]->GetScalarType();
		if (inData[1][i]->GetScalarType() != LabelType) {
			vtkErrorMacro(<< "Label maps must be of same type." );
			return;
		}
		if ( inData[1][i]->GetNumberOfScalarComponents() != 1 ) {
			vtkErrorMacro(<< "Label map can only have 1 component." );
			return;

		}
	}

    // this filter expects that inputs that have the same number of components
    if (inData[0][0]->GetNumberOfScalarComponents() != 1 )
    {
        vtkErrorMacro(<< "Execute: Image can only have one component.");
        return;
    }

    switch (inData[0][0]->GetScalarType())
    {
    vtkTemplateMacro(
                vtkImageAtlasLabelProbabilityExecute(this,inData[0],
                                             static_cast<VTK_TT *>(inPtr1),
                                             outData[0],
                                             static_cast<float *>(outPtr), outExt,
                                             inputVector[0]->GetNumberOfInformationObjects(),
											 id));
    default:
        vtkErrorMacro(<< "Execute: Unknown ScalarType");
        return;
    }


}

//----------------------------------------------------------------------------
int vtkImageAtlasLabelProbability::FillInputPortInformation(
        int port, vtkInformation* info)
{
	if( port == 0 ) info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(),1);
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    return 1;
}

//----------------------------------------------------------------------------
void vtkImageAtlasLabelProbability::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os,indent);

}
