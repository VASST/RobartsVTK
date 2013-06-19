#include "vtkCudaImageLogLikelihood.h"
#include "CUDA_loglikelihoodterm.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include <math.h>
#include <float.h>


vtkStandardNewMacro(vtkCudaImageLogLikelihood);

//----------------------------------------------------------------------------
vtkCudaImageLogLikelihood::vtkCudaImageLogLikelihood()
{
    this->NormalizeDataTerm = 0;
    this->LabelID = 1.0;
	this->HistogramSize = 512;
	this->RequiredAgreement = 0.8;
    this->SetNumberOfInputPorts(2);
}

vtkCudaImageLogLikelihood::~vtkCudaImageLogLikelihood(){

}

//----------------------------------------------------------------------------
// The output extent is the intersection.
int vtkCudaImageLogLikelihood::RequestInformation (
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
        vtkErrorMacro(<< "At least one label map must be specified.");
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
// This templated function executes the filter for any type of data.
// Handles the two input operations
template <class T, class TT>
void vtkCudaImageLogLikelihoodExecute(vtkCudaImageLogLikelihood *self,
                                  vtkImageData *in1Data, T *in1Ptr,
                                  vtkImageData **in2Data, TT *in2Ptr,
                                  vtkImageData *outData, float *outPtr,
                                  int numLabels)
{

    T* inputBuffer = (T*) in1Data->GetScalarPointer();
    int VolumeSize = in1Data->GetDimensions()[0]*
                     in1Data->GetDimensions()[1]*
                     in1Data->GetDimensions()[2];

	//initialize output and temporary memory
    float* outBuffer =  (float*)outData->GetScalarPointer();
	short* agreementGPU;
	CUDA_ILLT_GetRelevantBuffers(&agreementGPU,VolumeSize,self->GetStream());

	//figure out the agreement at each pixel
	short actualNumLabels = 0;
	for(int i = 0; i < numLabels; i++){
		if( !in2Data[i] ) continue;
		actualNumLabels++;
		self->ReserveGPU();
		switch (in2Data[i]->GetScalarType()){
		vtkTemplateMacro(
			CUDA_ILLT_IncrementInformation((VTK_TT *) in2Data[i]->GetScalarPointer(), (VTK_TT) self->GetLabelID(), agreementGPU, VolumeSize,self->GetStream()));
		default:
			vtkErrorWithObjectMacro(self,<< "Execute: Unknown ScalarType");
			return;
		}
	}

    // calculate normalized histogram and Calculate log likelihood dataterm
	float* histogramGPU = 0;
	if( in1Data->GetNumberOfScalarComponents() == 1 ){
		CUDA_ILLT_AllocateHistogram(&histogramGPU,self->GetHistogramSize(),self->GetStream());
		CUDA_ILLT_CalculateHistogramAndTerms(outBuffer,histogramGPU, self->GetHistogramSize(), agreementGPU, inputBuffer,
			(short)((double)actualNumLabels*self->GetRequiredAgreement()+0.99), VolumeSize, self->GetStream());
	}else if( in1Data->GetNumberOfScalarComponents() == 2 ){
		CUDA_ILLT_AllocateHistogram(&histogramGPU,self->GetHistogramSize()*self->GetHistogramSize(),self->GetStream());
		CUDA_ILLT_CalculateHistogramAndTerms2D(outBuffer,histogramGPU, agreementGPU, inputBuffer,
			(short)((double)actualNumLabels*self->GetRequiredAgreement()+0.99), VolumeSize, self->GetStream());
	}

	//return GPU memory
	//CUDA_ILLT_CopyBackResult(outputGPU,outBuffer,VolumeSize,self->GetStream());

}
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class T>
void vtkCudaImageLogLikelihoodExecute2(vtkCudaImageLogLikelihood *self,
                                  vtkImageData *in1Data, T *in1Ptr,
                                  vtkImageData **in2Data,
                                  vtkImageData *outData, float *outPtr,
                                  int numLabels, int lblType )
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
                vtkCudaImageLogLikelihoodExecute(self,in1Data,
                                             (T*) in1Ptr,
                                             in2Data,
											 static_cast<VTK_TT *>(scalarPtr),
                                             outData,
                                             outPtr,
                                             numLabels));
    default:
        vtkErrorWithObjectMacro(self, << "Execute: Unknown ScalarType");
        return;
    }

}

//----------------------------------------------------------------------------
// This method is passed a input and output datas, and executes the filter
// algorithm to fill the output from the inputs.
// It just executes a switch statement to call the correct function for
// the datas data types.
int vtkCudaImageLogLikelihood::RequestData(
        vtkInformation * vtkNotUsed( request ),
        vtkInformationVector ** inputVector,
        vtkInformationVector * outputVector )
{

	//collect the input image data
	vtkImageData**	inData[2];
	vtkImageData*	inputImage = vtkImageData::SafeDownCast(inputVector[0]->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
	inData[0] = &inputImage;
	void *inPtr1 = inputImage->GetScalarPointer();
	inData[1] = new vtkImageData* [inputVector[1]->GetNumberOfInformationObjects()];
		
	int numLabelMaps = 0;
	for(int i = 0; i < inputVector[1]->GetNumberOfInformationObjects(); i++){
		inData[1][i] = vtkImageData::SafeDownCast(inputVector[1]->GetInformationObject(i)->Get(vtkDataObject::DATA_OBJECT()));
		if(inData[1][i]) numLabelMaps++;
	}
    if (numLabelMaps == 0) {
        vtkErrorMacro(<< "At least one label map is required." );
        return -1;
    }
	
	//collect the output image data
	vtkImageData* outData = vtkImageData::SafeDownCast(outputVector->GetInformationObject(0)->Get(vtkDataObject::DATA_OBJECT()));
    int updateExtent[6];
    outputVector->GetInformationObject(0)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), updateExtent);
	outData->SetScalarTypeToFloat();
	outData->SetNumberOfScalarComponents(1);
	outData->SetExtent(updateExtent);
	outData->SetWholeExtent(updateExtent);
	outData->AllocateScalars();
	void *outPtr = outData->GetScalarPointer();

    // this filter expects the output datatype to be float.
    if (outData->GetScalarType() != VTK_FLOAT) {
        vtkErrorMacro(<< "Output data type must be float." );
        return -1;
    }

	// this filter expects the label map to be of type char
	int LabelType = -1;
	for(int i = 0; i < inputVector[1]->GetNumberOfInformationObjects(); i++){
		if( !inData[1][i] ) continue;
		if( LabelType == -1 ) LabelType = inData[1][i]->GetScalarType();
		if (inData[1][i]->GetScalarType() != LabelType) {
			vtkErrorMacro(<< "Label maps must be of same type." );
			return -1;
		}
		if ( inData[1][i]->GetNumberOfScalarComponents() != 1 ) {
			vtkErrorMacro(<< "Label map can only have 1 component." );
			return -1;
		}
	}

    // this filter expects that inputs that have the same number of components
    if (inData[0][0]->GetNumberOfScalarComponents() != 1 ) {
        vtkErrorMacro(<< "Execute: Image can only have one component.");
        return -1;
    }

    switch (inData[0][0]->GetScalarType()) {
    vtkTemplateMacro(
                vtkCudaImageLogLikelihoodExecute2(this,inData[0][0],
                                             static_cast<VTK_TT *>(inPtr1),
                                             inData[1],
                                             outData,
                                             static_cast<float *>(outPtr),
                                             inputVector[1]->GetNumberOfInformationObjects(),
											 LabelType));
    default:
        vtkErrorMacro(<< "Execute: Unknown ScalarType");
        return -1;
    }

	delete[] inData[1];

	return 1;
}

//----------------------------------------------------------------------------
int vtkCudaImageLogLikelihood::FillInputPortInformation(
        int port, vtkInformation* info)
{
	if( port == 1 ) info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(),1);
    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    return 1;
}

//----------------------------------------------------------------------------
void vtkCudaImageLogLikelihood::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os,indent);

}
