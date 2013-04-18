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

    T* inputBuffer = (T*) in1Data->GetScalarPointer();
    int volumeSize = in1Data->GetDimensions()[0]*
                     in1Data->GetDimensions()[1]*
                     in1Data->GetDimensions()[2];

    float* outBuffer =  (float*)outData->GetScalarPointer();
    //std::fill_n(outBuffer, volumeSize , 0.0f);

	//find the actual number of non-null labels
	int actualNumLabels = 0;
	for(int label = 0; label < numLabels; label++ ){
		if( in2Data[label] ) actualNumLabels++;
	}

    // calculate sample size
    int szSample = 0;
    for(int idx = 0; idx < volumeSize; idx++ ){
		int agree = 0;
		for(int label = 0; label < numLabels; label++ ){
			if( in2Data[label] )
				if( (int) ((TT*)in2Data[label]->GetScalarPointer())[idx] == self->GetLabelID() )
					agree++;
		}
		if( (double) agree >= self->GetRequiredAgreement()*(double)(actualNumLabels) )
			szSample++;
    }

    // acquire sample
    float *sample = new float[szSample];
    int count = 0;
    for(int idx = 0; idx < volumeSize; idx++ ){
		int agree = 0;
		for(int label = 0; label < numLabels; label++ ){
			if( in2Data[label] )
				if( (int) ((TT*)in2Data[label]->GetScalarPointer())[idx] == self->GetLabelID() )
					agree++;
		}
		if( (double) agree >= self->GetRequiredAgreement()*(double)(actualNumLabels) ){
            sample[count] = (float) inputBuffer[idx];
            count++;
        }
	}

    // calculate histogram and ML data term
    int szHist = 0;
    float resolution = self->GetHistogramResolution();

    float minVal = FLT_MAX;
    float maxVal = FLT_MIN;

    // Find min and max intensity values from sample
    for(int i = 0; i < szSample; i++){
        if ( sample[i] < minVal)
            minVal = sample[i];
        if ( sample[i] > maxVal)
            maxVal = sample[i];
    }

    //float resolution = 1;
    szHist = ((maxVal - minVal) / resolution) + 1;
    float *hist = new float[szHist];

    std::fill_n(hist, szHist , 0.0f);

    // Fill histogram bins
    for (int j = 0; j < szSample; j++){
        int histIdx = ((sample[j]-minVal)/resolution );
        hist[histIdx]++;
    }

    // Normalize histogram
    for (int i = 0; i < szHist; i++){
        hist[i] = hist[i]/ (float)szSample + 1.0e-10;

    }

    // Calculate log likelihood dataterm
    for (int idx = 0; idx < volumeSize; idx++){

        outBuffer[idx] =( inputBuffer[idx] < minVal ||  (int)((inputBuffer[idx]-minVal) / resolution) >= szHist) ?
            -log(1.0e-10) :
            (-log(hist[ (int)((inputBuffer[idx]-minVal) / resolution) ]));

        //// Normalize costs to [0,1]
        if( self->GetNormalizeDataTerm()){
            outBuffer[idx] = (outBuffer[idx] / -log(1.0e-10) );
        }

    }

    delete[] sample;
    delete[] hist;

    return;
}


//----------------------------------------------------------------------------
// This method is passed a input and output datas, and executes the filter
// algorithm to fill the output from the inputs.
// It just executes a switch statement to call the correct function for
// the datas data types.
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
                vtkImageLogLikelihoodExecute2(this,inData[0][0],
                                             static_cast<VTK_TT *>(inPtr1),
                                             inData[1],
                                             outData[0],
                                             static_cast<float *>(outPtr), outExt,
                                             inputVector[1]->GetNumberOfInformationObjects(),
											 LabelType,id));
    default:
        vtkErrorMacro(<< "Execute: Unknown ScalarType");
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
