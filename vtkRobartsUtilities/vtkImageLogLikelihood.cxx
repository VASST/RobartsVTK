#include "vtkImageLogLikelihood.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "vtkImageCast.h"

#include <math.h>
#include <float.h>


vtkStandardNewMacro(vtkImageLogLikelihood);

//----------------------------------------------------------------------------
vtkImageLogLikelihood::vtkImageLogLikelihood()
{
    this->NormalizeDataTerm = 0;
    this->LabelID = 1.0;
    this->HistogramResolution = 1.0f;
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
    vtkInformation *inInfo2 = inputVector[1]->GetInformationObject(0);

    int ext[6], ext2[6], idx;

    inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext);

    // two input take intersection

    if (!inInfo2)
    {
        vtkErrorMacro(<< "Second input must be specified for this operation.");
        return 1;
    }

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


    outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),ext,6);

    return 1;
}

//----------------------------------------------------------------------------
template <class TValue, class TIvar>
void vtkImageMathematicsClamp(TValue &value, TIvar ivar, vtkImageData *data)
{
  if (ivar < static_cast<TIvar>(data->GetScalarTypeMin()))
    {
    value = static_cast<TValue>(data->GetScalarTypeMin());
    }
  else if (ivar > static_cast<TIvar>(data->GetScalarTypeMax()))
    {
    value = static_cast<TValue>(data->GetScalarTypeMax());
    }
  else
    {
    value = static_cast<TValue>(ivar);
    }
}


//----------------------------------------------------------------------------
// This templated function executes the filter for any type of data.
// Handles the two input operations
template <class T>
void vtkImageLogLikelihoodExecute(vtkImageLogLikelihood *self,
                                  vtkImageData *in1Data, T *in1Ptr,
                                  vtkImageData *in2Data, T *in2Ptr,
                                  vtkImageData *outData, T *outPtr,
                                  int outExt[6], int id)
{


    vtkImageCast *in1Cast = vtkImageCast::New();
    vtkImageCast *in2Cast = vtkImageCast::New();

    in1Cast->SetInput(in1Data);
    in1Cast->SetOutputScalarTypeToFloat();
    in1Cast->Update();
    in2Cast->SetInput(in2Data);
    in2Cast->SetOutputScalarTypeToFloat();
    in2Cast->Update();

    float* inputBuffer = (float*) in1Cast->GetOutput()->GetScalarPointer();
    float* labelBuffer = (float*) in2Cast->GetOutput()->GetScalarPointer();

    // allocate output buffer
    outData->DeepCopy(in1Cast->GetOutput());

    int volumeSize = in2Cast->GetOutput()->GetDimensions()[0]*
                     in2Cast->GetOutput()->GetDimensions()[1]*
                     in2Cast->GetOutput()->GetDimensions()[2];

    float* outBuffer =  (float*)outData->GetScalarPointer();
    std::fill_n(outBuffer, volumeSize , 0.0f);

    // calculate sample size
    unsigned int szSample = 0;
    for(int idx = 0; idx < volumeSize; idx++ ){

        if( labelBuffer[idx] == self->GetLabelID())
            szSample++;
    }

    // acquire sample
    float *sample = new float[szSample];

    unsigned int count = 0;
    for(int idx = 0; idx < volumeSize; idx++ ){
        if( labelBuffer[idx] == self->GetLabelID() ){
            sample[count] = inputBuffer[idx];
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
    for (unsigned int idx = 0; idx < volumeSize; idx++){

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

    in1Cast->Delete();
    in2Cast->Delete();

    return;
}


//----------------------------------------------------------------------------
// This method is passed a input and output datas, and executes the filter
// algorithm to fill the output from the inputs.
// It just executes a switch statement to call the correct function for
// the datas data types.
void vtkImageLogLikelihood::ThreadedRequestData(
        vtkInformation * vtkNotUsed( request ),
        vtkInformationVector ** vtkNotUsed( inputVector ),
        vtkInformationVector * vtkNotUsed( outputVector ),
        vtkImageData ***inData,
        vtkImageData **outData,
        int outExt[6], int id)
{
    void *inPtr1;
    void *inPtr2;
    void *outPtr;

    inPtr1 = inData[0][0]->GetScalarPointerForExtent(outExt);
    inPtr2 = inData[1][0]->GetScalarPointerForExtent(outExt);
    outPtr = outData[0]->GetScalarPointerForExtent(outExt);

    if (!inData[1] || ! inData[1][0])
    {
        vtkErrorMacro(
                    "ImageMathematics requested to perform a two input operation "
                    "with only one input\n");
        return;
    }

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
                vtkImageLogLikelihoodExecute(this,inData[0][0],
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

//----------------------------------------------------------------------------
int vtkImageLogLikelihood::FillInputPortInformation(
        int port, vtkInformation* info)
{

    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    return 1;
}

//----------------------------------------------------------------------------
void vtkImageLogLikelihood::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os,indent);

}
