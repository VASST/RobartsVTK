#include "vtkDiceCoefficient.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"

#include "vtkImageCast.h"

#include <math.h>
#include <float.h>
#include <vtkVersion.h> //for VTK_MAJOR_VERSION


vtkStandardNewMacro(vtkDiceCoefficient);

//----------------------------------------------------------------------------
vtkDiceCoefficient::vtkDiceCoefficient()
{
    this->LabelID = 0;

    this->SetNumberOfInputPorts(2);
    this->SetNumberOfThreads(1);
}

vtkDiceCoefficient::~vtkDiceCoefficient(){

}

//----------------------------------------------------------------------------
// The output extent is the intersection.
int vtkDiceCoefficient::RequestInformation (
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
        vtkErrorMacro( "Second input must be specified for this operation.");
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
void vtkDiceCoefficient::vtkDiceCoefficientExecute(vtkDiceCoefficient *self,
                                  vtkImageData *in1Data, T *in1Ptr,
                                  vtkImageData *in2Data, T *in2Ptr,
                                  vtkImageData *outData, T *outPtr,
                                  int outExt[6], int id)
{


    vtkImageCast *in1Cast = vtkImageCast::New();
    vtkImageCast *in2Cast = vtkImageCast::New();

#if (VTK_MAJOR_VERSION < 6)
    in1Cast->SetInput(in1Data);
#else
    in1Cast->SetInputDataObject(in1Data);
#endif
    in1Cast->SetOutputScalarTypeToUnsignedChar();
    in1Cast->Update();
#if (VTK_MAJOR_VERSION < 6)
    in2Cast->SetInput(in2Data);
#else
    in2Cast->SetInputDataObject(in2Data);
#endif
    in2Cast->SetOutputScalarTypeToUnsignedChar();
    in2Cast->Update();

    unsigned char* lbl1Buffer = (unsigned char*) in1Cast->GetOutput()->GetScalarPointer();
    unsigned char* lbl2Buffer = (unsigned char*) in2Cast->GetOutput()->GetScalarPointer();

    // allocate output buffer
    outData->DeepCopy(in1Cast->GetOutput());

    int volumeSize = in2Cast->GetOutput()->GetDimensions()[0]*
                     in2Cast->GetOutput()->GetDimensions()[1]*
                     in2Cast->GetOutput()->GetDimensions()[2];

    unsigned char* outBuffer =  (unsigned char*)outData->GetScalarPointer();
    std::fill_n(outBuffer, volumeSize , 0.0f);

    // Calculate Dice Coefficient

    unsigned int numVx1 = 0;
    unsigned int numVx2 = 0;
    unsigned int numVxOverlap = 0;



    for(int idx = 0; idx < volumeSize; idx++){

        if (lbl1Buffer[idx] == this->LabelID)
      numVx1++;

    if (lbl2Buffer[idx] == this->LabelID)
      numVx2++;

    if (lbl1Buffer[idx] == this->LabelID && lbl2Buffer[idx] == this->LabelID)
      numVxOverlap++;

    }

    this->DiceCoefficient = ((double)(2*numVxOverlap))/(double)(numVx1+numVx2);

    in1Cast->Delete();
    in2Cast->Delete();

    return;
}


//----------------------------------------------------------------------------
// This method is passed a input and output datas, and executes the filter
// algorithm to fill the output from the inputs.
// It just executes a switch statement to call the correct function for
// the datas data types.
void vtkDiceCoefficient::ThreadedRequestData(
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
        vtkErrorMacro( "Execute: input1 ScalarType, "
                      <<  inData[0][0]->GetScalarType()
                      << ", must match output ScalarType "
                      << outData[0]->GetScalarType());
        return;
    }

    if (inData[1][0]->GetScalarType() != outData[0]->GetScalarType())
    {
        vtkErrorMacro( "Execute: input2 ScalarType, "
                      << inData[1][0]->GetScalarType()
                      << ", must match output ScalarType "
                      << outData[0]->GetScalarType());
        return;
    }

    // this filter expects that inputs that have the same number of components
    if (inData[0][0]->GetNumberOfScalarComponents() !=
            inData[1][0]->GetNumberOfScalarComponents())
    {
        vtkErrorMacro( "Execute: input1 NumberOfScalarComponents, "
                      << inData[0][0]->GetNumberOfScalarComponents()
                      << ", must match out input2 NumberOfScalarComponents "
                      << inData[1][0]->GetNumberOfScalarComponents());
        return;
    }

    switch (inData[0][0]->GetScalarType())
    {
    vtkTemplateMacro(
                this->vtkDiceCoefficientExecute(this,inData[0][0],
                                             static_cast<VTK_TT *>(inPtr1),
                                             inData[1][0],
                                             static_cast<VTK_TT *>(inPtr2),
                                             outData[0],
                                             static_cast<VTK_TT *>(outPtr), outExt,
                                             id));
    default:
        vtkErrorMacro( "Execute: Unknown ScalarType");


        return;
    }


}

//----------------------------------------------------------------------------
int vtkDiceCoefficient::FillInputPortInformation(
        int port, vtkInformation* info)
{

    info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkImageData");
    return 1;
}

//----------------------------------------------------------------------------
void vtkDiceCoefficient::PrintSelf(ostream& os, vtkIndent indent)
{
    this->Superclass::PrintSelf(os,indent);

}
