
#include "vtkImageMultiStatistics.h"
#include "vtkObjectFactory.h"
#include "vtkCommand.h"


//------------------------------------------------------------------------------
vtkImageMultiStatistics* vtkImageMultiStatistics::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkImageMultiStatistics");
  if(ret)
    {
    return (vtkImageMultiStatistics*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkImageMultiStatistics;
}

//----------------------------------------------------------------------------
vtkImageMultiStatistics::vtkImageMultiStatistics()
{
  this->NumberOfComponents = 0;
  this->AverageMagnitude = 0;
  this->StandardDeviation = 0;
  this->Count = 0;
}

vtkImageMultiStatistics::~vtkImageMultiStatistics()
{
}

void vtkImageMultiStatistics::SetInput(vtkImageData *input)
{
  this->vtkProcessObject::SetNthInput(0, input);
}

vtkImageData *vtkImageMultiStatistics::GetInput()
{
  if (this->NumberOfInputs < 1)
    {
      return NULL;
    }

  return (vtkImageData *)(this->Inputs[0]);
}

//----------------------------------------------------------------------------
double vtkImageMultiStatistics::GetAverageMagnitude(int component){
	this->Update(); 
	if( component < this->NumberOfComponents )
		return this->AverageMagnitude[component];
	else{
		vtkErrorMacro(<<"Cannot select component. Component not provided in input.");
		return 0.0;
	}
}

double vtkImageMultiStatistics::GetStandardDeviation(int component) {
	this->Update();
	if( component < this->NumberOfComponents )
		return this->StandardDeviation[component];
	else{
		vtkErrorMacro(<<"Cannot select component. Component not provided in input.");
		return 0.0;
	}
}

long int vtkImageMultiStatistics::GetCount() {
	this->Update();
	return this->Count;
}

//----------------------------------------------------------------------------
template <class T>
static void vtkImageMultiStatisticsExecute(vtkImageMultiStatistics *self, 
					  T *inPtr,
					  vtkImageData *inData,
					  double **AverageMagnitude,
					  double **StandardDeviation,
					  long int *Count, int N)
{
	int inIdxX, inIdxY, inIdxZ;
	vtkIdType inIncX, inIncY, inIncZ;
	int inIdxN;
	int wholeInExt[6];

	//use the output holders for temporary storage
	double* sum = *AverageMagnitude;
	double* sum_squared = *StandardDeviation;
	for(int i = 0; i < N; i++)
		sum[i] = sum_squared[i] = 0.0;

	T *curPtr;

	// Get increments to march through data 
	inData->GetWholeExtent(wholeInExt);
	inData->GetContinuousIncrements(wholeInExt, inIncX, inIncY, inIncZ);
	*Count = 0;
	
	// Loop through input dataset once to gather stats
	curPtr = inPtr;
	for (inIdxZ = wholeInExt[4]; inIdxZ <= wholeInExt[5]; inIdxZ++) {
		for (inIdxY = wholeInExt[2]; !self->AbortExecute && inIdxY <= wholeInExt[3]; inIdxY++) {
			for (inIdxX = wholeInExt[0]; inIdxX <= wholeInExt[1]; inIdxX++) {
				for( inIdxN = 0; inIdxN < N; inIdxN++ ){
					sum[inIdxN] += *curPtr;
					sum_squared[inIdxN] += *curPtr* *curPtr;
					curPtr++;
				}
				(*Count)++;
			}
			curPtr += inIncY;
		}
		curPtr += inIncZ;
	}

	//convert from sums and sums of squares to average and standard deviation
	for(int i = 0; i < N; i++){
		double SafeSum = sum[i];
		double SafeSquareSum = sum_squared[i];
		(*AverageMagnitude)[i] = SafeSum / (double)*Count;
		(*StandardDeviation)[i] = sqrt((SafeSquareSum * (double)*Count - SafeSum*SafeSum) /  ((double)*Count * ((double)*Count)));
	}

}



// Description:
// Make sure input is available then call the templated execute method to
// deal with the particular data type.
void vtkImageMultiStatistics::Update()
{
  vtkImageData *input = this->GetInput();
  void *inPtr;
  int wholeInExt[6];

  // make sure input is available
  if (!input)
    {
      vtkErrorMacro(<< "No input...can't execute!");
      return;
    }

  input->Update();
  input->GetWholeExtent(wholeInExt);
  inPtr = input->GetScalarPointerForExtent(wholeInExt);

  // get the number of input components
  int oldNumberOfComponents = this->NumberOfComponents;
  this->NumberOfComponents = input->GetNumberOfScalarComponents();
  if (this->NumberOfComponents == 0)
    {
      vtkErrorMacro(<< "Input must have at least 1 scalar component");
      return;
    }

  //update the size of the information holders
  if( this->NumberOfComponents != oldNumberOfComponents ){
	  delete [] this->AverageMagnitude;
	  delete [] this->StandardDeviation;
	  this->AverageMagnitude = new double[2*this->NumberOfComponents];
	  this->StandardDeviation = this->AverageMagnitude + this->NumberOfComponents;
  }

  if (input->GetMTime() > this->ExecuteTime ||
      this->GetMTime() > this->ExecuteTime )
    {
      this->InvokeEvent(vtkCommand::StartEvent, NULL);

      // reset Abort flag
      this->AbortExecute = 0;
      this->Progress = 0.0;
      switch (input->GetScalarType())
	{
	vtkTemplateMacro7(vtkImageMultiStatisticsExecute,
			  this, (VTK_TT *) (inPtr), input,
			  &(this->AverageMagnitude),
			  &(this->StandardDeviation),
			  &(this->Count), this->NumberOfComponents);
	default:
	  vtkErrorMacro(<< "Update: Unknown ScalarType");
	  return;
	}
      this->ExecuteTime.Modified();
      if (!this->AbortExecute)
	{
	this->UpdateProgress(1.0);
	}
      this->InvokeEvent(vtkCommand::EndEvent, NULL);
    }
  if (input->ShouldIReleaseData())
    {
    input->ReleaseData();
    }
}





void vtkImageMultiStatistics::PrintSelf(ostream& os, vtkIndent indent)
{
  vtkProcessObject::PrintSelf(os,indent);

  if (!this->GetInput())
    {
    return;
    }
  
  os << indent << "Components: " << this->NumberOfComponents << "\n";
  os << indent << "Count: " << this->Count << "\n";
  for( int i = 0; i < this->NumberOfComponents; i++){
	  os << indent << "AverageMagnitude: " << this->AverageMagnitude[i] << "\n";
	  os << indent << "StandardDeviation: " << this->StandardDeviation[i] << "\n";
  }

}

