#include "vtkHierarchicalMaxFlowSegmentation.h"
#include "vtkObjectFactory.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkPointData.h"
#include "vtkDataArray.h"
#include "vtkTreeDFSIterator.h"

vtkStandardNewMacro(vtkHierarchicalMaxFlowSegmentation);

vtkHierarchicalMaxFlowSegmentation::vtkHierarchicalMaxFlowSegmentation(){
	
	//configure the IO ports
	this->SetNumberOfInputPorts(1);
	this->SetNumberOfOutputPorts(1);

	//set the other values to defaults
	this->Hierarchy = 0;
	this->SmoothnessScalars.clear();

}

vtkHierarchicalMaxFlowSegmentation::~vtkHierarchicalMaxFlowSegmentation(){
	if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
	this->SmoothnessScalars.clear();
}

//------------------------------------------------------------

void vtkHierarchicalMaxFlowSegmentation::SetHierarchy(vtkTree* graph){
	if( graph != this->Hierarchy ){
		if( this->Hierarchy ) this->Hierarchy->UnRegister(this);
		this->Hierarchy = graph;
		if( this->Hierarchy ) this->Hierarchy->Register(this);
		this->Modified();
	}
}

vtkTree* vtkHierarchicalMaxFlowSegmentation::GetHierarchy(){
	return this->Hierarchy;
}

//------------------------------------------------------------

void vtkHierarchicalMaxFlowSegmentation::AddSmoothnessScalar(vtkIdType node, double value){
	if( value >= 0.0 ){
		this->SmoothnessScalars.insert(std::pair<vtkIdType,double>(node,value));
		this->Modified();
	}else{
		vtkErrorMacro(<<"Cannot use a negative smoothness value.");
	}
}

//------------------------------------------------------------

int vtkHierarchicalMaxFlowSegmentation::FillInputPortInformation(int i, vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  return this->Superclass::FillInputPortInformation(i,info);
}

void vtkHierarchicalMaxFlowSegmentation::SetInput(int idx, vtkDataObject *input)
{
  // Ask the superclass to connect the input.
  this->SetNthInputConnection(0, idx, (input ? input->GetProducerPort() : 0));
}

vtkDataObject *vtkHierarchicalMaxFlowSegmentation::GetInput(int idx)
{
  if (this->GetNumberOfInputConnections(0) <= idx)
    {
    return 0;
    }
  return vtkImageData::SafeDownCast(
    this->GetExecutive()->GetInputData(0, idx));
}

//----------------------------------------------------------------------------

int vtkHierarchicalMaxFlowSegmentation::RequestInformation(
  vtkInformation* request,
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	//check to make sure that the hierarchy is specified
	if( !this->Hierarchy ){
		vtkErrorMacro(<<"Hierarchy must be provided.");
		return -1;
	}

	return 1;
}

int vtkHierarchicalMaxFlowSegmentation::RequestUpdateExtent(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* outputVector)
{
	//check to make sure that the hierarchy is specified
	if( !this->Hierarchy ){
		vtkErrorMacro(<<"Hierarchy must be provided.");
		return -1;
	}
	return 1;
}

int vtkHierarchicalMaxFlowSegmentation::RequestData(vtkInformation *request, 
							vtkInformationVector **inputVector, 
							vtkInformationVector *outputVector){

	//check to make sure that the hierarchy is specified
	if( !this->Hierarchy ){
		vtkErrorMacro(<<"Hierarchy must be provided.");
		return -1;
	}

	//check to make sure that there is an image associated with each leaf node
	vtkTreeDFSIterator* iterator = vtkTreeDFSIterator::New();
	iterator->SetTree(this->Hierarchy);
	while( iterator->HasNext() ){
		vtkIdType node = iterator->Next();
		if( this->Hierarchy->IsLeaf(node) && !(inputVector[0])->GetInformationObject(node) ){
			vtkErrorMacro(<<"Missing data prior for leaf node.");
			return -1;
		}
	}
	


	return 1;
}