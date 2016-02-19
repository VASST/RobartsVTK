

#include "vtkRootedDirectedAcyclicGraphIterator.h"

//#include "vtkObjectFactory.h"
#include "vtkRootedDirectedAcyclicGraph.h"

vtkRootedDirectedAcyclicGraphIterator::vtkRootedDirectedAcyclicGraphIterator()
{
  this->DAG = NULL;
  this->RootVertex = -1;
  this->NextId = -1;
}

vtkRootedDirectedAcyclicGraphIterator::~vtkRootedDirectedAcyclicGraphIterator()
{
  if (this->DAG)
    {
    this->DAG->Delete();
    this->DAG = NULL;
    }
}

void vtkRootedDirectedAcyclicGraphIterator::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "DAG: " << this->DAG << endl;
  os << indent << "RootVertex: " << this->RootVertex << endl;
  os << indent << "NextId: " << this->NextId << endl;
}

void vtkRootedDirectedAcyclicGraphIterator::SetDAG(vtkRootedDirectedAcyclicGraph* dag)
{
  vtkDebugMacro( this->GetClassName() << " (" << this
                << "): setting DAG to " << dag );
  if (this->DAG != dag)
    {
    vtkRootedDirectedAcyclicGraph* temp = this->DAG;
    this->DAG = dag;
    if (this->DAG != NULL) { this->DAG->Register(this); }
    if (temp != NULL)
      {
      temp->UnRegister(this);
      }
    this->RootVertex = -1;
    this->Initialize();
    this->Modified();
    }
}

void vtkRootedDirectedAcyclicGraphIterator::SetRootVertex(vtkIdType vertex)
{
  if (this->RootVertex != vertex)
    {
    this->RootVertex = vertex;
    this->Initialize();
    this->Modified();
    }
}

vtkIdType vtkRootedDirectedAcyclicGraphIterator::Next()
{
  vtkIdType last = this->NextId;
  if(last != -1)
    {
    this->NextId = this->NextInternal();
    }
  return last;
}

bool vtkRootedDirectedAcyclicGraphIterator::HasNext()
{
  return this->NextId != -1;
}

void vtkRootedDirectedAcyclicGraphIterator::Restart()
{
  this->Initialize();
}
