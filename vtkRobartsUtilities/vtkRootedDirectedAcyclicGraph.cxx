
#include "vtkRootedDirectedAcyclicGraph.h"

#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkOutEdgeIterator.h"
#include "vtkSmartPointer.h"

#include <vtksys/stl/vector>

using vtksys_stl::vector;

vtkStandardNewMacro(vtkRootedDirectedAcyclicGraph);
//----------------------------------------------------------------------------
vtkRootedDirectedAcyclicGraph::vtkRootedDirectedAcyclicGraph()
{
  this->Root = -1;
  this->Level = 0;
}

//----------------------------------------------------------------------------
vtkRootedDirectedAcyclicGraph::~vtkRootedDirectedAcyclicGraph()
{
	if(this->Level)
	  delete this->Level;
}

//----------------------------------------------------------------------------
vtkIdType vtkRootedDirectedAcyclicGraph::GetChild(vtkIdType v, vtkIdType i)
{
  const vtkOutEdgeType *edges;
  vtkIdType nedges;
  this->GetOutEdges(v, edges, nedges);
  if (i < nedges)
    {
    return edges[i].Target;
    }
  return -1;
}

//----------------------------------------------------------------------------
vtkIdType vtkRootedDirectedAcyclicGraph::GetParent(vtkIdType v, vtkIdType i)
{
  const vtkInEdgeType *edges;
  vtkIdType nedges;
  this->GetInEdges(v, edges, nedges);
  if (nedges > 0)
    {
    return edges[i].Source;
    }
  return -1;
}

//----------------------------------------------------------------------------
vtkEdgeType vtkRootedDirectedAcyclicGraph::GetParentEdge(vtkIdType v, vtkIdType i)
{
  const vtkInEdgeType *edges;
  vtkIdType nedges;
  this->GetInEdges(v, edges, nedges);
  if (nedges > 0)
    {
    return vtkEdgeType(edges[i].Source, v, edges[0].Id);
    }
  return vtkEdgeType();
}

//----------------------------------------------------------------------------
vtkIdType vtkRootedDirectedAcyclicGraph::GetLevel(vtkIdType vertex)
{
  if (vertex < 0 || vertex >= this->GetNumberOfVertices())
    {
    return -1;
    }
  return this->Level[vertex];
}

//----------------------------------------------------------------------------
bool vtkRootedDirectedAcyclicGraph::IsLeaf(vtkIdType vertex)
{
  return (this->GetNumberOfChildren(vertex) == 0);
}
  
//----------------------------------------------------------------------------
vtkRootedDirectedAcyclicGraph *vtkRootedDirectedAcyclicGraph::GetData(vtkInformation *info)
{
  return info? vtkRootedDirectedAcyclicGraph::SafeDownCast(info->Get(DATA_OBJECT())) : 0;
}

//----------------------------------------------------------------------------
vtkRootedDirectedAcyclicGraph *vtkRootedDirectedAcyclicGraph::GetData(vtkInformationVector *v, int i)
{
  return vtkRootedDirectedAcyclicGraph::GetData(v->GetInformationObject(i));
}

//----------------------------------------------------------------------------
bool vtkRootedDirectedAcyclicGraph::IsStructureValid(vtkGraph *g)
{
  if (vtkRootedDirectedAcyclicGraph::SafeDownCast(g))
    {
    // Since a rooted DAG has the additional root propery, we need
    // to set that here.
    this->Root = (vtkRootedDirectedAcyclicGraph::SafeDownCast(g))->Root;
    return true;
    }

  // Empty graph is a valid tree.
  if (g->GetNumberOfVertices() == 0)
    {
    this->Root = -1;
    return true;
    }

  // Find the root and fail if there is more than one.
  vtkIdType root = -1;
  for (vtkIdType v = 0; v < g->GetNumberOfVertices(); ++v)
    {
    vtkIdType indeg = g->GetInDegree(v);
    if (indeg == 0 && root == -1)
      {
      // We found our first root.
      root = v;
      }
    else if (indeg == 0)
      {
      // We already found a root, so fail.
      return false;
      }
    }
  if (root < 0)
    {
    return false;
    }

  // Make sure the rooted DAG is connected with no directed cycles.
  vector<bool> visited(g->GetNumberOfVertices(), false);
  vector<bool> active(g->GetNumberOfVertices(), false);
  if(this->Level) delete this->Level;
  this->Level = new vtkIdType[g->GetNumberOfVertices()];
  for(int i = 0; i < g->GetNumberOfVertices(); i++)
	  this->Level[i] = -1;
  this->Level[root] = 0;
  vector<vtkIdType> stack;
  stack.push_back(root);
  vtkSmartPointer<vtkOutEdgeIterator> outIter = 
    vtkSmartPointer<vtkOutEdgeIterator>::New();
  while (!stack.empty())
    {
    vtkIdType v = stack.back();
    stack.pop_back();
    visited[v] = true;
    active[v] = true;
    g->GetOutEdges(v, outIter);
    while (outIter->HasNext())
      {
      vtkIdType id = outIter->Next().Target;
      if (!active[id])
        {
        stack.push_back(id);
		this->Level[id] = (this->Level[v] + 1 > this->Level[id]) ?
			this->Level[v] + 1 : this->Level[id];
        }
      else
        {
        return false;
        }
      }
	active[v] = false;
    }
  for (vtkIdType v = 0; v < g->GetNumberOfVertices(); ++v)
    {
    if (!visited[v])
      {
      return false;
      }
    }

  // Since a tree has the additional root propery, we need
  // to set that here.
  this->Root = root;

  return true;
}

//----------------------------------------------------------------------------
void vtkRootedDirectedAcyclicGraph::ReorderChildren(vtkIdType parent, vtkIdTypeArray *children)
{
  this->ReorderOutVertices(parent, children);
}

//----------------------------------------------------------------------------
void vtkRootedDirectedAcyclicGraph::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
  os << indent << "Root: " << this->Root << endl;
}
