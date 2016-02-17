#include "limits.h"
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
  this->UpLevel = 0;
  this->DownLevel = 0;
}

//----------------------------------------------------------------------------
vtkRootedDirectedAcyclicGraph::~vtkRootedDirectedAcyclicGraph()
{
  if(this->UpLevel)
  {
    delete this->UpLevel;
  }
  if(this->DownLevel)
  {
    delete this->DownLevel;
  }
}

//----------------------------------------------------------------------------
vtkIdType vtkRootedDirectedAcyclicGraph::GetNumberOfChildren(vtkIdType v)
{
  return this->GetOutDegree(v);
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
void vtkRootedDirectedAcyclicGraph::GetChildren(vtkIdType v, vtkAdjacentVertexIterator *it)
{
  this->GetAdjacentVertices(v, it);
}

//----------------------------------------------------------------------------
vtkIdType vtkRootedDirectedAcyclicGraph::GetNumberOfParents(vtkIdType v)
{
  return this->GetInDegree(v);
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
vtkIdType vtkRootedDirectedAcyclicGraph::GetUpLevel(vtkIdType vertex)
{
  if (vertex < 0 || vertex >= this->GetNumberOfVertices())
  {
    return -1;
  }
  return this->UpLevel[vertex];
}

vtkIdType vtkRootedDirectedAcyclicGraph::GetDownLevel(vtkIdType vertex)
{
  if (vertex < 0 || vertex >= this->GetNumberOfVertices())
  {
    return -1;
  }
  return this->DownLevel[vertex];
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
  if(this->UpLevel)
  {
    delete this->UpLevel;
  }
  if(this->DownLevel)
  {
    delete this->DownLevel;
  }
  this->UpLevel = new vtkIdType[g->GetNumberOfVertices()];
  this->DownLevel = new vtkIdType[g->GetNumberOfVertices()];
  for(int i = 0; i < g->GetNumberOfVertices(); i++)
  {
    this->UpLevel[i] = -1;
  }
  for(int i = 0; i < g->GetNumberOfVertices(); i++)
  {
    this->DownLevel[i] = INT_MAX;
  }
  this->UpLevel[root] = 0;
  this->DownLevel[root] = 0;
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
        this->UpLevel[id] = (this->UpLevel[v] + 1 > this->UpLevel[id]) ?
                            this->UpLevel[v] + 1 : this->UpLevel[id];
        this->DownLevel[id] = (this->DownLevel[v] + 1 < this->DownLevel[id]) ?
                              this->DownLevel[v] + 1 : this->DownLevel[id];
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

//----------------------------------------------------------------------------
int vtkRootedDirectedAcyclicGraph::GetDataObjectType()
{
  return VTK_DIRECTED_GRAPH;
}
