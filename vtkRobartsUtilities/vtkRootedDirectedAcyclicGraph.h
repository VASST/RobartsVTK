
#ifndef __vtkRootedDirectedAcyclicGraph_h
#define __vtkRootedDirectedAcyclicGraph_h

#include "vtkDirectedAcyclicGraph.h"

class vtkIdTypeArray;

class vtkRootedDirectedAcyclicGraph : public vtkDirectedAcyclicGraph
{
public:
  static vtkRootedDirectedAcyclicGraph *New();
  vtkTypeMacro(vtkRootedDirectedAcyclicGraph, vtkDirectedAcyclicGraph);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Return what type of dataset this is.
  virtual int GetDataObjectType() {return VTK_DIRECTED_GRAPH;}

  // Description:
  // Get the root vertex of the tree.
  vtkGetMacro(Root, vtkIdType);

  // Description:
  // Get the number of children of a vertex.
  vtkIdType GetNumberOfChildren(vtkIdType v)
    { return this->GetOutDegree(v); }

  // Description:
  // Get the i-th child of a parent vertex.
  vtkIdType GetChild(vtkIdType v, vtkIdType i);

  // Description:
  // Get the child vertices of a vertex.
  // This is a convenience method that functions exactly like
  // GetAdjacentVertices.
  void GetChildren(vtkIdType v, vtkAdjacentVertexIterator *it)
    { this->GetAdjacentVertices(v, it); }
  
  // Description:
  // Get the number of children of a vertex.
  vtkIdType GetNumberOfParents(vtkIdType v)
    { return this->GetInDegree(v); }

  // Description:
  // Get the parent of a vertex.
  vtkIdType GetParent(vtkIdType v, vtkIdType i);
  
  //BTX
  // Description:
  // Get the edge connecting the vertex to its parent.
  vtkEdgeType GetParentEdge(vtkIdType v, vtkIdType i);
  //ETX

  // Description:
  // Get the level of the vertex in the tree.  The root vertex has level 0.
  // Returns -1 if the vertex id is < 0 or greater than the number of vertices
  // in the tree.
  vtkIdType GetLevel(vtkIdType v);

  // Description:
  // Return whether the vertex is a leaf (i.e. it has no children).
  bool IsLeaf(vtkIdType vertex);

  //BTX
  // Description:
  // Retrieve a graph from an information vector.
  static vtkRootedDirectedAcyclicGraph *GetData(vtkInformation *info);
  static vtkRootedDirectedAcyclicGraph *GetData(vtkInformationVector *v, int i=0);
  //ETX

  // Description:
  // Reorder the children of a parent vertex.
  // The children array must contain all the children of parent,
  // just in a different order.
  // This does not change the topology of the tree.
  virtual void ReorderChildren(vtkIdType parent, vtkIdTypeArray *children);

protected:
  vtkRootedDirectedAcyclicGraph();
  ~vtkRootedDirectedAcyclicGraph();

  // Description:
  // Check the storage, and accept it if it is a valid
  // tree.
  virtual bool IsStructureValid(vtkGraph *g);

  // Description:
  // The root of the tree.
  vtkIdType Root;
  vtkIdType* Level;

private:
  vtkRootedDirectedAcyclicGraph(const vtkRootedDirectedAcyclicGraph&);  // Not implemented.
  void operator=(const vtkRootedDirectedAcyclicGraph&);  // Not implemented.
};

#endif