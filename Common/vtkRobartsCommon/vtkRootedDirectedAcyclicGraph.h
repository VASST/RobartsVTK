/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkRootedDirectedAcyclicGraph.h

  Copyright (c) 2016 John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __vtkRootedDirectedAcyclicGraph_h
#define __vtkRootedDirectedAcyclicGraph_h

#include "vtkRobartsCommonModule.h"

#include "vtkDirectedAcyclicGraph.h"

class vtkIdTypeArray;

class VTKROBARTSCOMMON_EXPORT vtkRootedDirectedAcyclicGraph : public vtkDirectedAcyclicGraph
{
public:
  static vtkRootedDirectedAcyclicGraph *New();
  vtkTypeMacro(vtkRootedDirectedAcyclicGraph, vtkDirectedAcyclicGraph);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Return what type of dataset this is.
  virtual int GetDataObjectType();

  // Description:
  // Get the root vertex of the tree.
  vtkGetMacro(Root, vtkIdType);

  // Description:
  // Get the number of children of a vertex.
  vtkIdType GetNumberOfChildren(vtkIdType v);

  // Description:
  // Get the i-th child of a parent vertex.
  vtkIdType GetChild(vtkIdType v, vtkIdType i);

  // Description:
  // Get the child vertices of a vertex.
  // This is a convenience method that functions exactly like
  // GetAdjacentVertices.
  void GetChildren(vtkIdType v, vtkAdjacentVertexIterator *it);

  // Description:
  // Get the number of children of a vertex.
  vtkIdType GetNumberOfParents(vtkIdType v);

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
  // Up Level is the first integer greater than all parents, and
  // Down Level is the first integer greater than at least one parent.
  // Returns -1 if the vertex id is < 0 or greater than the number of vertices
  // in the tree.
  vtkIdType GetUpLevel(vtkIdType v);
  vtkIdType GetDownLevel(vtkIdType v);

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
  vtkIdType* UpLevel;
  vtkIdType* DownLevel;

private:
  vtkRootedDirectedAcyclicGraph(const vtkRootedDirectedAcyclicGraph&);
  void operator=(const vtkRootedDirectedAcyclicGraph&);
};

#endif