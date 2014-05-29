

// .NAME vtkRootedDirectedAcyclicGraphIterator - Abstract class for iterator over a vtkRootedDirectedAcyclicGraph.
//
// .SECTION Description
// The base class for rooted DAG iterators vtkRootedDirectedAcyclicGraphForwardIterator and vtkRootedDirectedAcyclicGraphBackwardIterator.
// After setting up the iterator, the normal mode of operation is to
// set up a <code>while(iter->HasNext())</code> loop, with the statement
// <code>vtkIdType vertex = iter->Next()</code> inside the loop.
//
// .SECTION See Also
// vtkRootedDirectedAcyclicGraphBFSIterator vtkRootedDirectedAcyclicGraphDFSIterator

#ifndef __vtkRootedDirectedAcyclicGraphIterator_h
#define __vtkRootedDirectedAcyclicGraphIterator_h

#include "vtkObject.h"

class vtkRootedDirectedAcyclicGraph;

class vtkRootedDirectedAcyclicGraphIterator : public vtkObject
{
public:
  vtkTypeMacro(vtkRootedDirectedAcyclicGraphIterator, vtkObject);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/get the graph to iterate over.
  void SetDAG(vtkRootedDirectedAcyclicGraph* graph);
  vtkGetMacro(DAG, vtkRootedDirectedAcyclicGraph*);

  // Description: 
  // The DAG iterator will only iterate over the subgraph rooted at vertex.
  // If not set (or set to a negative value), starts at the root of the DAG.
  void SetRootVertex(vtkIdType vertex);
  vtkGetMacro(RootVertex, vtkIdType);

  // Description:
  // The next vertex visited in the graph.
  vtkIdType Next();

  // Description:
  // Return true when all vertices have been visited.
  bool HasNext();

  // Description:
  // Reset the iterator to its start vertex.
  void Restart();

protected:
  vtkRootedDirectedAcyclicGraphIterator();
  ~vtkRootedDirectedAcyclicGraphIterator();

  virtual void Initialize() = 0;
  virtual vtkIdType NextInternal() = 0;

  vtkRootedDirectedAcyclicGraph* DAG;
  vtkIdType RootVertex;
  vtkIdType NextId;

private:
  vtkRootedDirectedAcyclicGraphIterator(const vtkRootedDirectedAcyclicGraphIterator &);  // Not implemented.
  void operator=(const vtkRootedDirectedAcyclicGraphIterator &);        // Not implemented.
};

#endif
