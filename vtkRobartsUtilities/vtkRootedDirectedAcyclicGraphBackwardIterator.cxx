/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkRootedDirectedAcyclicGraphBackwardIterator.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkRootedDirectedAcyclicGraphBackwardIterator.h"

#include "vtkIntArray.h"
#include "vtkObjectFactory.h"
#include "vtkRootedDirectedAcyclicGraph.h"

#include <stack>
#include <queue>
using std::stack;
using std::queue;

class vtkRootedDirectedAcyclicGraphBackwardIteratorInternals
{
public:
  stack<vtkIdType> Stack;
};

vtkStandardNewMacro(vtkRootedDirectedAcyclicGraphBackwardIterator);

vtkRootedDirectedAcyclicGraphBackwardIterator::vtkRootedDirectedAcyclicGraphBackwardIterator()
{
  this->Internals = new vtkRootedDirectedAcyclicGraphBackwardIteratorInternals();
  this->Color = vtkIntArray::New();
}

vtkRootedDirectedAcyclicGraphBackwardIterator::~vtkRootedDirectedAcyclicGraphBackwardIterator()
{
  if (this->Internals)
    {
    delete this->Internals;
    this->Internals = NULL;
    }
  if (this->Color)
    {
    this->Color->Delete();
    this->Color = NULL;
    }
}

void vtkRootedDirectedAcyclicGraphBackwardIterator::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void vtkRootedDirectedAcyclicGraphBackwardIterator::Initialize()
{
  if (this->DAG == NULL)
    {
    return;
    }
  // Set all colors to white
  this->Color->Resize(this->DAG->GetNumberOfVertices());
  for (vtkIdType i = 0; i < this->DAG->GetNumberOfVertices(); i++)
    {
    this->Color->SetValue(i, this->WHITE);
    }
  if (this->RootVertex < 0)
    {
    this->RootVertex = this->DAG->GetRoot();
    }
  while (this->Internals->Stack.size())
    {
    this->Internals->Stack.pop();
    }

  // Find the first item
  if (this->DAG->GetNumberOfVertices() > 0)
    {
    this->NextId = this->NextInternal();
    }
  else
    {
    this->NextId = -1;
    }
  this->CurrentLevel = this->DAG->GetUpLevel(this->RootVertex);
}

vtkIdType vtkRootedDirectedAcyclicGraphBackwardIterator::NextInternal()
{
  if(this->Color->GetValue(this->RootVertex) == this->WHITE)
    {
    this->Color->SetValue(this->RootVertex, this->GRAY);

	queue<vtkIdType> Queue;
	Queue.push(this->RootVertex);
	while(Queue.size() > 0)
	  {
	  vtkIdType currentId = Queue.front();
	  this->Internals->Stack.push(currentId);
	  Queue.pop();

	  for(vtkIdType childNum = 0; childNum < this->DAG->GetNumberOfChildren(currentId); childNum++)
		{
		  vtkIdType childId = this->DAG->GetChild(currentId, childNum);
		  if(this->DAG->GetUpLevel(childId) != this->DAG->GetUpLevel(currentId) + 1) continue;
		  if(this->Color->GetValue(childId) == this->WHITE)
		  {
		  // Found a white vertex; make it gray, add it to the queue
		  this->Color->SetValue(childId, this->GRAY);
		  Queue.push(childId);
		  }
        }

	  this->Color->SetValue(currentId, this->BLACK);
	  }

    }

  while (this->Internals->Stack.size() > 0)
    {
	vtkIdType currentId = this->Internals->Stack.top();
    this->Internals->Stack.pop();
    return currentId;
    }
  return -1;
}
