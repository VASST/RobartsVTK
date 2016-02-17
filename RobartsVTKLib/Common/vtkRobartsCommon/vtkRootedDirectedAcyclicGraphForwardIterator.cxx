/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkRootedDirectedAcyclicGraphForwardIterator.cxx,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkRootedDirectedAcyclicGraphForwardIterator.h"

#include "vtkIntArray.h"
#include "vtkObjectFactory.h"
#include "vtkRootedDirectedAcyclicGraph.h"

#include <queue>
using std::queue;

class vtkRootedDirectedAcyclicGraphForwardIteratorInternals
{
public:
  queue<vtkIdType> Queue;
};

vtkStandardNewMacro(vtkRootedDirectedAcyclicGraphForwardIterator);

vtkRootedDirectedAcyclicGraphForwardIterator::vtkRootedDirectedAcyclicGraphForwardIterator()
{
  this->Internals = new vtkRootedDirectedAcyclicGraphForwardIteratorInternals();
  this->Color = vtkIntArray::New();
}

vtkRootedDirectedAcyclicGraphForwardIterator::~vtkRootedDirectedAcyclicGraphForwardIterator()
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

void vtkRootedDirectedAcyclicGraphForwardIterator::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void vtkRootedDirectedAcyclicGraphForwardIterator::Initialize()
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
  while (this->Internals->Queue.size())
    {
    this->Internals->Queue.pop();
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

vtkIdType vtkRootedDirectedAcyclicGraphForwardIterator::NextInternal()
{
  if(this->Color->GetValue(this->RootVertex) == this->WHITE)
    {
    this->Color->SetValue(this->RootVertex, this->GRAY);
    this->Internals->Queue.push(this->RootVertex);
    }

  while (this->Internals->Queue.size() > 0)
    {
    vtkIdType currentId = this->Internals->Queue.front();
    this->Internals->Queue.pop();

    for(vtkIdType childNum = 0; childNum < this->DAG->GetNumberOfChildren(currentId); childNum++)
      {
      vtkIdType childId = this->DAG->GetChild(currentId, childNum);
    if(this->DAG->GetUpLevel(childId) != this->DAG->GetUpLevel(currentId) + 1) continue;
      if(this->Color->GetValue(childId) == this->WHITE)
        {
        // Found a white vertex; make it gray, add it to the queue
        this->Color->SetValue(childId, this->GRAY);
        this->Internals->Queue.push(childId);
        }
      }

    this->Color->SetValue(currentId, this->BLACK);
    return currentId;
    }
  return -1;
}
