/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    $RCSfile: vtkRootedDirectedAcyclicGraphForwardIterator.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME vtkRootedDirectedAcyclicGraphForwardIterator - topological sort iterator through a vtkRootedDirectedAcyclicGraph
//
// .SECTION Description
// vtkRootedDirectedAcyclicGraphForwardIterator performs a topological sort of a rooted DAG.
//
// After setting up the iterator, the normal mode of operation is to
// set up a <code>while(iter->HasNext())</code> loop, with the statement
// <code>vtkIdType vertex = iter->Next()</code> inside the loop.
//
// .SECTION Thanks
// Thanks to John SH Baxter for submitting this class.

#ifndef __vtkRootedDirectedAcyclicGraphForwardIterator_h
#define __vtkRootedDirectedAcyclicGraphForwardIterator_h

#include "vtkRootedDirectedAcyclicGraphIterator.h"

class vtkRootedDirectedAcyclicGraphForwardIteratorInternals;
class vtkIntArray;

class vtkRobartsCommonExport vtkRootedDirectedAcyclicGraphForwardIterator : public vtkRootedDirectedAcyclicGraphIterator
{
public:
  static vtkRootedDirectedAcyclicGraphForwardIterator* New();
  vtkTypeMacro(vtkRootedDirectedAcyclicGraphForwardIterator, vtkRootedDirectedAcyclicGraphIterator);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

protected:
  vtkRootedDirectedAcyclicGraphForwardIterator();
  ~vtkRootedDirectedAcyclicGraphForwardIterator();

  virtual void Initialize();
  virtual vtkIdType NextInternal();

  vtkRootedDirectedAcyclicGraphForwardIteratorInternals* Internals;
  vtkIntArray* Color;

  //BTX
  enum ColorType
  {
    WHITE,
    GRAY,
    BLACK
  };
  //ETX

  vtkIdType CurrentLevel;

private:
  vtkRootedDirectedAcyclicGraphForwardIterator(const vtkRootedDirectedAcyclicGraphForwardIterator &);  // Not implemented.
  void operator=(const vtkRootedDirectedAcyclicGraphForwardIterator &);        // Not implemented.
};

#endif
