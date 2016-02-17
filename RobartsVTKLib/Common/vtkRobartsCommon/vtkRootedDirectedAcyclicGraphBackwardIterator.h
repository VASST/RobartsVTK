/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkRootedDirectedAcyclicGraphBackwardIterator.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// .NAME vtkRootedDirectedAcyclicGraphBackwardIterator - topological sort iterator through a vtkRootedDirectedAcyclicGraph
//
// .SECTION Description
// vtkRootedDirectedAcyclicGraphBackwardIterator performs a topological sort of a rooted DAG.
//
// After setting up the iterator, the normal mode of operation is to
// set up a <code>while(iter->HasNext())</code> loop, with the statement
// <code>vtkIdType vertex = iter->Next()</code> inside the loop.
//
// .SECTION Thanks
// Thanks to John SH Baxter for submitting this class.

#ifndef __vtkRootedDirectedAcyclicGraphBackwardIterator_h
#define __vtkRootedDirectedAcyclicGraphBackwardIterator_h

#include "vtkRootedDirectedAcyclicGraphIterator.h"

class vtkRootedDirectedAcyclicGraphBackwardIteratorInternals;
class vtkIntArray;

class VTKROBARTSCOMMON_EXPORT vtkRootedDirectedAcyclicGraphBackwardIterator : public vtkRootedDirectedAcyclicGraphIterator
{
public:
  static vtkRootedDirectedAcyclicGraphBackwardIterator* New();
  vtkTypeMacro(vtkRootedDirectedAcyclicGraphBackwardIterator, vtkRootedDirectedAcyclicGraphIterator);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

protected:
  vtkRootedDirectedAcyclicGraphBackwardIterator();
  ~vtkRootedDirectedAcyclicGraphBackwardIterator();

  virtual void Initialize();
  virtual vtkIdType NextInternal();

  vtkRootedDirectedAcyclicGraphBackwardIteratorInternals* Internals;
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
  vtkRootedDirectedAcyclicGraphBackwardIterator(const vtkRootedDirectedAcyclicGraphBackwardIterator &);  // Not implemented.
  void operator=(const vtkRootedDirectedAcyclicGraphBackwardIterator &);        // Not implemented.
};

#endif
