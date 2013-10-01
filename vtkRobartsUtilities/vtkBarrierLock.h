/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkBarrierLock.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkBarrierLock -
// .SECTION Description
// A lock that waits for a given number of threads to enter before releasing all of them
// .SECTION Caveats
// Will only work as well as the underlying vtk mutices
//

#ifndef __VTKBARRIERLOCK_H
#define __VTKBARRIERLOCK_H

#include "vtkObject.h"
#include "vtkMutexLock.h"
#include "vtkConditionVariable.h"
#include "vtkSetGet.h"

class vtkBarrierLock : public vtkObject {
public:
	static vtkBarrierLock *New();

	void Initialize(int numThreads);
	void DeInitialize();
	
	vtkSetMacro(Repeatable,bool);
	vtkGetMacro(Repeatable,bool);

	bool Query();
	void Enter();

protected:
	vtkBarrierLock();
	~vtkBarrierLock();
	
	vtkMutexLock* EntryLock;
	vtkConditionVariable* Condition;

	bool Repeatable;
	bool BarrierUsed;
	int NumEntered;
	int NumEnteredMax;

private:
	vtkBarrierLock(const vtkBarrierLock&);  // Not implemented.
	void operator=(const vtkBarrierLock&);  // Not implemented.
};

#endif //__VTKBARRIERLOCK_H