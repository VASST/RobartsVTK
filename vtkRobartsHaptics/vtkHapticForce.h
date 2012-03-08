#ifndef __vtkHapticForce_h
#define __vtkHapticForce_h

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"
#include "vtkForceFeedback.h"

class VTK_EXPORT vtkHapticForce : public vtkObject 
{
public:
    static vtkHapticForce *New();

    vtkTypeMacro(vtkHapticForce,vtkObject);

    void PrintSelf(ostream& os, vtkIndent indent);
	void AddForceModel(vtkForceFeedback * force);
void InsertForceModel(int position, vtkForceFeedback * force);
vtkForceFeedback * GetForceModel(int position);
int GetNumberOfFrames();

protected:
	vtkHapticForce();
	~vtkHapticForce();

private:
	//BTX
	std::vector<vtkForceFeedback *>  forceModel;
	int NumberOfFrames;
	//ETX
};

#endif
