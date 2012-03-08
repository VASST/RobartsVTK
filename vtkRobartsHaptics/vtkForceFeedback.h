#ifndef __vtkForceFeedback_h
#define __vtkForceFeedback_h

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"

class VTK_EXPORT vtkForceFeedback : public vtkObject 
{
public:
    static vtkForceFeedback *New();

    vtkTypeMacro(vtkForceFeedback,vtkObject);

    void PrintSelf(ostream& os, vtkIndent indent);

	virtual int GenerateForce(vtkMatrix4x4 * hapticPosition, double force[3]);
	~vtkForceFeedback();

protected:
	vtkForceFeedback();
	

private:

};

#endif
