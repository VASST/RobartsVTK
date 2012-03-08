//==============================================================================

//
//==============================================================================


#ifndef __vtkPolydataForce_h
#define __vtkPolydataForce_h

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

#include "vtkObject.h"
#include "vtkObjectFactory.h"
#include "vtkMatrix4x4.h"
#include "vtkForceFeedback.h"
#include "vtkPolyData.h"

class VTK_EXPORT vtkPolydataForce : public vtkForceFeedback
{
public:
    static vtkPolydataForce *New();
    vtkTypeMacro(vtkPolydataForce,vtkForceFeedback);

	void PrintSelf(ostream& os, vtkIndent indent);

	int GenerateForce(vtkMatrix4x4 * transformMatrix, double force[3]);
	int SetGamma(double gamma);
	void SetInput(vtkPolyData * poly);

protected:
	vtkPolydataForce();
	virtual ~vtkPolydataForce();
	double CalculateDistance(double x, double y, double z);
	void CalculateForce(double x, double y, double z, double force[3]);



private:

	vtkPolyData * poly;
	double gammaSigmoid;
	double scaleForce;
	double lastPos[3];
};

#endif
