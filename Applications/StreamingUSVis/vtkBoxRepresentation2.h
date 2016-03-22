
#ifndef __VTKBOXREPRESENTATION_H
#define __VTKBOXREPRESENTATION_H

#include <vtkBoxRepresentation.h>
#include <vtkPolyData.h>

class vtkBoxRepresentation2: public vtkBoxRepresentation{

public:

	// Description:
	// Instantiate the class.
	static vtkBoxRepresentation2 *New();

	void moveSelectedFace(double *p1, double *p2, double *dir,
                                    double *x1, double *x2, double *x3, double *x4,
                                    double *x5);
	void transformSelectedPlane(vtkTransform *);
	void setSelectedFace(int idx );
	void setSelectedOutlineEdge(int idx);

protected:
	vtkBoxRepresentation2();
	~vtkBoxRepresentation2();

};

#endif //__VTKBOXREPRESENTATION