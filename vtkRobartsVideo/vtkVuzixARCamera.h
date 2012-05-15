
#ifndef __vtkVuzixARCamera_H
#define __vtkVuzixARCamera_H

#include "vtkOpenGLCamera.h"
#include "vtkTransform.h"
#include "vtkPerspectiveTransform.h"
#include "vtkMatrix4x4.h"


class vtkVuzixARCamera : public vtkOpenGLCamera
{
public:
	static vtkVuzixARCamera *New();
	vtkTypeMacro(vtkVuzixARCamera,vtkCamera);
	
	void SetPixelwiseIntrinsicParameters( double fx, double fy, double cx, double cy);
	void SetFrameSize( double x, double y );

	vtkMatrix4x4 * 	GetPerspectiveTransformMatrix (double aspect, double nearz, double farz);
	vtkMatrix4x4 * 	GetProjectionTransformMatrix (double aspect, double nearz, double farz);
	vtkPerspectiveTransform * 	GetProjectionTransformObject (double aspect, double nearz, double farz);

protected:
	vtkVuzixARCamera();
	~vtkVuzixARCamera();

private:

	void			ComputeProjectionTransform(double nearz, double farz);
	
	double			fx;
	double			fy;
	double			cx;
	double			cy;
	double			sizeX;
	double			sizeY;

};

#endif