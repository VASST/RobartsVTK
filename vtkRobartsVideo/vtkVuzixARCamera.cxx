#include "vtkVuzixARCamera.h"
#include "vtkObjectFactory.h"
#include "vtkMath.h"

vtkStandardNewMacro(vtkVuzixARCamera);

//----------------------------------------------------------------------------
vtkVuzixARCamera::vtkVuzixARCamera(){
	fx = 854.067993;
	fy = 856.185364;
	cx = 307.904205;
	cy = 249.460327;
	sizeX = 640.0;
	sizeY = 480.0;
}

//----------------------------------------------------------------------------
vtkVuzixARCamera::~vtkVuzixARCamera(){
}

//----------------------------------------------------------------------------
void vtkVuzixARCamera::SetPixelwiseIntrinsicParameters( double fx, double fy, double cx, double cy){
	this->fx = fx;
	this->fy = fy;
	this->cx = cx;
	this->cy = cy;
}

void vtkVuzixARCamera::SetFrameSize( double x, double y ){
	this->sizeX = x;
	this->sizeY = y;
}


//----------------------------------------------------------------------------
#ifndef VTK_LEGACY_REMOVE
// Return the projection transform matrix. See ComputeProjectionTransform.
vtkMatrix4x4* vtkVuzixARCamera::GetPerspectiveTransformMatrix(double aspect,
                                                       double nearz,
                                                       double farz)
{
  VTK_LEGACY_REPLACED_BODY(vtkCamera::GetPerspectiveTransformMatrix,"VTK 5.4",
                           vtkCamera::GetProjectionTransformMatrix);
  return this->GetProjectionTransformMatrix(aspect, nearz,farz);
}
#endif

//----------------------------------------------------------------------------
// Return the projection transform matrix. See ComputeProjectionTransform.
vtkMatrix4x4* vtkVuzixARCamera::GetProjectionTransformMatrix(double aspect,
                                                      double nearz,
                                                      double farz)
{
  this->ComputeProjectionTransform(nearz, farz);

  // return the transform
  //return this->vtkCamera::GetProjectionTransformMatrix(480.0/640.0, nearz, farz);
  return this->ProjectionTransform->GetMatrix();
}

//----------------------------------------------------------------------------
// Return the projection transform object. See ComputeProjectionTransform.
vtkPerspectiveTransform* vtkVuzixARCamera::GetProjectionTransformObject(double aspect,
                                                                 double nearz,
                                                                 double farz)
{
  this->ComputeProjectionTransform(nearz, farz);

  // return the transform
  return ProjectionTransform;
}

//----------------------------------------------------------------------------
// Compute the projection transform matrix. This is used in converting
// between view and world coordinates.
void vtkVuzixARCamera::ComputeProjectionTransform(double nearz, double farz)
{

	this->ProjectionTransform->Identity();
	this->ProjectionTransform->PreMultiply();
	this->ProjectionTransform->AdjustZBuffer(-1.0,1.0,nearz,farz);

	double elements[16];
	elements[0] = 2.0 * this->fx / this->sizeX;
	elements[1] = 0.0;
	elements[2] = 2.0 * this->cx / this->sizeX - 1.0;
	elements[3] = 0.0;
	
	elements[4] = 0.0;
	elements[5] = 2.0 * this->fy / this->sizeY;
	elements[6] = 2.0 * this->cy / this->sizeY - 1.0;
	elements[7] = 0.0;
	
	elements[8] = 0.0;
	elements[9] = 0.0;
	elements[10] = (this->ClippingRange[1] + this->ClippingRange[0]) / (this->ClippingRange[0] - this->ClippingRange[1]);
	elements[11] = 2.0 * ( this->ClippingRange[1] * this->ClippingRange[0] / (this->ClippingRange[0] - this->ClippingRange[1]));
	
	elements[12] = 0.0;
	elements[13] = 0.0;
	elements[14] = -1.0;
	elements[15] = 0.0;
	
	this->ProjectionTransform->Concatenate( elements );

	// apply user defined transform last if there is one
	//if ( this->UserTransform ) this->ProjectionTransform->Concatenate( this->UserTransform->GetMatrix() );



}