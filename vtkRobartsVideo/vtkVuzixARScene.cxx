#include "vtkVuzixARScene.h"
#include "vtkRenderWindow.h"
#include "vtkObjectFactory.h"

#include <math.h>
#define PI 3.14159265

vtkStandardNewMacro(vtkVuzixARScene);

vtkVuzixARScene::vtkVuzixARScene(){

	//allocates the piece of the compilation
	leftEyeRenderer = vtkRenderer::New();
	rightEyeRenderer = vtkRenderer::New();
	leftEyeCamera = leftEyeRenderer->GetActiveCamera();
	rightEyeCamera = rightEyeRenderer->GetActiveCamera();

	//allocate the texture bases for the renderers and link them together
	leftEyePhysicalWorld = 0;
	leftEyeTexture = vtkTexture::New();
	leftEyeRenderer->SetBackgroundTexture( leftEyeTexture );
	leftEyeRenderer->SetTexturedBackground( true );
	rightEyePhysicalWorld = 0;
	rightEyeTexture = vtkTexture::New();
	rightEyeRenderer->SetBackgroundTexture( rightEyeTexture );
	rightEyeRenderer->SetTexturedBackground( true );

	//set some defaults for the poses
	trackedDevice = vtkTransform::New();
	trackedDevice->Identity();
	deviceToLeftEye = vtkTransform::New();
	deviceToLeftEye->Identity();
	deviceToRightEye = vtkTransform::New();
	deviceToRightEye->Identity();
	leftEyePose = vtkTransform::New();
	leftEyePose->PostMultiply();
	rightEyePose = vtkTransform::New();
	rightEyePose->PostMultiply();

	//allocate space for the temporary variables
	leftFocalPoint = vtkTransform::New();
	leftFocalPoint->PostMultiply();
	rightFocalPoint = vtkTransform::New();
	rightFocalPoint->PostMultiply();
	leftScreenScale = vtkTransform::New();
	rightScreenScale = vtkTransform::New();
	
	//initialize the screen sizes
	IdealScreenSizeLeft[0] = 0;
	IdealScreenSizeLeft[1] = 0;
	IdealScreenSizeRight[0] = 0;
	IdealScreenSizeRight[1] = 0;
	ActualScreenSizeLeft[0] = 0;
	ActualScreenSizeLeft[1] = 0;
	ActualScreenSizeRight[0] = 0;
	ActualScreenSizeRight[1] = 0;
	leftIdealFocus = 1.0;
	rightIdealFocus = 1.0;
	leftPrinciplePoint[0] = 0.0;
	leftPrinciplePoint[1] = 0.0;
	rightPrinciplePoint[0] = 0.0;
	rightPrinciplePoint[1] = 0.0;

}

vtkVuzixARScene::~vtkVuzixARScene(){
	leftEyeTexture->Delete();
	rightEyeTexture->Delete();
	leftEyeRenderer->Delete();
	rightEyeRenderer->Delete();
	leftEyeCamera->Delete();
	rightEyeCamera->Delete();
	leftEyePose->Delete();
	rightEyePose->Delete();
	leftFocalPoint->Delete();
	rightFocalPoint->Delete();
	deviceToLeftEye->Delete();
	deviceToRightEye->Delete();
	leftScreenScale->Delete();
	rightScreenScale->Delete();
}


void vtkVuzixARScene::Update(){

	//compute the new camera poses
	leftEyePose->Identity();
	rightEyePose->Identity();
	leftEyePose->Concatenate(deviceToLeftEye);
	rightEyePose->Concatenate(deviceToRightEye);
	leftEyePose->Concatenate(trackedDevice);
	rightEyePose->Concatenate(trackedDevice);

	//use a reasonable value for the focal length (only matters for 
	//resolution of the volume mapper)
	double focalLength = 0.01;

	//find the focal point of the left camera
	leftFocalPoint->Identity();
	leftFocalPoint->Translate( 0, 0, focalLength );
	leftFocalPoint->Concatenate( leftEyePose );
	
	//find the viewUp vector and position of the left camera
	double* leftViewUp = leftEyePose->TransformDoublePoint(0,-1,0);
	double leftPosition[3];
	leftEyePose->GetPosition(leftPosition);
	leftViewUp[0] -= leftPosition[0];
	leftViewUp[1] -= leftPosition[1];
	leftViewUp[2] -= leftPosition[2];
	
	//find the focal point of the right camera
	rightFocalPoint->Identity();
	rightFocalPoint->Translate( 0, 0, focalLength );
	rightFocalPoint->Concatenate( rightEyePose );
	
	//find the viewUp vector and position of the right camera
	double* rightViewUp = rightEyePose->TransformDoublePoint(0,-1,0);
	double rightPosition[3];
	rightEyePose->GetPosition(rightPosition);
	rightViewUp[0] -= rightPosition[0];
	rightViewUp[1] -= rightPosition[1];
	rightViewUp[2] -= rightPosition[2];

	//apply the poses to the camera
	leftEyeCamera->SetPosition( leftPosition );
	rightEyeCamera->SetPosition( rightPosition );
	leftEyeCamera->SetViewUp( leftViewUp );
	rightEyeCamera->SetViewUp( rightViewUp );
	leftEyeCamera->SetFocalPoint( leftFocalPoint->GetPosition() );
	rightEyeCamera->SetFocalPoint( rightFocalPoint->GetPosition() );

	//cross-check the frame sizes
	if( !IdealScreenSizeLeft[0] || !IdealScreenSizeLeft[1] ||
		!IdealScreenSizeRight[0] || !IdealScreenSizeRight[1] )
		UpdateFrameSizes();
	
	//update the left camera scaling
	int* size = leftEyeRenderer->GetSize();
	if( size[0] && size[1] && ( size[0] != ActualScreenSizeLeft[0] ||
		size[1] != ActualScreenSizeLeft[1] ) ){
		
		//update stored screen size and calculate required aspect ratios
		ActualScreenSizeLeft[0] = size[0];
		ActualScreenSizeLeft[1] = size[1];
		double aspectRatio = (double) size[0] / (double) size[1];
		double idealAspectRatio = (double) IdealScreenSizeLeft[0] / (double) IdealScreenSizeLeft[1];
		
		//update camera scale matrix for anisotropic scaling (adjusting for screen size and principle point)
		//TODO Add logic for the principle point
		leftScreenScale->Identity();
		if( idealAspectRatio < aspectRatio )
			rightScreenScale->Scale( idealAspectRatio / aspectRatio, 1, 1);
		else
			rightScreenScale->Scale( 1, idealAspectRatio / aspectRatio, 1);
	}

	//update the right camera scaling
	size = rightEyeRenderer->GetSize();
	if( size[0] && size[1] && ( size[0] != ActualScreenSizeRight[0] ||
		size[1] != ActualScreenSizeRight[1] ) ){
		
		//update stored screen size and calculate required aspect ratios
		ActualScreenSizeRight[0] = size[0];
		ActualScreenSizeRight[1] = size[1];
		double aspectRatio = (double) size[0] / (double) size[1];
		double idealAspectRatio = (double) ActualScreenSizeRight[0] / (double) ActualScreenSizeRight[1];

		//update camera scale matrix for anisotropic scaling (adjusting for screen size and principle point)
		//TODO Add logic for the principle point
		rightScreenScale->Identity();
		if( idealAspectRatio < aspectRatio )
			rightScreenScale->Scale( idealAspectRatio / aspectRatio, 1, 1);
		else
			rightScreenScale->Scale( 1, idealAspectRatio / aspectRatio, 1);
	}

}

//update the frame sizes on the cameras (used for determining the projection matrix through the view angle)
void vtkVuzixARScene::UpdateFrameSizes(){
	if( leftEyePhysicalWorld ){
		int extent[6];
		this->leftEyePhysicalWorld->GetExtent( extent );
		IdealScreenSizeLeft[0] = extent[1] - extent[0]+1;
		IdealScreenSizeLeft[1] = extent[3] - extent[2]+1;
		double viewAngle = std::atan( 0.5 * (double)(IdealScreenSizeLeft[1]-1) / leftIdealFocus ) * 360.0 / PI;
		this->leftEyeCamera->SetViewAngle(viewAngle);
	}
	if( rightEyePhysicalWorld ){
		int extent[6];
		this->rightEyePhysicalWorld->GetExtent( extent );
		IdealScreenSizeRight[0] = extent[1] - extent[0]+1;
		IdealScreenSizeRight[1] = extent[3] - extent[2]+1;
		double viewAngle = std::atan( 0.5 * (double)(IdealScreenSizeRight[1]-1) / rightIdealFocus ) * 360.0 / PI;
		this->rightEyeCamera->SetViewAngle(viewAngle);
	}
}

vtkRenderer* vtkVuzixARScene::GetLeftEyeView(){
	return leftEyeRenderer;
}

vtkRenderer* vtkVuzixARScene::GetRightEyeView(){
	return rightEyeRenderer;
}

void vtkVuzixARScene::SetLeftEyeSource( vtkImageData* eye ){
	this->leftEyeTexture->SetInput( (vtkDataObject*) eye );
	leftEyePhysicalWorld = eye;
	UpdateFrameSizes();
}

void vtkVuzixARScene::SetRightEyeSource( vtkImageData* eye ){
	this->rightEyeTexture->SetInput( (vtkDataObject*) eye );
	rightEyePhysicalWorld = eye;
	UpdateFrameSizes();
}

void vtkVuzixARScene::SetTrackedTransform( vtkTransform* t){
	trackedDevice = t;
}

void vtkVuzixARScene::SetLeftEyeTransform( vtkTransform* t){
	deviceToLeftEye = t;
	deviceToLeftEye->Inverse();
}

void vtkVuzixARScene::SetRightEyeTransform( vtkTransform* t){
	deviceToRightEye = t;
	deviceToRightEye->Inverse();
}

void vtkVuzixARScene::SetLeftEyePixelwiseIntrinsicParameters(	double fx,
																double fy,
																double cx,
																double cy ){
	//get the relative focus in terms of the view angle
	leftIdealFocus = 0.5 * (fx + fy);
	double viewAngle = std::atan( 0.5 * (double)(IdealScreenSizeLeft[1]-1) / leftIdealFocus ) * 360.0 / PI;
	this->leftEyeCamera->SetViewAngle(viewAngle);
	
	//save the principle point for later camera shifting
	leftPrinciplePoint[0] = cx;
	leftPrinciplePoint[1] = cy;
}

void vtkVuzixARScene::SetRightEyePixelwiseIntrinsicParameters(	double fx,
																double fy,
																double cx,
																double cy ){
	//get the relative focus in terms of the view angle
	rightIdealFocus = 0.5 * (fx + fy);
	double viewAngle = std::atan( 0.5 * (double)(IdealScreenSizeRight[1]-1) / rightIdealFocus ) * 360.0 / PI;
	this->rightEyeCamera->SetViewAngle(viewAngle);

	//save the principle point for later camera shifting
	rightPrinciplePoint[0] = cx;
	rightPrinciplePoint[1] = cy;
}