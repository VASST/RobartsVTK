#include "vtkVuzixARScene.h"
#include "vtkObjectFactory.h"

vtkStandardNewMacro(vtkVuzixARScene);

vtkVuzixARScene::vtkVuzixARScene(){

	//allocates the piece of the compilation
	leftEyeRenderer = vtkRenderer::New();
	rightEyeRenderer = vtkRenderer::New();
	leftEyeCamera = vtkVuzixARCamera::New();
	leftEyeRenderer->SetActiveCamera(leftEyeCamera);
	rightEyeCamera = vtkVuzixARCamera::New();
	rightEyeRenderer->SetActiveCamera(rightEyeCamera);

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

	//allocate space for the temporary variables
	leftFocalPoint = vtkTransform::New();
	leftFocalPoint->PostMultiply();
	rightFocalPoint = vtkTransform::New();
	rightFocalPoint->PostMultiply();

}

vtkVuzixARScene::~vtkVuzixARScene(){
	leftEyeTexture->Delete();
	rightEyeTexture->Delete();
	leftEyeRenderer->Delete();
	rightEyeRenderer->Delete();
	leftEyeCamera->Delete();
	rightEyeCamera->Delete();
	leftFocalPoint->Delete();
	rightFocalPoint->Delete();
	deviceToLeftEye->Delete();
	deviceToRightEye->Delete();
}


void vtkVuzixARScene::Update(){

	//update the frame sizes
	UpdateFrameSizes();

	//tell the cameras that they have been modified
	leftEyeCamera->Modified();
	rightEyeCamera->Modified();
}

//update the frame sizes on the cameras (used for determining the projection matrix)
void vtkVuzixARScene::UpdateFrameSizes(){
	if( leftEyePhysicalWorld ){
		int extent[6];
		this->leftEyePhysicalWorld->GetExtent( extent );
		this->leftEyeCamera->SetFrameSize(	(double) (extent[1] - extent[0]+1),
											(double) (extent[3] - extent[2]+1) );
	}
	if( rightEyePhysicalWorld ){
		int extent[6];
		this->rightEyePhysicalWorld->GetExtent( extent );
		this->rightEyeCamera->SetFrameSize(	(double) (extent[1] - extent[0]+1),
											(double) (extent[3] - extent[2]+1) );
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
}

void vtkVuzixARScene::SetRightEyeSource( vtkImageData* eye ){
	this->rightEyeTexture->SetInput( (vtkDataObject*) eye );
	rightEyePhysicalWorld = eye;

}

void vtkVuzixARScene::SetTrackedTransform( vtkTransform* t){
	trackedDevice = t;
	leftEyeCamera->SetUserTransform(t);
	rightEyeCamera->SetUserTransform(t);
}

void vtkVuzixARScene::SetLeftEyeTransform( vtkTransform* t){
	deviceToLeftEye = t;
	deviceToLeftEye->Inverse();
	
	//use a reasonable value for the focal length (only matters for 
	//resolution of the volume mapper)
	double focalLength = 1.0;

	//find the focal point of the left camera
	leftFocalPoint->Identity();
	leftFocalPoint->Translate( 0, 0, focalLength );
	leftFocalPoint->Concatenate( deviceToLeftEye );
	
	//find the viewUp vector and position of the left camera
	double* leftViewUp = deviceToLeftEye->TransformDoublePoint(0,-1,0);
	double leftPosition[3];
	deviceToLeftEye->GetPosition(leftPosition);
	leftViewUp[0] -= leftPosition[0];
	leftViewUp[1] -= leftPosition[1];
	leftViewUp[2] -= leftPosition[2];

	//apply the poses to the camera
	leftEyeCamera->SetPosition( leftPosition );
	leftEyeCamera->SetViewUp( leftViewUp );
	leftEyeCamera->SetFocalPoint( leftFocalPoint->GetPosition() );

}

void vtkVuzixARScene::SetRightEyeTransform( vtkTransform* t){
	deviceToRightEye = t;
	deviceToRightEye->Inverse();

	//use a reasonable value for the focal length (only matters for 
	//resolution of the volume mapper)
	double focalLength = 1.0;

	//find the focal point of the right camera
	rightFocalPoint->Identity();
	rightFocalPoint->Translate( 0, 0, focalLength );
	rightFocalPoint->Concatenate( deviceToRightEye );
	
	//find the viewUp vector and position of the right camera
	double* rightViewUp = deviceToRightEye->TransformDoublePoint(0,-1,0);
	double rightPosition[3];
	deviceToRightEye->GetPosition(rightPosition);
	rightViewUp[0] -= rightPosition[0];
	rightViewUp[1] -= rightPosition[1];
	rightViewUp[2] -= rightPosition[2];

	//apply the poses to the camera
	rightEyeCamera->SetPosition( rightPosition );
	rightEyeCamera->SetViewUp( rightViewUp );
	rightEyeCamera->SetFocalPoint( rightFocalPoint->GetPosition() );
}

void vtkVuzixARScene::SetLeftEyePixelwiseIntrinsicParameters(	double fx,
																double fy,
																double cx,
																double cy ){
	this->leftEyeCamera->SetPixelwiseIntrinsicParameters(fx,fy,cx,cy);
}

void vtkVuzixARScene::SetRightEyePixelwiseIntrinsicParameters(	double fx,
																double fy,
																double cx,
																double cy ){
	this->rightEyeCamera->SetPixelwiseIntrinsicParameters(fx,fy,cx,cy);
}