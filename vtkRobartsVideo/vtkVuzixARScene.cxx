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
	leftEyePose = vtkTransform::New();
	leftEyePose->PostMultiply();
	rightEyePose = vtkTransform::New();
	rightEyePose->PostMultiply();

}

vtkVuzixARScene::~vtkVuzixARScene(){
	leftEyeRenderer->Delete();
	rightEyeRenderer->Delete();
	leftEyePose->Delete();
	rightEyePose->Delete();
}


void vtkVuzixARScene::Update(){

	//compute the new camera poses
	leftEyePose->Identity();
	rightEyePose->Identity();
	leftEyePose->Concatenate(deviceToLeftEye);
	rightEyePose->Concatenate(deviceToRightEye);
	leftEyePose->Concatenate(trackedDevice);
	rightEyePose->Concatenate(trackedDevice);

	double focalLength = 1.0;

	vtkTransform* leftFocalPoint = vtkTransform::New();
	leftFocalPoint->Identity();
	leftFocalPoint->PostMultiply();
	leftFocalPoint->Translate( 0, 0, focalLength );
	leftFocalPoint->Concatenate( leftEyePose );
	
	double* leftViewUp = leftEyePose->TransformDoublePoint(0,-1,0);
	double leftPosition[3];
	leftEyePose->GetPosition(leftPosition);
	leftViewUp[0] -= leftPosition[0];
	leftViewUp[1] -= leftPosition[1];
	leftViewUp[2] -= leftPosition[2];

	vtkTransform* rightFocalPoint = vtkTransform::New();
	rightFocalPoint->Identity();
	rightFocalPoint->PostMultiply();
	rightFocalPoint->Translate( 0, 0, focalLength );
	rightFocalPoint->Concatenate( rightEyePose );

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

	UpdateFrameSizes();

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

void vtkVuzixARScene::AddViewProp( vtkProp* p ){
	leftEyeRenderer->AddViewProp(p);
	rightEyeRenderer->AddViewProp(p);
}

void vtkVuzixARScene::RemoveViewProp( vtkProp* p ){
	leftEyeRenderer->RemoveViewProp(p);
	rightEyeRenderer->RemoveViewProp(p);
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
	this->leftEyeCamera->SetPixelwiseIntrinsicParameters(fx,fy,cx,cy);
}

void vtkVuzixARScene::SetRightEyePixelwiseIntrinsicParameters(	double fx,
																double fy,
																double cx,
																double cy ){
	this->rightEyeCamera->SetPixelwiseIntrinsicParameters(fx,fy,cx,cy);
}