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
	
	IdealLeftFocus = 1.0;
	IdealRightFocus = 1.0;

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

	//find the focal point of the left camera
	leftFocalPoint->Identity();
	leftFocalPoint->Translate( 0, 0, IdealLeftFocus );
	leftFocalPoint->Concatenate( deviceToLeftEye );

	//apply the poses to the camera
	leftEyeCamera->SetPosition( deviceToLeftEye->GetPosition() );
	leftEyeCamera->SetFocalPoint( leftFocalPoint->GetPosition() );
	leftEyeCamera->OrthogonalizeViewUp();

}

void vtkVuzixARScene::SetRightEyeTransform( vtkTransform* t){
	deviceToRightEye = t;
	deviceToRightEye->Inverse();

	//find the focal point of the right camera
	rightFocalPoint->Identity();
	rightFocalPoint->Translate( 0, 0, IdealRightFocus );
	rightFocalPoint->Concatenate( deviceToRightEye );
	
	//apply the poses to the camera
	rightEyeCamera->SetPosition( deviceToRightEye->GetPosition() );
	rightEyeCamera->SetFocalPoint( rightFocalPoint->GetPosition() );
	rightEyeCamera->OrthogonalizeViewUp();
}

void vtkVuzixARScene::SetLeftEyePixelwiseIntrinsicParameters(	double fx,
																double fy,
																double cx,
																double cy ){
	this->leftEyeCamera->SetPixelwiseIntrinsicParameters(fx,fy,cx,cy);
	IdealLeftFocus = 0.5 * (fx + fy);

	//find the focal point of the left camera
	leftFocalPoint->Identity();
	leftFocalPoint->Translate( 0, 0, IdealLeftFocus );
	leftFocalPoint->Concatenate( deviceToLeftEye );

	//apply the poses to the camera
	leftEyeCamera->SetPosition( deviceToLeftEye->GetPosition() );
	leftEyeCamera->SetFocalPoint( leftFocalPoint->GetPosition() );
	leftEyeCamera->OrthogonalizeViewUp();
}

void vtkVuzixARScene::SetRightEyePixelwiseIntrinsicParameters(	double fx,
																double fy,
																double cx,
																double cy ){
	this->rightEyeCamera->SetPixelwiseIntrinsicParameters(fx,fy,cx,cy);
	IdealRightFocus = 0.5 * (fx + fy);

	//find the focal point of the right camera
	rightFocalPoint->Identity();
	rightFocalPoint->Translate( 0, 0, IdealRightFocus );
	rightFocalPoint->Concatenate( deviceToRightEye );
	
	//apply the poses to the camera
	rightEyeCamera->SetPosition( deviceToRightEye->GetPosition() );
	rightEyeCamera->SetFocalPoint( rightFocalPoint->GetPosition() );
	rightEyeCamera->OrthogonalizeViewUp();
}