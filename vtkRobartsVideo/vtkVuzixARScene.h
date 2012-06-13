
#ifndef __VTKVUZIXARSCENE_H
#define __VTKVUZIXARSCENE_H

#include "vtkRenderer.h"
#include "vtkTransform.h"
#include "vtkCamera.h"
#include "vtkTexture.h"
#include "vtkImageData.h"
#include "vtkActor.h"
#include "vtkVideoSource.h"

class vtkVuzixARScene : public vtkObject  {
public:

	/** @brief VTK compatible constructor method
	 *
	 */
	static vtkVuzixARScene* New();

	//
	void			Update();

	//methods for grabbing the renderers (to add to a render window)
	vtkRenderer*	GetLeftEyeView();
	vtkRenderer*	GetRightEyeView();
	
	//methods for setting the source of the video feeds
	void			SetLeftEyeSource( vtkImageData* );
	void			SetRightEyeSource( vtkImageData* );

	//method for setting the tracked device and rigid transforms from
	//tracked device to camera
	void			SetTrackedTransform( vtkTransform* ); //no assumptions about constancy
	void			SetLeftEyeTransform( vtkTransform* ); //assumes landmark transform is constant
	void			SetRightEyeTransform( vtkTransform* ); //assumes landmark transform is constant
	void			SetLeftEyePixelwiseIntrinsicParameters( double fx,
															double fy,
															double cx,
															double cy );
	void			SetRightEyePixelwiseIntrinsicParameters(double fx,
															double fy,
															double cx,
															double cy );
	void			UpdateFrameSizes();

protected:
	vtkVuzixARScene();
	~vtkVuzixARScene();

private:

	//renderer information for the two views
	vtkRenderer*	leftEyeRenderer;
	vtkRenderer*	rightEyeRenderer;
	vtkCamera*		leftEyeCamera;
	vtkCamera*		rightEyeCamera;
	vtkTexture*		leftEyeTexture;
	vtkTexture*		rightEyeTexture;
	vtkImageData*	leftEyePhysicalWorld;
	vtkImageData*	rightEyePhysicalWorld;

	//pose information for the cameras
	vtkTransform*	trackedDevice;
	vtkTransform*	deviceToLeftEye;
	vtkTransform*	deviceToRightEye;
	vtkTransform*	leftEyePose;
	vtkTransform*	rightEyePose;

	//temporary transforms used
	vtkTransform*	leftFocalPoint;
	vtkTransform*	rightFocalPoint;
	vtkTransform*	leftScreenScale;
	vtkTransform*	rightScreenScale;
	
	int IdealScreenSizeLeft[2];
	int IdealScreenSizeRight[2];
	int ActualScreenSizeLeft[2];
	int ActualScreenSizeRight[2];
	double leftIdealFocus;
	double rightIdealFocus;
	double leftPrinciplePoint[2];
	double rightPrinciplePoint[2];

};

#endif