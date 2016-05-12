#include <iostream>

// VTK includes
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkImageImport.h>
#include <vtkTexture.h>
#include <vtkOpenGLRenderWindow.h> 
#include <vtkOpenGLProperty.h>
#include <vtkOpenGLRenderer.h>
#include <vtkDefaultPass.h>
#include <vtkRenderPass.h>
#include <vtkLightsPass.h>
#include <vtkCameraPass.h>
#include <vtkRenderPassCollection.h>
#include <vtkSequencePass.h>
#include <vtkGaussianBlurPass.h>
#include <vtkCommand.h>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Testing class
#include "vtkKeyholePass.h"

class vtkWindowEventCallback : public vtkCommand{

public: 
	static vtkWindowEventCallback *New(){
		return new vtkWindowEventCallback;
	}
	vtkWindowEventCallback(){ this->size = 120;
								this->gamma = 5.0;
									x = y = 256;
									this->pinned = true;}

	virtual void Execute(vtkObject *caller, unsigned long eventid, void* callData){

		vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::SafeDownCast(caller);

		if( eventid == vtkCommand::MouseMoveEvent && !this->pinned){
			x = iren->GetEventPosition()[0];
			y = iren->GetEventPosition()[1];
		}
		if( eventid == vtkCommand::LeftButtonPressEvent )
			this->pinned = ( this->pinned == true )? false: true; 
		if( eventid == vtkCommand::MouseWheelForwardEvent )
			this->size += 5;
		if( eventid == vtkCommand::MouseWheelBackwardEvent )
			this->size -= 5;
		if( eventid == vtkCommand::RightButtonPressEvent )
			this->gamma += 0.5;
		if( eventid == vtkCommand::KeyPressEvent ){
			// Reset everything
			char *c = iren->GetKeySym();
			if( *c == 'r' ){
				this->size = 120;
				x = 256;
				y = 256;
				this->gamma = 5.0;
			}
		}

		// Set keyhole parameters. 
		keyholePass->SetKeyholeParameters(x, y, size, this->gamma);

		iren->GetRenderWindow()->Render();
		
	}

	vtkKeyholePass *keyholePass;

private:
	int size;
	double gamma;
	int x, y;

	bool pinned;
};
	

int main(){
	
	vtkSmartPointer< vtkSphereSource > sphere = vtkSmartPointer< 
													vtkSphereSource >::New();
	sphere->SetPhiResolution( 100 );
	sphere->SetThetaResolution( 100 );
	sphere->SetRadius( 5 );

	vtkSmartPointer< vtkPolyDataMapper > mapper = vtkSmartPointer< 
														vtkPolyDataMapper >::New();
	mapper->SetInputConnection( sphere->GetOutputPort() );

	vtkSmartPointer< vtkActor > actor = vtkSmartPointer<			
													vtkActor >::New();
	actor->SetMapper( mapper );
	actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
	actor->GetProperty()->SetOpacity(1);

	vtkSmartPointer< vtkRenderer > ren = vtkSmartPointer< 
											vtkRenderer >::New();
	vtkOpenGLRenderer *glRen = vtkOpenGLRenderer::SafeDownCast( ren );
	glRen->AddActor( actor );
	//glRen->SetBackground(0.0, 0.0, 255.0);

	vtkSmartPointer< vtkRenderWindow > renwin = vtkSmartPointer< 
													vtkRenderWindow >::New();

	vtkOpenGLRenderWindow *glRenWin = vtkOpenGLRenderWindow::SafeDownCast( renwin );
	glRenWin->AddRenderer( glRen );
	glRenWin->SetWindowName( "Test" );
	glRenWin->SetSize(512, 512);
	glRenWin->SetAlphaBitPlanes( 1 );

	vtkSmartPointer< vtkRenderWindowInteractor > iren = vtkSmartPointer< 
															vtkRenderWindowInteractor >::New();
	iren->SetRenderWindow( renwin );

	// Read the background picture and the mask and set it 
	cv::Mat background = cv::imread("./Data/container.jpg"); 
	cv::Mat mask = cv::imread("./Data/mask_2.png");

	uchar *backgroundData = new uchar[background.total()*4];
	cv::Mat background_RGBA(background.size(), CV_8UC4, backgroundData);
	cv::cvtColor(background, background_RGBA, CV_BGR2RGBA, 4);

	uchar *maskData = new uchar[mask.total()*4];
	cv::Mat mask_RGBA(mask.size(), CV_8UC4, maskData);
	cv::cvtColor(mask, mask_RGBA, cv::COLOR_BGR2RGBA, 4);

	cv::cvtColor( background, background, CV_RGBA2BGRA );
	cv::cvtColor( mask, mask, CV_RGBA2BGRA );
	// Flip the image to compensate for the difference in coordinate systems in VTK and OpenCV
	cv::flip( background, background, 0);
	cv::flip( mask, mask, 0 );

	vtkImageImport *backgroundImport = vtkImageImport::New();
	backgroundImport->SetDataOrigin( 0, 0, 0);
	backgroundImport->SetDataSpacing( 1, 1, 1);
	backgroundImport->SetWholeExtent(0, background.cols-1 ,0 , background.rows-1, 1, 1);
	backgroundImport->SetDataExtentToWholeExtent();
	backgroundImport->SetDataScalarTypeToUnsignedChar();
	backgroundImport->SetNumberOfScalarComponents( 4 );
	backgroundImport->SetImportVoidPointer( background_RGBA.data );
	backgroundImport->Update();

	vtkImageImport *maskImport = vtkImageImport::New();
	maskImport->SetDataOrigin( 0, 0, 0);
	maskImport->SetDataSpacing( 1, 1, 1);
	maskImport->SetWholeExtent(0, mask.cols-1 ,0 , mask.rows-1, 1, 1);
	maskImport->SetDataExtentToWholeExtent();
	maskImport->SetDataScalarTypeToUnsignedChar();
	maskImport->SetNumberOfScalarComponents( 4 );
	maskImport->SetImportVoidPointer( mask_RGBA.data );
	maskImport->Update();
	
	// Test code
	vtkSmartPointer< vtkKeyholePass > keyholePass = vtkSmartPointer<				
														vtkKeyholePass >::New();
	// Set background image
	keyholePass->SetBackgroundImage( backgroundImport->GetOutput() );
	// Set mask image
	keyholePass->SetMaskImage( maskImport->GetOutput() );

	// Set keyhole parameters
	keyholePass->SetKeyholeParameters(256, 256, 150, 5.0);
	keyholePass->SetHardKeyholeEdges( false );


	vtkLightsPass *lights = vtkLightsPass::New();
	vtkDefaultPass *defaultPass = vtkDefaultPass::New();
	vtkCameraPass *cameraP = vtkCameraPass::New();

	vtkRenderPassCollection *passes = vtkRenderPassCollection::New();
	passes->AddItem(lights);
	passes->AddItem(defaultPass);

	vtkSequencePass *seq = vtkSequencePass::New();
	seq->SetPasses( passes );

	cameraP->SetDelegatePass( seq );

	keyholePass->SetDelegatePass( cameraP );
	
	glRen->SetPass( keyholePass );

	vtkSmartPointer< vtkWindowEventCallback > call_back = vtkSmartPointer< vtkWindowEventCallback >::New();
	call_back->keyholePass = keyholePass;
	iren->AddObserver(vtkCommand::KeyPressEvent , call_back); 
	iren->AddObserver(vtkCommand::MouseWheelForwardEvent, call_back);
	iren->AddObserver(vtkCommand::MouseWheelBackwardEvent, call_back);
	iren->AddObserver(vtkCommand::MouseMoveEvent, call_back);
	iren->AddObserver( vtkCommand::LeftButtonPressEvent, call_back);
	iren->AddObserver( vtkCommand::RightButtonPressEvent, call_back);
		
	iren->Initialize();
	iren->Start();

	return 0;
}
