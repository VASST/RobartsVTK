//For video capture
#include "vtkDirectShowVideoSource.h"
#include "vtkVuzixARScene.h"

//general use
#include "vtkImageData.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkTexture.h"


// ---------------------------------------------------------------------------------------------------
// Main Program
int main(int argc, char** argv){

	//set up the video source for capturing
	vtkDirectShowVideoSource* videoSourceLeft = vtkDirectShowVideoSource::New();
	videoSourceLeft->SetVideoSourceNumber(0);
	videoSourceLeft->SetFrameRate(20);
	videoSourceLeft->SetFrameBufferSize(10);
	videoSourceLeft->Initialize();
	videoSourceLeft->Record();
	vtkDirectShowVideoSource* videoSourceRight = vtkDirectShowVideoSource::New();
	videoSourceRight->SetVideoSourceNumber(1);
	videoSourceRight->SetFrameRate(20);
	videoSourceRight->SetFrameBufferSize(10);
	videoSourceRight->Initialize();
	videoSourceRight->Record();

	//set up a texture to display in the renderer
	vtkVuzixARScene* arScene = vtkVuzixARScene::New();
	arScene->SetLeftEyeSource( videoSourceLeft->GetOutput() );
	arScene->SetRightEyeSource( videoSourceRight->GetOutput() );
	vtkRenderer* renderLeft = arScene->GetLeftEyeView();
	vtkRenderer* renderRight = arScene->GetRightEyeView();
	renderLeft->SetViewport(0,0,0.5,1);
	renderRight->SetViewport(0.5,0,1,1);

	//set up the remainder of the VTK pipeline
	vtkRenderWindow* window = vtkRenderWindow::New();
	window->AddRenderer(renderLeft);
	window->AddRenderer(renderRight);
	window->Render();

	//apply keyhole planes widget and interactor
	vtkRenderWindowInteractor* interactor = vtkRenderWindowInteractor::New();
	interactor->SetRenderWindow( window );

	//start the process
	interactor->Initialize();
	interactor->Start();
	videoSourceLeft->Stop();
	videoSourceRight->Stop();
	videoSourceLeft->Play();
	videoSourceRight->Play();
	interactor->Initialize();
	interactor->Start();

	//clean up pipeline
	interactor->Delete();
	window->Delete();
	arScene->Delete();
	videoSourceLeft->Delete();
	videoSourceRight->Delete();

}

//#include "vtkTexture.h"
//#include "vtkOpenCVVideoSource.h"
//#include "vtkRenderer.h"
//#include "vtkRenderWindow.h"
//#include "vtkRenderWindowInteractor.h"
//
//#include <iostream>
//using namespace std;
//
//int main ()
//{
//	//creats a new inputfeed object
//	vtkOpenCVVideoSource *InputFeed = vtkOpenCVVideoSource::New();
//	vtkTexture *texture = vtkTexture::New();//creats texture, which works as a background
//	vtkRenderer *ren1 = vtkRenderer::New();//use to control what happens in the window
//	vtkRenderWindow *renwin = vtkRenderWindow::New();//the window itself
//	vtkRenderWindowInteractor *interact=vtkRenderWindowInteractor::New();//an interactor for the mouse
//
//	InputFeed->SetVideoSourceNumber(0);//Sets the video input to the first thing it sees
//	InputFeed->Initialize();
//	InputFeed->Record();
//
//	texture->SetInput(InputFeed->GetOutput());
//	ren1->SetBackgroundTexture(texture);
//	ren1->SetTexturedBackground(true);
//	
//	renwin->AddRenderer(ren1);
//	renwin->SetInteractor(interact);
//	renwin->Render();
//	interact->Start();
//
//	InputFeed->Stop();
//
//} 