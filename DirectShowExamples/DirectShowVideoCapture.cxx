//For video capture
#include "vtkDirectShowVideoSource.h"
#include "vtkVideoSource2.h"

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
	vtkDirectShowVideoSource* videoSource = vtkDirectShowVideoSource::New();
	for( int i = 0; i < videoSource->GetNumberOfDevices(); i++ ){
		std::cout << videoSource->GetDeviceName(i) << std::endl;
	}
	videoSource->SetVideoSourceNumber(2);
	videoSource->SetFrameRate(20);
	videoSource->SetFrameBufferSize(100);
	videoSource->Initialize();
	videoSource->Record();

	//set up a texture to display in the renderer
	vtkTexture* background = vtkTexture::New();
	background->SetInput( videoSource->GetOutput() );

	//set up the remainder of the VTK pipeline
	vtkRenderer* renderer = vtkRenderer::New();
	renderer->ResetCamera();
	renderer->SetBackgroundTexture(background);
	renderer->SetTexturedBackground(true);
	vtkRenderWindow* window = vtkRenderWindow::New();
	window->AddRenderer( renderer );
	window->Render();

	//apply keyhole planes widget and interactor
	vtkRenderWindowInteractor* interactor = vtkRenderWindowInteractor::New();
	interactor->SetRenderWindow( window );

	//start the process
	interactor->Initialize();
	interactor->Start();
	videoSource->Stop();
	videoSource->Play();
	interactor->Initialize();
	interactor->Start();

	//clean up pipeline
	interactor->Delete();
	window->Delete();
	renderer->Delete();
	background->Delete();
	videoSource->Delete();

}