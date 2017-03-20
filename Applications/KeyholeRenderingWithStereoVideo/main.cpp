#include <iostream>

//OpenCV includes
#include <opencv2/opencv.hpp>

//vtk includes
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCamera.h>
#include <vtkImageImport.h>
#include <vtkDefaultPass.h>
#include <vtkCameraPass.h>
#include <vtkLightsPass.h>
#include <vtkSequencePass.h> 
#include <vtkRenderPassCollection.h>
#include <vtkOpenGLRenderer.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkTextureMapToPlane.h>
#include <vtkPlaneSource.h>
#include <vtkProperty.h>
#include <vtkOpenGLCamera.h>
#include <vtksys/CommandLineArguments.hxx>

#include "vtkKeyholePass.h"

//#define DEBUG_ON

class vtkWindowEventCallback : public vtkCommand
{
public:
	static vtkWindowEventCallback* New()
	{
		return new vtkWindowEventCallback;
	}
	vtkWindowEventCallback()
	{
		this->frame_idx = 0;
		this->size = 120;
		this->gamma = 5.0;
		x = y = 256;
		this->pinned = true;
	}

	virtual void Execute(vtkObject* caller, unsigned long eventid, void* callData)
	{
		vtkRenderWindowInteractor* renWindowInteractor = vtkRenderWindowInteractor::SafeDownCast(caller);

		if (eventid == vtkCommand::MouseMoveEvent && !this->pinned)
		{
			x = renWindowInteractor->GetEventPosition()[0];
			y = renWindowInteractor->GetEventPosition()[1];
		}
		if (eventid == vtkCommand::LeftButtonPressEvent)
		{
			this->pinned = (this->pinned == true) ? false : true;
		}
		if (eventid == vtkCommand::MouseWheelForwardEvent)
		{
			this->size += 5;
		}
		if (eventid == vtkCommand::MouseWheelBackwardEvent)
		{
			this->size -= 5;
		}
		if (eventid == vtkCommand::RightButtonPressEvent)
		{
			this->gamma += 0.5;
		}
		if (eventid == vtkCommand::KeyPressEvent)
		{
			// Reset everything
			char* c = renWindowInteractor->GetKeySym();
			if (*c == 'r')
			{
				this->size = 120;
				x = 256;
				y = 256;
				this->gamma = 5.0;
			}
		}

		/* Read video and update background texture */
		if (frame_idx == capture->get(CV_CAP_PROP_FRAME_COUNT))
		{
			frame_idx = 0;
			capture->set(CV_CAP_PROP_POS_FRAMES, frame_idx);
		}

		capture->read(*background);
		cv::flip(*background, *background, 0);
		cv::cvtColor(*background, *background_RGBA, CV_BGR2RGBA, 4);

		// split left and right images
		int width = background->cols/2;
		int height = background->rows;

		cv::Rect roi(0, 0, width, height);
		((*background_RGBA)(roi)).copyTo(*left_img);
		roi.x = width;
		((*background_RGBA)(roi)).copyTo(*right_img); 

		leftImageImport->SetImportVoidPointer(left_img->data);
		leftImageImport->Modified();

		rightImageImport->SetImportVoidPointer(right_img->data);
		rightImageImport->Modified();


		leftTexture->Modified();
		rightTexture->Modified();

		// Set keyhole parameters.
		keyholePass->SetLeftKeyholeParameters(x, y, size, this->gamma);
		keyholePass->SetRightKeyholeParameters(x+10, y, size, this->gamma);


		renWindowInteractor->GetRenderWindow()->Render();
		frame_idx++;
		
		
#ifdef DEBUG_ON
		cv::imshow("Input Left Image", *left_img);
		cv::imshow("Input Right Image", *right_img);
#endif
		

	}

	vtkImageImport *leftImageImport, *rightImageImport;
	vtkTexture *leftTexture, *rightTexture;
	vtkKeyholePass *keyholePass;
	cv::VideoCapture* capture;
	cv::Mat *background;
	cv::Mat *background_RGBA;
	cv::Mat *left_img, *right_img;
	int frame_idx;

private:
	int size;
	double gamma;
	int x, y;

	bool pinned;
};


 
int main(int argc, char** argv)
{

	std::string backgroundVideoFile;

	vtksys::CommandLineArguments args;
	args.Initialize(argc, argv);

	args.AddArgument("--background-image-file", vtksys::CommandLineArguments::EQUAL_ARGUMENT, &backgroundVideoFile, "The background picture filename");

	// Input arguments error checking
	if (!args.Parse())
	{
		std::cerr << "Problem parsing arguments" << std::endl;
		std::cout << "Help: " << args.GetHelp() << std::endl;
		exit(EXIT_FAILURE);
	}
	if (backgroundVideoFile.empty())
	{
		backgroundVideoFile = DEFAULT_BACKGROUND_FILE;
	}

	cv::VideoCapture capture = cv::VideoCapture(backgroundVideoFile);
	int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	uchar* backgroundData = new uchar[width*height*sizeof(unsigned char)* 4];
	cv::Mat background_RGBA(cv::Size(width, height), CV_8UC4, backgroundData);
	cv::Mat background = cv::Mat(height, width, CV_8UC3);
	cv::Mat left_img = cv::Mat(height, width/2, CV_8UC4);
	cv::Mat right_img = cv::Mat(height, width/2, CV_8UC4);

	vtkSmartPointer<vtkImageImport> leftImageImport = vtkSmartPointer<vtkImageImport>::New();
	leftImageImport->SetDataOrigin(0, 0, 0);
	leftImageImport->SetDataSpacing(1, 1, 1);
	leftImageImport->SetWholeExtent(0, width / 2 - 1, 0, height - 1, 1, 1);
	leftImageImport->SetDataExtentToWholeExtent();
	leftImageImport->SetDataScalarTypeToUnsignedChar();
	leftImageImport->SetNumberOfScalarComponents(4);
	leftImageImport->SetImportVoidPointer(left_img.data);
	leftImageImport->Update();

	vtkSmartPointer<vtkImageImport> rightImageImport = vtkSmartPointer<vtkImageImport>::New();
	rightImageImport->SetDataOrigin(0, 0, 0);
	rightImageImport->SetDataSpacing(1, 1, 1);
	rightImageImport->SetWholeExtent(0, width / 2 - 1, 0, height - 1, 1, 1);
	rightImageImport->SetDataExtentToWholeExtent();
	rightImageImport->SetDataScalarTypeToUnsignedChar();
	rightImageImport->SetNumberOfScalarComponents(4);
	rightImageImport->SetImportVoidPointer(right_img.data);
	rightImageImport->Update();

	vtkSmartPointer<vtkTexture> leftImageTex = vtkSmartPointer<vtkTexture>::New();
	leftImageTex->SetInputConnection(leftImageImport->GetOutputPort());
	vtkSmartPointer<vtkTexture> rightImageTex = vtkSmartPointer<vtkTexture>::New();
	rightImageTex->SetInputConnection(rightImageImport->GetOutputPort());
	

	vtkSmartPointer<vtkPlaneSource> leftImagePlane = vtkSmartPointer<vtkPlaneSource>::New();
	leftImagePlane->SetCenter(0.0, 0.0, 0.0);
	leftImagePlane->SetNormal(0.0, 0.0, 1.0);
	vtkSmartPointer<vtkPlaneSource> rightImagePlane = vtkSmartPointer<vtkPlaneSource>::New();
	rightImagePlane->SetCenter(0.0, 0.0, 0.0);
	rightImagePlane->SetNormal(0.0, 0.0, 1.0);

	vtkSmartPointer<vtkTextureMapToPlane> leftImageTexturePlane = vtkSmartPointer<vtkTextureMapToPlane>::New();
	leftImageTexturePlane->SetInputConnection(leftImagePlane->GetOutputPort());
	vtkSmartPointer<vtkTextureMapToPlane> rightImageTexturePlane = vtkSmartPointer<vtkTextureMapToPlane>::New();
	rightImageTexturePlane->SetInputConnection(rightImagePlane->GetOutputPort());

	vtkSmartPointer<vtkPolyDataMapper> leftPlaneMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	leftPlaneMapper->SetInputConnection(leftImageTexturePlane->GetOutputPort());
	vtkSmartPointer<vtkPolyDataMapper> rightPlaneMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	rightPlaneMapper->SetInputConnection(rightImageTexturePlane->GetOutputPort());

	vtkSmartPointer<vtkActor> leftTexturedPlane = vtkSmartPointer<vtkActor>::New();
	leftTexturedPlane->SetMapper(leftPlaneMapper);
	leftTexturedPlane->SetTexture(leftImageTex);
	leftTexturedPlane->GetProperty()->SetOpacity(0.0);

	vtkSmartPointer<vtkActor> rightTexturedPlane = vtkSmartPointer<vtkActor>::New();
	rightTexturedPlane->SetMapper(rightPlaneMapper);
	rightTexturedPlane->SetTexture(rightImageTex);
	rightTexturedPlane->GetProperty()->SetOpacity(0.0);

	vtkSmartPointer< vtkRenderer > ren = vtkSmartPointer< vtkRenderer >::New();
	vtkOpenGLRenderer* glRenderer = vtkOpenGLRenderer::SafeDownCast(ren);


	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	vtkOpenGLRenderWindow* glRenWin = vtkOpenGLRenderWindow::SafeDownCast(renderWindow);
	glRenWin->AddRenderer(glRenderer);
	glRenWin->SetWindowName("Stereo_Rendering_Example");
	glRenWin->SetSize(width / 2, height);
	glRenWin->SetAlphaBitPlanes(1);
	glRenWin->StereoCapableWindowOn();
	glRenWin->StereoRenderOn();
	glRenWin->SetStereoTypeToDresden();

	// Setup camera
	vtkOpenGLCamera *cam = vtkOpenGLCamera::New();
	glRenderer->SetActiveCamera(cam);

	// Create an actor
	vtkSmartPointer< vtkSphereSource > sphere = vtkSmartPointer< vtkSphereSource >::New();
	sphere->SetPhiResolution(50);
	sphere->SetThetaResolution(50);
	sphere->SetRadius(10);

	vtkSmartPointer< vtkPolyDataMapper > sphereMapper = vtkSmartPointer< vtkPolyDataMapper >::New();
	sphereMapper->SetInputConnection(sphere->GetOutputPort());

	vtkSmartPointer< vtkActor > sphereActor = vtkSmartPointer< vtkActor >::New();
	sphereActor->SetMapper(sphereMapper);
	sphereActor->SetPosition(0, 0, -200); // Place the actor at a particular distance from the origin
	sphereActor->GetProperty()->SetColor(1, 0, 0);

	glRenderer->AddActor(sphereActor);

	// Add background texture last!
	glRenderer->AddViewProp(leftTexturedPlane);
	glRenderer->AddViewProp(rightTexturedPlane);

	vtkSmartPointer<vtkRenderWindowInteractor> renWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renWindowInteractor->SetRenderWindow(glRenWin);


	// Render Passes
	vtkSmartPointer<vtkKeyholePass> keyholePass = vtkSmartPointer<vtkKeyholePass>::New();

	// Set keyhole parameters
	keyholePass->SetLeftKeyholeParameters(320, 240, 150, 5.0);
	keyholePass->SetRightKeyholeParameters(320+10, 240, 150, 5.0);
	keyholePass->SetHardKeyholeEdges(false);

	vtkSmartPointer<vtkLightsPass> lightsPass = vtkSmartPointer<vtkLightsPass>::New();
	vtkSmartPointer<vtkDefaultPass> defaultPass = vtkSmartPointer<vtkDefaultPass>::New();
	vtkSmartPointer<vtkCameraPass> cameraPass = vtkSmartPointer<vtkCameraPass>::New();

	vtkSmartPointer<vtkRenderPassCollection> passes = vtkSmartPointer<vtkRenderPassCollection>::New();
	passes->AddItem(lightsPass);
	passes->AddItem(defaultPass);

	vtkSmartPointer<vtkSequencePass> sequencePass = vtkSmartPointer<vtkSequencePass>::New();
	sequencePass->SetPasses(passes);

	cameraPass->SetDelegatePass(sequencePass);
	keyholePass->SetDelegatePass(cameraPass);

	glRenderer->SetPass(keyholePass);


	vtkSmartPointer<vtkWindowEventCallback> call_back = vtkSmartPointer<vtkWindowEventCallback>::New();
	call_back->capture = &capture;
	call_back->background = &background;
	call_back->background_RGBA = &background_RGBA;
	call_back->leftImageImport = leftImageImport;
	call_back->rightImageImport = rightImageImport;
	call_back->left_img = &left_img;
	call_back->right_img = &right_img;
	call_back->leftTexture = leftImageTex;
	call_back->rightTexture = rightImageTex;
	call_back->keyholePass = keyholePass;


	renWindowInteractor->AddObserver(vtkCommand::TimerEvent, call_back);
	renWindowInteractor->AddObserver(vtkCommand::KeyPressEvent, call_back);
	renWindowInteractor->AddObserver(vtkCommand::MouseWheelForwardEvent, call_back);
	renWindowInteractor->AddObserver(vtkCommand::MouseWheelBackwardEvent, call_back);
	renWindowInteractor->AddObserver(vtkCommand::MouseMoveEvent, call_back);
	renWindowInteractor->AddObserver(vtkCommand::LeftButtonPressEvent, call_back);
	renWindowInteractor->AddObserver(vtkCommand::RightButtonPressEvent, call_back);
	renWindowInteractor->Initialize();

	int interactorTimerID = renWindowInteractor->CreateRepeatingTimer(1000.0 / 30.0);

	renWindowInteractor->Start();
 
	return 0;
}