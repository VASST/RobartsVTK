/*==========================================================================

Copyright (c) 2016 Uditha L. Jayarathne, ujayarat@robarts.ca

Use, modification and redistribution of the software, in source or
binary forms, are permitted provided that the following terms and
conditions are met:

1) Redistribution of the source code, in verbatim or modified
form, must retain the above copyright notice, this license,
the following disclaimer, and any notices that refer to this
license and/or the following disclaimer.

2) Redistribution in binary form must include the above copyright
notice, a copy of this license and the following disclaimer
in the documentation or with other materials provided with the
distribution.

3) Modified copies of the source code must be clearly marked as such,
and must not be misrepresented as verbatim copies of the source code.

THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.
=========================================================================*/

// STL includes
#include <iostream>

// VTK includes
#include <vtkActor.h>
#include <vtkCameraPass.h>
#include <vtkCommand.h>
#include <vtkDefaultPass.h>
#include <vtkImageImport.h>
#include <vtkLightsPass.h>
#include <vtkOpenGLProperty.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLRenderer.h>
#include <vtkPlaneSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderPass.h>
#include <vtkRenderPassCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSequencePass.h>
#include <vtkSmartPointer.h>
#include <vtkSphereSource.h>
#include <vtkTexture.h>
#include <vtkTextureMapToPlane.h>
#include <vtkCamera.h>
#include <vtkOpenGLCamera.h>
#include <vtkMatrix4x4.h>
#include <vtksys/CommandLineArguments.hxx>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Testing class
#include "vtkKeyholePass.h"

class vtkWindowEventCallback : public vtkCommand
{
public:
  static vtkWindowEventCallback* New()
  {
    return new vtkWindowEventCallback;
  }
  vtkWindowEventCallback()
  {
    this->size = 250;
    this->gamma = 5.0;
    x = 320;
    y = 240;
    this->pinned = true;

    double m[16] = { 1, 0, 0, 10,
                     0, 1, 0, 10,
                     0, 0, 1, 200,
                     0, 0, 0, 1
                   };
    mat = vtkMatrix4x4::New();
    mat->DeepCopy(m);
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
        this->size = 250;
        x = 256;
        y = 256;
        this->gamma = 5.0;
      }
    }

    /* Read video and update background texture */
    // TODO: texture update code goes here
    capture->read(*background);
    cv::flip(*background, *background, 0);
    cv::cvtColor(*background, *background_RGBA, CV_BGR2RGBA, 4);


    imgImport->SetImportVoidPointer((*background_RGBA).data);
    imgImport->Modified();

    texture->Modified();
    actor->Modified();

    sphere1->SetUserMatrix(mat);
    sphere2->SetUserMatrix(mat);
    sphere3->SetUserMatrix(mat);

    // Set keyhole parameters.
    keyholePass->SetLeftKeyholeParameters(x, y, size, this->gamma);

    renWindowInteractor->GetRenderWindow()->Render();
  }

  vtkKeyholePass* keyholePass;
  vtkImageImport* imgImport;
  vtkTexture* texture;
  cv::VideoCapture* capture;
  vtkMatrix4x4* mat;
  cv::Mat* background;
  cv::Mat* background_RGBA;
  vtkActor* actor, *sphere1, *sphere2, *sphere3;

private:
  int size;
  double gamma;
  int x, y;

  bool pinned;
};

int main(int argc, char** argv)
{
  std::string backgroundVideoFile;
  std::string maskImageFile;

  vtksys::CommandLineArguments args;
  args.Initialize(argc, argv);

  args.AddArgument("--background-image-file", vtksys::CommandLineArguments::EQUAL_ARGUMENT, &backgroundVideoFile, "The background picture filename");
  args.AddArgument("--mask-image-file", vtksys::CommandLineArguments::EQUAL_ARGUMENT, &maskImageFile, "The mask picture filename");

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
  if (maskImageFile.empty())
  {
    maskImageFile = DEFAULT_MASK_FILE;
  }

  vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
  sphere->SetPhiResolution(100);
  sphere->SetThetaResolution(100);
  sphere->SetRadius(5);
  sphere->SetCenter(0, 0, 0);

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(sphere->GetOutputPort());

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer <vtkActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
  actor->GetProperty()->SetOpacity(1);

  // Second actor
  vtkSmartPointer< vtkSphereSource > sphere2 = vtkSmartPointer< vtkSphereSource >::New();
  sphere2->SetPhiResolution(100);
  sphere2->SetThetaResolution(100);
  sphere2->SetRadius(5);
  sphere2->SetCenter(10, 0, 0);

  vtkSmartPointer< vtkPolyDataMapper > mapper2 = vtkSmartPointer< vtkPolyDataMapper>::New();
  mapper2->SetInputConnection(sphere2->GetOutputPort());

  vtkSmartPointer< vtkActor > actor2 = vtkSmartPointer< vtkActor >::New();
  actor2->SetMapper(mapper2);
  actor2->GetProperty()->SetColor(0.0, 1.0, 0.0);
  actor2->GetProperty()->SetOpacity(1);

  // 3rd Actor
  vtkSmartPointer< vtkSphereSource > sphere3 = vtkSmartPointer< vtkSphereSource >::New();
  sphere3->SetPhiResolution(100);
  sphere3->SetThetaResolution(100);
  sphere3->SetRadius(5);
  sphere3->SetCenter(5, sqrt(75), 0);

  vtkSmartPointer< vtkPolyDataMapper > mapper3 = vtkSmartPointer< vtkPolyDataMapper>::New();
  mapper3->SetInputConnection(sphere3->GetOutputPort());

  vtkSmartPointer< vtkActor > actor3 = vtkSmartPointer< vtkActor >::New();
  actor3->SetMapper(mapper3);
  actor3->GetProperty()->SetColor(0.0, 0.0, 1.0);
  actor3->GetProperty()->SetOpacity(1);

  // Read the background picture and the mask and set it .
  cv::VideoCapture capture = cv::VideoCapture(backgroundVideoFile);
  cv::Mat mask = cv::imread(maskImageFile);
  int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
  int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

  uchar* backgroundData = new uchar[width * height * sizeof(unsigned char) * 4];
  cv::Mat background_RGBA(cv::Size(width, height), CV_8UC4, backgroundData);
  cv::Mat background = cv::Mat(width, height, CV_8UC3);

  uchar* maskData = new uchar[mask.total() * 4];
  cv::Mat mask_RGBA(mask.size(), CV_8UC4, maskData);
  cv::cvtColor(mask, mask_RGBA, cv::COLOR_BGR2RGBA, 4);

  vtkSmartPointer<vtkImageImport> backgroundImport = vtkSmartPointer<vtkImageImport>::New();
  backgroundImport->SetDataOrigin(0, 0, 0);
  backgroundImport->SetDataSpacing(1, 1, 1);
  backgroundImport->SetWholeExtent(0, width - 1, 0, height - 1, 1, 1);
  backgroundImport->SetDataExtentToWholeExtent();
  backgroundImport->SetDataScalarTypeToUnsignedChar();
  backgroundImport->SetNumberOfScalarComponents(4);
  backgroundImport->SetImportVoidPointer(background_RGBA.data);
  backgroundImport->Update();

  vtkSmartPointer<vtkImageImport> maskImport = vtkSmartPointer<vtkImageImport>::New();
  maskImport->SetDataOrigin(0, 0, 0);
  maskImport->SetDataSpacing(1, 1, 1);
  maskImport->SetWholeExtent(0, mask.cols - 1, 0, mask.rows - 1, 1, 1);
  maskImport->SetDataExtentToWholeExtent();
  maskImport->SetDataScalarTypeToUnsignedChar();
  maskImport->SetNumberOfScalarComponents(4);
  maskImport->SetImportVoidPointer(mask_RGBA.data);
  maskImport->Update();

  vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
  plane->SetCenter(0.0, 0.0, 0.0);
  plane->SetNormal(0.0, 0.0, 1.0);

  // Apply Texture
  vtkSmartPointer<vtkTexture> forgroundTex = vtkSmartPointer<vtkTexture>::New();
  forgroundTex->SetInputConnection(backgroundImport->GetOutputPort());

  vtkSmartPointer<vtkTextureMapToPlane> texturePlane = vtkSmartPointer<vtkTextureMapToPlane>::New();
  texturePlane->SetInputConnection(plane->GetOutputPort());

  vtkSmartPointer<vtkPolyDataMapper> planeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  planeMapper->SetInputConnection(texturePlane->GetOutputPort());

  vtkSmartPointer<vtkActor> texturedPlane = vtkSmartPointer<vtkActor>::New();
  texturedPlane->SetMapper(planeMapper);
  texturedPlane->SetTexture(forgroundTex);
  texturedPlane->SetVisibility(0);

  vtkSmartPointer<vtkRenderer> ren = vtkSmartPointer<vtkRenderer>::New();
  ren->AddActor(actor);
  ren->AddActor(actor2);
  ren->AddActor(actor3);

  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(ren);
  renderWindow->SetWindowName("Keyhole_Rendering_Example");
  renderWindow->SetSize(width, height);
  renderWindow->SetAlphaBitPlanes(1);

  vtkSmartPointer<vtkRenderWindowInteractor> renWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renWindowInteractor->SetRenderWindow(renderWindow);

  // Add texturedPlane as an actor
  ren->AddViewProp(texturedPlane);

  // Setup camera
  double viewAngle = 2 * atan((480 / 2.0) / 775) * 180 / (4 * atan(1.0));
  double center_x = (640 - 320) / ((640 - 1) / 2.0) - 1;
  double center_y = 240 / ((480 - 1) / 2.0) - 1;
  vtkOpenGLCamera* cam = vtkOpenGLCamera::SafeDownCast(ren->GetActiveCamera());
  cam->SetViewAngle(viewAngle);
  cam->SetPosition(0, 0, 0);
  cam->SetViewUp(0, -1, 0);
  cam->SetFocalPoint(0, 0, 775);
  cam->SetWindowCenter(center_x, center_y);
  cam->SetClippingRange(0.01, 1000.01);


  vtkSmartPointer<vtkKeyholePass> keyholePass = vtkSmartPointer<vtkKeyholePass>::New();

  // Set keyhole parameters
  keyholePass->SetLeftKeyholeParameters(320, 240, 250, 0.02);
  keyholePass->SetHardKeyholeEdges(false);
  keyholePass->SetBackgroundColor(0, 0, 0);
  keyholePass->SetD1(10);
  keyholePass->SetVisualizationMode(vtkKeyholePass::MODE_NO_KEYHOLE);

  // Set render passes.
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

  ren->SetPass(keyholePass);

  vtkSmartPointer<vtkWindowEventCallback> call_back = vtkSmartPointer<vtkWindowEventCallback>::New();
  call_back->keyholePass = keyholePass;
  call_back->capture = &capture;
  call_back->background = &background;
  call_back->background_RGBA = &background_RGBA;
  call_back->imgImport = backgroundImport;
  call_back->texture = forgroundTex;
  call_back->actor = texturedPlane;
  call_back->sphere1 = actor;
  call_back->sphere2 = actor2;
  call_back->sphere3 = actor3;

  renWindowInteractor->AddObserver(vtkCommand::KeyPressEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::MouseWheelForwardEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::MouseWheelBackwardEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::MouseMoveEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::LeftButtonPressEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::RightButtonPressEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::TimerEvent, call_back);

  renWindowInteractor->Initialize();

  int interactorTimerID = renWindowInteractor->CreateRepeatingTimer(1000.0 / 30.0);

  renWindowInteractor->Start();

  return 0;
}