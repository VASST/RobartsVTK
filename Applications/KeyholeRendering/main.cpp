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

    // Set keyhole parameters.
    keyholePass->SetKeyholeParameters(x, y, size, this->gamma);

    renWindowInteractor->GetRenderWindow()->Render();
  }

  vtkKeyholePass* keyholePass;

private:
  int size;
  double gamma;
  int x, y;

  bool pinned;
};

int main(int argc, char** argv)
{
  std::string backgroundImageFile;
  std::string maskImageFile;

  vtksys::CommandLineArguments args;
  args.Initialize(argc, argv);

  args.AddArgument("--background-image-file", vtksys::CommandLineArguments::EQUAL_ARGUMENT, &backgroundImageFile, "The background picture filename");
  args.AddArgument("--mask-image-file", vtksys::CommandLineArguments::EQUAL_ARGUMENT, &maskImageFile, "The mask picture filename");

  // Input arguments error checking
  if (!args.Parse())
  {
    std::cerr << "Problem parsing arguments" << std::endl;
    std::cout << "Help: " << args.GetHelp() << std::endl;
    exit(EXIT_FAILURE);
  }
  if (backgroundImageFile.empty())
  {
    backgroundImageFile = DEFAULT_BACKGROUND_FILE;
  }
  if (maskImageFile.empty())
  {
    maskImageFile = DEFAULT_MASK_FILE;
  }

  vtkSmartPointer<vtkSphereSource> sphere = vtkSmartPointer<vtkSphereSource>::New();
  sphere->SetPhiResolution(100);
  sphere->SetThetaResolution(100);
  sphere->SetRadius(5);

  vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(sphere->GetOutputPort());

  vtkSmartPointer<vtkActor> actor = vtkSmartPointer <vtkActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(1.0, 0.0, 0.0);
  actor->GetProperty()->SetOpacity(1);

  vtkSmartPointer<vtkRenderer> ren = vtkSmartPointer<vtkRenderer>::New();
  vtkOpenGLRenderer* glRenderer = vtkOpenGLRenderer::SafeDownCast(ren);
  glRenderer->AddActor(actor);

  vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();

  vtkOpenGLRenderWindow* glRenWin = vtkOpenGLRenderWindow::SafeDownCast(renderWindow);
  glRenWin->AddRenderer(glRenderer);
  glRenWin->SetWindowName("Keyhole_Rendering_Example");
  glRenWin->SetSize(512, 512);
  glRenWin->SetAlphaBitPlanes(1);
  glRenWin->StereoCapableWindowOn();
  glRenWin->SetStereoTypeToSplitViewportHorizontal();

  vtkSmartPointer<vtkRenderWindowInteractor> renWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renWindowInteractor->SetRenderWindow(renderWindow);

  // Read the background picture and the mask and set it .
  cv::Mat background = cv::imread(backgroundImageFile);
  cv::Mat mask = cv::imread(maskImageFile);

  uchar* backgroundData = new uchar[background.total() * 4];
  cv::Mat background_RGBA(background.size(), CV_8UC4, backgroundData);
  cv::cvtColor(background, background_RGBA, CV_BGR2RGBA, 4);

  uchar* maskData = new uchar[mask.total() * 4];
  cv::Mat mask_RGBA(mask.size(), CV_8UC4, maskData);
  cv::cvtColor(mask, mask_RGBA, cv::COLOR_BGR2RGBA, 4);

  vtkSmartPointer<vtkImageImport> backgroundImport = vtkSmartPointer<vtkImageImport>::New();
  backgroundImport->SetDataOrigin(0, 0, 0);
  backgroundImport->SetDataSpacing(1, 1, 1);
  backgroundImport->SetWholeExtent(0, background.cols - 1 , 0 , background.rows - 1, 1, 1);
  backgroundImport->SetDataExtentToWholeExtent();
  backgroundImport->SetDataScalarTypeToUnsignedChar();
  backgroundImport->SetNumberOfScalarComponents(4);
  backgroundImport->SetImportVoidPointer(background_RGBA.data);
  backgroundImport->Update();

  vtkSmartPointer<vtkImageImport> maskImport = vtkSmartPointer<vtkImageImport>::New();
  maskImport->SetDataOrigin(0, 0, 0);
  maskImport->SetDataSpacing(1, 1, 1);
  maskImport->SetWholeExtent(0, mask.cols - 1 , 0 , mask.rows - 1, 1, 1);
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

  glRenderer->AddViewProp(texturedPlane);

  // Test code
  vtkSmartPointer<vtkKeyholePass> keyholePass = vtkSmartPointer<vtkKeyholePass>::New();

  // Set keyhole parameters
  keyholePass->SetKeyholeParameters(256, 256, 150, 5.0);
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
  call_back->keyholePass = keyholePass;
  renWindowInteractor->AddObserver(vtkCommand::KeyPressEvent , call_back);
  renWindowInteractor->AddObserver(vtkCommand::MouseWheelForwardEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::MouseWheelBackwardEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::MouseMoveEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::LeftButtonPressEvent, call_back);
  renWindowInteractor->AddObserver(vtkCommand::RightButtonPressEvent, call_back);

  renWindowInteractor->Initialize();
  renWindowInteractor->Start();

  return 0;
}