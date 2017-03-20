/*==========================================================================

  Copyright (c) 2015 Uditha L. Jayarathne, ujayarat@robarts.ca

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

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

// Qt includes
#include <QMainWindow>
#include <QLabel>
#include <QSignalMapper>
#include <QMenuBar>

// VTK includes
#include <QVTKWidget.h>
#include <vtkBoxRepresentation.h>
#include <vtkBoxWidget.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkColorTransferFunction.h>
#include <vtkCommand.h>
#include <vtkImageActor.h>
#include <vtkImageCanvasSource2D.h>
#include <vtkImageClip.h>
#include <vtkImageFlip.h>
#include <vtkImageImport.h>
#include <vtkImageMapper3D.h>
#include <vtkImageViewer2.h>
#include <vtkInteractorStyleImage.h>
#include <vtkMatrix4x4.h>
#include <vtkMetaImageWriter.h>
#include <vtkPNGWriter.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPlanes.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTransform.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkWindowToImageFilter.h>
#include <vtkXMLUtilities.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLProperty.h>
#include <vtkOpenGLRenderer.h>
#include <vtkDefaultPass.h>
#include <vtkRenderPass.h>
#include <vtkLightsPass.h>
#include <vtkCameraPass.h>
#include <vtkRenderPassCollection.h>
#include <vtkSequencePass.h>
#include <vtkTexture.h>
#include <vtkPolyDataMapper.h>
#include <vtkTextureMapToPlane.h>
#include <vtkPlaneSource.h>

// Testing includes
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkImageExtractComponents.h>

// PLUS includes
#include <vtkPlusChannel.h>
#include <vtkPlusDataSource.h>
#include <vtkPlusDevice.h>
#include <PlusTrackedFrame.h>
#include <vtkPlusTrackedFrameList.h>
#include <vtkPlusSavedDataSource.h>
#include <PlusConfigure.h>
#include <vtkPlusVolumeReconstructor.h>
#include <vtkPlusDataCollector.h>
#include <vtkPlusSequenceIO.h>
#include <vtkPlusTransformRepository.h>

// RobartsVTK includes
#include "qTransferFunctionDefinitionWidget.h"
#include "qTransferFunctionWindowWidget.h"
#include "vtkCuda1DVolumeMapper.h"
#include "vtkCuda2DInExLogicVolumeMapper.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCuda2DVolumeMapper.h"
#include "vtkCudaFunctionPolygonReader.h"
#include "vtkKeyholePass.h"

// OpenCV includes
#include <opencv2/videoio.hpp>

#include "vtkCLVolumeReconstruction.h"

#include <iostream>
#include <chrono> // For timing

//# define D_TIMING
//# define ALIGNMENT_DEBUG

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

    vtkRenderWindowInteractor* iren = vtkRenderWindowInteractor::SafeDownCast(caller);

    if (eventid == vtkCommand::MouseMoveEvent && !this->pinned)
    {
      x = iren->GetEventPosition()[0];
      y = iren->GetEventPosition()[1];
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
      char* c = iren->GetKeySym();
      if (*c == 'r')
      {
        this->size = 120;
        x = 256;
        y = 256;
        this->gamma = 5.0;
      }
    }

    // Set keyhole parameters.
    keyholePass->SetLeftKeyholeParameters(x, y, size, this->gamma);

    iren->GetRenderWindow()->Render();

  }

  vtkKeyholePass* keyholePass;

private:
  int size;
  double gamma;
  int x, y;

  bool pinned;
};


class vtkUSEventCallback : public vtkCommand
{
public:
  static vtkUSEventCallback* New()
  {
    return new vtkUSEventCallback;
  }

  virtual void Execute(vtkObject* caller, unsigned long, void*);

  vtkPlusTrackedFrameList* PlusTrackedFrames;
  vtkPlusTransformRepository* repository;
  vtkPlusVolumeReconstructor* reconstructor;
  vtkCLVolumeReconstruction* accRecon;
  vtkImageData* usVolume;
  vtkSmartVolumeMapper* volMapper;
  vtkCuda1DVolumeMapper* cudaMapper;
  vtkCuda2DVolumeMapper* cudaMapper2;
  vtkImageViewer2* Viewer;
  vtkImageFlip* imgFlip;
  vtkRenderWindowInteractor* Iren;
  PlusTransformName TransformName;
  vtkImageData* ImageData;
  cv::VideoCapture* _camCapture;
  int n_frames;
  vtkImageImport* _imgImport;
  vtkRenderWindow* _camRenWin;
  vtkRenderWindow* _volRenWin;
  vtkVolume* _vol;
  vtkRenderWindow* _augmentedRenWin;
  vtkRenderer* _volRenderer;
  vtkTexture* _camImgTexture;
  vtkTransform* _boxTransform;
  vtkTransform* _transform;
  vtkBoxWidget* _boxWidget;
  vtkPlanes* _boxPlanes;
  vtkCuda2DInExLogicVolumeMapper* _inExMapper;
  QVTKWidget* _screen;
  QLabel* Info;
  vtkWindowToImageFilter* _win2Img;
  vtkPNGWriter* _imgWriter;
  vtkKeyholePass* _keyholePass;
  std::string current_mapper;
  bool sc_capture_on;
  unsigned int index;
};

namespace Ui
{
  class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  explicit MainWindow(QWidget* parent = 0);
  ~MainWindow();

public slots:
  void onStartButtonClick(const QString&);
  void onScanTypeRadioButtonClick(const QString&);
  void onSaveVolumeButtonClick(const QString&);
  void ontf1ButtonClick(const QString&);
  void ontf2ButtonClick(const QString&);
  void ontfInExButtonClick(const QString&);
  void onScCaptureRadioButtonClick(const QString&);

protected:
  /* Initialize PLUS pipeline */
  int InitPLUSPipeline();

  /* Initialize VTK pipeline */
  int InitVTKPipeline();

  /* Initialize OpenCV variables */
  int InitCVPipeline();

  /* Initialize PLUS-bypass pipeline */
  int InitPLUSBypassPipeline();

  /* Setup VTK Camera from intrinsics */
  void SetupVTKCamera(cv::Mat, double, double, vtkCamera*);
  void SetupVTKCamera(cv::Mat, vtkCamera*);

  /* Setup Volume Rendering Pipeline */
  void SetupVolumeRenderingPipeline();

  /* Setup AR-Volume Rendering Pipeline */
  void SetupARVolumeRenderingPipeline();

  /* Setup US Volume Reconstruction Pipeline */
  int SetupVolumeReconstructionPipeline();

  int GetExtentFromTrackedFrameList(vtkPlusTrackedFrameList*, vtkPlusTransformRepository*,
                                    double spacing, int*, double*);

  void GetFirstFramePosition(PlusTrackedFrame*,  vtkPlusTransformRepository*, double*);

protected:
  /* Structure to hold camera video properties */
  struct VideoCaptureProperties
  {
    int framerate;
    int frame_width, frame_height;
    int n_frames;
  } cam_video_prop;

  std::string inputConfigFileName;
  std::string inputVideoBufferMetafile;
  std::string inputTrackerBufferMetafile;
  std::string inputTransformName;
  std::string scan_type;
  std::string video_filename;
  std::string calibration_filename;

  /* Camera/US frame rate. Viewers are updated at this rate */
  double frame_rate;

  /* ID for interactor Timer. Used to start/stop the corresponding timer */
  int interactorTimerID;

  int frame_counter;
  bool inputRepeat, streamingON;

  std::string outputDir;

  /* Camera intrinsics and distortion params. Data is read into these matrices from the XML files */
  cv::Mat intrinsics, distortion_params;

  /* Capture handle for capturing from file */
  cv::VideoCapture cam_capture;
  cv::Mat cam_frame;

  vtkMatrix4x4* matrix;
  /* Members associated with data streaming with PLUS Library */
  vtkSmartPointer< vtkXMLDataElement > configRootElement;
  vtkSmartPointer< vtkPlusDataCollector > dataCollector;
  vtkPlusDevice* videoDevice;
  vtkPlusDevice* trackerDevice;

  /* Tracked US data is read from a SequenceMeta file into a vtkPlusTrackedFrameList */
  vtkSmartPointer< vtkPlusTrackedFrameList > trackedUSFrameList;

  /* Tracked US framelist used for online 3DUS reconstruction */
  vtkSmartPointer< vtkPlusTrackedFrameList > PlusTrackedFrameList4Recon;

  vtkSmartPointer< vtkPlusVolumeReconstructor >   volumeReconstructor;
  vtkSmartPointer< vtkCLVolumeReconstruction >    acceleratedVolumeReconstructor;
  vtkSmartPointer< vtkPlusTransformRepository >   repository;
  vtkSmartPointer< vtkRenderer >                  camImgRenderer;
  vtkSmartPointer< vtkRenderWindow >              camImgRenWin;
  vtkSmartPointer< vtkRenderWindow >              augmentedRenWin;
  vtkSmartPointer< vtkImageData >                 reconstructedVol;
  vtkSmartPointer< vtkImageData >                 usImageData;
  vtkSmartPointer< vtkImageData >                 usVolume;
  vtkSmartPointer< vtkImageViewer2 >              usViewer;
  vtkSmartPointer< vtkImageClip >                 usImageClip;
  vtkSmartPointer< vtkImageFlip >                 usImageFlip;
  vtkSmartPointer< vtkImageImport >               camImgImport;
  vtkSmartPointer< vtkTexture >                   camImgTexture;
  vtkSmartPointer< vtkImageActor >                camImgActor;
  vtkPlusChannel*                                 aChannel;
  vtkSmartPointer< vtkRenderer >                  us_renderer;
  vtkSmartPointer<vtkRenderer >                   endo_renderer;
  vtkSmartPointer< vtkRenderWindowInteractor >    interactor;
  vtkSmartPointer< vtkMetaImageWriter >           metaWriter;
  vtkSmartPointer< vtkUSEventCallback >           us_callback;

  /* Members for volume rendering */
  vtkSmartPointer< vtkSmartVolumeMapper >           volumeMapper;
  vtkSmartPointer< vtkCuda1DVolumeMapper >          cudaVolumeMapper;
  vtkSmartPointer< vtkCuda2DVolumeMapper >          cuda2DVolumeMapper;
  vtkSmartPointer< vtkCuda2DTransferFunction >      cuda2DTransferFun;
  vtkSmartPointer< vtkCuda2DTransferFunction >      backgroundTF;
  vtkSmartPointer< vtkCudaFunctionPolygonReader >   polyReader;
  vtkSmartPointer< vtkCuda2DInExLogicVolumeMapper > inExVolumeMapper;
  vtkSmartPointer< vtkBoxWidget >                   box;
  vtkSmartPointer< vtkTransform >                   boxTransform;
  vtkSmartPointer< vtkTransform >                   transform;
  vtkSmartPointer< vtkPlanes >                      boxPlanes;
  vtkSmartPointer< vtkImageCanvasSource2D >         background;
  vtkSmartPointer< vtkVolumeProperty >              volumeProperty;
  vtkSmartPointer< vtkPiecewiseFunction >           compositeOpacity;
  vtkSmartPointer< vtkColorTransferFunction >       color;
  vtkSmartPointer< vtkVolume >                      volume;
  vtkSmartPointer< vtkRenderer >                    volRenderer;
  vtkSmartPointer< vtkRenderWindow >                volRenWin;
  qTransferFunctionWindowWidget*                    tfWidget;
  vtkSmartPointer< vtkSphereSource >                sphere;
  vtkSmartPointer< vtkActor >                       actor;
  vtkSmartPointer< vtkWindowToImageFilter >         windowToImage;
  vtkSmartPointer< vtkPNGWriter >                   imageWriter;

  /* Members for keyhole rendering */
  vtkSmartPointer< vtkLightsPass >            lightsPass;
  vtkSmartPointer< vtkDefaultPass >           defaultPass;
  vtkSmartPointer< vtkCameraPass >            cameraPass;
  vtkSmartPointer< vtkKeyholePass >           keyholePass;
  vtkSmartPointer< vtkRenderPassCollection >  passCollection;
  vtkSmartPointer< vtkSequencePass >          sequencePass;
  vtkSmartPointer< vtkPlaneSource >           foregroundPlane;
  vtkSmartPointer< vtkTextureMapToPlane >     foregroundTexturePlane;
  vtkSmartPointer< vtkPolyDataMapper >        foregroundMapper;
  vtkSmartPointer< vtkActor >                 foregroundTexturedPlane;

private:
  Ui::MainWindow* ui;
};

#endif // MAINWINDOW_H