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

#include <QMainWindow>
#include <QLabel>
#include <QSignalMapper>
#include <QMenuBar>

// VTK includes
#include "vtkMatrix4x4.h"
#include "vtkSmartPointer.h"
#include "vtkXMLUtilities.h"
#include "vtkSavedDataSource.h"
#include "vtkImageViewer2.h"
#include "vtkTextActor.h"
#include "vtkTextProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRfProcessor.h"
#include "vtkCallbackCommand.h"
#include "vtkRenderer.h"
#include <vtkCommand.h> 
#include <vtkMetaImageWriter.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkVolumeProperty.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkVolume.h>
#include <vtkImageImport.h>
#include <vtkTexture.h>
#include <vtkCamera.h>
#include <vtkImageClip.h>
#include <vtkImageActor.h>
#include <vtkImageMapper3D.h>
#include <vtkImageFlip.h>
#include <vtkImageCanvasSource2D.h>
#include <vtkBoxWidget.h>
#include <vtkBoxRepresentation.h>
#include <vtkTransform.h>
#include <vtkPlanes.h>
#include <vtkRendererCollection.h>
#include <vtkInteractorStyleImage.h>
#include <vtkCellArray.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

// for testing
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>

// PLUS includes
#include <VolumeReconstruction/vtkVolumeReconstructor.h>
#include <PlusCommon/vtkTrackedFrameList.h>
#include <PlusCommon/TrackedFrame.h>
#include <DataCollection/vtkPlusDevice.h>
#include <DataCollection/vtkPlusDataSource.h>
#include <DataCollection/vtkPlusChannel.h>
#include <vtkTransformRepository.h>
#include <vtkDataCollector.h>
#include <PlusConfigure.h>
#include <vtkTrackedFrameList.h>
#include <vtkSequenceIO.h>

// RobartsVTK includes
#include "vtkCuda1DVolumeMapper.h"
#include "vtkCuda2DVolumeMapper.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaFunctionPolygonReader.h"
#include "vtkCuda2DInExLogicVolumeMapper.h"
#include "qTransferFunctionDefinitionWidget.h"
#include "qTransferFunctionWindowWidget.h"

// OpenCV includes
#include "cv.h"
#include "highgui.h"

#include "CudaReconstruction.h"

#include <iostream> 
#include <chrono> // For timing

# define D_TIMING 
//# define ALIGNMENT_DEBUG

class vtkUSEventCallback : public vtkCommand{

public: 
	static vtkUSEventCallback *New(){
		return new vtkUSEventCallback;
	}

	virtual void Execute(vtkObject *caller, unsigned long, void*){

		vtkSmartPointer< vtkMatrix4x4 > tFrame2Tracker = vtkSmartPointer< vtkMatrix4x4 >::New();

		// Repeat playback
		if( index > trackedFrames->GetNumberOfTrackedFrames()-1 )
			index = 0;

		TrackedFrame *trackedFrame = trackedFrames->GetTrackedFrame( index );

/*		// Update transform repository
		if ( repository->SetTransforms(*trackedFrame) != PLUS_SUCCESS ){
		  LOG_ERROR("Failed to update transform repository with frame"  );
		  return ;
		}

		// Add this tracked frame to the reconstructor
		if ( reconstructor->AddTrackedFrame(trackedFrame, repository) != PLUS_SUCCESS ){
			  LOG_ERROR("Failed to add tracked frame to volume with frame"); 
			  return;
		}

		// Retrieve the reconstructed volume and pass it to the volume renderer
		reconstructor->GetReconstructedVolume( usVolume ); */

#ifdef D_TIMING
		// Start timing
		auto t_start = std::chrono::high_resolution_clock::now();
#endif

		/* Accelerated Volume Reconstruction */
		// TODO: ImageFlip: make a pipeline connection
		imgFlip->SetInputData( trackedFrame->GetImageData()->GetImage() );
		imgFlip->Update();


		accRecon->SetInputImageData( trackedFrame->GetTimestamp(), imgFlip->GetOutput() );
		
		trackedFrame->GetCustomFrameTransform( TransformName, tFrame2Tracker );
		accRecon->SetInputPoseData( trackedFrame->GetTimestamp(), tFrame2Tracker );

		accRecon->UpdateReconstruction();
		accRecon->GetOutputVolume( usVolume );

		if( !std::strcmp( current_mapper.c_str(), "1D_MAPPER") ){
			usVolume->Modified();
			cudaMapper->SetInput( usVolume );
			cudaMapper->Modified();

#ifdef ALIGNMENT_DEBUG
			_boxTransform->Identity();
			_boxTransform->Concatenate( tFrame2Tracker );
			_boxTransform->Update();
#endif
		}
		else if ( !std::strcmp( current_mapper.c_str(), "2D_MAPPER") ){			
			cudaMapper2->SetInput( usVolume );
			cudaMapper2->Modified();
			//cudaMapper2->Update();
		}
		else if( !std::strcmp( current_mapper.c_str(), "INEX_MAPPER") ){
			_inExMapper->SetInput( usVolume );
			_inExMapper->Modified();
			//_inExMapper->Update();

			_boxWidget->SetInputData( usVolume );
			//_boxWidget->Modified();

		}		

		//volMapper->SetInputData( usVolume );
		//volMapper->Modified();
		//_volRenderer->ResetCamera();
		//_volRenWin->Render();

		// Display the tracked frame
		if ( trackedFrame->GetImageData()->IsImageValid() ){
			// Display image if it's valid
			if (trackedFrame->GetImageData()->GetImageType()==US_IMG_BRIGHTNESS || trackedFrame->GetImageData()->GetImageType()==US_IMG_RGB_COLOR){

				// B mode
				this->ImageData->DeepCopy(trackedFrame->GetImageData()->GetImage());  
			}

			this->Viewer->SetInputData_vtk5compatible( ImageData );
			this->Viewer->Modified();
		}


		if (TransformName.IsValid())
		{
			std::ostringstream ss;
			ss.precision( 2 ); 
			TrackedFrameFieldStatus status;			
			if (trackedFrame->GetCustomFrameTransformStatus(TransformName, status) == PLUS_SUCCESS 
				&& status == FIELD_OK ){
				
				trackedFrame->GetCustomFrameTransform(TransformName, tFrame2Tracker); 

				if ( !std::strcmp( current_mapper.c_str(), "INEX_MAPPER") ){	
					// Update boxWidget transform
					_boxTransform->Identity();
					_boxTransform->Concatenate( _transform );
					_boxTransform->Concatenate( tFrame2Tracker );
					_boxTransform->Update();

					_boxWidget->SetTransform( _boxTransform );
					_boxWidget->Modified();

					// Set the box at the right location
					_boxWidget->GetPlanes( _boxPlanes );

					// Now set the keyhole planes
					_inExMapper->SetKeyholePlanes( _boxPlanes );
					_inExMapper->Modified();
					_inExMapper->Update();
				}

				_boxTransform->Identity();
					_boxTransform->Concatenate( _transform );
					_boxTransform->Concatenate( tFrame2Tracker );
					_boxTransform->Update();

					ss  << std::fixed 
				  << "Tracking Info: \n"
				  << tFrame2Tracker->GetElement(0,0) << "   " << tFrame2Tracker->GetElement(0,1) << "   " << tFrame2Tracker->GetElement(0,2) << "   " << tFrame2Tracker->GetElement(0,3) << "\n"
				  << tFrame2Tracker->GetElement(1,0) << "   " << tFrame2Tracker->GetElement(1,1) << "   " << tFrame2Tracker->GetElement(1,2) << "   " << tFrame2Tracker->GetElement(1,3) << "\n"
				  << tFrame2Tracker->GetElement(2,0) << "   " << tFrame2Tracker->GetElement(2,1) << "   " << tFrame2Tracker->GetElement(2,2) << "   " << tFrame2Tracker->GetElement(2,3) << "\n"
				  << tFrame2Tracker->GetElement(3,0) << "   " << tFrame2Tracker->GetElement(3,1) << "   " << tFrame2Tracker->GetElement(3,2) << "   " << tFrame2Tracker->GetElement(3,3) << "\n"; 

				  //repository->SetTransform(PlusTransformName("Frame","Tracker"), tFrame2Tracker);
				  Info->setText(QString(ss.str().c_str()));
			  }
			  else{

				std::string strTransformName; 
				TransformName.GetTransformName(strTransformName); 
				ss  << "Transform '" << strTransformName << "' is invalid ..."; 
			  }
		}


		// Render camera image
		cv::Mat cam_frame;
		bool success = _camCapture->read( cam_frame );
		int c_frame = _camCapture->get( CV_CAP_PROP_POS_FRAMES);

		if( c_frame < n_frames-2 ){
					
			cv::cvtColor( cam_frame, cam_frame, CV_RGB2BGR );
			// Flip the image to compensate for the difference in coordinate systems in VTK and OpenCV
			cv::flip( cam_frame, cam_frame, 0);
			_imgImport->SetImportVoidPointer( cam_frame.data );

			_camImgTexture->Modified();
			_camRenWin->Render();
			this->Viewer->Render();

			_augmentedRenWin->GetRenderers()->GetFirstRenderer()->ResetCameraClippingRange();
			//_augmentedRenWin->GetRenderers()->GetFirstRenderer()->ResetCamera();
			_augmentedRenWin->Render();
			_screen->repaint();

			if( sc_capture_on ){

				// Captures the screen and save the image
				std::string dir = "./screen_captures/";
				std::string prefix = "CAP_";
				std::string ext = ".png";

				_win2Img->Update();
				_win2Img->Modified();
				char num[5];
				itoa(index, num, 10);
				std::string filename = dir + prefix + num + ext;
				_imgWriter->SetFileName( filename.c_str() );
				_imgWriter->Update();
				_imgWriter->Write();
			}


#ifdef D_TIMING
		// End time
		auto t_end = std::chrono::high_resolution_clock::now();
		std::cout << "Elapsed time (Reconstruction + Visualization) : " 
				  << std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()
				  << " us." << std::endl;
#endif

		}
		else{
			index = 0;
			_camCapture->set( CV_CAP_PROP_POS_FRAMES, index);
		}

		index++;
	  }

	  vtkTrackedFrameList *trackedFrames;
	  vtkTransformRepository *repository;
	  vtkVolumeReconstructor *reconstructor;
	  CudaReconstruction *accRecon;
	  vtkImageData *usVolume;
	  vtkSmartVolumeMapper *volMapper;
	  vtkCuda1DVolumeMapper *cudaMapper;
	  vtkCuda2DVolumeMapper *cudaMapper2;
	  vtkImageViewer2 *Viewer;
	  vtkImageFlip *imgFlip;
	  vtkRenderWindowInteractor *Iren;
	  PlusTransformName TransformName; 
	  vtkImageData *ImageData;
	  cv::VideoCapture *_camCapture;
	  int n_frames;
	  vtkImageImport *_imgImport;
	  vtkRenderWindow *_camRenWin;
	  vtkRenderWindow *_volRenWin;
	  vtkVolume *_vol;
	  vtkRenderWindow *_augmentedRenWin;
	  vtkRenderer *_volRenderer;
	  vtkTexture *_camImgTexture;
	  vtkTransform *_boxTransform;
	  vtkTransform *_transform;
	  vtkBoxWidget *_boxWidget;
	  vtkPlanes *_boxPlanes;
	  vtkCuda2DInExLogicVolumeMapper *_inExMapper;
	  QVTKWidget * _screen;
	  QLabel *Info;
	  vtkWindowToImageFilter *_win2Img;
	  vtkPNGWriter *_imgWriter;
	  std::string current_mapper;
	  bool sc_capture_on;
	  int index;
};



namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
	void onStartButtonClick(const QString &);
	void onScanTypeRadioButtonClick(const QString &);
	void onSaveVolumeButtonClick(const QString &);
	void ontf1ButtonClick(const QString &);
	void ontf2ButtonClick(const QString &);
	void ontfInExButtonClick(const QString &);
	void onScCaptureRadioButtonClick(const QString &);

private:
    Ui::MainWindow *ui;

	/* Structure to hold camera video properties */ 
	struct VideoCaptureProperties{
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
	vtkSmartPointer< vtkDataCollector > dataCollector;
	vtkPlusDevice* videoDevice;
	vtkPlusDevice* trackerDevice;

	/* Tracked US data is read from a SequenceMeta file into a vtkTrackedFrameList */
	vtkSmartPointer< vtkTrackedFrameList > trackedUSFrameList;

	/* Tracked US framelist used for online 3DUS reconstruction */
	vtkSmartPointer< vtkTrackedFrameList > trackedFrameList4Recon;

	vtkSmartPointer< vtkVolumeReconstructor >  volumeReconstructor;
	CudaReconstruction * acceleratedVolumeReconstructor;
	vtkSmartPointer< vtkTransformRepository > repository;
	vtkSmartPointer< vtkRenderer > camImgRenderer;
	vtkSmartPointer< vtkRenderWindow > camImgRenWin;
	vtkSmartPointer< vtkRenderWindow > augmentedRenWin;
	vtkSmartPointer< vtkImageData > reconstructedVol;
	vtkSmartPointer< vtkImageData > usImageData;
	vtkSmartPointer< vtkImageData > usVolume;
	vtkSmartPointer< vtkImageViewer2 > usViewer;
	vtkSmartPointer< vtkImageClip > usImageClip;
	vtkSmartPointer< vtkImageFlip > usImageFlip;
	vtkSmartPointer< vtkImageImport > camImgImport;
	vtkSmartPointer< vtkTexture > camImgTexture;
	vtkSmartPointer< vtkImageActor > camImgActor;
	vtkPlusChannel* aChannel;
	vtkSmartPointer< vtkRenderer > us_renderer;
	vtkSmartPointer<vtkRenderer > endo_renderer;
	vtkSmartPointer< vtkRenderWindowInteractor > interactor;
	vtkSmartPointer< vtkMetaImageWriter > metaWriter;
	vtkSmartPointer< vtkUSEventCallback > us_callback;

	/* Members for volume rendering */
	vtkSmartPointer< vtkSmartVolumeMapper > volumeMapper;
	vtkSmartPointer< vtkCuda1DVolumeMapper > cudaVolumeMapper;
	vtkSmartPointer< vtkCuda2DVolumeMapper > cuda2DVolumeMapper;
	vtkSmartPointer< vtkCuda2DTransferFunction > cuda2DTransferFun;
	vtkSmartPointer< vtkCuda2DTransferFunction > backgroundTF;
	vtkSmartPointer< vtkCudaFunctionPolygonReader > polyReader;
	vtkSmartPointer< vtkCuda2DInExLogicVolumeMapper > inExVolumeMapper;
	vtkSmartPointer< vtkBoxWidget > box;
	vtkSmartPointer< vtkTransform > boxTransform;
	vtkSmartPointer< vtkTransform > transform;
	vtkSmartPointer< vtkPlanes > boxPlanes;
	vtkSmartPointer< vtkImageCanvasSource2D > background;
	vtkSmartPointer< vtkVolumeProperty > volumeProperty;
	vtkSmartPointer< vtkPiecewiseFunction > compositeOpacity;
	vtkSmartPointer< vtkColorTransferFunction > color;
	vtkSmartPointer< vtkVolume > volume;
	vtkSmartPointer< vtkRenderer > volRenderer;
	vtkSmartPointer< vtkRenderWindow > volRenWin;
	transferFunctionWindowWidget * tfWidget;
	vtkSmartPointer< vtkSphereSource > sphere;
	vtkSmartPointer< vtkActor > actor;
	vtkSmartPointer< vtkWindowToImageFilter > windowToImage;
	vtkSmartPointer< vtkPNGWriter > imageWriter;

	/* Initialize PLUS pipeline */
	int init_PLUS_Pipeline();

	/* Initialize VTK pipeline */
	int init_VTK_Pipeline();

	/* Initialize OpenCV variables */
	int init_CV_Pipeline();

	/* Initialize PLUS-bypass pipeline */
	int init_PLUS_Bypass_Pipeline();

	/* Setup VTK Camera from intrinsics */
	void setup_VTK_Camera(cv::Mat, double, double, vtkCamera*);
	void setup_VTK_Camera(cv::Mat, vtkCamera*);

	/* Setup Volume Rendering Pipeline */
	void setup_VolumeRendering_Pipeline();

	/* Setup AR-Volume Rendering Pipeline */
	void setup_ARVolumeRendering_Pipeline();

	/* Setup US Volume Reconstruction Pipeline */
	int setup_VolumeReconstruction_Pipeline();

	int get_extent_from_trackedList(vtkTrackedFrameList *, vtkTransformRepository *,
										double spacing, int *, double *);

	void get_first_frame_position(TrackedFrame *,  vtkTransformRepository *, double *);

};

#endif // MAINWINDOW_H
