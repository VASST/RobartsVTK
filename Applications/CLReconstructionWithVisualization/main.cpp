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

#include <iostream>
#include <string>
// For timing
#include <Windows.h>
#include <stdint.h>

// Use PLUS just to read data from sequence meta file
#include <vtkPlusSequenceIO.h>
#include <vtkPlusTrackedFrameList.h>
#include <vtkPlusVolumeReconstructor.h> // Use PLUS volume reconstructor for comparison purposes.
#include <vtkPlusTransformRepository.h>
#include <PlusTrackedFrame.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkMetaImageWriter.h>
#include <vtkImageFlip.h>
#include <vtkVolumeProperty.h>
#include <vtkColorTransferFunction.h>
#include <vtkPiecewiseFunction.h> 
#include <vtkVolume.h>
#include <vtkRenderer.h> 
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderWindow.h>
#include <vtkSmartVolumeMapper.h> 
#include <vtkImageData.h>
#include <vtkCamera.h> 
#include <vtkCommand.h> 
#include <vtkMetaImageReader.h>

#include <vtkCLVolumeReconstruction.h>
#include <vtkCuda1DVolumeMapper.h>

/* Sets output extent and origin from a vtkTrackedFrameList */
int get_extent_from_trackedList(vtkPlusTrackedFrameList *, vtkPlusTransformRepository *,
	double spacing, int *, double *);

class vtkTimerCallback : public vtkCommand
{
public:
	static vtkTimerCallback *New()
	{
		vtkTimerCallback *cb = new vtkTimerCallback;
		return cb;
	}

	virtual void Execute(vtkObject *caller, unsigned long eventID, void *vtkNotUsed(callData))
	{
		if (idx == trackedFrameList->GetNumberOfTrackedFrames() - 1)
			idx = 0;

		// Set Image Dada
		PlusTrackedFrame *trackedFrame = trackedFrameList->GetTrackedFrame(idx);
		imgFlip->SetInputData(trackedFrame->GetImageData()->GetImage());
		imgFlip->Modified();
		imgFlip->Update();
		//reconstructor->SetInputImageData(trackedFrame->GetTimestamp(), imgFlip->GetOutput());

		// Set pose Data
		trackedFrame->GetCustomFrameTransform(transformName, tFrame2Tracker);
		//reconstructor->SetInputPoseData(trackedFrame->GetTimestamp(), tFrame2Tracker);


		imagePose->SetMatrix(tFrame2Tracker);
		//imagePose->Modified();
		reconstructor->SetInputData(imgFlip->GetOutput());
		reconstructor->Update();


		// Update Reconstruction
		//reconstructor->UpdateReconstruction();
		//reconstructor->GetOutputVolume(outputVolume);
		//outputVolume->Modified();

		// Visualize
		//cudaMapper->SetInputData(reconstructor->GetOutput());
		//volumeMapper->SetInputData(reconstructor->GetOutput());


		// Update rendering pipeline
		ren->Render();

		std::cout << "Rendered.. " << std::endl;

		idx++;
	}

private:
	int idx = 0;

public:
	vtkSmartPointer< vtkPlusTrackedFrameList > trackedFrameList;
	vtkSmartPointer< vtkCLVolumeReconstruction > reconstructor;
	vtkSmartPointer< vtkCuda1DVolumeMapper > cudaMapper;
	vtkSmartPointer< vtkImageFlip > imgFlip;
	vtkSmartPointer< vtkVolume > usVolume;
	vtkSmartPointer< vtkRenderer > ren;
	vtkSmartPointer< vtkRenderWindow > renwin;
	vtkSmartPointer < vtkImageData > outputVolume;
	PlusTransformName transformName;
	vtkSmartPointer< vtkMatrix4x4 > tFrame2Tracker;
	vtkSmartPointer< vtkTransform > imagePose;
	vtkSmartPointer< vtkSmartVolumeMapper > volumeMapper;
};

// This is for timing. 
int gettimeofday(struct timeval * tp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}

int main(){

	// Read in Data
	vtkSmartPointer< vtkPlusTrackedFrameList > trackedFrameList = vtkSmartPointer< vtkPlusTrackedFrameList >::New();
	// Read Sequence Meta file in the tracked framelist
	vtkPlusSequenceIO::Read(std::string("tracked_us_video.mha"), trackedFrameList);
	PlusTransformName transformName = PlusTransformName("Probe", "Tracker");
	vtkSmartPointer< vtkMatrix4x4 > tFrame2Tracker = vtkSmartPointer< vtkMatrix4x4 >::New();

	vtkSmartPointer< vtkImageFlip > imgFlip = vtkSmartPointer< vtkImageFlip >::New();
	imgFlip->SetFilteredAxis(1);


	// Initialize PLUS VolumeReconstructor
	std::string inputConfigFileName = "config.xml";
	vtkSmartPointer< vtkXMLDataElement> configRootElement = vtkSmartPointer< vtkXMLDataElement >::New();
	if (PlusXmlUtils::ReadDeviceSetConfigurationFromFile(configRootElement, inputConfigFileName.c_str()) == PLUS_FAIL){
		LOG_ERROR("Unable to read configuration from file " << inputConfigFileName.c_str());
		return EXIT_FAILURE;
	}

	vtkSmartPointer< vtkPlusVolumeReconstructor > volumeReconstructor = vtkSmartPointer< vtkPlusVolumeReconstructor >::New();

	vtkSmartPointer< vtkPlusTransformRepository > repository = vtkSmartPointer< vtkPlusTransformRepository >::New();
	/* Read Coordinate system definitions from XML data */
	if (repository->ReadConfiguration(configRootElement) != PLUS_SUCCESS){
		LOG_ERROR("Configuration incorrect for vtkTransformRepository.");
		exit(EXIT_FAILURE);
	}

	/* Read Coordinate system definitions from XML data */
	if (volumeReconstructor->ReadConfiguration(configRootElement) != PLUS_SUCCESS){
		LOG_ERROR("Configuration incorrect for vtkTransformRepository.");
		exit(EXIT_FAILURE);
	}

	std::string errorDetail;
	/* Set output Extent from the input data. During streaming a sout scan may be necessary to set the output
	extent. */
	if (volumeReconstructor->SetOutputExtentFromFrameList(trackedFrameList, repository, errorDetail) != PLUS_SUCCESS){
		LOG_ERROR("Setting up Output Extent from FrameList:" + errorDetail);
		exit(EXIT_FAILURE);
	}




	// Set-up visualization pipeline
	vtkMetaImageReader *reader = vtkMetaImageReader::New();
	reader->SetFileName("3DUS.mhd");
	reader->Update();

	vtkSmartPointer< vtkCuda1DVolumeMapper > cudaMapper = vtkSmartPointer< vtkCuda1DVolumeMapper >::New();
	cudaMapper->UseCUDAOpenGLInteroperability();
	cudaMapper->SetBlendModeToComposite();
	cudaMapper->SetInputData(reader->GetOutput()); 

	// Create vtkCLVolumeReconstructor
	vtkSmartPointer< vtkCLVolumeReconstruction > recon = vtkSmartPointer< vtkCLVolumeReconstruction >::New();
	recon->SetDevice(0);


	// calibration matrix
	float us_cal_mat[12] = { 0.0727, 0.0076, -0.0262, -12.6030,
		-0.0030, 0.0118, -0.9873, -7.8930,
		-0.0069, 0.0753, 0.1568, 1.0670 };

	recon->SetProgramSourcePath("./kernels.cl");
	recon->SetBScanSize(820, 616);
	recon->SetBScanSpacing(0.077, 0.073);

	int extent[6] = { 0, 0, 0, 0, 0, 0 };
	double origin[3] = { 0, 0, 0 };
	double output_spacing = 0.5;

	get_extent_from_trackedList(trackedFrameList, repository, output_spacing, extent, origin);
	recon->SetOutputExtent(extent[0], extent[1], extent[2], extent[3], extent[4], extent[5]);
	recon->SetOutputSpacing(output_spacing);
	recon->SetOutputOrigin(origin[0], origin[1], origin[2]);
	recon->SetCalMatrix(us_cal_mat);
	recon->Initialize();
	recon->StartReconstruction();
	vtkTransform *pose = vtkTransform::New();
	recon->SetImagePoseTransform(pose);

	std::cout << "vtkCLReconstruction initialized. " << std::endl;


	// Meta image writer
	vtkSmartPointer< vtkMetaImageWriter > writer = vtkSmartPointer< vtkMetaImageWriter >::New();
	writer->SetFileName("3DUS-output.mhd");

	vtkImageData *outputVolume = vtkImageData::New();
	outputVolume->SetExtent(extent);
	outputVolume->SetDimensions(extent[1] - extent[0],
		extent[3] - extent[2],
		extent[5] - extent[4]);
	outputVolume->SetSpacing(output_spacing, output_spacing, output_spacing);
	outputVolume->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
	//recon->GetOutputVolume(outputVolume);	
	outputVolume->Modified();
	recon->SetOutput(outputVolume);


	vtkSmartPointer< vtkSmartVolumeMapper > volumeMapper = vtkSmartPointer< vtkSmartVolumeMapper >::New();
	volumeMapper->SetBlendModeToComposite();
	volumeMapper->SetInputData(outputVolume); 

	vtkSmartPointer< vtkVolumeProperty > volumeProperty = vtkSmartPointer< vtkVolumeProperty >::New();
	volumeProperty->ShadeOff();
	volumeProperty->SetInterpolationType(VTK_LINEAR_INTERPOLATION);

	vtkSmartPointer< vtkPiecewiseFunction > compositeOpacity = vtkSmartPointer< vtkPiecewiseFunction >::New();
	compositeOpacity->AddPoint(0.0, 0.0);
	compositeOpacity->AddPoint(75.72, 0.079);
	compositeOpacity->AddPoint(176.15, 0.98);
	compositeOpacity->AddPoint(255.0, 1.0);
	volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.

	vtkSmartPointer< vtkColorTransferFunction > colorTransferFun = vtkSmartPointer< vtkColorTransferFunction >::New();
	colorTransferFun->AddRGBPoint(0.0, 0.0, 0.0, 1.0);
	colorTransferFun->AddRGBPoint(40.0, 0.0, 0.1, 0.0);
	colorTransferFun->AddRGBPoint(255.0, 1.0, 0.0, 0.0);
	volumeProperty->SetColor(colorTransferFun);

	vtkSmartPointer< vtkVolume > usVolume = vtkSmartPointer< vtkVolume >::New();
	usVolume->SetMapper(cudaMapper);
	//usVolume->SetMapper(volumeMapper);
	//usVolume->SetProperty(volumeProperty);
	//usVolume->SetOrigin(0, 0, 0);
	//usVolume->SetPosition(0, 0, 0);
	//cudaMapper->SetInputData(outputVolume);
	//usVolume->Modified();	

	vtkSmartPointer< vtkRenderer > ren = vtkSmartPointer< vtkRenderer >::New();
	ren->AddViewProp(usVolume);

	int win_size[2] = { 640, 480 };
	/*vtkCamera *cam = ren->GetActiveCamera();
	cam->SetWindowCenter(win_size[0]/2.0, win_size[1]/2.0);
	double viewAngle = 2 * atan((win_size[1] / 2.0) / 250) * 180 / (4 * atan(1.0));
	cam->SetViewAngle(viewAngle);
	cam->SetPosition(0, 0, 0);
	cam->SetViewUp(0, -1, 0);
	cam->SetFocalPoint(0, 0, 615);
	cam->SetClippingRange(0.01, 1000.01);
	cam->Modified(); */

	vtkSmartPointer< vtkRenderWindow > renwin = vtkSmartPointer< vtkRenderWindow >::New();
	//renwin->SetSize(win_size);
	renwin->AddRenderer(ren);
	//ren->ResetCamera();
	//ren->ResetCameraClippingRange();

	
	vtkSmartPointer< vtkTimerCallback > callback = vtkSmartPointer< vtkTimerCallback >::New();
	callback->trackedFrameList = trackedFrameList;
	callback->reconstructor = recon;
	callback->cudaMapper = cudaMapper;
	callback->volumeMapper = volumeMapper;
	callback->imgFlip = imgFlip;
	callback->usVolume = usVolume;
	callback->ren = ren;
	callback->renwin = renwin;
	callback->outputVolume = outputVolume;
	callback->transformName = transformName;
	callback->tFrame2Tracker = tFrame2Tracker;
	callback->imagePose = pose; 

	vtkSmartPointer< vtkRenderWindowInteractor > iren = vtkSmartPointer< vtkRenderWindowInteractor >::New();
	iren->SetRenderWindow(renwin);

	iren->AddObserver(vtkCommand::TimerEvent, callback);
	iren->Initialize();

	int timerID = iren->CreateRepeatingTimer(1000.0 / 30.0);
	//renwin->Render();
	iren->Start();

	std::cout << "Reconstruction done. " << std::endl;
	std::cout << "Writing output to file. " << std::endl;

	writer->SetInputData(outputVolume);
	writer->Write();

	return 0;
}


int get_extent_from_trackedList(vtkPlusTrackedFrameList *frameList, vtkPlusTransformRepository *repository, double spacing,
	int * outputExtent, double * origin){

	PlusTransformName imageToReferenceTransformName;
	imageToReferenceTransformName = PlusTransformName("Image", "Tracker");

	if (frameList == NULL){
		LOG_ERROR("Failed to set output extent from tracked frame list - input frame list is NULL");
		return -1;
	}

	if (frameList->GetNumberOfTrackedFrames() == 0){

		LOG_ERROR("Failed to set output extent from tracked frame list - input frame list is empty");
		return -1;
	}

	if (repository == NULL){

		LOG_ERROR("Failed to set output extent from tracked frame list - input transform repository is NULL");
		return -1;
	}

	double extent_Ref[6] = {
		VTK_DOUBLE_MAX, VTK_DOUBLE_MIN,
		VTK_DOUBLE_MAX, VTK_DOUBLE_MIN,
		VTK_DOUBLE_MAX, VTK_DOUBLE_MIN
	};

	const int numberOfFrames = frameList->GetNumberOfTrackedFrames();
	int numberOfValidFrames = 0;
	for (int frameIndex = 0; frameIndex < numberOfFrames; ++frameIndex) {

		PlusTrackedFrame* frame = frameList->GetTrackedFrame(frameIndex);

		if (repository->SetTransforms(*frame) != PLUS_SUCCESS){

			LOG_ERROR("Failed to update transform repository with tracked frame!");
			continue;
		}

		// Get transform
		bool isMatrixValid(false);
		vtkSmartPointer<vtkMatrix4x4> imageToReferenceTransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
		if (repository->GetTransform(imageToReferenceTransformName, imageToReferenceTransformMatrix, &isMatrixValid) != PLUS_SUCCESS){

			std::string strImageToReferenceTransformName;
			imageToReferenceTransformName.GetTransformName(strImageToReferenceTransformName);

			LOG_WARNING("Failed to get transform '" << strImageToReferenceTransformName << "' from transform repository!");
			continue;
		}

		if (isMatrixValid)
		{
			numberOfValidFrames++;

			// Get image (only the frame extents will be used)
			vtkImageData* frameImage = frameList->GetTrackedFrame(frameIndex)->GetImageData()->GetImage();

			// Output volume is in the Reference coordinate system.

			// Prepare the four corner points of the input US image.
			int* frameExtent = frameImage->GetExtent();
			std::vector< double* > corners_ImagePix;
			double minX = frameExtent[0];
			double maxX = frameExtent[1];
			double minY = frameExtent[2];
			double maxY = frameExtent[3];

			double c0[4] = { minX, minY, 0, 1 };
			double c1[4] = { minX, maxY, 0, 1 };
			double c2[4] = { maxX, minY, 0, 1 };
			double c3[4] = { maxX, maxY, 0, 1 };

			corners_ImagePix.push_back(c0);
			corners_ImagePix.push_back(c1);
			corners_ImagePix.push_back(c2);
			corners_ImagePix.push_back(c3);

			// Transform the corners to Reference and expand the extent if needed
			for (unsigned int corner = 0; corner < corners_ImagePix.size(); ++corner){

				double corner_Ref[4] = { 0, 0, 0, 1 }; // position of the corner in the Reference coordinate system
				imageToReferenceTransformMatrix->MultiplyPoint(corners_ImagePix[corner], corner_Ref);

				for (int axis = 0; axis < 3; axis++){
					if (corner_Ref[axis] < extent_Ref[axis * 2]){
						// min extent along this coord axis has to be decreased
						extent_Ref[axis * 2] = corner_Ref[axis];
					}

					if (corner_Ref[axis] > extent_Ref[axis * 2 + 1]){

						// max extent along this coord axis has to be increased
						extent_Ref[axis * 2 + 1] = corner_Ref[axis];
					}
				}
			}

		}
	}


	// Set the output extent from the current min and max values, using the user-defined image resolution.
	outputExtent[1] = int((extent_Ref[1] - extent_Ref[0]) / spacing);
	outputExtent[3] = int((extent_Ref[3] - extent_Ref[2]) / spacing);
	outputExtent[5] = int((extent_Ref[5] - extent_Ref[4]) / spacing);

	origin[0] = extent_Ref[0];
	origin[1] = extent_Ref[2];
	origin[2] = extent_Ref[4];

	return 0;
}