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

#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <vtkMetaImageReader.h>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
	ui->trackingInfo_Label->setText(QString("Tracking Info:"));
	ui->scantype_RadioButton->setChecked(true);
	ui->start_Button->setText(QString("Start"));
	ui->scCaptureButton->setChecked( false );
	streamingON = false;

	// UI Signal Mapping
	QSignalMapper *signalMapper = new QSignalMapper(this);
	signalMapper->setMapping(ui->start_Button, "START_BTN");
	signalMapper->setMapping(ui->scantype_RadioButton, "SCANTYPE_BTN");
	signalMapper->setMapping(ui->saveVolumeButton, "SAVEVOL_BTN");
	signalMapper->setMapping(ui->tf1_button, "TF1_BTN");
	signalMapper->setMapping(ui->tf2_button, "TF2_BTN");
	signalMapper->setMapping(ui->tfInEx_button, "TFINEX_BTN");
	signalMapper->setMapping(ui->scCaptureButton, "SCCAPTURE_BTN");
	QObject::connect(signalMapper, SIGNAL(mapped(const QString &)), this, SLOT(onStartButtonClick(const QString &)));
	QObject::connect(signalMapper, SIGNAL(mapped(const QString &)), this, SLOT(onScanTypeRadioButtonClick(const QString &)));
	QObject::connect(signalMapper, SIGNAL(mapped(const QString &)), this, SLOT(onSaveVolumeButtonClick(const QString &)));
	QObject::connect(signalMapper, SIGNAL(mapped(const QString &)), this, SLOT(ontf1ButtonClick(const QString &)));
	QObject::connect(signalMapper, SIGNAL(mapped(const QString &)), this, SLOT(ontf2ButtonClick(const QString &)));
	QObject::connect(signalMapper, SIGNAL(mapped(const QString &)), this, SLOT(ontfInExButtonClick(const QString &)));
	QObject::connect(signalMapper, SIGNAL(mapped(const QString &)), this, SLOT(onScCaptureRadioButtonClick(const QString &)));
	QObject::connect(ui->start_Button, SIGNAL(clicked()), signalMapper, SLOT(map()) );
	QObject::connect(ui->scantype_RadioButton, SIGNAL(clicked()), signalMapper, SLOT(map()));
	QObject::connect(ui->saveVolumeButton, SIGNAL(clicked()), signalMapper, SLOT(map()));
	QObject::connect(ui->tf1_button, SIGNAL(clicked()), signalMapper, SLOT(map()));
	QObject::connect(ui->tf2_button, SIGNAL(clicked()), signalMapper, SLOT(map()));
	QObject::connect(ui->tfInEx_button, SIGNAL(clicked()), signalMapper, SLOT(map()));
	QObject::connect(ui->scCaptureButton, SIGNAL(clicked()), signalMapper, SLOT(map()));

	inputRepeat = false;
	scan_type = "SCOUT";
	inputConfigFileName = "config.xml";
	video_filename = "CAP_201587T184044_UND.avi";
	calibration_filename = "left_calibration.xml";
	frame_rate = 30;

	init_CV_Pipeline();
	//init_PLUS_Pipeline();
	init_PLUS_Bypass_Pipeline();
	init_VTK_Pipeline();

	outputDir = "./Data/";

	metaWriter = vtkSmartPointer< vtkMetaImageWriter >::New();
	metaWriter->SetFileName( std::string(outputDir + "output.mhd").c_str());
}

MainWindow::~MainWindow()
{
    delete ui;
}

/* Initialize PLUS pipeline */
int MainWindow::init_PLUS_Pipeline(){
	
	inputTrackerBufferMetafile.clear();
	inputVideoBufferMetafile.clear();

	configRootElement = vtkSmartPointer<vtkXMLDataElement>::New();
	if (PlusXmlUtils::ReadDeviceSetConfigurationFromFile(configRootElement, inputConfigFileName.c_str())==PLUS_FAIL){  
		LOG_ERROR("Unable to read configuration from file " << inputConfigFileName.c_str()); 
		return EXIT_FAILURE;
	}

	vtkPlusConfig::GetInstance()->SetDeviceSetConfigurationData(configRootElement);

	dataCollector = vtkSmartPointer< vtkDataCollector >::New(); 

	if( dataCollector->ReadConfiguration( configRootElement ) != PLUS_SUCCESS ){
		LOG_ERROR("Configuration incorrect for vtkDataCollectorTest1.");
		exit( EXIT_FAILURE );
	}
	
	videoDevice = NULL;
	trackerDevice = NULL;
	volumeReconstructor = NULL;

	if ( ! inputVideoBufferMetafile.empty() ){
		if( dataCollector->GetDevice(videoDevice, "VideoDevice") != PLUS_SUCCESS ){
			LOG_ERROR("Unable to locate the device with Id=\"VideoDevice\". Check config file.");
			exit(EXIT_FAILURE);
		}
		
		vtkSavedDataSource* videoSource = dynamic_cast<vtkSavedDataSource*>(videoDevice); 
		if ( videoSource == NULL ){
			LOG_ERROR( "Unable to cast video source to vtkSavedDataSource." );
			exit( EXIT_FAILURE );
		}
    
		videoSource->SetSequenceFile(inputVideoBufferMetafile.c_str()); 
		videoSource->SetRepeatEnabled(inputRepeat);

	}
	
	if ( ! inputTrackerBufferMetafile.empty() )	{
		if( dataCollector->GetDevice(trackerDevice, "TrackerDevice") != PLUS_SUCCESS ){
			LOG_ERROR("Unable to locate the device with Id=\"TrackerDevice\". Check config file.");
			exit(EXIT_FAILURE);
		}
   
		vtkSavedDataSource* tracker = dynamic_cast<vtkSavedDataSource*>(trackerDevice); 
		if ( tracker == NULL ){
			LOG_ERROR( "Unable to cast tracker to vtkSavedDataSource" );
			exit( EXIT_FAILURE );
		}
    
		tracker->SetSequenceFile(inputTrackerBufferMetafile.c_str()); 
		tracker->SetRepeatEnabled(inputRepeat); 
	}

	if ( dataCollector->Connect() != PLUS_SUCCESS ){
		LOG_ERROR("Failed to connect to devices!" ); 
		exit( EXIT_FAILURE );
	}
	
	if( dataCollector->GetDevice(videoDevice, "TrackedVideoDevice") != PLUS_SUCCESS ){
      
		LOG_ERROR("Unable to locate the device with Id=\"TrackedVideoDevice\". Check config file.");
        exit(EXIT_FAILURE);
    }	

    if( videoDevice->GetOutputChannelByName(aChannel, "TrackedVideoStream") != PLUS_SUCCESS ){

		LOG_ERROR("Unable to locate the channel with Id=\"TrackedVideoStream\". Check config file.");
        exit(EXIT_FAILURE);
    }
	
	if ( dataCollector->Start() != PLUS_SUCCESS ){
		LOG_ERROR("Failed to start data collection!" ); 
		exit( EXIT_FAILURE );
	}
}

/* Initialize PLUS Bypass pipeline */
int MainWindow::init_PLUS_Bypass_Pipeline(){
	
	trackedUSFrameList = vtkSmartPointer<vtkTrackedFrameList>::New(); 
	// Read sequence file into tracked frame list
	vtkSequenceIO::Read(std::string("tracked_us_video.mha"), trackedUSFrameList);

	// Setup US clip region
	usImageClip = vtkSmartPointer< vtkImageClip >::New();
	usImageClip->ClipDataOff();
	usImageClip->SetOutputWholeExtent(159, 663, 41, 575, 0, 0);

	// Setup US Image Flip
	usImageFlip = vtkSmartPointer< vtkImageFlip >::New();
	usImageFlip->SetFilteredAxis( 1 ); // flip the Y axis

	configRootElement = vtkSmartPointer<vtkXMLDataElement>::New();
	if (PlusXmlUtils::ReadDeviceSetConfigurationFromFile(configRootElement, inputConfigFileName.c_str())==PLUS_FAIL){  
		LOG_ERROR("Unable to read configuration from file " << inputConfigFileName.c_str()); 
		return EXIT_FAILURE;
	}

	vtkPlusConfig::GetInstance()->SetDeviceSetConfigurationData(configRootElement);

	// Setup VolumeReconstruction pipeline
	setup_VolumeReconstruction_Pipeline();

	// GPU Accelerated Reconstructor
	acceleratedVolumeReconstructor = new CudaReconstruction();

	// calibration matrix
	float us_cal_mat[12] = {0.0727,    0.0076,   -0.0262,  -12.6030,
						   -0.0030,    0.0118,   -0.9873,   -7.8930,
						   -0.0069,    0.0753,    0.1568,    1.0670};

	acceleratedVolumeReconstructor->SetProgramSourcePath("..\\src\\acceleratedReconstruction\\kernels.cl");
	acceleratedVolumeReconstructor->SetBScanSize( 820, 616);
	acceleratedVolumeReconstructor->SetBScanSpacing(0.077, 0.073);

	int extent[6] = {0, 0, 0, 0, 0, 0};
	double origin[3] = {0, 0, 0};
	double output_spacing = 0.5;
	
	get_extent_from_trackedList( trackedUSFrameList, repository, output_spacing, extent, origin);
	acceleratedVolumeReconstructor->SetOutputExtent( extent[0], extent[1], extent[2], extent[3], extent[4], extent[5]);
	acceleratedVolumeReconstructor->SetOutputSpacing( output_spacing );
	acceleratedVolumeReconstructor->SetOutputOrigin(origin[0], origin[1], origin[2]);
	acceleratedVolumeReconstructor->SetCalMatrix(us_cal_mat);
	acceleratedVolumeReconstructor->Initialize();
	acceleratedVolumeReconstructor->StartReconstruction();

	return 0;
}

/* Initilize VTK pipeline */
int MainWindow::init_VTK_Pipeline(){

	// Setup US Image Viewer
	usViewer = vtkSmartPointer< vtkImageViewer2 >::New();
	usViewer->SetColorWindow( 255 );
	usViewer->SetColorLevel( 127.5 );	
	// Resize to scale to the QWidget
	usViewer->SetSize( this->ui->US_View->geometry().width(), 
						this->ui->US_View->geometry().height());
	usViewer->SetupInteractor(this->ui->US_View->GetInteractor());
	this->ui->US_View->SetRenderWindow( usViewer->GetRenderWindow() );	

	usImageData = vtkSmartPointer<vtkImageData>::New();

	// Setup image import for camera images
	camImgImport = vtkSmartPointer< vtkImageImport >::New();
	camImgImport->SetDataOrigin( 0, 0, 0);
	camImgImport->SetDataSpacing( 1, 1, 1);
	camImgImport->SetWholeExtent( 0, cam_video_prop.frame_width - 1, 0, cam_video_prop.frame_height -1, 1, 1);
	camImgImport->SetDataExtentToWholeExtent();
	camImgImport->SetDataScalarTypeToUnsignedChar();
	camImgImport->SetNumberOfScalarComponents( 3 );
		
	// Set camera image as the background texture
	camImgTexture = vtkSmartPointer< vtkTexture >::New();
	camImgTexture->SetInputConnection( camImgImport->GetOutputPort() );

	camImgRenderer = vtkSmartPointer< vtkRenderer >::New();
	camImgRenderer->SetBackgroundTexture( camImgTexture );
	camImgRenderer->TexturedBackgroundOn();

	// Setup VTK camera
	vtkCamera *vtkCam = camImgRenderer->GetActiveCamera();
	//setup_VTK_Camera(intrinsics, vtkCam);
	setup_VTK_Camera(intrinsics, this->ui->Endoscopic_View->width(), this->ui->Endoscopic_View->height(), vtkCam);

	// Create Camera Render Window
	camImgRenWin = vtkSmartPointer< vtkRenderWindow >::New();
	camImgRenWin->AddRenderer( camImgRenderer );
	this->ui->Endoscopic_View->SetRenderWindow( camImgRenWin );

	// Set up Cam Image as a layer in Augmented View
	/*camImgActor = vtkSmartPointer< vtkImageActor >::New();
	camImgActor->GetMapper()->SetInputConnection( camImgImport->GetOutputPort() );

	double fx( intrinsics.at<double>(0,0) ), fy( intrinsics.at<double>(1,1) );
	double cx( intrinsics.at<double>(0,2) ), cy( intrinsics.at<double>(1,2) );
	double center_x = (cam_video_prop.frame_width - cx)/((cam_video_prop.frame_width-1)/2.0) - 1;
	double center_y = cy/((cam_video_prop.frame_height-1)/2.0) - 1;

	vtkTransform *transform = vtkTransform::New();
	transform->PostMultiply();	

	vtkMatrix4x4 *mat = vtkMatrix4x4::New();*/

	/*double m_transform[16] = {1, 0, 0, -cx*fy/fx,
							  0, 1, 0, -cy, 
							  0, 0, 1, fy, 
							  0, 0, 0, 1};
	mat->DeepCopy( m_transform );
	camImgActor->SetUserMatrix(mat);*/
	
	// Setup volume Rendering pipeline for online US visualization 
	// Make sure that volumeReconstructor is initialized before this. 
	//setup_VolumeRendering_Pipeline();
	setup_ARVolumeRendering_Pipeline();

	// Setup observers
	us_callback = vtkSmartPointer< vtkUSEventCallback >::New();
	//us_callback->DataCollector = dataCollector;
	//us_callback->BroadcastChannel = aChannel;
	us_callback->trackedFrames = trackedUSFrameList;
	us_callback->usVolume = usVolume;
	us_callback->repository = repository;
	us_callback->reconstructor = volumeReconstructor;
	us_callback->accRecon = acceleratedVolumeReconstructor;
	us_callback->volMapper = volumeMapper;
	us_callback->cudaMapper = cudaVolumeMapper;
	us_callback->Viewer = usViewer;
	us_callback->imgFlip = usImageFlip;
	us_callback->index = 0;
	us_callback->Info = this->ui->trackingInfo_Label;
	us_callback->Iren = this->ui->US_View->GetInteractor();
	us_callback->ImageData = usImageData;
	us_callback->_camCapture = &cam_capture ;
	us_callback->n_frames = cam_video_prop.n_frames;
	us_callback->_camRenWin = camImgRenWin;
	us_callback->_volRenWin = volRenWin;
	us_callback->_volRenderer = volRenderer;
	us_callback->_augmentedRenWin = augmentedRenWin;
	us_callback->_imgImport = camImgImport;
	us_callback->_camImgTexture = camImgTexture;
	us_callback->TransformName = PlusTransformName("Probe", "Tracker" );
	us_callback->_boxWidget = box;
	us_callback->_boxTransform = boxTransform;
	us_callback->_boxPlanes = boxPlanes;
	us_callback->_transform = transform;
	us_callback->_inExMapper = inExVolumeMapper;
	us_callback->cudaMapper2 = cuda2DVolumeMapper;
	us_callback->_vol = volume;
	us_callback->_screen = this->ui->augmentedView;
	us_callback->current_mapper = "1D_MAPPER";
	us_callback->sc_capture_on = false;

	this->ui->US_View->GetInteractor()->AddObserver( vtkCommand::TimerEvent, us_callback);
	//augmentedRenWin->GetInteractor()->Initialize();

	return 0;
}

/* Initialize CV Pipeline */
int MainWindow::init_CV_Pipeline(){
	
	// TODO : Test th

	// Read the calibration file.
	std::cout << calibration_filename << std::endl;
	cv::FileStorage calib_file(calibration_filename, cv::FileStorage::READ);
	if(!calib_file.isOpened()){
		std::cout << "Could not open the left calibration file" << std::endl;
		return false;
	}

	calib_file["Intrinsics"] >> intrinsics;
	calib_file["Distortion_Parameters"] >> distortion_params;

	calib_file.release();

	std::cerr << "Calibration params:" << std::endl;
	std::cerr << "Intrinsics: " << std::endl;
	std::cerr << intrinsics << std::endl;
	std::cerr << "Distortions: " << std::endl;
	std::cerr << distortion_params << std::endl;
	
	cam_capture = cv::VideoCapture( video_filename );
	
	if( !cam_capture.isOpened() ){
		std::cerr << "Cannot open the camera video file. " << std::endl;
		return false;
	}
	
	cam_video_prop.n_frames	   = (int)cam_capture.get( CV_CAP_PROP_FRAME_COUNT );
	cam_video_prop.frame_width = (int)cam_capture.get( CV_CAP_PROP_FRAME_WIDTH );
	cam_video_prop.frame_height= (int)cam_capture.get( CV_CAP_PROP_FRAME_HEIGHT);
	cam_video_prop.framerate   = (int)cam_capture.get( CV_CAP_PROP_FPS );

	std::cout << " Playing back " << video_filename
			  << " at " << cam_video_prop.framerate << "fps" 
			  << " ["   << cam_video_prop.n_frames << ", " << cam_video_prop.frame_width
			  << "x"    << cam_video_prop.frame_height
			  << " frames]" << std::endl;

	return true;

}

/* Public slot onStartButtonClick() */
void MainWindow::onStartButtonClick(const QString &str){

	if( str == "START_BTN"){
		if(!streamingON){
			// Start timer that repeats at 1/fps
			interactorTimerID = this->ui->US_View->GetInteractor()->CreateRepeatingTimer(1000.0/frame_rate);
			this->ui->start_Button->setText(QString("Stop"));

			// Set streaming ON
			streamingON = true;
		}
		else{
			// Stop repeating timer
			this->ui->US_View->GetInteractor()->DestroyTimer( interactorTimerID );
			this->ui->start_Button->setText(QString("Start"));

			// Set streaming OFF
			streamingON = false; 
		}
	}
}

/* Public slot onScanTypeRadioButtonClikc() */
void MainWindow::onScanTypeRadioButtonClick(const QString &str){
	if(str == "SCANTYPE_BTN"){
		std::cout << this->ui->scantype_RadioButton->isChecked() << std::endl;

	}
}

/* Public slot onSaveVolumeButtonClick() */
void MainWindow::onSaveVolumeButtonClick( const QString &str){

	if( str == "SAVEVOL_BTN"){
		
		vtkImageData * data = vtkImageData::New();
		acceleratedVolumeReconstructor->GetOutputVolume( data );

		// Write volume to meta file
		metaWriter->SetInputData( data );

		auto now = std::chrono::system_clock::now();
		auto now_c = std::chrono::system_clock::to_time_t( now );
		std::stringstream ss;
		ss << outputDir << "3DUS-" << std::put_time( std::localtime( &now_c ), "%Y-%m-%d-%H%M%S");
		ss << ".mhd";
		metaWriter->SetFileName( ss.str().c_str() );

		metaWriter->Modified();
		metaWriter->Write();

		data->Delete();
	}
}

/* Public slot ontf1ButtonClick() */
void MainWindow::ontf1ButtonClick( const QString &str ){
	
	if( str == "TF1_BTN" ){

		volume->SetMapper( this->cudaVolumeMapper );
		volume->SetProperty(volumeProperty);
		volume->Modified();

		vtkCamera* renCam = vtkCamera::New();
		//setup_VTK_Camera(intrinsics, renCam);
		setup_VTK_Camera(intrinsics, this->ui->augmentedView->width(), this->ui->augmentedView->height(), renCam);
		volRenderer->SetActiveCamera( renCam );
		volRenderer->ResetCameraClippingRange();

		this->us_callback->current_mapper.assign( "1D_MAPPER" );

		augmentedRenWin->Render();
		this->ui->augmentedView->repaint();
	}
}

/* Public slot ontf2ButtonClick() */
void MainWindow::ontf2ButtonClick( const QString &str ){
	
	if( str == "TF2_BTN" ){

		volume->SetMapper( this->cuda2DVolumeMapper);

		vtkCamera* renCam = vtkCamera::New();
		//setup_VTK_Camera(intrinsics, renCam);
		setup_VTK_Camera(intrinsics, this->ui->augmentedView->width(), this->ui->augmentedView->height(), renCam);
		volRenderer->SetActiveCamera( renCam );
		volRenderer->ResetCameraClippingRange();

		this->us_callback->current_mapper.assign( "2D_MAPPER" );

		augmentedRenWin->Render();
		this->ui->augmentedView->repaint();
	}
}

/* Public slot ontfInExButtonClick */
void MainWindow::ontfInExButtonClick( const QString &str ){
	
	if( str == "TFINEX_BTN" ){
		volume->SetMapper( this->inExVolumeMapper);

		// Set up box widget to get the keyhole. Only for inExvolumemapper
		double box_size[3] = { 180, 150, 170 };
		double bounds[6] = { -box_size[0]/2, box_size[0]/2, -box_size[1]/2, box_size[1]/2,	
								-box_size[2]/2, box_size[2] };

		double box_calibration[16] = { 1, 0, 0, 20, 
										0.0, 1, 0, 10, 
										0, 0, 1, 0, 
										0, 0, 0, 1};
		transform->GetMatrix()->DeepCopy( box_calibration );
	
		box->SetPlaceFactor( 1.01 );
		//box->SetInputData( imageData );
		box->PlaceWidget( bounds );
		box->InsideOutOn();
		box->GetOutlineProperty()->SetOpacity( 1 );
		box->GetHandleProperty()->SetOpacity( 1 );
		box->GetSelectedFaceProperty()->SetOpacity( 1 );
		box->GetFaceProperty()->SetOpacity( 1 );

		volRenderer->AddViewProp( box->GetProp3D() );

		vtkCamera* renCam = vtkCamera::New();
		//setup_VTK_Camera(intrinsics, renCam);
		setup_VTK_Camera(intrinsics, this->ui->augmentedView->width(), this->ui->augmentedView->height(), renCam);
		volRenderer->SetActiveCamera( renCam );
		volRenderer->ResetCameraClippingRange();

		this->us_callback->current_mapper.assign( "INEX_MAPPER" );

		augmentedRenWin->Render();
		this->ui->augmentedView->repaint();
	}
}

/* Publick slot onScCaptureButtonClick */
void MainWindow::onScCaptureRadioButtonClick( const QString &str){

	if( str == "SCCAPTURE_BTN" ){

		if( this->ui->scCaptureButton->isChecked() ){

		// Setup for screen capturing
		windowToImage = vtkSmartPointer< vtkWindowToImageFilter >::New();
		windowToImage->SetInput( augmentedRenWin );
		windowToImage->SetMagnification( 1 );
		windowToImage->SetInputBufferTypeToRGB();
		windowToImage->ReadFrontBufferOff();

		imageWriter = vtkSmartPointer< vtkPNGWriter >::New();
		imageWriter->SetInputConnection( windowToImage->GetOutputPort() );

		this->us_callback->sc_capture_on = true;
		us_callback->_win2Img = windowToImage;
		us_callback->_imgWriter = imageWriter;
		}
		else
			this->us_callback->sc_capture_on = false;
	}

}


/* Setup VTK Camera */
void MainWindow::setup_VTK_Camera(cv::Mat mat, double win_width, double win_height, vtkCamera* vtkCam){

	double fx( mat.at<double>(0,0) ), fy( mat.at<double>(1,1) );
	double cx( mat.at<double>(0,2) ), cy( mat.at<double>(1,2) );

	double width(win_width);

	if(  win_height != cam_video_prop.frame_height || cam_video_prop.frame_width != win_width ){
		 
		double factor = static_cast<double>(win_height)/static_cast<double>(cam_video_prop.frame_height);
		fy = fy*factor;

		cx = factor*cx;

		int expectedWinSize = cvRound( factor*static_cast<double>(cam_video_prop.frame_width));
		if( expectedWinSize != win_width ){
			
			int diff = (win_width - expectedWinSize)/2;
			cx = cx + diff;
		}

		cy = factor*cy;
	}

	double center_x = (win_width - cx)/((win_width-1)/2.0) - 1;
	double center_y = cy/((win_height-1)/2.0) - 1;
	double viewAngle = 2*atan((win_height/2.0)/fy)*180/(4*atan(1.0));

	vtkCam->SetViewAngle( viewAngle );
	vtkCam->SetWindowCenter( center_x, center_y);
	vtkCam->SetPosition( 0, 0, 0); // Camera spatial position
	vtkCam->SetViewUp(0, -1, 0);
	vtkCam->SetFocalPoint(0, 0, fy);

	vtkCam->SetClippingRange(0.01, 1000.01);
	vtkCam->Modified();

	return; 
}

/* Setup VTK Camera */
void MainWindow::setup_VTK_Camera(cv::Mat mat, vtkCamera* vtkCam){

	double fx( mat.at<double>(0,0) ), fy( mat.at<double>(1,1) );
	double cx( mat.at<double>(0,2) ), cy( mat.at<double>(1,2) );

	double center_x = (cam_video_prop.frame_width - cx)/((cam_video_prop.frame_width-1)/2.0) - 1;
	double center_y = cy/((cam_video_prop.frame_height-1)/2.0) - 1;
	double viewAngle = 2*atan(((cam_video_prop.frame_height-1)/2.0)/fy)*180/(4*atan(1.0));

	vtkCam->SetViewAngle( viewAngle );
	vtkCam->SetPosition( 0, 0, 0);
	vtkCam->SetViewUp(0, -1, 0);
	vtkCam->SetFocalPoint(0, 0, fy);
	vtkCam->SetWindowCenter( center_x, center_y);
	vtkCam->SetClippingRange(0.01, 1000.01);
	vtkCam->Modified();

	return; 
}

/* Setup Volume Rendering Pipeline */
void MainWindow::setup_VolumeRendering_Pipeline(){
	
	volumeMapper = vtkSmartPointer< vtkSmartVolumeMapper >::New();
	volumeMapper->SetBlendModeToComposite();
	volumeMapper->SetRequestedRenderMode( vtkSmartVolumeMapper::GPURenderMode);

	vtkMetaImageReader *reader = vtkMetaImageReader::New();
	reader->SetFileName("3DUS.mhd");
	reader->Update();
	vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();  
	imageData->DeepCopy( reader->GetOutputDataObject( 0 ) ); 
	imageData->Modified();

	cudaVolumeMapper = vtkSmartPointer< vtkCuda1DVolumeMapper >::New();
	cudaVolumeMapper->UseFullVTKCompatibility();
	cudaVolumeMapper->SetBlendModeToComposite();

	volumeProperty = vtkSmartPointer< vtkVolumeProperty >::New();
	volumeProperty->ShadeOff();
	volumeProperty->SetInterpolationType( VTK_LINEAR_INTERPOLATION );

	compositeOpacity = vtkSmartPointer< vtkPiecewiseFunction >::New();
	/* TODO: Get these values from the UI */
	compositeOpacity->AddPoint(0.0,0.0);
	compositeOpacity->AddPoint(75.72,0.079);
	compositeOpacity->AddPoint(176.15,0.98);
	compositeOpacity->AddPoint(255.0,1.0);
	volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.

	color = vtkSmartPointer< vtkColorTransferFunction >::New();
	/* TODO: Get these values from the UI */
	color->AddRGBPoint(0.0  ,0.0,0.0,1.0);
	color->AddRGBPoint(40.0  ,0.0,0.1,0.0);
	color->AddRGBPoint(255.0,1.0,0.0,0.0);
	volumeProperty->SetColor(color);

	// Set up 2D Cuda volume mapper
	volume = vtkSmartPointer<vtkVolume>::New();
	volume->SetMapper(cudaVolumeMapper);
	volume->SetProperty(volumeProperty);
	/*vtkSmartPointer< vtkMatrix4x4 > userMatrix = vtkSmartPointer< vtkMatrix4x4 >::New();
	trackedUSFrameList->GetTrackedFrame( 0 )->GetCustomFrameTransform(PlusTransformName("Probe", "Tracker" ), 
												userMatrix );
	volume->SetUserMatrix( userMatrix ); */

	volRenderer = vtkSmartPointer< vtkRenderer >::New();
	volRenderer->SetBackground(0, 0, 0);
	volRenderer->AddViewProp( volume );
	volRenderer->SetBackgroundTexture( camImgTexture );
	volRenderer->TexturedBackgroundOn();
	volRenderer->ResetCamera();

	volRenWin = vtkSmartPointer< vtkRenderWindow >::New();
	augmentedRenWin = vtkSmartPointer< vtkRenderWindow >::New();
	//volRenWin->AddRenderer( volRenderer );
	augmentedRenWin->AddRenderer( volRenderer );
	//augmentedRenWin->AddRenderer( volRenderer );
	/* TODO: Setup the render position */
	vtkCamera* renCam = volRenderer->GetActiveCamera();
	setup_VTK_Camera(intrinsics, this->ui->augmentedView->width(), this->ui->augmentedView->height(), renCam);

	//volRenderer->SetActiveCamera( renCam );

	/*tfWidget = new transferFunctionWindowWidget( this->ui->tfControlWidget, this->ui->menuBar );
	tfWidget->SetRenderWindow( volRenWin );
	tfWidget->SetRenderer( volRenderer );
	tfWidget->SetMapper( cuda2DVolumeMapper );
	tfWidget->SetScreen( this->ui->augmentedView );
	tfWidget->Initialize();

	tfWidget->SetImageData( imageData );
	tfWidget->LoadedImageData();*/
	
	this->ui->volrenViewer->SetRenderWindow( volRenWin );
	this->ui->augmentedView->SetRenderWindow( augmentedRenWin );
}

/* Setup AR-Volume Rendering Pipeline */
void MainWindow::setup_ARVolumeRendering_Pipeline(){

	boxTransform = vtkSmartPointer< vtkTransform >::New();
	boxTransform->PostMultiply();
	//box = vtkSmartPointer< vtkBoxWidget >::New();

	usVolume = vtkImageData::New();
	int _extent[6];
	acceleratedVolumeReconstructor->GetOutputExtent( _extent );
	usVolume->SetExtent( _extent );
	usVolume->SetExtent( _extent );
	usVolume->SetDimensions( _extent[1] - _extent[0], 
							_extent[3] - _extent[2], 
							_extent[5] - _extent[4] );
	//usVolume->SetOrigin( acceleratedVolumeReconstructor->GetOrigin() );
	usVolume->SetSpacing( acceleratedVolumeReconstructor->GetSpacing() );
	usVolume->AllocateScalars( VTK_UNSIGNED_CHAR, 1);
	usVolume->Modified();
	// Initialize buffers to 0
	double volume_size = (_extent[1] - _extent[0] )*(_extent[3] - _extent[2])*(_extent[5] - _extent[4] );
	memset(usVolume->GetScalarPointer(0,0,0), 0, sizeof(unsigned char)*volume_size);
				
	vtkMetaImageReader *reader = vtkMetaImageReader::New();
	reader->SetFileName("./Data/3DUS-2016-01-14-111925.mhd");
	reader->Update();
	//usVolume->DeepCopy( reader->GetOutputDataObject( 0 ) ); 
	//usVolume->Modified();
	
	// Set US Calibration Transform
	transform = vtkSmartPointer< vtkTransform >::New();
	boxPlanes = vtkSmartPointer< vtkPlanes >::New();
	
	// Set up transfer function
	cuda2DTransferFun = vtkSmartPointer< vtkCuda2DTransferFunction >::New();
	backgroundTF = vtkSmartPointer< vtkCuda2DTransferFunction >::New();
	polyReader = vtkSmartPointer< vtkCudaFunctionPolygonReader >::New();
	polyReader->SetFileName( "./Data/tf4.2tf" );
	polyReader->Read();

	for( int i=0; i<polyReader->GetNumberOfOutputs(); i++)
		cuda2DTransferFun->AddFunctionObject( polyReader->GetOutput( i ) );

	cuda2DTransferFun->Modified();

	polyReader->SetFileName("./Data/background.2tf");
	polyReader->Read();
	for( int i=0; i<polyReader->GetNumberOfOutputs(); i++)
		backgroundTF->AddFunctionObject( polyReader->GetOutput( i ) );

	// Set up mappers
	cuda2DVolumeMapper = vtkSmartPointer< vtkCuda2DVolumeMapper >::New();
	cuda2DVolumeMapper->SetInputData( usVolume );
	cuda2DVolumeMapper->SetFunction( cuda2DTransferFun );

	// Set up mapper
	inExVolumeMapper = vtkSmartPointer< vtkCuda2DInExLogicVolumeMapper >::New();
	inExVolumeMapper->SetInputData( usVolume );
	inExVolumeMapper->SetVisualizationFunction( cuda2DTransferFun );
	inExVolumeMapper->SetInExLogicFunction( backgroundTF );
	inExVolumeMapper->SetUseBlackKeyhole( false );
	//inExVolumeMapper->SetDistanceShadingConstants( 0.0, 0.0, 0.0 );
	//inExVolumeMapper->SetCelShadingConstants( 1.0, 0.01, 0.1);
	inExVolumeMapper->Modified();

	cudaVolumeMapper = vtkSmartPointer< vtkCuda1DVolumeMapper >::New();
	cudaVolumeMapper->SetInputData( usVolume );
	cudaVolumeMapper->UseFullVTKCompatibility();
	cudaVolumeMapper->SetBlendModeToComposite();

	volumeProperty = vtkSmartPointer< vtkVolumeProperty >::New();
	volumeProperty->ShadeOff();
	volumeProperty->SetInterpolationType( VTK_LINEAR_INTERPOLATION );

	compositeOpacity = vtkSmartPointer< vtkPiecewiseFunction >::New();
	/* TODO: Get these values from the UI */
	compositeOpacity->AddPoint(0.0,0.0);
	compositeOpacity->AddPoint(75.72,0.079);
	compositeOpacity->AddPoint(176.15,0.98);
	compositeOpacity->AddPoint(255.0,1.0);
	volumeProperty->SetScalarOpacity(compositeOpacity); // composite first.

	color = vtkSmartPointer< vtkColorTransferFunction >::New();
	/* TODO: Get these values from the UI */
	color->AddRGBPoint(0.0  ,0.0,0.0,1.0);
	color->AddRGBPoint(40.0  ,0.0,0.1,0.0);
	color->AddRGBPoint(255.0,1.0,0.0,0.0);
	volumeProperty->SetColor(color);

	// Set up volume
	volume = vtkSmartPointer<vtkVolume>::New();
	double _origin[3];
	acceleratedVolumeReconstructor->GetOrigin( _origin );

	get_first_frame_position(trackedUSFrameList->GetTrackedFrame(0), repository, _origin);

	volume->SetOrigin( _origin );
	//volume->SetPosition( _origin[0], _origin[1], _origin[2] );
	volume->SetScale( 0.5, 0.5, 0.5 );
	volume->SetMapper( this->cudaVolumeMapper); // Default mapper
	ui->tf1_button->setChecked( true );

	// set up Renderer	
	volRenderer = vtkSmartPointer< vtkRenderer >::New();
	volRenderer->SetBackgroundTexture( camImgTexture );
	volRenderer->TexturedBackgroundOn();	
	volRenderer->AddViewProp( volume );

#ifdef ALIGNMENT_DEBUG
	vtkSphereSource * sphere = vtkSphereSource::New();
	sphere->SetRadius( 2 );

	vtkPolyDataMapper *mapper = vtkPolyDataMapper::New();
	mapper->SetInputConnection( sphere->GetOutputPort() );

	vtkActor *actor = vtkActor::New();
	actor->SetMapper( mapper );
	actor->SetUserTransform( boxTransform );

	volRenderer->AddActor( actor );
	camImgRenderer->AddActor( actor );
#endif

	// Set up vtk camera
	vtkCamera* renCam = vtkCamera::New();
	setup_VTK_Camera(intrinsics, this->ui->augmentedView->width(), this->ui->augmentedView->height(), renCam);
	//setup_VTK_Camera(intrinsics, renCam);
	volRenderer->SetActiveCamera( renCam );
	volRenderer->ResetCameraClippingRange();	

	// Set up render window
	augmentedRenWin = vtkSmartPointer< vtkRenderWindow >::New();
	augmentedRenWin->AddRenderer( volRenderer );
	
	this->ui->augmentedView->SetRenderWindow( augmentedRenWin );
	augmentedRenWin->Render();
	this->ui->augmentedView->repaint();
}

/* Setup Volume Reconstruction Pipeline */
int MainWindow::setup_VolumeReconstruction_Pipeline(){

	volumeReconstructor = vtkSmartPointer< vtkVolumeReconstructor >::New();
	/* Configure volumeReconstor from XML data */
	if(volumeReconstructor->ReadConfiguration( configRootElement ) != PLUS_SUCCESS){
		LOG_ERROR("Configuration incorrect for vtkVolumeReconstructor.");
		exit( EXIT_FAILURE );
	}

	repository = vtkSmartPointer< vtkTransformRepository >::New();
	/* Read Coordinate system definitions from XML data */
	if( repository->ReadConfiguration( configRootElement )  != PLUS_SUCCESS ){
		LOG_ERROR("Configuration incorrect for vtkTransformRepository.");
		exit( EXIT_FAILURE );
	}

	std::string errorDetail;
	/* Set output Extent from the input data. During streaming a sout scan may be necessary to set the output
	   extent. */
	if (volumeReconstructor->SetOutputExtentFromFrameList(trackedUSFrameList, repository, errorDetail) != PLUS_SUCCESS){
		LOG_ERROR("Setting up Output Extent from FrameList:" + errorDetail );
		exit( EXIT_FAILURE );
	}

	return PLUS_SUCCESS;
}

int MainWindow::get_extent_from_trackedList(vtkTrackedFrameList *frameList, vtkTransformRepository *repository, double spacing,  
											int * outputExtent, double * origin){

	PlusTransformName imageToReferenceTransformName;
	imageToReferenceTransformName = PlusTransformName("Image", "Tracker");

    if ( frameList == NULL ){
	     LOG_ERROR("Failed to set output extent from tracked frame list - input frame list is NULL");
	     return -1; 
	}
	
	if ( frameList->GetNumberOfTrackedFrames() == 0){
	    
	     LOG_ERROR("Failed to set output extent from tracked frame list - input frame list is empty");
	     return -1; 
	}
	 
	if ( repository == NULL ){

	     LOG_ERROR("Failed to set output extent from tracked frame list - input transform repository is NULL");
	     return -1; 
	 }
	 
	double extent_Ref[6]={
	     VTK_DOUBLE_MAX, VTK_DOUBLE_MIN,
	     VTK_DOUBLE_MAX, VTK_DOUBLE_MIN,
	     VTK_DOUBLE_MAX, VTK_DOUBLE_MIN
	};

	const int numberOfFrames = frameList->GetNumberOfTrackedFrames();
	int numberOfValidFrames = 0;
	for (int frameIndex = 0; frameIndex < numberOfFrames; ++frameIndex ) {
		
		TrackedFrame* frame = frameList->GetTrackedFrame( frameIndex );
	     
	    if ( repository->SetTransforms(*frame) != PLUS_SUCCESS ){
	       
			LOG_ERROR("Failed to update transform repository with tracked frame!"); 
			continue;
	     }
	 
	     // Get transform
	     bool isMatrixValid(false); 
	     vtkSmartPointer<vtkMatrix4x4> imageToReferenceTransformMatrix=vtkSmartPointer<vtkMatrix4x4>::New();
	     if ( repository->GetTransform(imageToReferenceTransformName, imageToReferenceTransformMatrix, &isMatrixValid ) != PLUS_SUCCESS ){
	      
			 std::string strImageToReferenceTransformName; 
			 imageToReferenceTransformName.GetTransformName(strImageToReferenceTransformName); 
			
			 LOG_WARNING("Failed to get transform '"<<strImageToReferenceTransformName<<"' from transform repository!"); 
			 continue;
	     }
	 
	     if ( isMatrixValid )
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

		   double c0[ 4 ] = { minX, minY, 0,  1 };
		   double c1[ 4 ] = { minX, maxY, 0,  1 };
	       double c2[ 4 ] = { maxX, minY, 0,  1 };
		   double c3[ 4 ] = { maxX, maxY, 0,  1 };
		   
		   corners_ImagePix.push_back( c0 );
		   corners_ImagePix.push_back( c1 );
	       corners_ImagePix.push_back( c2 );
	       corners_ImagePix.push_back( c3 );

		   // Transform the corners to Reference and expand the extent if needed
		   for ( unsigned int corner = 0; corner < corners_ImagePix.size(); ++corner ){

			   double corner_Ref[ 4 ] = { 0, 0, 0, 1 }; // position of the corner in the Reference coordinate system
			   imageToReferenceTransformMatrix->MultiplyPoint( corners_ImagePix[corner], corner_Ref );

			    for ( int axis = 0; axis < 3; axis ++ ){
					if ( corner_Ref[axis] < extent_Ref[axis*2] ){
						// min extent along this coord axis has to be decreased
						extent_Ref[axis*2]=corner_Ref[axis];
					}	
					
					if ( corner_Ref[axis] > extent_Ref[axis*2+1] ){
	        
						// max extent along this coord axis has to be increased
						extent_Ref[axis*2+1]=corner_Ref[axis];
					}
				}
		   }

	     }
	   }
	 
	   	 
	   // Set the output extent from the current min and max values, using the user-defined image resolution.
	   outputExtent[ 1 ] = int( ( extent_Ref[1] - extent_Ref[0] ) / spacing );
	   outputExtent[ 3 ] = int( ( extent_Ref[3] - extent_Ref[2] ) / spacing );
	   outputExtent[ 5 ] = int( ( extent_Ref[5] - extent_Ref[4] ) / spacing );

	   origin[0] = extent_Ref[0];
	   origin[1] = extent_Ref[2];
	   origin[2] = extent_Ref[4];

	   return 0;	   
}

void MainWindow::get_first_frame_position(TrackedFrame *frame,  vtkTransformRepository *repository, double *pos){

	PlusTransformName  transformName = PlusTransformName("Probe", "Tracker" );
	PlusTransformName  imageToReferenceTransformName = PlusTransformName("Image", "Tracker");


	vtkSmartPointer< vtkMatrix4x4 > pose = vtkSmartPointer< vtkMatrix4x4 >::New();
	vtkSmartPointer<vtkMatrix4x4> imageToReferenceTransformMatrix=vtkSmartPointer<vtkMatrix4x4>::New();

	frame->GetCustomFrameTransform( transformName, pose);

	// Get the location of the first pixel
	int* frameExtent = frame->GetImageData()->GetImage()->GetExtent();
	//double minX = 159;//frameExtent[0];
	//double minY = 41;//frameExtent[2];
	double minX = 0;
	double minY = 0;

	double c0[4]   = { minX, minY, 0,  1 };

	bool isMatrixValid(false);
	if ( repository->GetTransform(imageToReferenceTransformName, imageToReferenceTransformMatrix, &isMatrixValid ) != PLUS_SUCCESS ){
	      
			 std::string strImageToReferenceTransformName; 
			 imageToReferenceTransformName.GetTransformName(strImageToReferenceTransformName); 
			
			 LOG_WARNING("Failed to get transform '"<<strImageToReferenceTransformName<<"' from transform repository!"); 
			 return;
	}

	double corner_pos[ 4 ] = { 0, 0, 0, 1 }; // position of the corner in the Reference coordinate system
	//imageToReferenceTransformMatrix->MultiplyPoint( c0, corner_pos );
	pose->MultiplyPoint( c0, corner_pos );

	pos[0] = corner_pos[0];
	pos[1] = corner_pos[1];
	pos[2] = corner_pos[2];
}

//----------------------------------------------------------------------------
void vtkUSEventCallback::Execute(vtkObject *caller, unsigned long, void*)
{
  vtkSmartPointer< vtkMatrix4x4 > tFrame2Tracker = vtkSmartPointer< vtkMatrix4x4 >::New();

  // Repeat playback
  if( index > trackedFrames->GetNumberOfTrackedFrames()-1 )
  {
    index = 0;
  }

  TrackedFrame *trackedFrame = trackedFrames->GetTrackedFrame( index );

  /*    // Update transform repository
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

  if( !std::strcmp( current_mapper.c_str(), "1D_MAPPER") )
  {
    usVolume->Modified();
    cudaMapper->SetInputData( usVolume );
    cudaMapper->Modified();

#ifdef ALIGNMENT_DEBUG
    _boxTransform->Identity();
    _boxTransform->Concatenate( tFrame2Tracker );
    _boxTransform->Update();
#endif
  }
  else if ( !std::strcmp( current_mapper.c_str(), "2D_MAPPER") )
  {
    cudaMapper2->SetInputData( usVolume );
    cudaMapper2->Modified();
    //cudaMapper2->Update();
  }
  else if( !std::strcmp( current_mapper.c_str(), "INEX_MAPPER") )
  {
    _inExMapper->SetInputData( usVolume );
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
  if ( trackedFrame->GetImageData()->IsImageValid() )
  {
    // Display image if it's valid
    if (trackedFrame->GetImageData()->GetImageType()==US_IMG_BRIGHTNESS || trackedFrame->GetImageData()->GetImageType()==US_IMG_RGB_COLOR)
    {

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
      && status == FIELD_OK )
    {

      trackedFrame->GetCustomFrameTransform(TransformName, tFrame2Tracker);

      if ( !std::strcmp( current_mapper.c_str(), "INEX_MAPPER") )
      {
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
    else
    {

      std::string strTransformName;
      TransformName.GetTransformName(strTransformName);
      ss  << "Transform '" << strTransformName << "' is invalid ...";
    }
  }

  // Render camera image
  cv::Mat cam_frame;
  bool success = _camCapture->read( cam_frame );
  int c_frame = _camCapture->get( CV_CAP_PROP_POS_FRAMES);

  if( c_frame < n_frames-2 )
  {

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

    if( sc_capture_on )
    {

      // Captures the screen and save the image
      std::string dir = "./screen_captures/";
      std::string prefix = "CAP_";
      std::string ext = ".png";

      _win2Img->Update();
      _win2Img->Modified();
      std::stringstream ss;
      ss << index;
      std::string filename = dir + prefix + ss.str() + ext;
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
  else
  {
    index = 0;
    _camCapture->set( CV_CAP_PROP_POS_FRAMES, index);
  }

  index++;
}
