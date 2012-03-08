//This is a set of sample code showing how to use the
//CUDA accelerated ray caster and segmentor.

//For transfer function
#include "vtkCuda2DTransferClassificationFunction.h"
#include "vtkCudaFunctionSquare.h"

//For ray-caster
#include "vtkCudaVolumeMapper.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPointData.h"
#include "vtkBoxWidget.h"
#include "vtkCommand.h"
#include "vtkPlanes.h"
#include "vtkImageMask.h"

//For segmentor
#include "vtkCudaSegmentor.h"
#include "vtkCleanPolyData.h"

//general use
#include "vtkImageData.h"
#include "vtkPolyData.h"
#include "vtkAlgorithm.h"
#include "vtkMetaImageReader.h"
#include "vtkPolyDataMapper.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkProperty.h"
#include "vtkCamera.h"

// Callback for moving the planes from the box widget to the mapper
class vtkBoxWidgetCallback : public vtkCommand
{
public:
  static vtkBoxWidgetCallback *New()
    { return new vtkBoxWidgetCallback; }
  virtual void Execute(vtkObject *caller, unsigned long, void*)
    {
      vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
      if (this->Mapper)
        {
        vtkPlanes *planes = vtkPlanes::New();
        widget->GetPlanes(planes);
        this->Mapper->SetClippingPlanes(planes);
        planes->Delete();
        }
    }
  void SetMapper(vtkCudaVolumeMapper* m) 
    { this->Mapper = m; }

protected:
  vtkBoxWidgetCallback() 
    { this->Mapper = 0; }

  vtkCudaVolumeMapper *Mapper;
};

int main(int argc, char** argv){

	// parse input parameters
	int count = 1;
	bool clip = 0;
	bool temporal = 1;
	bool interact = 0;
	bool segment = 0;
	int dness = 2;
	while ( count < argc ){
		if ( !strcmp( argv[count], "-Clip") ){
			clip = 1;
			count++;
		}else if ( !strcmp( argv[count], "-2D") ){
			dness = 2;
			count++;
		}else if ( !strcmp( argv[count], "-3D") ){
			dness = 3;
			count++;
		}else if ( !strcmp( argv[count], "-Dynamic") ){
			temporal = 1;
			count++;
		}else if ( !strcmp( argv[count], "-Interact") ){
			interact = 1;
			count++;
		}else if ( !strcmp( argv[count], "-Segment") ){
			segment = 1;
			count++;
		}else{
			count ++;
		}
	}
	
	//create a renderer to add the scene to
	vtkRenderer* renderer = vtkRenderer::New();
	vtkImageData* input = 0;

	vtkVolume *volume = 0;
	vtkCudaVolumeMapper* rayCaster = 0;

	vtkCudaSegmentor* segmentor = 0;
	vtkCleanPolyData* redundancyRemoveHeart = 0;
	vtkCleanPolyData* redundancyRemoveInterior = 0;
	vtkPolyDataMapper* mapperHeart = 0;
	vtkActor* actorHeart = 0;
	vtkPolyDataMapper* mapperInterior = 0;
	vtkActor* actorInterior = 0;

// ---------- TRANSFER FUNCTION SECTION ---------- 
	//Note: defining a transfer function is common to both the segmentor and volume ray caster
	
	//define a transfer function
	vtkCuda2DTransferClassificationFunction* transferFunction = vtkCuda2DTransferClassificationFunction::New();
	
	vtkCudaFunctionSquare* heartSection = vtkCudaFunctionSquare::New();
	vtkCudaFunctionSquare* interiorSection = vtkCudaFunctionSquare::New();
	
	heartSection->SetSizeAndPosition( 800, 1600, 0, 52 ); //define the bounds of the intensity and gradient
	interiorSection->SetSizeAndPosition( 1600, 3000, 0, 26 );

	heartSection->SetColour( 1.0, 0.75, 0.8 ); //set the colour to pink
	interiorSection->SetColour( 1.0, 0.0, 0.0 );

	heartSection->SetOpacity( 0.99 ); //set the opacity to visible
	interiorSection->SetOpacity( 0.99 ); 

	heartSection->SetIdentifier( 1 ); //give it a unique identification number
	interiorSection->SetIdentifier( 2 );

	transferFunction->AddFunctionObject( heartSection ); //add it to the transfer/classification function to use
	transferFunction->AddFunctionObject( interiorSection );
	transferFunction->SignalUpdate();



// ---------- SEGMENTATION SECTION ---------- 

	if(segment){
		
		//collect the file name and information
		char* fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p01.mhd")+1];
		sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p01.mhd" );

		//load in the image data
		vtkMetaImageReader* metaReader = vtkMetaImageReader::New();
		metaReader->SetFileName(fileName);
		metaReader->Update();
		input = metaReader->GetOutput();
		
		//create the segmentor and give it the required input, triangulation and classification function information
		//note: updating the segmentor triggers the segmentation process
		segmentor = vtkCudaSegmentor::New();
		segmentor->SetInput(input);
		segmentor->SetFunction(transferFunction);
		segmentor->SetConstraintThreshold( 0.5 );
		segmentor->SetSmoothingCoefficient( 0.2 );
		segmentor->SetConstraintType( 1 ); //set constraints to implicit
		segmentor->SetMeshingType( 64 ); //set it to use 64 case face generation (rather than 12 case face generation)
		segmentor->Update();

		//display the output (reduce it first for best results)
		//note: the segmentor creates a separate polydata for each surface
		redundancyRemoveHeart = vtkCleanPolyData::New();
		redundancyRemoveInterior = vtkCleanPolyData::New();
		redundancyRemoveHeart->SetInput(segmentor->GetOutput(1));
		redundancyRemoveInterior->SetInput(segmentor->GetOutput(2));
		vtkPolyData* outputHeart = redundancyRemoveHeart->GetOutput();
		vtkPolyData* outputInterior = redundancyRemoveInterior->GetOutput();

		//declare the remainder of the pipeline
		mapperHeart = vtkPolyDataMapper::New();
		actorHeart = vtkActor::New();
		mapperInterior = vtkPolyDataMapper::New();
		actorInterior = vtkActor::New();

		//format the remainder of the pipeline
		mapperHeart->SetInput(outputHeart);
		actorHeart->SetMapper(mapperHeart);
		actorHeart->GetProperty()->SetColor(1,0.75,0.8);
		actorHeart->GetProperty()->SetOpacity(0.05);

		mapperInterior->SetInput(outputInterior);
		actorInterior->SetMapper(mapperInterior);
		actorInterior->GetProperty()->SetColor(1,0,0);
		actorInterior->GetProperty()->SetOpacity(1.0);

		renderer->AddActor(actorHeart);
		renderer->AddActor(actorInterior);


// ---------- RAY CASTING SECTION ----------
	}else{
		//define the volume
		volume = vtkVolume::New();

		//define the ray caster and set it to use the transfer function
		rayCaster = vtkCudaVolumeMapper::New();
		rayCaster->SetFunction( transferFunction );

		//set the shading parameters (defaults: no shading) and input
		rayCaster->SetGoochShadingConstants( 0.4, 3, 1 );
		rayCaster->SetGradientShadingConstants( 0.3 );
		rayCaster->SetDepthShadingConstants( 0.8 );

		//collect the file name and information
		if(!temporal){ //if we are only displaying one image

			//collect the file name and information
			char* fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p01.mhd")+1];
			sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p01.mhd" );

			//load in the image data
			vtkMetaImageReader* metaReader = vtkMetaImageReader::New();
			metaReader->SetFileName(fileName);
			metaReader->Update();
			input = metaReader->GetOutput();
			rayCaster->SetInput(input,0);
			rayCaster->SetNumberOfFrames(1);

		}else{ //if we are displaying a 4D dataset

			for(int i = 0; i < 20; i++){

				//change filename to increment in heart cycle
				char* fileName;
				if(i == 0){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p01.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p01.mhd" );
				}else if(i == 1){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p02.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p02.mhd" );
				}else if(i == 2){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p03.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p03.mhd" );
				}else if(i == 3){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p04.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p04.mhd" );
				}else if(i == 4){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p05.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p05.mhd" );
				}else if(i == 5){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p06.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p06.mhd" );
				}else if(i == 6){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p07.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p07.mhd" );
				}else if(i == 7){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p08.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p08.mhd" );
				}else if(i == 8){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p09.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p09.mhd" );
				}else if(i == 9){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p10.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p10.mhd" );
				}else if(i == 10){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p11.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p11.mhd" );
				}else if(i == 11){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p12.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p12.mhd" );
				}else if(i == 12){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p13.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p13.mhd" );
				}else if(i == 13){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p14.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p14.mhd" );
				}else if(i == 14){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p15.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p15.mhd" );
				}else if(i == 15){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p16.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p16.mhd" );
				}else if(i == 16){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p17.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p17.mhd" );
				}else if(i == 17){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p18.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p18.mhd" );
				}else if(i == 18){
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p19.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p19.mhd" );
				}else{
					fileName = new char[strlen("E:\\jbaxter\\data\\metas\\JTM45-p20.mhd")+1];
					sprintf( fileName, "%s", "E:\\jbaxter\\data\\metas\\JTM45-p20.mhd" );
				}

				//load in the image data
				vtkMetaImageReader *metaReader = vtkMetaImageReader::New();
				metaReader->SetFileName(fileName);
				metaReader->Update();
				input = metaReader->GetOutput();
				vtkAlgorithm* reader=metaReader;
				rayCaster->SetInput( input, i );
			}

			//set the mapper's frame parameters
			rayCaster->SetNumberOfFrames(20);
			rayCaster->SetFrameRate(20.0);
		}

		//format the remainder of the visualization pipeline
		volume->SetMapper(rayCaster);
		renderer->AddVolume(volume);
	}

//------------- DISPLAY -----------------

	// Add a box widget
	vtkRenderWindow *renWin = vtkRenderWindow::New();
	vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();
	renWin->AddRenderer(renderer);
	iren->SetRenderWindow(renWin);

	//set up a clipping box
	if(clip && !segment){
		vtkBoxWidget *box = vtkBoxWidget::New();
		box->SetInteractor(iren);
		box->SetPlaceFactor(1.01);
		box->SetInput(input);
			
		box->SetDefaultRenderer(renderer);
		box->InsideOutOn();
		box->PlaceWidget();
		vtkBoxWidgetCallback *callback = vtkBoxWidgetCallback::New();
		callback->SetMapper(rayCaster);
		box->AddObserver(vtkCommand::InteractionEvent, callback);
		callback->Delete();
		box->EnabledOn();
		box->GetSelectedFaceProperty()->SetOpacity(0.0);
	}
	
	//set to render in 3D
	if(dness == 3){
		renWin->StereoCapableWindowOn();
		renWin->SetStereoTypeToInterlaced();
		renWin->StereoRenderOn();
		if(!segment) rayCaster->SetRenderOutputScaleFactor( 2.0 );
	}else{
		if(!segment) rayCaster->SetRenderOutputScaleFactor( 1.2 );
	}

	renderer->SetBackground(1,1,1);
	renWin->SetSize(1200, 900);
	renderer->ResetCamera();
	renderer->GetActiveCamera()->Azimuth(180.0);
	renWin->Render();

	//if we have a dynamic demo, do a rotation first
	if(temporal){
		for(int i = 0; i < 720; i++){
			renderer->GetActiveCamera()->Azimuth(0.5);
			renWin->Render();
		}
	}

	//start the render window
	if(interact){
		iren->Initialize();
		iren->Start();
	}

	//clear the transfer function
	transferFunction->Delete();
	heartSection->Delete();
	interiorSection->Delete();

	//clear the display pipeline
	iren->Delete();
	renWin->Delete();
	renderer->Delete();

	if(segment){
		//clear the segmentation pipeline
		segmentor->Delete();
		redundancyRemoveHeart->Delete();
		redundancyRemoveInterior->Delete();
		mapperHeart->Delete();
		actorHeart->Delete();
		mapperInterior->Delete();
		actorInterior->Delete();
	}else{
		//clear the ray cast pipeline
		rayCaster->Delete();
		volume->Delete();
	}


	return 0;
}