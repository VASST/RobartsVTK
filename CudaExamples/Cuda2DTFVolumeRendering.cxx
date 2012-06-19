////For transfer function
//#include "vtkCuda2DTransferFunction.h"
//#include "vtkCudaFunctionPolygon.h"
//#include "vtkCudaFunctionPolygonReader.h"
//
////For ray-caster
//#include "vtkCudaDualImageVolumeMapper.h"
//#include "vtkVolume.h"
//#include "vtkVolumeProperty.h"
//#include "vtkPiecewiseFunction.h"
//#include "vtkColorTransferFunction.h"
//#include "vtkBoxWidget.h"
//#include "vtkCommand.h"
//#include "vtkPlanes.h"
//
////general use
//#include "vtkImageData.h"
//#include "vtkImageAppendComponents.h"
//#include "vtkMetaImageReader.h"
//#include "vtkActor.h"
//#include "vtkRenderer.h"
//#include "vtkRenderWindow.h"
//#include "vtkRenderWindowInteractor.h"
//#include "vtkProperty.h"
//#include "vtkCamera.h"
//
//// ---------------------------------------------------------------------------------------------------
//// Callback for moving the planes from the box widget to the mapper
//class vtkBoxWidgetCallback : public vtkCommand
//{
//public:
//  static vtkBoxWidgetCallback *New()
//    { return new vtkBoxWidgetCallback; }
//  virtual void Execute(vtkObject *caller, unsigned long, void*)
//    {
//      vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
//      if (this->Mapper)
//        {
//        vtkPlanes *planes = vtkPlanes::New();
//        widget->GetPlanes(planes);
//        this->Mapper->SetKeyholePlanes(planes);
//        planes->Delete();
//        }
//    }
//  void SetMapper(vtkCudaDualImageVolumeMapper* m) 
//    { this->Mapper = m; }
//
//protected:
//  vtkBoxWidgetCallback() 
//    { this->Mapper = 0; }
//
//  vtkCudaDualImageVolumeMapper *Mapper;
//};
//
//// ---------------------------------------------------------------------------------------------------
//// Main Program
//int main(int argc, char** argv){
//
//	//retrieve the image
//	std::cout << "Enter META image filename" << std::endl;
//	std::string filename = "";
//	//std::getline(std::cin, filename);
//	//filename = "E:\\jbaxter\\data\\Chamberlain.mhd";
//	//filename = "E:\\jbaxter\\data\\4D Heart\\JTM45-flip-p01.mhd";
//	//filename = "E:\\jbaxter\\4J\\Tumor2_hyperintense.mhd";
//	filename = "E:\\jbaxter\\data\\brain\\t1undiffused.mhd";
//	vtkMetaImageReader* imReader1 = vtkMetaImageReader::New();
//	imReader1->SetFileName(filename.c_str());
//	imReader1->Update();
//	
//	filename = "E:\\jbaxter\\data\\brain\\t2undiffused.mhd";
//	vtkMetaImageReader* imReader2 = vtkMetaImageReader::New();
//	imReader2->SetFileName(filename.c_str());
//	imReader2->Update();
//
//	vtkImageAppendComponents* appender = vtkImageAppendComponents::New();
//	appender->SetInput(0,imReader1->GetOutput());
//	appender->SetInput(1,imReader2->GetOutput());
//	appender->Update();
//
//	//create the transfer function (or load from file for 2D)
//	std::cout << "Enter 2D transfer function (.2tf) filename" << std::endl;
//	//std::getline(std::cin, filename);
//	//filename = "E:\\jbaxter\\data\\Chamberlain.2tf";
//	//filename = "E:\\jbaxter\\data\\4D Heart\\heart.2tf";
//	filename = "E:\\jbaxter\\data\\brain\\testingDual.2tf";
//	//filename = "E:\\jbaxter\\4J\\Tumour.2tf";
//	vtkCuda2DTransferFunction* viz = vtkCuda2DTransferFunction::New();
//	vtkCudaFunctionPolygonReader* vizReader = vtkCudaFunctionPolygonReader::New();
//	vizReader->SetFileName(filename.c_str());
//	vizReader->Read();
//	for(int i = 0; i < vizReader->GetNumberOfOutputs(); i++){
//		viz->AddFunctionObject( vizReader->GetOutput(i) );
//	}
//	viz->Modified();
//
//	//create the transfer function (or load from file for 2D)
//	std::cout << "Enter 2D INEX transfer function (.2tf) filename" << std::endl;
//	//std::getline(std::cin, filename);
//	filename = "E:\\jbaxter\\4J\\J-WholeHead.2tf";
//	vtkCuda2DTransferFunction* inex = vtkCuda2DTransferFunction::New();
//	vtkCudaFunctionPolygonReader* inexReader = vtkCudaFunctionPolygonReader::New();
//	inexReader->SetFileName(filename.c_str());
//	inexReader->Read();
//	for(int i = 0; i < inexReader->GetNumberOfOutputs(); i++){
//		inexReader->GetOutput(i)->SetOpacity(1);
//		inex->AddFunctionObject( inexReader->GetOutput(i) );
//	}
//	inex->Modified();
//
//	//assemble the ray caster
//	vtkCudaDualImageVolumeMapper* mapper = vtkCudaDualImageVolumeMapper::New();
//	mapper->SetDevice(0);
//	mapper->SetInput( appender->GetOutput() );
//	//mapper->SetInput( imReader1->GetOutput() );
//	mapper->SetDistanceShadingConstants(0.0,0.0,1.0);
//	mapper->SetFunction( viz );
//	//mapper->SetVisualizationFunction( viz );
//	//mapper->SetInExLogicFunction( inex );
//
//	//assemble the VTK pipeline
//	vtkVolume* volume = vtkVolume::New();
//	volume->SetMapper( mapper );
//	vtkRenderer* renderer = vtkRenderer::New();
//	renderer->AddVolume( volume );
//	renderer->ResetCamera();
//	renderer->SetBackground(1.0,1.0,1.0);
//	vtkRenderWindow* window = vtkRenderWindow::New();
//	window->AddRenderer( renderer );
//	window->Render();
//
//	//apply keyhole planes widget and interactor
//	vtkRenderWindowInteractor* interactor = vtkRenderWindowInteractor::New();
//	interactor->SetRenderWindow( window );
//	vtkBoxWidget* clippingPlanes = vtkBoxWidget::New();
//	clippingPlanes->SetInteractor( interactor );
//	clippingPlanes->SetPlaceFactor(1.01);
//	clippingPlanes->SetInput( appender->GetOutput() );
//	clippingPlanes->SetDefaultRenderer(renderer);
//	clippingPlanes->InsideOutOn();
//	clippingPlanes->PlaceWidget();
//	vtkBoxWidgetCallback *callback = vtkBoxWidgetCallback::New();
//	callback->SetMapper( mapper );
//	clippingPlanes->AddObserver(vtkCommand::InteractionEvent, callback);
//	callback->Delete();
//	clippingPlanes->EnabledOn();
//	clippingPlanes->GetSelectedFaceProperty()->SetOpacity(0.0);
//
//	//start the process
//	interactor->Initialize();
//	interactor->Start();
//
//	//clean up pipeline
//	interactor->Delete();
//	window->Delete();
//	volume->Delete();
//	renderer->Delete();
//	clippingPlanes->Delete();
//	mapper->Delete();
//	viz->Delete();
//	vizReader->Delete();
//	appender->Delete();
//	imReader1->Delete();
//	imReader2->Delete();
//
//
//}



///** @file main.cxx
// *
// *  @brief Sample volume ray casting program
// *
// *  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
// *  @note First documented on June 14, 2012
// *
// */
//
////For ray-caster
//#include "vtkCuda1DVolumeMapper.h"
//#include "vtkVolume.h"
//#include "vtkVolumeProperty.h"
//#include "vtkPiecewiseFunction.h"
//#include "vtkColorTransferFunction.h"
//#include "vtkBoxWidget.h"
//#include "vtkCommand.h"
//#include "vtkPlanes.h"
//
////for VTK ray caster
//#include "vtkGPUVolumeRayCastMapper.h"
//
////general use
//#include "vtkImageData.h"
//#include "vtkMetaImageReader.h"
//#include "vtkActor.h"
//#include "vtkRenderer.h"
//#include "vtkRenderWindow.h"
//#include "vtkRenderWindowInteractor.h"
//#include "vtkProperty.h"
//#include "vtkCamera.h"
//
//// ---------------------------------------------------------------------------------------------------
//// Callback for moving the planes from the box widget to the mapper
//class vtkBoxWidgetKeyholeCallback : public vtkCommand
//{
//public:
//  static vtkBoxWidgetKeyholeCallback *New()
//    { return new vtkBoxWidgetKeyholeCallback; }
//  virtual void Execute(vtkObject *caller, unsigned long, void*)
//    {
//      vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
//      if (this->Mapper)
//        {
//        vtkPlanes *planes = vtkPlanes::New();
//        widget->GetPlanes(planes);
//        this->Mapper->SetClippingPlanes(planes);
//        planes->Delete();
//        }
//    }
//  void SetMapper(vtkCuda1DVolumeMapper* m) 
//    { this->Mapper = m; }
//
//protected:
//  vtkBoxWidgetKeyholeCallback() 
//    { this->Mapper = 0; }
//
//  vtkCuda1DVolumeMapper *Mapper;
//};
//
//// ---------------------------------------------------------------------------------------------------
//// Main Program
//int main(int argc, char** argv){
//
//	//retrieve the image
//	vtkMetaImageReader* imReader = vtkMetaImageReader::New();
//	imReader->SetFileName("E:\\jbaxter\\data\\Chamberlain.mhd");
//	imReader->Update();
//
//	//create the transfer function
//	vtkPiecewiseFunction* opacityFun = vtkPiecewiseFunction::New();
//	opacityFun->AddPoint(0,0);
//	opacityFun->AddPoint(1000,1);
//	vtkColorTransferFunction* colourFun = vtkColorTransferFunction::New();
//	colourFun->AddRGBPoint(0,1,0,0);
//
//	//assemble the ray caster
//	vtkCuda1DVolumeMapper* mapper1 = vtkCuda1DVolumeMapper::New();
//	mapper1->SetInput( imReader->GetOutput() );
//	mapper1->SetGradientShadingConstants( 0.0 );
//	vtkGPUVolumeRayCastMapper* mapper2 = vtkGPUVolumeRayCastMapper::New();
//	mapper2->SetInput( imReader->GetOutput() );
//
//	//assemble the VTK pipeline
//	vtkVolume* volume1 = vtkVolume::New();
//	volume1->SetMapper( mapper1 );
//	vtkVolume* volume2 = vtkVolume::New();
//	volume2->SetMapper( mapper2 );
//	vtkVolumeProperty* volProperty1 = vtkVolumeProperty::New();
//	volProperty1->SetScalarOpacity(opacityFun);
//	volProperty1->SetColor(colourFun);
//	volProperty1->Modified();
//	volume1->SetProperty(volProperty1);
//	volume1->Update();
//	vtkVolumeProperty* volProperty2 = vtkVolumeProperty::New();
//	volProperty2->SetScalarOpacity(opacityFun);
//	volProperty2->SetColor(colourFun);
//	volProperty2->Modified();
//	volume2->SetProperty(volProperty2);
//	volume2->Update();
//	vtkRenderer* renderer1 = vtkRenderer::New();
//	renderer1->AddVolume( volume1 );
//	renderer1->ResetCamera();
//	renderer1->SetBackground(1.0,1.0,1.0);
//	renderer1->SetViewport(0.5,0,1,1);
//	vtkRenderer* renderer2 = vtkRenderer::New();
//	renderer2->SetViewport(0,0,0.5,1);
//	renderer2->AddVolume( volume2 );
//	renderer2->ResetCamera();
//	renderer2->SetBackground(1.0,1.0,1.0);
//	vtkRenderWindow* window = vtkRenderWindow::New();
//	window->AddRenderer( renderer1 );
//	window->AddRenderer( renderer2 );
//	window->Render();
//
//	//apply clipping planes widget and interactor
//	vtkRenderWindowInteractor* interactor = vtkRenderWindowInteractor::New();
//	interactor->SetRenderWindow( window );
//	vtkBoxWidget* viewBox = vtkBoxWidget::New();
//	viewBox->SetInteractor( interactor );
//	viewBox->SetPlaceFactor(1.01);
//	viewBox->SetInput( imReader->GetOutput() );
//	viewBox->SetDefaultRenderer(renderer1);
//	viewBox->InsideOutOn();
//	viewBox->PlaceWidget();
//	vtkBoxWidgetKeyholeCallback *viewBoxCallback = vtkBoxWidgetKeyholeCallback::New();
//	viewBoxCallback->SetMapper( mapper1 );
//	viewBox->AddObserver(vtkCommand::InteractionEvent, viewBoxCallback);
//	viewBoxCallback->Delete();
//	viewBox->EnabledOn();
//	viewBox->GetSelectedFaceProperty()->SetOpacity(0.0);
//	viewBox->GetHandleProperty()->SetOpacity(1);
//	viewBox->GetFaceProperty()->SetColor(0.0, 0.0, 0.0);
//
//
//	//start the process
//	interactor->Initialize();
//	interactor->Start();
//
//	//clean up pipeline
//	interactor->Delete();
//	window->Delete();
//	renderer1->Delete();
//	renderer2->Delete();
//	viewBox->Delete();
//	volume1->Delete();
//	mapper1->Delete();
//	volume2->Delete();
//	mapper2->Delete();
//	imReader->Delete();
//
//
//}

/** @file main.cxx
*
*  @brief Sample volume ray casting program
*
*  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
*  @note First documented on June 14, 2012
*
*/

//For ray-caster
#include "vtkCuda1DVolumeMapper.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkPiecewiseFunction.h"
#include "vtkColorTransferFunction.h"
#include "vtkBoxWidget.h"
#include "vtkCommand.h"
#include "vtkPlanes.h"

//for VTK ray caster
#include "vtkGPUVolumeRayCastMapper.h"

//general use
#include "vtkImageData.h"
#include "vtkMetaImageReader.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkProperty.h"
#include "vtkCamera.h"
#include <vtkSmartPointer.h>

// ---------------------------------------------------------------------------------------------------
// Callback for moving the planes from the box widget to the mapper
class vtkBoxWidgetKeyholeCallback : public vtkCommand
    {
    public:
        static vtkBoxWidgetKeyholeCallback *New()
            { return new vtkBoxWidgetKeyholeCallback; }
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
        void SetMapper(vtkCuda1DVolumeMapper* m) 
            { this->Mapper = m; }

    protected:
        vtkBoxWidgetKeyholeCallback() 
            { this->Mapper = 0; }

        vtkCuda1DVolumeMapper *Mapper;
    };

// ---------------------------------------------------------------------------------------------------
// Main Program
int main(int argc, char** argv){

    //retrieve the image
    vtkSmartPointer< vtkMetaImageReader > imReader = 
        vtkSmartPointer< vtkMetaImageReader >::New();
    //imReader->SetFileName("C:\\external\\src\\4Elvis-VolumeMapper\\Chamberlain.mhd");
    imReader->SetFileName("CT-chest.mhd" );
    imReader->Update();

    //create the transfer function
    vtkSmartPointer< vtkPiecewiseFunction > opacityFun = 
        vtkSmartPointer< vtkPiecewiseFunction >::New();

    opacityFun->AddPoint(0,    0.00);
    opacityFun->AddPoint(500,  0.15);
    opacityFun->AddPoint(1000, 0.15);
    opacityFun->AddPoint(1150, 0.85);

    vtkSmartPointer< vtkColorTransferFunction > colourFun = 
        vtkSmartPointer< vtkColorTransferFunction >::New();
    colourFun->AddRGBPoint(0,    0.0, 0.0, 0.0);
    colourFun->AddRGBPoint(500,  1.0, 0.5, 0.3);
    colourFun->AddRGBPoint(1000, 1.0, 0.5, 0.3);
    colourFun->AddRGBPoint(1150, 1.0, 1.0, 0.9);
    //assemble the ray caster
    vtkSmartPointer< vtkCuda1DVolumeMapper > mapper1 = 
        vtkSmartPointer< vtkCuda1DVolumeMapper >::New();
    mapper1->SetInput( imReader->GetOutput() );
    mapper1->SetGradientShadingConstants( 0.0 );

    vtkSmartPointer< vtkGPUVolumeRayCastMapper > mapper2 = 
        vtkSmartPointer< vtkGPUVolumeRayCastMapper >::New();
    mapper2->SetInput( imReader->GetOutput() );

    //assemble the VTK pipeline
    vtkSmartPointer< vtkVolume > volume1 = 
        vtkSmartPointer< vtkVolume >::New();
    volume1->SetMapper( mapper1 );

    vtkSmartPointer< vtkVolume > volume2 = 
        vtkSmartPointer< vtkVolume >::New();
    volume2->SetMapper( mapper2 );

    vtkSmartPointer< vtkVolumeProperty > volProperty1 = 
        vtkSmartPointer< vtkVolumeProperty >::New();
    volProperty1->SetScalarOpacity(opacityFun);
    volProperty1->SetGradientOpacity(opacityFun);
	volProperty1->DisableGradientOpacityOn();
    volProperty1->SetColor(colourFun);
	volProperty1->SetShade(1);
	volProperty1->SetAmbient(0.2);
	volProperty1->SetDiffuse(0);
	volProperty1->SetSpecular(1);
	volProperty1->SetSpecularPower(1.0);
    volProperty1->Modified();
    volume1->SetProperty(volProperty1);
    volume1->Update();

    vtkSmartPointer< vtkVolumeProperty > volProperty2 = 
        vtkSmartPointer< vtkVolumeProperty >::New();
    volProperty2->SetScalarOpacity(opacityFun);
    volProperty2->SetGradientOpacity(opacityFun);
	volProperty2->DisableGradientOpacityOn();
    volProperty2->SetColor(colourFun);
	volProperty2->SetShade(1);
	volProperty2->SetAmbient(0.2);
	volProperty2->SetDiffuse(0);
	volProperty2->SetSpecular(1);
	volProperty2->SetSpecularPower(1.0);
    volProperty2->Modified();
    volume2->SetProperty(volProperty2);
    volume2->Update();


    vtkSmartPointer< vtkRenderer > renderer1 = 
        vtkSmartPointer< vtkRenderer >::New();
    renderer1->AddVolume( volume1 );
    renderer1->ResetCamera();
    renderer1->SetViewport(0.5,0,1,1);


    vtkSmartPointer< vtkRenderer > renderer2 = 
        vtkSmartPointer< vtkRenderer >::New();
    renderer2->SetViewport(0,0,0.5,1);
    renderer2->AddVolume( volume2 );
    renderer2->ResetCamera();
    vtkSmartPointer< vtkRenderWindow > window = 
        vtkSmartPointer< vtkRenderWindow >::New();
    window->AddRenderer( renderer1 );
    window->AddRenderer( renderer2 );
    window->Render();

    //apply clipping planes widget and interactor
    vtkSmartPointer< vtkRenderWindowInteractor > interactor = 
        vtkSmartPointer< vtkRenderWindowInteractor >::New();
    interactor->SetRenderWindow( window );
    /*
    vtkBoxWidget* viewBox = vtkBoxWidget::New();
    viewBox->SetInteractor( interactor );
    viewBox->SetPlaceFactor(1.01);
    viewBox->SetInput( imReader->GetOutput() );
    viewBox->SetDefaultRenderer(renderer1);
    viewBox->InsideOutOn();
    viewBox->PlaceWidget();
    vtkBoxWidgetKeyholeCallback *viewBoxCallback = vtkBoxWidgetKeyholeCallback::New();
    viewBoxCallback->SetMapper( mapper1 );
    viewBox->AddObserver(vtkCommand::InteractionEvent, viewBoxCallback);
    viewBoxCallback->Delete();
    viewBox->EnabledOn();
    viewBox->GetSelectedFaceProperty()->SetOpacity(0.0);
    viewBox->GetHandleProperty()->SetOpacity(1);
    viewBox->GetFaceProperty()->SetColor(0.0, 0.0, 0.0);
    */

    //start the process
    interactor->Initialize();
    interactor->Start();

    //clean up pipeline
    /*
    interactor->Delete();
    window->Delete();
    renderer1->Delete();
    renderer2->Delete();
    viewBox->Delete();
    volume1->Delete();
    mapper1->Delete();
    volume2->Delete();
    mapper2->Delete();
    imReader->Delete();
    */


    }