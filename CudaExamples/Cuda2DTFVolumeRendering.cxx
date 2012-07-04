//For transfer function
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaFunctionPolygon.h"
#include "vtkCudaFunctionPolygonReader.h"

//For ray-caster
#include "vtkCudaDualImageVolumeMapper.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkPiecewiseFunction.h"
#include "vtkColorTransferFunction.h"
#include "vtkBoxWidget.h"
#include "vtkCommand.h"
#include "vtkPlanes.h"

//general use
#include "vtkImageData.h"
#include "vtkMetaImageReader.h"
#include "vtkImageAppendComponents.h"
#include "vtkActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkProperty.h"
#include "vtkCamera.h"

// ---------------------------------------------------------------------------------------------------
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
	 this->Mapper->SetKeyholePlanes(planes);
	 planes->Delete();
	 }
}
void SetMapper(vtkCudaDualImageVolumeMapper* m) 
 { this->Mapper = m; }

protected:
vtkBoxWidgetCallback() 
 { this->Mapper = 0; }

vtkCudaDualImageVolumeMapper *Mapper;
};

// ---------------------------------------------------------------------------------------------------
// Main Program
int main(int argc, char** argv){

	 //retrieve the first image
	 std::cout << "Enter META image1 filename" << std::endl;
	 std::string filename = "";
	 //std::getline(std::cin, filename);
	 //filename = "E:\\jbaxter\\data\\Chamberlain.mhd";
	 filename = "E:\\jbaxter\\data\\brain\\t1undiffused.mhd";
	 vtkMetaImageReader* imReader1 = vtkMetaImageReader::New();
	 imReader1->SetFileName(filename.c_str());
	 imReader1->Update();

	 //retrieve the second image
	 std::cout << "Enter META image2 filename" << std::endl;
	 //std::getline(std::cin, filename);
	 //filename = "E:\\jbaxter\\data\\Chamberlain.mhd";
	 filename = "E:\\jbaxter\\data\\brain\\t2undiffused.mhd";
	 vtkMetaImageReader* imReader2 = vtkMetaImageReader::New();
	 imReader2->SetFileName(filename.c_str());
	 imReader2->Update();

	 //append them together
	 vtkImageAppendComponents* appender = vtkImageAppendComponents::New();
	 appender->SetInput(0, imReader1->GetOutput());
	 appender->SetInput(1, imReader2->GetOutput());
	 appender->Update();

	 //create the transfer function (or load from file for 2D)
	 std::cout << "Enter 2D transfer function (.2tf) filename" << std::endl;
	 //std::getline(std::cin, filename);
	 //filename = "E:\\jbaxter\\data\\Chamberlain.2tf";
	 filename = "E:\\jbaxter\\data\\brain\\testingDual.2tf";
	 vtkCuda2DTransferFunction* viz = vtkCuda2DTransferFunction::New();
	 vtkCudaFunctionPolygonReader* vizReader = vtkCudaFunctionPolygonReader::New();
	 vizReader->SetFileName(filename.c_str());
	 vizReader->Read();
	 for(int i = 0; i < vizReader->GetNumberOfOutputs(); i++){
			 viz->AddFunctionObject( vizReader->GetOutput(i) );
	 }
	 viz->Modified();

	 std::cout << "Enter 2D InEx transfer function (.2tf) filename" << std::endl;
	 //std::getline(std::cin, filename);
	 filename = "E:\\jbaxter\\data\\Chamberlain.2tf";
	 vtkCuda2DTransferFunction* inex = vtkCuda2DTransferFunction::New();
	 vtkCudaFunctionPolygonReader* inexReader = vtkCudaFunctionPolygonReader::New();
	 inexReader->SetFileName(filename.c_str());
	 inexReader->Read();
	 for(int i = 0; i < inexReader->GetNumberOfOutputs(); i++){
			 inex->AddFunctionObject( inexReader->GetOutput(i) );
	 }
	 inex->Modified();

	 //assemble the ray caster
	 vtkCudaDualImageVolumeMapper* mapper = vtkCudaDualImageVolumeMapper::New();
	 mapper->SetInput( appender->GetOutput() );
	 //mapper->SetVisualizationFunction( viz );
	 //mapper->SetInExLogicFunction( inex );
	 mapper->SetFunction(viz);

	 //assemble the VTK pipeline
	 vtkVolume* volume = vtkVolume::New();
	 volume->SetMapper( mapper );
	 vtkRenderer* renderer = vtkRenderer::New();
	 renderer->AddVolume( volume );
	 renderer->ResetCamera();
	 renderer->SetBackground(1.0,1.0,1.0);
	 vtkRenderWindow* window = vtkRenderWindow::New();
	 window->AddRenderer( renderer );
	 window->Render();

	 //apply keyhole planes widget and interactor
	 vtkRenderWindowInteractor* interactor = vtkRenderWindowInteractor::New();
	 interactor->SetRenderWindow( window );
	 vtkBoxWidget* clippingPlanes = vtkBoxWidget::New();
	 clippingPlanes->SetInteractor( interactor );
	 clippingPlanes->SetPlaceFactor(1.01);
	 clippingPlanes->SetInput( appender->GetOutput() );
	 clippingPlanes->SetDefaultRenderer(renderer);
	 clippingPlanes->InsideOutOn();
	 clippingPlanes->PlaceWidget();
	 vtkBoxWidgetCallback *callback = vtkBoxWidgetCallback::New();
	 callback->SetMapper( mapper );
	 clippingPlanes->AddObserver(vtkCommand::InteractionEvent, callback);
	 callback->Delete();
	 clippingPlanes->EnabledOn();
	 clippingPlanes->GetSelectedFaceProperty()->SetOpacity(0.0);

	 //start the process
	 interactor->Initialize();
	 interactor->Start();

	 //clean up pipeline
	 interactor->Delete();
	 window->Delete();
	 volume->Delete();
	 renderer->Delete();
	 clippingPlanes->Delete();
	 mapper->Delete();
	 viz->Delete();
	 vizReader->Delete();
	 inex->Delete();
	 inexReader->Delete();
	 appender->Delete();
	 imReader1->Delete();
	 imReader2->Delete();

}