//For transfer function
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaFunctionPolygon.h"
#include "vtkCudaFunctionPolygonReader.h"

//For ray-caster
#include "vtkCuda2DInExLogicVolumeMapper.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkPiecewiseFunction.h"
#include "vtkColorTransferFunction.h"
#include "vtkBoxWidget.h"
#include "vtkCommand.h"
#include "vtkPlanes.h"

//general use
#include "vtkImageData.h"
#include "vtkImageAppendComponents.h"
#include "vtkMetaImageReader.h"
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
  void SetMapper(vtkCuda2DInExLogicVolumeMapper* m) 
    { this->Mapper = m; }

protected:
  vtkBoxWidgetCallback() 
    { this->Mapper = 0; }

  vtkCuda2DInExLogicVolumeMapper *Mapper;
};

// ---------------------------------------------------------------------------------------------------
// Main Program
int main(int argc, char** argv){

	//retrieve the image
	std::cout << "Enter META image filename" << std::endl;
	std::string filename = "";
	//std::getline(std::cin, filename);
	//filename = "E:\\jbaxter\\data\\Chamberlain.mhd";
	//filename = "E:\\jbaxter\\data\\4D Heart\\JTM45-flip-p01.mhd";
	filename = "E:\\jbaxter\\4J\\Tumor2_hyperintense.mhd";
	//filename = "E:\\jbaxter\\data\\brain\\t1undiffused.mhd";
	vtkMetaImageReader* imReader1 = vtkMetaImageReader::New();
	imReader1->SetFileName(filename.c_str());
	imReader1->Update();
	
	filename = "E:\\jbaxter\\data\\brain\\t2undiffused.mhd";
	vtkMetaImageReader* imReader2 = vtkMetaImageReader::New();
	imReader2->SetFileName(filename.c_str());
	imReader2->Update();

	vtkImageAppendComponents* appender = vtkImageAppendComponents::New();
	appender->SetInput(0,imReader1->GetOutput());
	appender->SetInput(1,imReader2->GetOutput());
	appender->Update();

	//create the transfer function (or load from file for 2D)
	std::cout << "Enter 2D transfer function (.2tf) filename" << std::endl;
	//std::getline(std::cin, filename);
	//filename = "E:\\jbaxter\\data\\Chamberlain.2tf";
	//filename = "E:\\jbaxter\\data\\4D Heart\\heart.2tf";
	//filename = "E:\\jbaxter\\data\\brain\\testingDual.2tf";
	filename = "E:\\jbaxter\\4J\\Tumour.2tf";
	vtkCuda2DTransferFunction* viz = vtkCuda2DTransferFunction::New();
	vtkCudaFunctionPolygonReader* vizReader = vtkCudaFunctionPolygonReader::New();
	vizReader->SetFileName(filename.c_str());
	vizReader->Read();
	for(int i = 0; i < vizReader->GetNumberOfOutputs(); i++){
		viz->AddFunctionObject( vizReader->GetOutput(i) );
	}
	viz->Modified();

	//create the transfer function (or load from file for 2D)
	std::cout << "Enter 2D INEX transfer function (.2tf) filename" << std::endl;
	//std::getline(std::cin, filename);
	filename = "E:\\jbaxter\\4J\\J-WholeHead.2tf";
	vtkCuda2DTransferFunction* inex = vtkCuda2DTransferFunction::New();
	vtkCudaFunctionPolygonReader* inexReader = vtkCudaFunctionPolygonReader::New();
	inexReader->SetFileName(filename.c_str());
	inexReader->Read();
	for(int i = 0; i < inexReader->GetNumberOfOutputs(); i++){
		inexReader->GetOutput(i)->SetOpacity(1);
		inex->AddFunctionObject( inexReader->GetOutput(i) );
	}
	inex->Modified();

	//assemble the ray caster
	vtkCuda2DInExLogicVolumeMapper* mapper = vtkCuda2DInExLogicVolumeMapper::New();
	mapper->SetDevice(0);
	mapper->SetInput( imReader1->GetOutput() );
	mapper->SetDistanceShadingConstants(0.0,0.0,1.0);
	mapper->SetVisualizationFunction( viz );
	mapper->SetInExLogicFunction( inex );

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
	appender->Delete();
	imReader1->Delete();
	imReader2->Delete();


}