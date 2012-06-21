
/** @file main.cxx
*
*  @brief Sample volume ray casting program
*
*  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
*  @note First documented on June 14, 2012
*
*/

//For ray-caster
#include "vtkCuda2DVolumeMapper.h"
#include "vtkVolume.h"
#include "vtkVolumeProperty.h"
#include "vtkCuda2DTransferFunction.h"
#include "vtkCudaFunctionPolygon.h"
#include "vtkCudaFunctionPolygonReader.h"
#include "vtkBoxWidget.h"
#include "vtkCommand.h"
#include "vtkPlanes.h"

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
    void SetMapper(vtkCuda2DVolumeMapper* m) 
      { this->Mapper = m; }

  protected:
    vtkBoxWidgetKeyholeCallback() 
      { this->Mapper = 0; }

    vtkCuda2DVolumeMapper *Mapper;
  };

// ---------------------------------------------------------------------------------------------------
// Main Program
int main(int argc, char** argv){

  //retrieve the image
  vtkSmartPointer< vtkMetaImageReader > imReader = 
    vtkSmartPointer< vtkMetaImageReader >::New();
  imReader->SetFileName("E:\\jbaxter\\data\\Chamberlain.mhd");
  imReader->Update();

  //create the transfer function
  vtkSmartPointer< vtkCudaFunctionPolygonReader > tfReader = 
	  vtkSmartPointer< vtkCudaFunctionPolygonReader >::New();
  tfReader->SetFileName("E:\\jbaxter\\data\\Chamberlain.2tf");
  tfReader->Read();
  vtkSmartPointer< vtkCuda2DTransferFunction > func = 
	  vtkSmartPointer< vtkCuda2DTransferFunction >::New();
  for( int i = 0; i < tfReader->GetNumberOfOutputs(); i++ )
	  func->AddFunctionObject( tfReader->GetOutput(i) );

  //assemble the ray caster
  vtkSmartPointer< vtkCuda2DVolumeMapper > mapper = 
    vtkSmartPointer< vtkCuda2DVolumeMapper >::New();
  mapper->SetInput( imReader->GetOutput() );
  mapper->SetFunction( func );
  mapper->SetGradientShadingConstants( 0.5 );

  //assemble the VTK pipeline
  vtkSmartPointer< vtkVolume > volume = 
    vtkSmartPointer< vtkVolume >::New();
  volume->SetMapper( mapper );

  vtkSmartPointer< vtkVolumeProperty > volProperty = 
    vtkSmartPointer< vtkVolumeProperty >::New();
  volume->Update();

  vtkSmartPointer< vtkRenderer > renderer = 
    vtkSmartPointer< vtkRenderer >::New();
  renderer->AddVolume( volume );
  renderer->ResetCamera();
  renderer->SetBackground(1.0,1.0,1.0);

  vtkSmartPointer< vtkRenderWindow > window = 
    vtkSmartPointer< vtkRenderWindow >::New();
  window->AddRenderer( renderer );
  window->Render();

  //apply clipping planes widget and interactor
  vtkSmartPointer< vtkRenderWindowInteractor > interactor = 
    vtkSmartPointer< vtkRenderWindowInteractor >::New();
  interactor->SetRenderWindow( window );

  //start the process
  interactor->Initialize();
  interactor->Start();

  }