
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
  imReader->SetFileName("CT-chest.mhd" );
  imReader->Update();

  //create the transfer function
  vtkSmartPointer< vtkPiecewiseFunction > opacityFun = 
    vtkSmartPointer< vtkPiecewiseFunction >::New();
  /*
  opacityFun->AddPoint(0,    0.00);
  opacityFun->AddPoint(500,  0.15);
  opacityFun->AddPoint(1000, 0.15);
  opacityFun->AddPoint(1150, 0.85);
  */
  opacityFun->AddPoint(-3024, 0);
  opacityFun->AddPoint(143.556, 0);
  opacityFun->AddPoint(166.222, 0.686275);
  opacityFun->AddPoint(214.389, 0.696078);
  opacityFun->AddPoint(419.736, 0.833333);
  opacityFun->AddPoint(3071, 0.803922);

  vtkSmartPointer< vtkPiecewiseFunction > gradFun =
    vtkSmartPointer< vtkPiecewiseFunction >::New();
  gradFun->AddPoint( 0, 1 );
  gradFun->AddPoint( 255, 1 );

  vtkSmartPointer< vtkColorTransferFunction > colourFun = 
    vtkSmartPointer< vtkColorTransferFunction >::New();
  colourFun->AddRGBPoint(-3024, 0, 0, 0);
  colourFun->AddRGBPoint(143.556, 0.615686, 0.356863, 0.184314);
  colourFun->AddRGBPoint(166.222, 0.882353, 0.603922, 0.290196);
  colourFun->AddRGBPoint(214.389, 1, 1, 1);
  colourFun->AddRGBPoint(419.736, 1, 0.937033, 0.954531);
  colourFun->AddRGBPoint(3071, 0.827451, 0.658824, 1);

  /*
  colourFun->AddRGBPoint(0,    0.0, 0.0, 0.0);
  colourFun->AddRGBPoint(500,  1.0, 0.5, 0.3);
  colourFun->AddRGBPoint(1000, 1.0, 0.5, 0.3);
  colourFun->AddRGBPoint(1150, 1.0, 1.0, 0.9);
  */
  //assemble the ray caster
  vtkSmartPointer< vtkCuda1DVolumeMapper > mapper = 
    vtkSmartPointer< vtkCuda1DVolumeMapper >::New();
  mapper->SetInput( imReader->GetOutput() );
  mapper->SetGradientShadingConstants( 0.0 );

  //assemble the VTK pipeline
  vtkSmartPointer< vtkVolume > volume = 
    vtkSmartPointer< vtkVolume >::New();
  volume->SetMapper( mapper );

  vtkSmartPointer< vtkVolumeProperty > volProperty = 
    vtkSmartPointer< vtkVolumeProperty >::New();
  volProperty->SetScalarOpacity(opacityFun);
  volProperty->SetColor(colourFun);
  volProperty->SetGradientOpacity( gradFun );
  volProperty->SetDiffuse( 0.9 );
  volProperty->SetAmbient( 0.1 );
  volProperty->SetSpecular( 0.2 );
  volProperty->SetSpecularPower( 10 );
  volProperty->SetShade( 1 );
  volProperty->Modified();
  volume->SetProperty(volProperty);
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