//
///** @file main.cxx
//*
//*  @brief Sample volume ray casting program
//*
//*  @author John Stuart Haberl Baxter (Dr. Peter's Lab at Robarts Research Institute)
//*  @note First documented on June 14, 2012
//*
//*/
//
////For ray-caster
//#include "vtkCuda2DInExLogicVolumeMapper.h"
//#include "vtkVolume.h"
//#include "vtkVolumeProperty.h"
//#include "vtkCuda2DTransferFunction.h"
//#include "vtkCudaFunctionPolygon.h"
//#include "vtkCudaFunctionPolygonReader.h"
//#include "vtkBoxWidget.h"
//#include "vtkCommand.h"
//#include "vtkPlanes.h"
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
//#include <vtkSmartPointer.h>
//
//// ---------------------------------------------------------------------------------------------------
//// Callback for moving the planes from the box widget to the mapper
//class vtkBoxWidgetKeyholeCallback : public vtkCommand
//  {
//  public:
//    static vtkBoxWidgetKeyholeCallback *New()
//      { return new vtkBoxWidgetKeyholeCallback; }
//    virtual void Execute(vtkObject *caller, unsigned long, void*)
//      {
//      vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
//      if (this->Mapper)
//        {
//        vtkPlanes *planes = vtkPlanes::New();
//        widget->GetPlanes(planes);
//        this->Mapper->SetClippingPlanes(planes);
//        planes->Delete();
//        }
//      }
//    void SetMapper(vtkCuda2DInExLogicVolumeMapper* m) 
//      { this->Mapper = m; }
//
//  protected:
//    vtkBoxWidgetKeyholeCallback() 
//      { this->Mapper = 0; }
//
//    vtkCuda2DInExLogicVolumeMapper *Mapper;
//  };
//
//// ---------------------------------------------------------------------------------------------------
//// Main Program
//int main(int argc, char** argv){
//
//  //retrieve the image
//  vtkSmartPointer< vtkMetaImageReader > imReader = 
//    vtkSmartPointer< vtkMetaImageReader >::New();
//  imReader->SetFileName("E:\\jbaxter\\data\\Chamberlain.mhd");
//  imReader->Update();
//
//  //create the transfer function
//  vtkSmartPointer< vtkCudaFunctionPolygonReader > tfReader = 
//	  vtkSmartPointer< vtkCudaFunctionPolygonReader >::New();
//  tfReader->SetFileName("E:\\jbaxter\\data\\Chamberlain.2tf");
//  tfReader->Read();
//  vtkSmartPointer< vtkCuda2DTransferFunction > func = 
//	  vtkSmartPointer< vtkCuda2DTransferFunction >::New();
//  for( int i = 0; i < tfReader->GetNumberOfOutputs(); i++ )
//	  func->AddFunctionObject( tfReader->GetOutput(i) );
//
//  //assemble the ray caster
//  vtkSmartPointer< vtkCuda2DInExLogicVolumeMapper > mapper = 
//    vtkSmartPointer< vtkCuda2DInExLogicVolumeMapper >::New();
//  mapper->SetInput( imReader->GetOutput() );
//  mapper->SetVisualizationFunction( func );
//  mapper->SetInExLogicFunction( func );
//  mapper->SetGradientShadingConstants( 0.5 );
//
//  //assemble the VTK pipeline
//  vtkSmartPointer< vtkVolume > volume = 
//    vtkSmartPointer< vtkVolume >::New();
//  volume->SetMapper( mapper );
//
//  vtkSmartPointer< vtkVolumeProperty > volProperty = 
//    vtkSmartPointer< vtkVolumeProperty >::New();
//  volume->Update();
//
//  vtkSmartPointer< vtkRenderer > renderer = 
//    vtkSmartPointer< vtkRenderer >::New();
//  renderer->AddVolume( volume );
//  renderer->ResetCamera();
//  renderer->SetBackground(1.0,1.0,1.0);
//
//  vtkSmartPointer< vtkRenderWindow > window = 
//    vtkSmartPointer< vtkRenderWindow >::New();
//  window->AddRenderer( renderer );
//  window->Render();
//
//  //apply clipping planes widget and interactor
//  vtkSmartPointer< vtkRenderWindowInteractor > interactor = 
//    vtkSmartPointer< vtkRenderWindowInteractor >::New();
//  interactor->SetRenderWindow( window );
//
//  //start the process
//  interactor->Initialize();
//  interactor->Start();
//
//  }

#include "vtkSmartPointer.h"
#include "vtkCT2USSimulation.h"
#include "vtkCudaCT2USSimulation.h"
#include "vtkMINCImageReader.h"
#include "vtkTransform.h"
#include "vtkBMPWriter.h"
#include "vtkImageCast.h"

// ---------------------------------------------------------------------------------------------------
// Main Program
int main(int argc, char** argv){
	
  //retrieve the image
  vtkSmartPointer< vtkMINCImageReader > imReader = 
    vtkSmartPointer< vtkMINCImageReader >::New();
  imReader->SetFileName("E:\\jbaxter\\Code\\spine_NDI_AIGS_RSNA\\data\\NDI-phantom-box-SE2-crop.mnc");
  imReader->Update();
  
  vtkSmartPointer< vtkTransform > probePose =
	  vtkSmartPointer< vtkTransform >::New();
  probePose->Identity();
  probePose->Translate( imReader->GetOutput()->GetCenter() );

  vtkSmartPointer< vtkCT2USSimulation > ct2us =
	  vtkSmartPointer< vtkCT2USSimulation >::New();
  ct2us->SetInput( imReader->GetOutput() );
  ct2us->SetNumberOfThreads(100);
  ct2us->SetTransform( probePose );
  ct2us->SetProbeWidth(0,0);
  ct2us->SetFarClippingDepth(100);
  ct2us->SetNearClippingDepth(0);
  ct2us->SetFanAngle(180,0);
  ct2us->SetLinearCombinationAlpha(0.72);
  ct2us->SetLinearCombinationBeta(20);
  ct2us->SetLinearCombinationBias(0);
  ct2us->SetLogarithmicScaleFactor(1.0);
  ct2us->SetTotalReflectionThreshold(4000000);
  ct2us->SetDensityScaleModel(0.20,-887);
  ct2us->SetOutputResolution(1000,1000,1);
  ct2us->Modified();
  ct2us->Update();

 /* vtkSmartPointer< vtkCudaCT2USSimulation > cuda_ct2us =
	  vtkSmartPointer< vtkCudaCT2USSimulation >::New();
  cuda_ct2us->SetInput( imReader->GetOutput() );
  cuda_ct2us->SetTransform( probePose );
  cuda_ct2us->SetProbeWidth(0,0);
  cuda_ct2us->SetFarClippingDepth(100);
  cuda_ct2us->SetNearClippingDepth(0);
  cuda_ct2us->SetFanAngle(180,0);
  cuda_ct2us->SetLinearCombinationAlpha(0.72);
  cuda_ct2us->SetLinearCombinationBeta(20);
  cuda_ct2us->SetLinearCombinationBias(0);
  cuda_ct2us->SetLogarithmicScaleFactor(1.0);
  cuda_ct2us->SetTotalReflectionThreshold(4000000);
  cuda_ct2us->SetDensityScaleModel(0.20,-887);
  cuda_ct2us->SetOutputResolution(1000,1000,1);
  cuda_ct2us->Modified();
  cuda_ct2us->Update();*/
  
  vtkSmartPointer< vtkBMPWriter > writer1 =
	  vtkSmartPointer< vtkBMPWriter >::New();
  writer1->SetInput(ct2us->GetOutput());
  writer1->SetFileName("E:\\jbaxter\\test.bmp");
  writer1->Write();
  //vtkSmartPointer< vtkBMPWriter > writer2 =
	 // vtkSmartPointer< vtkBMPWriter >::New();
  //writer2->SetInput(cuda_ct2us->GetOutput());
  //writer2->SetFileName("E:\\jbaxter\\test_cuda.bmp");
  //writer2->Write();
  
}