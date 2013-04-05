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
//#include "vtkMetaImageReader.h"
//#include "vtkImageAppendComponents.h"
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
//static vtkBoxWidgetCallback *New()
// { return new vtkBoxWidgetCallback; }
//virtual void Execute(vtkObject *caller, unsigned long, void*)
// {
//   vtkBoxWidget *widget = reinterpret_cast<vtkBoxWidget*>(caller);
//   if (this->Mapper)
//	 {
//	 vtkPlanes *planes = vtkPlanes::New();
//	 widget->GetPlanes(planes);
//	 this->Mapper->SetKeyholePlanes(planes);
//	 planes->Delete();
//	 }
//}
//void SetMapper(vtkCudaDualImageVolumeMapper* m) 
// { this->Mapper = m; }
//
//protected:
//vtkBoxWidgetCallback() 
// { this->Mapper = 0; }
//
//vtkCudaDualImageVolumeMapper *Mapper;
//};
//
//// ---------------------------------------------------------------------------------------------------
//// Main Program
//int main(int argc, char** argv){
//
//	 //retrieve the first image
//	 std::cout << "Enter META image1 filename" << std::endl;
//	 std::string filename = "";
//	 //std::getline(std::cin, filename);
//	 //filename = "E:\\jbaxter\\data\\Chamberlain.mhd";
//	 filename = "E:\\jbaxter\\data\\brain\\t1undiffused.mhd";
//	 vtkMetaImageReader* imReader1 = vtkMetaImageReader::New();
//	 imReader1->SetFileName(filename.c_str());
//	 imReader1->Update();
//
//	 //retrieve the second image
//	 std::cout << "Enter META image2 filename" << std::endl;
//	 //std::getline(std::cin, filename);
//	 //filename = "E:\\jbaxter\\data\\Chamberlain.mhd";
//	 filename = "E:\\jbaxter\\data\\brain\\t2undiffused.mhd";
//	 vtkMetaImageReader* imReader2 = vtkMetaImageReader::New();
//	 imReader2->SetFileName(filename.c_str());
//	 imReader2->Update();
//
//	 //append them together
//	 vtkImageAppendComponents* appender = vtkImageAppendComponents::New();
//	 appender->SetInput(0, imReader1->GetOutput());
//	 appender->SetInput(1, imReader2->GetOutput());
//	 appender->Update();
//
//	 //create the transfer function (or load from file for 2D)
//	 std::cout << "Enter 2D transfer function (.2tf) filename" << std::endl;
//	 //std::getline(std::cin, filename);
//	 //filename = "E:\\jbaxter\\data\\Chamberlain.2tf";
//	 filename = "E:\\jbaxter\\data\\brain\\testingDual.2tf";
//	 vtkCuda2DTransferFunction* viz = vtkCuda2DTransferFunction::New();
//	 vtkCudaFunctionPolygonReader* vizReader = vtkCudaFunctionPolygonReader::New();
//	 vizReader->SetFileName(filename.c_str());
//	 vizReader->Read();
//	 for(int i = 0; i < vizReader->GetNumberOfOutputs(); i++){
//			 viz->AddFunctionObject( vizReader->GetOutput(i) );
//	 }
//	 viz->Modified();
//
//	 std::cout << "Enter 2D InEx transfer function (.2tf) filename" << std::endl;
//	 //std::getline(std::cin, filename);
//	 filename = "E:\\jbaxter\\data\\Chamberlain.2tf";
//	 vtkCuda2DTransferFunction* inex = vtkCuda2DTransferFunction::New();
//	 vtkCudaFunctionPolygonReader* inexReader = vtkCudaFunctionPolygonReader::New();
//	 inexReader->SetFileName(filename.c_str());
//	 inexReader->Read();
//	 for(int i = 0; i < inexReader->GetNumberOfOutputs(); i++){
//			 inex->AddFunctionObject( inexReader->GetOutput(i) );
//	 }
//	 inex->Modified();
//
//	 //assemble the ray caster
//	 vtkCudaDualImageVolumeMapper* mapper = vtkCudaDualImageVolumeMapper::New();
//	 mapper->SetInput( appender->GetOutput() );
//	 //mapper->SetVisualizationFunction( viz );
//	 //mapper->SetInExLogicFunction( inex );
//	 mapper->SetFunction(viz);
//
//	 //assemble the VTK pipeline
//	 vtkVolume* volume = vtkVolume::New();
//	 volume->SetMapper( mapper );
//	 vtkRenderer* renderer = vtkRenderer::New();
//	 renderer->AddVolume( volume );
//	 renderer->ResetCamera();
//	 renderer->SetBackground(1.0,1.0,1.0);
//	 vtkRenderWindow* window = vtkRenderWindow::New();
//	 window->AddRenderer( renderer );
//	 window->Render();
//
//	 //apply keyhole planes widget and interactor
//	 vtkRenderWindowInteractor* interactor = vtkRenderWindowInteractor::New();
//	 interactor->SetRenderWindow( window );
//	 vtkBoxWidget* clippingPlanes = vtkBoxWidget::New();
//	 clippingPlanes->SetInteractor( interactor );
//	 clippingPlanes->SetPlaceFactor(1.01);
//	 clippingPlanes->SetInput( appender->GetOutput() );
//	 clippingPlanes->SetDefaultRenderer(renderer);
//	 clippingPlanes->InsideOutOn();
//	 clippingPlanes->PlaceWidget();
//	 vtkBoxWidgetCallback *callback = vtkBoxWidgetCallback::New();
//	 callback->SetMapper( mapper );
//	 clippingPlanes->AddObserver(vtkCommand::InteractionEvent, callback);
//	 callback->Delete();
//	 clippingPlanes->EnabledOn();
//	 clippingPlanes->GetSelectedFaceProperty()->SetOpacity(0.0);
//
//	 //start the process
//	 interactor->Initialize();
//	 interactor->Start();
//
//	 //clean up pipeline
//	 interactor->Delete();
//	 window->Delete();
//	 volume->Delete();
//	 renderer->Delete();
//	 clippingPlanes->Delete();
//	 mapper->Delete();
//	 viz->Delete();
//	 vizReader->Delete();
//	 inex->Delete();
//	 inexReader->Delete();
//	 appender->Delete();
//	 imReader1->Delete();
//	 imReader2->Delete();
//
//}


#include "vtkMetaImageReader.h"
#include "vtkMetaImageWriter.h"
#include "vtkImageAppendComponents.h"
#include "vtkImageExtractComponents.h"
#include "vtkCudaKohonenGenerator.h"
#include "vtkCudaKohonenApplication.h"
#include "vtkCudaKohonenReprojector.h"
#include "vtkImageCast.h"
#include "vtkImageLogarithmicScale.h"

#include "vtkImageGradientMagnitude.h"

#include "vtkImageMultiStatistics.h"

#include "vtkImageMask.h"

//
//int main(int argc, char** argv ){
//	
//	vtkMetaImageReader* imReader1 = vtkMetaImageReader::New();
//	vtkMetaImageReader* imReader2 = vtkMetaImageReader::New();
//	vtkMetaImageReader* imReader3 = vtkMetaImageReader::New();
//	vtkMetaImageReader* imReader4 = vtkMetaImageReader::New();
//	vtkMetaImageReader* imReader5 = vtkMetaImageReader::New();
//	vtkMetaImageReader* imReader6 = vtkMetaImageReader::New();
//	vtkMetaImageReader* imReader7 = vtkMetaImageReader::New();
//	//vtkMetaImageReader* maskReader = vtkMetaImageReader::New();
//	/*
//	imReader1->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\DESPOT1HIFI_T1Map_thalami.mhd");
//	imReader2->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\DESPOTFM_T2Map_thalami.mhd");
//	imReader3->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\dti_FA_thalami.mhd");
//	imReader4->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\dti_MD_thalami.mhd");
//	imReader5->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\PositionalX.mhd");
//	imReader6->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\PositionalY.mhd");
//	imReader7->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\PositionalZ.mhd");
//	maskReader->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\Thalami.mhd");*/
//	imReader1->SetFileName( argv[1] );
//	imReader2->SetFileName( argv[2] );
//	imReader3->SetFileName( argv[3] );
//	imReader4->SetFileName( argv[4] );
//	imReader5->SetFileName( argv[5] );
//	imReader6->SetFileName( argv[6] );
//	imReader7->SetFileName( argv[7] );
//	//maskReader->SetFileName( argv[8] );
//	imReader1->Update();
//	imReader2->Update();
//	imReader3->Update();
//	imReader4->Update();
//	imReader5->Update();
//	imReader6->Update();
//	imReader7->Update();
//	//maskReader->Update();
//	
//	vtkImageCast* caster1 = vtkImageCast::New();
//	caster1->SetOutputScalarTypeToFloat();
//	caster1->SetInput(imReader1->GetOutput());
//	caster1->Update();
//	vtkImageCast* caster2 = vtkImageCast::New();
//	caster2->SetOutputScalarTypeToFloat();
//	caster2->SetInput(imReader2->GetOutput());
//	caster2->Update();
//	vtkImageCast* caster3 = vtkImageCast::New();
//	caster3->SetOutputScalarTypeToFloat();
//	caster3->SetInput(imReader3->GetOutput());
//	caster3->Update();
//	vtkImageCast* caster4 = vtkImageCast::New();
//	caster4->SetOutputScalarTypeToFloat();
//	caster4->SetInput(imReader4->GetOutput());
//	caster4->Update();
//	vtkImageCast* caster5 = vtkImageCast::New();
//	caster5->SetOutputScalarTypeToFloat();
//	caster5->SetInput(imReader5->GetOutput());
//	caster5->Update();
//	vtkImageCast* caster6 = vtkImageCast::New();
//	caster6->SetOutputScalarTypeToFloat();
//	caster6->SetInput(imReader6->GetOutput());
//	caster6->Update();
//	vtkImageCast* caster7 = vtkImageCast::New();
//	caster7->SetOutputScalarTypeToFloat();
//	caster7->SetInput(imReader7->GetOutput());
//	caster7->Update();
//	//vtkImageCast* mapCaster = vtkImageCast::New();
//	//mapCaster->SetOutputScalarTypeToChar();
//	//mapCaster->SetInput(maskReader->GetOutput());
//	//mapCaster->Update();
//	
//	vtkImageGradientMagnitude* mag1 = vtkImageGradientMagnitude::New();
//	mag1->SetInput(caster1->GetOutput());
//	mag1->Update();
//	vtkImageGradientMagnitude* mag2 = vtkImageGradientMagnitude::New();
//	mag2->SetInput(caster2->GetOutput());
//	mag2->Update();
//	
//	vtkImageLogarithmicScale* scaler1 = vtkImageLogarithmicScale::New();
//	scaler1->SetInput(mag1->GetOutput());
//	scaler1->SetConstant(0.1);
//	scaler1->Update();
//	vtkImageLogarithmicScale* scaler2 = vtkImageLogarithmicScale::New();
//	scaler2->SetInput(mag2->GetOutput());
//	scaler2->SetConstant(0.1);
//	scaler2->Update();
//	
//	vtkImageAppendComponents* appender = vtkImageAppendComponents::New();
//	appender->SetInput(0, (vtkDataObject*) caster1->GetOutput());
//	appender->SetInput(1, (vtkDataObject*) caster2->GetOutput());
//	appender->SetInput(2, (vtkDataObject*) scaler1->GetOutput());
//	appender->SetInput(3, (vtkDataObject*) scaler2->GetOutput());
//	appender->SetInput(4, (vtkDataObject*) caster3->GetOutput());
//	appender->SetInput(5, (vtkDataObject*) caster4->GetOutput());
//	appender->SetInput(6, (vtkDataObject*) caster5->GetOutput());
//	appender->SetInput(7, (vtkDataObject*) caster6->GetOutput());
//	appender->SetInput(8, (vtkDataObject*) caster7->GetOutput());
//	appender->Update();
//
////	vtkCudaKohonenGenerator* kohonen = vtkCudaKohonenGenerator::New();
////	kohonen->SetInput(0,appender->GetOutput());
////	kohonen->SetInput(1,mapCaster->GetOutput());
////	kohonen->SetUseAllVoxelsFlag(true);
////	kohonen->SetNumberOfIterations(2000);
////	kohonen->SetAlphaInitial(0.5);
////	kohonen->SetAlphaDecay(0.997);
////	kohonen->SetWidthInitial(2.0);
////	kohonen->SetWidthDecay(0.997);
////	kohonen->SetBatchSize(0.5);
////	kohonen->SetWeight(0,500);
////	kohonen->SetWeight(1,500);
////	kohonen->SetWeight(2,10);
////	kohonen->SetWeight(3,10);
////	kohonen->SetWeight(4,300);
////	kohonen->SetWeight(5,300);
////	kohonen->SetWeight(6,50);
////	kohonen->SetWeight(7,50);
////	kohonen->SetWeight(8,50);
////	kohonen->SetKohonenMapSize(512,512);
////	kohonen->Update();
////	//vtkMetaImageReader* kohonen = vtkMetaImageReader::New();
////	//kohonen->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\kohonen-thalamus-t1t2g1g2AM.mhd");
////	//kohonen->Update();
////
////	vtkCudaKohonenApplication* applier = vtkCudaKohonenApplication::New();
////	applier->SetInput(0,appender->GetOutput());
////	applier->SetInput(1,kohonen->GetOutput());
////	applier->SetWeight(0,5000);
////	applier->SetWeight(1,5000);
////	applier->SetWeight(2,100);
////	applier->SetWeight(3,100);
////	applier->SetWeight(4,3000);
////	applier->SetWeight(5,3000);
////	applier->SetWeight(6,500);
////	applier->SetWeight(7,500);
////	applier->SetWeight(8,500);
////	applier->Update();
////	//vtkMetaImageReader* applier = vtkMetaImageReader::New();
////	//applier->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\imageK-t1t2g1g2AM.mhd");
////	//applier->Update();
////	//
////	//vtkImageAppendComponents* appender2 = vtkImageAppendComponents::New();
////	//appender2->SetInput(0, (vtkDataObject*) appender->GetOutput());
////	//appender2->SetInput(1, (vtkDataObject*) applier->GetOutput());
////	//appender2->Update();
////	//
////	//vtkImageMultiStatistics* multiStats1 = vtkImageMultiStatistics::New();
////	//multiStats1->SetInput(0,appender->GetOutput());
////	//multiStats1->SetInput(1,maskReader->GetOutput());
////	//multiStats1->SetEntropyResolution(64);
////	//multiStats1->Update();
////	//multiStats1->Print(std::cout);
////	//std::cout << "Original Image Entropy: " << multiStats1->GetTotalEntropy() << std::endl;
////
////	//vtkImageMultiStatistics* multiStats2 = vtkImageMultiStatistics::New();
////	//multiStats2->SetInput(0,applier->GetOutput());
////	//multiStats2->SetInput(1,maskReader->GetOutput());
////	//multiStats2->SetEntropyResolution(512);
////	//multiStats2->Update();
////	//multiStats2->Print(std::cout);
////	//std::cout << "Manifold Image Entropy: " << multiStats2->GetTotalEntropy() << std::endl;
////
////	//vtkImageMultiStatistics* multiStats3 = vtkImageMultiStatistics::New();
////	//multiStats3->SetInput(0,appender2->GetOutput());
////	//multiStats3->SetInput(1,maskReader->GetOutput());
////	//multiStats3->SetEntropyResolution(512);
////	//multiStats3->Update();
////	//std::cout << "Combined Image Entropy: " << multiStats3->GetTotalEntropy() << std::endl;
////
////	//vtkCudaKohonenReprojector* reprojection = vtkCudaKohonenReprojector::New();
////	//reprojection->SetInput(0, applier->GetOutput());
////	//reprojection->SetInput(1, kohonen->GetOutput());
//
//	vtkMetaImageWriter* writer = vtkMetaImageWriter::New();
//	writer->SetCompression(false);
////	//writer->SetInput( reprojection->GetOutput() );
////	//writer->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\imageR-t1t2g1g2AM.mhd");
////	//writer->SetRAWFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\imageR-t1t2g1g2AM.raw");
////	//writer->Update();
//	//writer->Write();
//	writer->SetInput( appender->GetOutput() );
//	writer->SetFileName( argv[8] );
//	writer->SetRAWFileName( argv[9] );
//	writer->Update();
//	writer->Write();
////	//writer->SetInput( applier->GetOutput() );
////	//writer->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\imageK-thalami-t1t2g1g2AM.mhd");
////	//writer->SetRAWFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\imageK-thalami-t1t2g1g2AM.raw");
////	//writer->Update();
////	//writer->Write();
////	//writer->SetInput( kohonen->GetOutput() );
////	//writer->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\kohonen-thalamus-t1t2g1g2AM.mhd");
////	//writer->SetRAWFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\kohonen-thalamus-t1t2g1g2AM.raw");
////	//writer->Update();
////	//writer->Write();
//
//	writer->Delete();
////	//reprojection->Delete();
////	//multiStats1->Delete();
////	//multiStats2->Delete();
////	//multiStats3->Delete();
////	//appender2->Delete();
////	//applier->Delete();
//	//kohonen->Delete();
//	scaler1->Delete();
//	scaler2->Delete();
//	mag1->Delete();
//	mag2->Delete();
//	caster1->Delete();
//	caster2->Delete();
//	caster3->Delete();
//	caster4->Delete();
//	caster5->Delete();
//	caster6->Delete();
//	caster7->Delete();
//	//mapCaster->Delete();
//	appender->Delete();
//	imReader1->Delete();
//	imReader2->Delete();
//	imReader3->Delete();
//	imReader4->Delete();
//	imReader5->Delete();
//	imReader6->Delete();
//	imReader7->Delete();
//	//maskReader->Delete();
//}

//int main(int argc, char** argv ){
//
//	//get names of training files
//	char** trainingFiles = new char*[20];
//	trainingFiles[0] = "E:\\jbaxter\\data\\brain\\EPI_P008\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[1] = "E:\\jbaxter\\data\\brain\\EPI_P014\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[2] = "E:\\jbaxter\\data\\brain\\EPI_P016\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[3] = "E:\\jbaxter\\data\\brain\\EPI_P018\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[4] = "E:\\jbaxter\\data\\brain\\EPI_V042\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[5] = "E:\\jbaxter\\data\\brain\\EPI_V043\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[6] = "E:\\jbaxter\\data\\brain\\EPI_V044\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[7] = "E:\\jbaxter\\data\\brain\\EPI_V045\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[8] = "E:\\jbaxter\\data\\brain\\EPI_V042\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[9] = "E:\\jbaxter\\data\\brain\\EPI_V043\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[10] = "E:\\jbaxter\\data\\brain\\EPI_V044\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[11] = "E:\\jbaxter\\data\\brain\\EPI_V045\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[12] = "E:\\jbaxter\\data\\brain\\EPI_V046\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[13] = "E:\\jbaxter\\data\\brain\\EPI_V047\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[14] = "E:\\jbaxter\\data\\brain\\EPI_V048\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[15] = "E:\\jbaxter\\data\\brain\\EPI_V049\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[16] = "E:\\jbaxter\\data\\brain\\EPI_V051\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[17] = "E:\\jbaxter\\data\\brain\\EPI_V052\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[18] = "E:\\jbaxter\\data\\brain\\EPI_V054\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	trainingFiles[19] = "E:\\jbaxter\\data\\brain\\EPI_V055\\imageO-thalamus-t1t2g1g2ADXYZ.mhd";
//	
//	char** trainingMasks = new char*[20];
//	trainingMasks[0] = "E:\\jbaxter\\data\\brain\\EPI_P008\\Thalami_char.mhd";
//	trainingMasks[1] = "E:\\jbaxter\\data\\brain\\EPI_P014\\Thalami_char.mhd";
//	trainingMasks[2] = "E:\\jbaxter\\data\\brain\\EPI_P016\\Thalami_char.mhd";
//	trainingMasks[3] = "E:\\jbaxter\\data\\brain\\EPI_P018\\Thalami_char.mhd";
//	trainingMasks[4] = "E:\\jbaxter\\data\\brain\\EPI_V042\\Thalami_char.mhd";
//	trainingMasks[5] = "E:\\jbaxter\\data\\brain\\EPI_V043\\Thalami_char.mhd";
//	trainingMasks[6] = "E:\\jbaxter\\data\\brain\\EPI_V044\\Thalami_char.mhd";
//	trainingMasks[7] = "E:\\jbaxter\\data\\brain\\EPI_V045\\Thalami_char.mhd";
//	trainingMasks[8] = "E:\\jbaxter\\data\\brain\\EPI_V042\\Thalami_char.mhd";
//	trainingMasks[9] = "E:\\jbaxter\\data\\brain\\EPI_V043\\Thalami_char.mhd";
//	trainingMasks[10] = "E:\\jbaxter\\data\\brain\\EPI_V044\\Thalami_char.mhd";
//	trainingMasks[11] = "E:\\jbaxter\\data\\brain\\EPI_V045\\Thalami_char.mhd";
//	trainingMasks[12] = "E:\\jbaxter\\data\\brain\\EPI_V046\\Thalami_char.mhd";
//	trainingMasks[13] = "E:\\jbaxter\\data\\brain\\EPI_V047\\Thalami_char.mhd";
//	trainingMasks[14] = "E:\\jbaxter\\data\\brain\\EPI_V048\\Thalami_char.mhd";
//	trainingMasks[15] = "E:\\jbaxter\\data\\brain\\EPI_V049\\Thalami_char.mhd";
//	trainingMasks[16] = "E:\\jbaxter\\data\\brain\\EPI_V051\\Thalami_char.mhd";
//	trainingMasks[17] = "E:\\jbaxter\\data\\brain\\EPI_V052\\Thalami_char.mhd";
//	trainingMasks[18] = "E:\\jbaxter\\data\\brain\\EPI_V054\\Thalami_char.mhd";
//	trainingMasks[19] = "E:\\jbaxter\\data\\brain\\EPI_V055\\Thalami_char.mhd";
//
//	vtkCudaKohonenApplication* applier = vtkCudaKohonenApplication::New();
//
//	//vtkCudaKohonenGenerator* generator = vtkCudaKohonenGenerator::New();
//	//generator->SetUseMaskFlag(true);
//	//generator->SetUseAllVoxelsFlag(true);
//	//generator->SetNumberOfIterations(2000);
//	//generator->SetAlphaInitial(0.2);
//	//generator->SetAlphaProlong(1000);
//	//generator->SetAlphaDecay(0.997);
//	//generator->SetAlphaBaseline(0.0);
//	//generator->SetWidthInitial(0.75);
//	//generator->SetWidthProlong(10);
//	//generator->SetWidthDecay(0.99);
//	//generator->SetWidthBaseline(0.0);
//	//generator->SetBatchSize(0.5);
//	//generator->SetWeightNormalization(false);
//	//generator->SetWeight(0,     1.9764); //corresponds to normalized 50
//	//generator->SetWeight(1,    10.2848); //corresponds to normalized 50
//	//generator->SetWeight(2,   130.3753); //corresponds to normalized 1
//	//generator->SetWeight(3,   177.2710); //corresponds to normalized 1
//	//generator->SetWeight(4,  7516.4512); //corresponds to normalized 50
//	//generator->SetWeight(5, 84983.6797); //corresponds to normalized 2
//	//generator->SetWeight(6,    90.9091); //corresponds to normalized 10
//	//generator->SetWeight(7,   131.5789); //corresponds to normalized 10
//	//generator->SetWeight(8,   172.4138); //corresponds to normalized 10
//	//generator->SetKohonenMapSize(512,512);
//	vtkMetaImageReader* generator = vtkMetaImageReader::New();
//	generator->SetFileName("E:\\jbaxter\\data\\brain\\kohonen\\kohonenen-thalami-t1t2g1g2AMXYZ.mhd");
//	generator->Update();
//
//	for( int i = 0; i < 1; i++ ){
//		vtkMetaImageReader* reader = vtkMetaImageReader::New();
//		reader->SetFileName(trainingFiles[i]);
//		reader->Update();
//
//		vtkMetaImageReader* maskReader = vtkMetaImageReader::New();
//		maskReader->SetFileName(trainingMasks[i]);
//		maskReader->Update();
//
//		applier->SetInput(1, generator->GetOutput());
//		applier->SetInput(0, reader->GetOutput());
//		applier->SetInput(2, maskReader->GetOutput() );
//		applier->Update();
//
//		vtkMetaImageWriter* writer = vtkMetaImageWriter::New();
//		writer->SetCompression(false);
//		writer->SetInput( applier->GetOutput() );
//		writer->SetFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\imageK-t1t2g1g2AMXYZ-2.mhd");
//		writer->SetRAWFileName("E:\\jbaxter\\data\\brain\\EPI_P008\\imageK-t1t2g1g2AMXYZ-2.raw");
//		writer->Update();
//		writer->Write();
//
//		reader->Delete();
//		maskReader->Delete();
//		writer->Delete();
//
//	}
//	
//	generator->Delete();
//	applier->Delete();
//
//	delete trainingMasks;
//	delete trainingFiles;
//
//}

#include "vtkHierarchicalMaxFlowSegmentation.h"
#include "vtkMutableDirectedGraph.h"
#include "vtkTree.h"

int main( int argc, char** argv ){

	vtkMutableDirectedGraph* hierarchy = vtkMutableDirectedGraph::New();
	vtkIdType source =		hierarchy->AddVertex();
	vtkIdType higherSide =	hierarchy->AddChild(source);
	vtkIdType lowerSide =	hierarchy->AddChild(source);
	vtkIdType high =		hierarchy->AddChild(higherSide);
	vtkIdType mid_high =	hierarchy->AddChild(higherSide);
	vtkIdType low =			hierarchy->AddChild(lowerSide);
	vtkIdType mid_low =		hierarchy->AddChild(lowerSide);
	vtkTree* roHierarchy = vtkTree::New();
	roHierarchy->CheckedDeepCopy(hierarchy);
	hierarchy->Delete();

	vtkMetaImageReader* highReader = vtkMetaImageReader::New();
	highReader->SetFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\cost_high.mhd");
	highReader->Update();
	vtkMetaImageReader* midReader = vtkMetaImageReader::New();
	midReader->SetFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\cost_mid.mhd");
	midReader->Update();
	vtkMetaImageReader* lowReader = vtkMetaImageReader::New();
	lowReader->SetFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\cost_low.mhd");
	lowReader->Update();
	vtkMetaImageReader* gradReader = vtkMetaImageReader::New();
	gradReader->SetFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\gradMag.mhd");
	gradReader->Update();

	vtkHierarchicalMaxFlowSegmentation* segmentation = vtkHierarchicalMaxFlowSegmentation::New();
	segmentation->SetInput(source,		gradReader->GetOutput());
	segmentation->SetInput(higherSide,	gradReader->GetOutput());
	//segmentation->SetInput(lowerSide,	gradReader->GetOutput());
	segmentation->SetInput(high,		highReader->GetOutput());
	segmentation->SetInput(mid_high,	midReader->GetOutput());
	segmentation->SetInput(mid_low,		midReader->GetOutput());
	segmentation->SetInput(low,			lowReader->GetOutput());
	segmentation->SetInput(higherSide,	0);
	segmentation->SetHierarchy(roHierarchy);
	segmentation->AddSmoothnessScalar(source,1);
	segmentation->AddSmoothnessScalar(higherSide,0.5);
	segmentation->AddSmoothnessScalar(lowerSide,0.5);
	segmentation->SetNumberOfIterations(1000);
	segmentation->Update();

	vtkImageData* highOut = (vtkImageData*) segmentation->GetOutput(high);
	vtkImageData* midHighOut = (vtkImageData*) segmentation->GetOutput(mid_high);
	vtkImageData* midLowOut = (vtkImageData*) segmentation->GetOutput(mid_low);
	vtkImageData* lowOut = (vtkImageData*) segmentation->GetOutput(low);

	vtkMetaImageWriter* writer = vtkMetaImageWriter::New();
	writer->SetCompression(false);
	writer->SetFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\high_out.mhd");
	writer->SetRAWFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\high_out.raw");
	writer->SetInput(highOut);
	writer->Update();
	writer->Write();
	writer->SetFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\midHigh_out.mhd");
	writer->SetRAWFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\midHigh_out.raw");
	writer->SetInput(midHighOut);
	writer->Update();
	writer->Write();
	writer->SetFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\midLow_out.mhd");
	writer->SetRAWFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\midLow_out.raw");
	writer->SetInput(midLowOut);
	writer->Update();
	writer->Write();
	writer->SetFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\low_out.mhd");
	writer->SetRAWFileName("E:\\jbaxter\\data\\AbstractData\\2013-03-28-TestingHMF\\low_out.raw");
	writer->SetInput(lowOut);
	writer->Update();
	writer->Write();

	writer->Delete();
	segmentation->Delete();
	highReader->Delete();
	midReader->Delete();
	lowReader->Delete();
	gradReader->Delete();
	roHierarchy->Delete();

}