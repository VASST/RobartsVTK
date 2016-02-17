#include <QtGui>
#include <QApplication>
#include "FluoroPredViz.h"

int main(int argc, char** argv){

  QApplication a(argc, argv);
  FluoroPredViz* w = new FluoroPredViz();
  if(w->GetSuccessInit() == 0) w->show();
  else return 0;
  return a.exec();

}

#include "vtkDICOMImageReader.h"
#include "vtkMetaImageWriter.h"
#include "vtkImageFrangiFilter.h"
#include "vtkImageCast.h"
#include "vtkImageGaussianSmooth.h"

#include <vtkVersion.h> //for VTK_MAJOR_VERSION

int main1(int argc, char** argv){

  vtkDICOMImageReader* Reader = vtkDICOMImageReader::New();
  Reader->SetDirectoryName("E:\\jbaxter\\data\\CTA\\Recon 2- 4cc sec 120cc - 3");
  Reader->Update();

  vtkImageCast* Cast = vtkImageCast::New();
#if (VTK_MAJOR_VERSION < 6)
  Cast->SetInput(Reader->GetOutput());
#else
  Cast->SetInputConnection(Reader->GetOutputPort());
#endif
  Cast->SetOutputScalarTypeToDouble();
  Cast->Update();

  vtkImageGaussianSmooth* Smooth = vtkImageGaussianSmooth::New();
  Smooth->SetStandardDeviation(1.5,1.5,1.5);
  Smooth->SetRadiusFactors(15,15,15);
#if (VTK_MAJOR_VERSION < 6)
  Smooth->SetInput(Cast->GetOutput());
#else
  Smooth->SetInputConnection(Cast->GetOutputPort());
#endif
  Smooth->Update();

  vtkImageFrangiFilter* Frangi = vtkImageFrangiFilter::New();
#if (VTK_MAJOR_VERSION < 6)
  Frangi->SetInput(Smooth->GetOutput());
#else
  Frangi->SetInputConnection(Smooth->GetOutputPort());
#endif
  Frangi->SetSheet(0);
  Frangi->SetLine(1);
  Frangi->SetBlob(0);
  Frangi->Update();
  
  vtkMetaImageWriter* Writer = vtkMetaImageWriter::New();
  Writer->SetFileName("E:\\jbaxter\\data\\CTA\\Recon2.mhd");
  Writer->SetRAWFileName("E:\\jbaxter\\data\\CTA\\Recon2.raw");
  Writer->SetCompression(false);
#if (VTK_MAJOR_VERSION < 6)
  Writer->SetInput(Frangi->GetOutput());
#else
  Writer->SetInputConnection(Frangi->GetOutputPort());
#endif

  Writer->Write();

  Reader->Delete();
  Cast->Delete();
  Smooth->Delete();
  Frangi->Delete();
  Writer->Delete();

  return 0;
}