#include <QtGui>
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
#if (VTK_MAJOR_VERSION <= 5)
  Cast->SetInput(Reader->GetOutput());
#else
  Cast->SetInputConnection(Reader->GetOutputPort());
#endif
  Cast->SetOutputScalarTypeToDouble();
  Cast->Update();

  vtkImageGaussianSmooth* Smooth = vtkImageGaussianSmooth::New();
  Smooth->SetStandardDeviation(1.5,1.5,1.5);
  Smooth->SetRadiusFactors(15,15,15);
#if (VTK_MAJOR_VERSION <= 5)
  Smooth->SetInput(Cast->GetOutput());
#else
  Smooth->SetInputConnection(Cast->GetOutputPort());
#endif
  Smooth->Update();

  //vtkImageFrangiFilter* Frangi = vtkImageFrangiFilter::New();
  //Frangi->SetInput(Smooth->GetOutput());
  //Frangi->SetSheet(0);
  //Frangi->SetLine(1);
  //Frangi->SetBlob(0);
  //Frangi->Update();
  
  vtkMetaImageWriter* Writer = vtkMetaImageWriter::New();
  Writer->SetFileName("E:\\jbaxter\\data\\CTA\\Recon2.mhd");
  Writer->SetRAWFileName("E:\\jbaxter\\data\\CTA\\Recon2.raw");
  Writer->SetCompression(false);
#if (VTK_MAJOR_VERSION <= 5)
  Writer->SetInput(Cast->GetOutput());
#else
  Writer->SetInputConnection(Cast->GetOutputPort());
#endif

  Writer->Write();

  Reader->Delete();
  Cast->Delete();
  Smooth->Delete();
  //Frangi->Delete();
  Writer->Delete();

  return 0;

}

//#include "vtkTree.h"
//#include "vtkMutableDirectedGraph.h"
//#include "vtkTreeWriter.h"
//#include "vtkStringArray.h"
//#include "vtkDoubleArray.h"
//#include "vtkIntArray.h"
//#include "vtkDataSet.h"
//#include "vtkDataSetAttributes.h"
//
//int main(int arc, char** argv){
//
//  //create nodes
//  vtkMutableDirectedGraph* mut = vtkMutableDirectedGraph::New();
//  vtkIdType source = mut->AddVertex();
//  vtkIdType bkg = mut->AddChild(source);
//  vtkIdType brain = mut->AddChild(source);
//  vtkIdType eCSF = mut->AddChild(brain);
//  vtkIdType inte = mut->AddChild(brain);
//  vtkIdType cgm = mut->AddChild(inte);
//  vtkIdType sub = mut->AddChild(inte);
//  vtkIdType wm = mut->AddChild(sub);
//  vtkIdType subw = mut->AddChild(sub);
//  vtkIdType sgm = mut->AddChild(subw);
//  vtkIdType vent = mut->AddChild(subw);
//
//  //create input filenames
//  vtkStringArray* DataCosts = vtkStringArray::New();
//  DataCosts->SetName("DataTerm");
//  mut->GetVertexData()->AddArray(DataCosts);
//  DataCosts->InsertValue(bkg,    "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\cost0");
//  DataCosts->InsertValue(eCSF,  "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\cost1");
//  DataCosts->InsertValue(cgm,    "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\cost2");
//  DataCosts->InsertValue(wm,    "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\cost3");
//  DataCosts->InsertValue(sgm,    "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\cost4");
//  DataCosts->InsertValue(vent,  "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\cost5");
//
//  //create smoothness filenames
//  vtkStringArray* SmoothCosts = vtkStringArray::New();
//  SmoothCosts->SetName("SmoothnessTerm");
//  mut->GetVertexData()->AddArray(SmoothCosts);
//
//
//  //create smoothness alphas
//  vtkDoubleArray* Alphas = vtkDoubleArray::New();
//  Alphas->SetName("Alpha");
//  mut->GetVertexData()->AddArray(Alphas);
//  Alphas->InsertValue(source,    0);
//  Alphas->InsertValue(bkg,    0);
//  Alphas->InsertValue(brain,  1);
//  Alphas->InsertValue(eCSF,  0);
//  Alphas->InsertValue(inte,  1);
//  Alphas->InsertValue(cgm,  0);
//  Alphas->InsertValue(sub,  1);
//  Alphas->InsertValue(wm,    0);
//  Alphas->InsertValue(subw,  1);
//  Alphas->InsertValue(sgm,  0);
//  Alphas->InsertValue(vent,  1);
//
//  //create output filenames
//  vtkStringArray* Output = vtkStringArray::New();
//  Output->SetName("OutputLocation");
//  mut->GetVertexData()->AddArray(Output);
//  Output->InsertValue(bkg,  "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\l0");
//  Output->InsertValue(eCSF,  "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\l1");
//  Output->InsertValue(cgm,  "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\l2");
//  Output->InsertValue(wm,    "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\l3");
//  Output->InsertValue(sgm,  "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\l4");
//  Output->InsertValue(vent,  "E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\l5");
//  
//
//  //create indentifier numbers
//  vtkIntArray* Ident = vtkIntArray::New();
//  Ident->SetName("Identifier");
//  mut->GetVertexData()->AddArray(Ident);
//  Ident->InsertValue(source,  -1);
//  Ident->InsertValue(bkg,    0);
//  Ident->InsertValue(brain,  -1);
//  Ident->InsertValue(eCSF,  1);
//  Ident->InsertValue(inte,  -1);
//  Ident->InsertValue(cgm,    2);
//  Ident->InsertValue(sub,    -1);
//  Ident->InsertValue(wm,    3);
//  Ident->InsertValue(subw,  -1);
//  Ident->InsertValue(sgm,    4);
//  Ident->InsertValue(vent,  5);
//
//  //create and write tree
//  vtkTree* Tree = vtkTree::New();
//  Tree->CheckedDeepCopy(mut);
//  Tree->Update();
//  vtkTreeWriter* TreeWriter = vtkTreeWriter::New();
//  TreeWriter->SetInput(Tree);
//  TreeWriter->SetFileName("E:\\jbaxter\\data\\testingPARAMTUNER\\brainseg\\tree.vtk");
//  TreeWriter->Write();
//
//  //cleanup
//  DataCosts->Delete();
//  SmoothCosts->Delete();
//  Output->Delete();
//  Ident->Delete();
//  Alphas->Delete();
//  Tree->Delete();
//  TreeWriter->Delete();
//}