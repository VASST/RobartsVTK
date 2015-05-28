#include "vtkRootedDirectedAcyclicGraph.h"
#include "vtkMutableDirectedGraph.h"
#include "vtkRootedDirectedAcyclicGraphIterator.h"
#include "vtkRootedDirectedAcyclicGraphForwardIterator.h"
#include "vtkRootedDirectedAcyclicGraphBackwardIterator.h"
#include <iostream>

#include "vtkDirectedAcyclicGraphMaxFlowSegmentation.h"
#include "vtkCudaDirectedAcyclicGraphMaxFlowSegmentation.h"

#include "vtkImageExtractComponents.h"
#include "vtkImageCast.h"
#include "vtkBMPReader.h"
#include "vtkMetaImageWriter.h"
#include "vtkImageData.h"

#include "vtkFloatArray.h"
#include "vtkDataSetAttributes.h"

int main(int argc, char** argv){
  vtkMutableDirectedGraph* mut = vtkMutableDirectedGraph::New();
  vtkIdType source = mut->AddVertex();
  vtkIdType bkg = mut->AddVertex();
  vtkIdType c1 = mut->AddVertex();
  vtkIdType c2 = mut->AddVertex();
  vtkIdType l1 = mut->AddVertex();
  vtkIdType l2 = mut->AddVertex();
  vtkIdType l3 = mut->AddVertex();
  
  vtkFloatArray* Weights = vtkFloatArray::New();
  Weights->SetName("Weights");
  Weights->InsertValue((mut->AddEdge(source,bkg)).Id,1.0f);
  Weights->InsertValue((mut->AddEdge(source,c1)).Id,1.0f);
  Weights->InsertValue((mut->AddEdge(source,c2)).Id,1.0f);
  Weights->InsertValue((mut->AddEdge(source,l1)).Id,0.5f);
  Weights->InsertValue((mut->AddEdge(source,l2)).Id,0.5f);
  Weights->InsertValue((mut->AddEdge(c1,l1)).Id,0.5f);
  Weights->InsertValue((mut->AddEdge(c2,l2)).Id,0.5f);
  Weights->InsertValue((mut->AddEdge(c1,l3)).Id,0.5f);
  Weights->InsertValue((mut->AddEdge(c2,l3)).Id,0.5f);
  
  //vtkIdType c1 = mut->AddChild(source);
  //vtkIdType c2 = mut->AddChild(c1);
  //vtkIdType l1 = mut->AddChild(c1);
  //vtkIdType l2 = mut->AddChild(c2);
  //vtkIdType l3 = mut->AddChild(c2);

  vtkRootedDirectedAcyclicGraph* DAG = vtkRootedDirectedAcyclicGraph::New();
  DAG->CheckedShallowCopy(mut);
  //DAG->GetEdgeData()->AddArray(Weights);

  vtkBMPReader* cost0 = vtkBMPReader::New();
  cost0->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\cost0.bmp");
  vtkBMPReader* cost1 = vtkBMPReader::New();
  cost1->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\cost1.bmp");
  vtkBMPReader* cost2 = vtkBMPReader::New();
  cost2->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\cost2.bmp");
  vtkBMPReader* cost3 = vtkBMPReader::New();
  cost3->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\cost3.bmp");

  vtkImageExtractComponents* extract0 = vtkImageExtractComponents::New();
  extract0->SetComponents(0);
  extract0->SetInput(cost0->GetOutput());
  vtkImageCast* cast0 = vtkImageCast::New();
  cast0->SetOutputScalarTypeToFloat();
  cast0->SetInput(extract0->GetOutput());
  vtkImageExtractComponents* extract1 = vtkImageExtractComponents::New();
  extract1->SetComponents(0);
  extract1->SetInput(cost1->GetOutput());
  vtkImageCast* cast1 = vtkImageCast::New();
  cast1->SetOutputScalarTypeToFloat();
  cast1->SetInput(extract1->GetOutput());
  vtkImageExtractComponents* extract2 = vtkImageExtractComponents::New();
  extract2->SetComponents(0);
  extract2->SetInput(cost2->GetOutput());
  vtkImageCast* cast2 = vtkImageCast::New();
  cast2->SetOutputScalarTypeToFloat();
  cast2->SetInput(extract2->GetOutput());
  vtkImageExtractComponents* extract3 = vtkImageExtractComponents::New();
  extract3->SetComponents(0);
  extract3->SetInput(cost3->GetOutput());
  vtkImageCast* cast3 = vtkImageCast::New();
  cast3->SetOutputScalarTypeToFloat();
  cast3->SetInput(extract3->GetOutput());

  vtkCudaDirectedAcyclicGraphMaxFlowSegmentation* dagmf = vtkCudaDirectedAcyclicGraphMaxFlowSegmentation::New();
  dagmf->SetStructure(DAG);
  dagmf->SetDataInput(bkg,cast0->GetOutput());
  dagmf->SetDataInput(l1,cast1->GetOutput());
  dagmf->SetDataInput(l2,cast2->GetOutput());
  dagmf->SetDataInput(l3,cast3->GetOutput());
  dagmf->AddSmoothnessScalar(bkg,0.01);
  dagmf->AddSmoothnessScalar(l1, 0.01);
  dagmf->AddSmoothnessScalar(l2, 0.01);
  dagmf->AddSmoothnessScalar(l3, 0.01);
  dagmf->AddSmoothnessScalar(c1, 50);
  dagmf->AddSmoothnessScalar(c2, 50);
  //dagmf->AddSmoothnessScalar(bkg,0);
  //dagmf->AddSmoothnessScalar(l1, 0);
  //dagmf->AddSmoothnessScalar(l2, 0);
  //dagmf->AddSmoothnessScalar(l3, 0);
  //dagmf->AddSmoothnessScalar(c1, 0);
  //dagmf->AddSmoothnessScalar(c2, 0);
  dagmf->SetCC(0.01);
  dagmf->SetStepSize(0.1);
  dagmf->SetNumberOfIterations(100);
  dagmf->Update();
  
  vtkImageData* test0 = vtkImageData::New();
  test0->ShallowCopy((vtkImageData*) dagmf->GetOutput(bkg));
  vtkImageData* test1 = vtkImageData::New();
  test1->ShallowCopy((vtkImageData*) dagmf->GetOutput(l1));
  vtkImageData* test2 = vtkImageData::New();
  test2->ShallowCopy((vtkImageData*) dagmf->GetOutput(l2));
  vtkImageData* test3 = vtkImageData::New();
  test3->ShallowCopy((vtkImageData*) dagmf->GetOutput(l3));

  vtkMetaImageWriter* writer = vtkMetaImageWriter::New();
  writer->SetInput(test0);
  writer->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\l0.mhd");
  writer->SetRAWFileName("E:\\jbaxter\\data\\DAGMF_testing\\l0.raw");
  writer->Update();
  writer->Write();
  writer->SetInput(test1);
  writer->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\l1.mhd");
  writer->SetRAWFileName("E:\\jbaxter\\data\\DAGMF_testing\\l1.raw");
  writer->Update();
  writer->Write();
  writer->SetInput(test2);
  writer->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\l2.mhd");
  writer->SetRAWFileName("E:\\jbaxter\\data\\DAGMF_testing\\l2.raw");
  writer->Update();
  writer->Write();
  writer->SetInput(test3);
  writer->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\l3.mhd");
  writer->SetRAWFileName("E:\\jbaxter\\data\\DAGMF_testing\\l3.raw");
  writer->Update();
  writer->Write();
  writer->Delete();

  test0->Delete();
  test1->Delete();
  test2->Delete();
  test3->Delete();
  dagmf->Delete();
  cost0->Delete();
  cost1->Delete();
  cost2->Delete();
  cost3->Delete();
  cast0->Delete();
  cast1->Delete();
  cast2->Delete();
  cast3->Delete();
  extract0->Delete();
  extract1->Delete();
  extract2->Delete();
  extract3->Delete();
  DAG->Delete();
  mut->Delete();
  Weights->Delete();
  return 0;
}
