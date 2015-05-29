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

#include <vtkSmartPointer.h>

int main(int argc, char** argv){
  vtkSmartPointer<vtkMutableDirectedGraph> mut = vtkSmartPointer<vtkMutableDirectedGraph>::New();
  vtkIdType source = mut->AddVertex();
  vtkIdType bkg = mut->AddVertex();
  vtkIdType c1 = mut->AddVertex();
  vtkIdType c2 = mut->AddVertex();
  vtkIdType l1 = mut->AddVertex();
  vtkIdType l2 = mut->AddVertex();
  vtkIdType l3 = mut->AddVertex();
  
  vtkSmartPointer<vtkFloatArray> Weights = vtkSmartPointer<vtkFloatArray>::New();
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

  vtkSmartPointer<vtkRootedDirectedAcyclicGraph> DAG = vtkSmartPointer<vtkRootedDirectedAcyclicGraph>::New();
  DAG->CheckedShallowCopy(mut);
  //DAG->GetEdgeData()->AddArray(Weights);

  vtkSmartPointer<vtkBMPReader> cost0 = vtkSmartPointer<vtkBMPReader>::New();
  cost0->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\cost0.bmp");
  vtkSmartPointer<vtkBMPReader> cost1 = vtkSmartPointer<vtkBMPReader>::New();
  cost1->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\cost1.bmp");
  vtkSmartPointer<vtkBMPReader> cost2 = vtkSmartPointer<vtkBMPReader>::New();
  cost2->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\cost2.bmp");
  vtkSmartPointer<vtkBMPReader> cost3 = vtkSmartPointer<vtkBMPReader>::New();
  cost3->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\cost3.bmp");

  vtkSmartPointer<vtkImageExtractComponents> extract0 = vtkSmartPointer<vtkImageExtractComponents>::New();
  extract0->SetComponents(0);
  extract0->SetInputConnection(cost0->GetOutputPort());
  vtkSmartPointer<vtkImageCast> cast0 = vtkSmartPointer<vtkImageCast>::New();
  cast0->SetOutputScalarTypeToFloat();
  cast0->SetInputConnection(extract0->GetOutputPort());
  vtkSmartPointer<vtkImageExtractComponents> extract1 = vtkSmartPointer<vtkImageExtractComponents>::New();
  extract1->SetComponents(0);
  extract1->SetInputConnection(cost1->GetOutputPort());
  vtkSmartPointer<vtkImageCast> cast1 = vtkSmartPointer<vtkImageCast>::New();
  cast1->SetOutputScalarTypeToFloat();
  cast1->SetInputConnection(extract1->GetOutputPort());
  vtkSmartPointer<vtkImageExtractComponents> extract2 = vtkSmartPointer<vtkImageExtractComponents>::New();
  extract2->SetComponents(0);
  extract2->SetInputConnection(cost2->GetOutputPort());
  vtkSmartPointer<vtkImageCast> cast2 = vtkSmartPointer<vtkImageCast>::New();
  cast2->SetOutputScalarTypeToFloat();
  cast2->SetInputConnection(extract2->GetOutputPort());
  vtkSmartPointer<vtkImageExtractComponents> extract3 = vtkSmartPointer<vtkImageExtractComponents>::New();
  extract3->SetComponents(0);
  extract3->SetInputConnection(cost3->GetOutputPort());
  vtkSmartPointer<vtkImageCast> cast3 = vtkSmartPointer<vtkImageCast>::New();
  cast3->SetOutputScalarTypeToFloat();
  cast3->SetInputConnection(extract3->GetOutputPort());

  vtkSmartPointer<vtkCudaDirectedAcyclicGraphMaxFlowSegmentation> dagmf =
	  vtkSmartPointer<vtkCudaDirectedAcyclicGraphMaxFlowSegmentation>::New();
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
  
  vtkSmartPointer<vtkImageData> test0 = vtkSmartPointer<vtkImageData>::New();
  test0->ShallowCopy((vtkImageData*) dagmf->GetOutput(bkg));
  vtkSmartPointer<vtkImageData> test1 = vtkSmartPointer<vtkImageData>::New();
  test1->ShallowCopy((vtkImageData*) dagmf->GetOutput(l1));
  vtkSmartPointer<vtkImageData> test2 = vtkSmartPointer<vtkImageData>::New();
  test2->ShallowCopy((vtkImageData*) dagmf->GetOutput(l2));
  vtkSmartPointer<vtkImageData> test3 = vtkSmartPointer<vtkImageData>::New();
  test3->ShallowCopy((vtkImageData*) dagmf->GetOutput(l3));

  vtkSmartPointer<vtkMetaImageWriter> writer = vtkSmartPointer<vtkMetaImageWriter>::New();
#if (VTK_MAJOR_VERSION <= 5)
	writer->SetInput(test0);
#else
	writer->SetInputData(test0);
#endif
  writer->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\l0.mhd");
  writer->SetRAWFileName("E:\\jbaxter\\data\\DAGMF_testing\\l0.raw");
  writer->Update();
  writer->Write();
#if (VTK_MAJOR_VERSION <= 5)
	writer->SetInput(test1);
#else
	writer->SetInputData(test1);
#endif
  writer->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\l1.mhd");
  writer->SetRAWFileName("E:\\jbaxter\\data\\DAGMF_testing\\l1.raw");
  writer->Update();
  writer->Write();
#if (VTK_MAJOR_VERSION <= 5)
	writer->SetInput(test2);
#else
	writer->SetInputData(test2);
#endif
  writer->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\l2.mhd");
  writer->SetRAWFileName("E:\\jbaxter\\data\\DAGMF_testing\\l2.raw");
  writer->Update();
  writer->Write();
#if (VTK_MAJOR_VERSION <= 5)
	writer->SetInput(test3);
#else
	writer->SetInputData(test3);
#endif
  writer->SetFileName("E:\\jbaxter\\data\\DAGMF_testing\\l3.mhd");
  writer->SetRAWFileName("E:\\jbaxter\\data\\DAGMF_testing\\l3.raw");
  writer->Update();
  writer->Write();

  return 0;
}
