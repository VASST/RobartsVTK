/*=========================================================================

Program:   Robarts Visualization Toolkit

Copyright (c) John Stuart Haberl Baxter, Robarts Research Institute

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkBMPReader.h"
#include "vtkCudaDirectedAcyclicGraphMaxFlowSegmentation.h"
#include "vtkDataSetAttributes.h"
#include "vtkDirectedAcyclicGraphMaxFlowSegmentation.h"
#include "vtkFloatArray.h"
#include "vtkImageCast.h"
#include "vtkImageData.h"
#include "vtkImageExtractComponents.h"
#include "vtkMetaImageWriter.h"
#include "vtkMutableDirectedGraph.h"
#include "vtkRootedDirectedAcyclicGraph.h"
#include "vtkRootedDirectedAcyclicGraphBackwardIterator.h"
#include "vtkRootedDirectedAcyclicGraphForwardIterator.h"
#include "vtkRootedDirectedAcyclicGraphIterator.h"
#include "vtksys/CommandLineArguments.hxx"
#include "vtksys/SystemTools.hxx"
#include <iostream>
#include <vtkSmartPointer.h>
#include <vtkVersion.h>

int main(int argc, char** argv)
{
  // Check command line arguments.
  bool printHelp(false);
  std::string outputDirectory;

  vtksys::CommandLineArguments args;
  args.Initialize( argc, argv );

  args.AddArgument("--help", vtksys::CommandLineArguments::NO_ARGUMENT, &printHelp, "Print this help.");
  args.AddArgument( "--output-dir", vtksys::CommandLineArguments::EQUAL_ARGUMENT, &outputDirectory, "Name of the output directory." );

  if ( !args.Parse() )
  {
    std::cerr << "Problem parsing arguments." << std::endl;
    std::cout << "Help: " << args.GetHelp() << std::endl;
    exit(EXIT_FAILURE);
  }

  if ( printHelp )
  {
    std::cout << args.GetHelp() << std::endl;
    exit(EXIT_SUCCESS);
  }

  if( !vtksys::SystemTools::FileExists(outputDirectory.c_str(), false) && !vtksys::SystemTools::MakeDirectory(outputDirectory) )
  {
    std::cerr << "Output directory doesn't exist and can't be created." << std::endl;
  }

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

  vtkSmartPointer<vtkRootedDirectedAcyclicGraph> DAG = vtkSmartPointer<vtkRootedDirectedAcyclicGraph>::New();
  DAG->CheckedShallowCopy(mut);

  vtkSmartPointer<vtkBMPReader> cost0 = vtkSmartPointer<vtkBMPReader>::New();
  cost0->SetFileName( std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/cost0.bmp").c_str() );
  vtkSmartPointer<vtkBMPReader> cost1 = vtkSmartPointer<vtkBMPReader>::New();
  cost1->SetFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/cost1.bmp").c_str());
  vtkSmartPointer<vtkBMPReader> cost2 = vtkSmartPointer<vtkBMPReader>::New();
  cost2->SetFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/cost2.bmp").c_str());
  vtkSmartPointer<vtkBMPReader> cost3 = vtkSmartPointer<vtkBMPReader>::New();
  cost3->SetFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/cost3.bmp").c_str());

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
  dagmf->SetDataInputConnection(bkg,cast0->GetOutputPort());
  dagmf->SetDataInputConnection(l1,cast1->GetOutputPort());
  dagmf->SetDataInputConnection(l2,cast2->GetOutputPort());
  dagmf->SetDataInputConnection(l3,cast3->GetOutputPort());
  dagmf->AddSmoothnessScalar(bkg,0.01);
  dagmf->AddSmoothnessScalar(l1, 0.01);
  dagmf->AddSmoothnessScalar(l2, 0.01);
  dagmf->AddSmoothnessScalar(l3, 0.01);
  dagmf->AddSmoothnessScalar(c1, 50);
  dagmf->AddSmoothnessScalar(c2, 50);
  dagmf->SetCC(0.01);
  dagmf->SetStepSize(0.1);
  dagmf->SetNumberOfIterations(100);
  dagmf->Update();

  vtkSmartPointer<vtkImageData> test0 = vtkSmartPointer<vtkImageData>::New();
  test0->ShallowCopy((vtkImageData*) dagmf->GetOutputDataObject(bkg));
  vtkSmartPointer<vtkImageData> test1 = vtkSmartPointer<vtkImageData>::New();
  test1->ShallowCopy((vtkImageData*) dagmf->GetOutputDataObject(l1));
  vtkSmartPointer<vtkImageData> test2 = vtkSmartPointer<vtkImageData>::New();
  test2->ShallowCopy((vtkImageData*) dagmf->GetOutputDataObject(l2));
  vtkSmartPointer<vtkImageData> test3 = vtkSmartPointer<vtkImageData>::New();
  test3->ShallowCopy((vtkImageData*) dagmf->GetOutputDataObject(l3));

  vtkSmartPointer<vtkMetaImageWriter> writer = vtkSmartPointer<vtkMetaImageWriter>::New();
  writer->SetInputData(test0);
  writer->SetFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/l0.mhd").c_str());
  writer->SetRAWFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/l0.raw").c_str());
  writer->Update();
  writer->Write();
  writer->SetInputData(test1);
  writer->SetFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/l1.mhd").c_str());
  writer->SetRAWFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/l1.raw").c_str());
  writer->Update();
  writer->Write();
  writer->SetInputData(test2);
  writer->SetFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/l2.mhd").c_str());
  writer->SetRAWFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/l2.raw").c_str());
  writer->Update();
  writer->Write();
  writer->SetInputData(test3);
  writer->SetFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/l3.mhd").c_str());
  writer->SetRAWFileName(std::string(vtksys::SystemTools::GetFilenamePath(outputDirectory) + "/l3.raw").c_str());
  writer->Update();
  writer->Write();

  return 0;
}
