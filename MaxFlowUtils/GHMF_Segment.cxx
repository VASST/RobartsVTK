#include "vtkCudaHierarchicalMaxFlowSegmentation2.h"
#include "vtkCudaImageVote.h"
#include "vtkMetaImageReader.h"
#include "vtkMetaImageWriter.h"

#include "vtkTree.h"
#include "vtkTreeReader.h"
#include "vtkTreeDFSIterator.h"
#include "vtkType.h"
#include "vtkDataSetAttributes.h"
#include "vtkStringArray.h"
#include "vtkDoubleArray.h"
#include "vtkIntArray.h"

#include <string>

#include "vtkImageExtractComponents.h"
#include "vtkImageCast.h"
#include "vtkMutableDirectedGraph.h"
#include "vtkTreeWriter.h"
#include "vtkBMPReader.h"
int main1(int argc, char** argv){

	vtkMutableDirectedGraph* mut = vtkMutableDirectedGraph::New();
	vtkIdType source = mut->AddVertex();
	vtkIdType l0 = mut->AddChild(source);
	vtkIdType l1 = mut->AddChild(source);
	vtkIdType l2 = mut->AddChild(source);

	vtkStringArray* DataTerms = vtkStringArray::New();
	DataTerms->SetName("DataTerm");
	DataTerms->InsertValue(source,"");
	DataTerms->InsertValue(l0,"E:\\jbaxter\\Publications\\2014_jmi_metaoptimization\\cost0.mhd");
	DataTerms->InsertValue(l1,"E:\\jbaxter\\Publications\\2014_jmi_metaoptimization\\cost1.mhd");
	DataTerms->InsertValue(l2,"E:\\jbaxter\\Publications\\2014_jmi_metaoptimization\\cost2.mhd");
	
	vtkStringArray* SmoothnessTerm = vtkStringArray::New();
	SmoothnessTerm->SetName("SmoothnessTerm");
	SmoothnessTerm->InsertValue(source,"");
	SmoothnessTerm->InsertValue(l0,"");
	SmoothnessTerm->InsertValue(l1,"");
	SmoothnessTerm->InsertValue(l2,"");
	
	vtkStringArray* OutputLocation = vtkStringArray::New();
	OutputLocation->SetName("OutputLocation");
	OutputLocation->InsertValue(source,"");
	OutputLocation->InsertValue(l0,"E:\\jbaxter\\Publications\\2014_jmi_metaoptimization\\l0");
	OutputLocation->InsertValue(l1,"E:\\jbaxter\\Publications\\2014_jmi_metaoptimization\\l1");
	OutputLocation->InsertValue(l2,"E:\\jbaxter\\Publications\\2014_jmi_metaoptimization\\l2");

	vtkDoubleArray* Alphas = vtkDoubleArray::New();
	Alphas->SetName("Alphas");
	Alphas->InsertTuple1(source,1);
	Alphas->InsertTuple1(l0,1);
	Alphas->InsertTuple1(l1,1);
	Alphas->InsertTuple1(l2,1);
	
	vtkIntArray* Identifier = vtkIntArray::New();
	Identifier->SetName("Identifier");
	Identifier->InsertTuple1(source,-1);
	Identifier->InsertTuple1(l0,0);
	Identifier->InsertTuple1(l1,1);
	Identifier->InsertTuple1(l2,2);
	
	mut->GetVertexData()->AddArray(DataTerms);
	mut->GetVertexData()->AddArray(SmoothnessTerm);
	mut->GetVertexData()->AddArray(OutputLocation);
	mut->GetVertexData()->AddArray(Alphas);
	mut->GetVertexData()->AddArray(Identifier);
	mut->Update();
	vtkTree* Tree = vtkTree::New();
	Tree->CheckedDeepCopy(mut);

	for(int i = 0; i < Tree->GetVertexData()->GetNumberOfArrays(); i++){
		if( Tree->GetVertexData()->GetAbstractArray(i) ) 
		std::cout << Tree->GetVertexData()->GetAbstractArray(i)->GetName() << "\t: " <<  Tree->GetVertexData()->GetAbstractArray(i) << std::endl;
		else
		std::cout << "unknown\t: " <<  Tree->GetVertexData()->GetAbstractArray(i) << std::endl;
	}

	vtkTreeWriter* Writer  = vtkTreeWriter::New();
	Writer->SetFileName("E:\\jbaxter\\Publications\\2014_jmi_metaoptimization\\tree.vtk");
	Writer->SetInput(Tree);
	Writer->Write();

	Writer->Delete();
	Tree->Delete();
	mut->Delete();
	DataTerms->Delete();
	SmoothnessTerm->Delete();
	OutputLocation->Delete();
	Alphas->Delete();
	Identifier->Delete();

	return 0;

}

int main(int argc, char** argv){

	//check number of iterations
	int NumIts = std::atoi(argv[2]);
	int NumDev = std::atoi(argv[3]);

	//load tree filename and output filename
	std::string TreeFilename = std::string(argv[1]);
	vtkTreeReader* TreeReader = vtkTreeReader::New();
	TreeReader->SetFileName(TreeFilename.c_str());
	TreeReader->Update();
	vtkTree* Tree = TreeReader->GetOutput();

	//create segmenter
	vtkCudaHierarchicalMaxFlowSegmentation2* Segmenter = vtkCudaHierarchicalMaxFlowSegmentation2::New();
	Segmenter->SetHierarchy(Tree);
	Segmenter->SetNumberOfIterations(NumIts);
	Segmenter->ClearDevices();
	for(int i = 0; i < NumDev; i++)
		Segmenter->AddDevice(std::atoi(argv[4+i]));

	//get information arrays
	vtkStringArray* DataTerms = (vtkStringArray*) Tree->GetVertexData()->GetAbstractArray("DataTerm");
	vtkStringArray* SmoothTerms = (vtkStringArray*) Tree->GetVertexData()->GetAbstractArray("SmoothnessTerm");
	vtkStringArray* OutLoc = (vtkStringArray*) Tree->GetVertexData()->GetAbstractArray("OutputLocation");
	vtkDoubleArray* Alphas = (vtkDoubleArray*) Tree->GetVertexData()->GetAbstractArray("Alpha");
	vtkIntArray* Identifiers = (vtkIntArray*) Tree->GetVertexData()->GetAbstractArray("Identifier"); 
	if(!DataTerms)
		std::cout << "No data terms provided." << std::endl;

	//add data terms and smoothness terms
	vtkTreeDFSIterator* Iterator = vtkTreeDFSIterator::New();
	Iterator->SetTree(Tree);
	Iterator->SetStartVertex(Tree->GetRoot());
	while(Iterator->HasNext()){
		vtkIdType Node = Iterator->Next();
		if( Node == Tree->GetRoot() ) continue;
		
		//read in Data term
		if( Tree->IsLeaf(Node) ){
			vtkMetaImageReader* Reader = vtkMetaImageReader::New();
			Reader->SetFileName( DataTerms->GetValue(Node).c_str() );
			Reader->Update();
			Segmenter->SetDataInput(Node, (vtkDataObject*) Reader->GetOutput());
			Reader->Delete();
		}

		//read in smoothness term
		if( SmoothTerms && SmoothTerms->GetValue(Node).length() > 0 ){
			vtkMetaImageReader* Reader = vtkMetaImageReader::New();
			Reader->SetFileName( DataTerms->GetValue(Node).c_str() );
			Reader->Update();
			Segmenter->SetSmoothnessInput(Node, (vtkDataObject*) Reader->GetOutput());
			Reader->Delete();
		}

		//read in alpha
		if(Alphas)
			Segmenter->AddSmoothnessScalar(Node, Alphas->GetValue(Node) );

	}
	Iterator->Delete();

	//run segmentation
	Segmenter->Update();

	//output files
	Iterator = vtkTreeDFSIterator::New();
	Iterator->SetTree(Tree);
	Iterator->SetStartVertex(Tree->GetRoot());
	while(Iterator->HasNext()){
		vtkIdType Node = Iterator->Next();
		if( !Tree->IsLeaf(Node) || OutLoc->GetValue(Node).length() < 1 ) continue;
		std::string OutFileBase = OutLoc->GetValue(Node);
		std::string OutFileMHD = OutFileBase + ".mhd";
		std::string OutFileRAW = OutFileBase + ".raw";
		vtkMetaImageWriter* Writer = vtkMetaImageWriter::New();
		Writer->SetFileName( OutFileMHD.c_str() );
		Writer->SetRAWFileName( OutFileRAW.c_str() );
		Writer->SetInput(Segmenter->GetOutput(Node));
		Writer->Write();
		Writer->Delete();
	}
	Iterator->Delete();

	//output merged file
	if( argv[4+NumDev] ){
		std::string OutFileBase = std::string(argv[4+NumDev]);
		std::string OutFileMHD = OutFileBase + ".mhd";
		std::string OutFileRAW = OutFileBase + ".raw";
		
		vtkCudaImageVote* Voter = vtkCudaImageVote::New();
		Voter->SetDevice(std::atoi(argv[4]));
		Voter->SetOutputDataType(VTK_INT);
		Iterator = vtkTreeDFSIterator::New();
		Iterator->SetTree(Tree);
		Iterator->SetStartVertex(Tree->GetRoot());
		while(Iterator->HasNext()){
			vtkIdType Node = Iterator->Next();
			if( !Tree->IsLeaf(Node) ) continue;
			Voter->SetInput( Identifiers->GetValue(Node), Segmenter->GetOutput(Node) );
		}
		Iterator->Delete();
		Voter->Update();
		vtkMetaImageWriter* Writer = vtkMetaImageWriter::New();
		Writer->SetFileName(OutFileMHD.c_str());
		Writer->SetRAWFileName(OutFileRAW.c_str());
		Writer->SetInput((vtkDataObject*)Voter->GetOutput());
		Writer->Write();
		Writer->Delete();
		Voter->Delete();
	}

	//cleanup
	Segmenter->Delete();
	TreeReader->Delete();

	return 0;
}