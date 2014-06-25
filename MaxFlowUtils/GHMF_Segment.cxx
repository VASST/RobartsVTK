/*------------------------------------------------------------------------------//
GHMF_Segment.exe

Description:
This file is a utility for applying the GHMF solver. It will take a brief description
of the GHMF hierarchy with associated smoothness/data terms and alphas as well as
identifiers and output values

Usage:\t TreeFilename NumberOfIterations NumberOfDevices Device1 ... DeviceN [OutputFilename]

The tree is saved in a VTK file with the following attributes:
"DataTerm": (mandatory) filename for the data term
"SmoothnessTerm": (optional) filename for the smoothness term
"OutputLocation": (optional) filename to save probabilistic label to
"Alpha": (optional) filename for the smoothness term
"Identifier": (mandatory iff OutputFilename is specified) unique label integer to associate
				in merged file (discrete segmentation)

//------------------------------------------------------------------------------*/

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
#include <algorithm>

void showHelpMessage(){
	std::cerr << "Usage:\t TreeFilename NumberOfIterations NumberOfDevices Device1 ... DeviceN " <<
		"[-output OutputFilename] [-step StepSize] [-cc VanishingRatio]" << std::endl;
}

void showLongHelpMessage(){
	std::cerr <<
	"Description:" << std::endl <<
	"This file is a utility for applying the GHMF solver. It will take a brief description" <<
	"of the GHMF hierarchy with associated smoothness/data terms and alphas as well as" <<
	"identifiers and output values" << std::endl <<
	std::endl <<
	"Usage:\t TreeFilename NumberOfIterations NumberOfDevices Device1 ... DeviceN " <<
		"[-output OutputFilename] [-step StepSize] [-cc VanishingRatio]" << std::endl <<
	"The tree is saved in a VTK file with the following attributes:" << std::endl <<
	"\"DataTerm\": (mandatory) filename for the data term" << std::endl <<
	"\"SmoothnessTerm\": (optional) filename for the smoothness term" << std::endl <<
	"\"OutputLocation\": (optional) filename to save probabilistic label to" << std::endl <<
	"\"Alpha\": (optional) filename for the smoothness term" << std::endl <<
	"\"Identifier\": (mandatory iff OutputFilename is specified) unique label integer to associate" <<
					" in merged file (discrete segmentation)" << std::endl;
}

int main(int argc, char** argv){
	
	if(argc < 2){
		showHelpMessage();
		return 0;
	}

	//check for explicit help message
	std::string FirstArgument = std::string(argv[1]);
	std::transform(FirstArgument.begin(), FirstArgument.end(), FirstArgument.begin(), ::tolower);
	int found = (int) FirstArgument.find(std::string("help"));
	if(found!=std::string::npos){
		showLongHelpMessage();
		return 0;
	}

	//make sure there are enough arguments
	if(argc < 5){
		showHelpMessage();
		return 0;
	}

	//check number of iterations
	int NumIts = std::atoi(argv[2]);
	int NumDev = std::atoi(argv[3]);
	int NumFlags = argc - (3+NumDev);
	double Tau = 0.1;
	double CC = 0.25;
	bool hasOutput = false;
	std::string OutFileBase = "";

	//read in flags
	for(int i = 0; i < NumFlags; i++){
		std::string command = std::string(argv[4+NumDev+2*i]);
		if( command.compare("-output") )
			{hasOutput = true; OutFileBase = std::string(argv[5+NumDev+2*i]); }
		else if( command.compare("-step") )
			Tau = std::atof(argv[5+NumDev+2*i]);
		else if( command.compare("-cc") )
			CC = std::atof(argv[5+NumDev+2*i]);
		else
			{showHelpMessage(); return 0;}
	}

	//load tree filename and output filename
	std::string TreeFilename = std::string(argv[1]);
	vtkTreeReader* TreeReader = vtkTreeReader::New();
	TreeReader->SetFileName(TreeFilename.c_str());
	TreeReader->Update();
	vtkTree* Tree = TreeReader->GetOutput();

	//create segmenter
	vtkCudaHierarchicalMaxFlowSegmentation2* Segmenter = vtkCudaHierarchicalMaxFlowSegmentation2::New();
	Segmenter->SetStructure(Tree);
	Segmenter->SetNumberOfIterations(NumIts);
	Segmenter->SetStepSize(Tau);
	Segmenter->SetCC(CC);
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
		std::cout << "No data terms provided in tree file. (No field named \"DataTerm\".)" << std::endl;

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
			Reader->SetFileName( SmoothTerms->GetValue(Node).c_str() );
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
	if( hasOutput && Identifiers ){
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
	}else if(argv[4+NumDev]){
		std::cerr << "Identifiers not given in tree file. (No field named \"Identifier\".) " << std::endl;
	}

	//cleanup
	Segmenter->Delete();
	TreeReader->Delete();

	return 0;
}