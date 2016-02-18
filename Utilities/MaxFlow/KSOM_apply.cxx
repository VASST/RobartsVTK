/*------------------------------------------------------------------------------//
KSOM_train.exe

Description:
This file is a utility for applying the GHMF solver. It will take a brief description
of the GHMF hierarchy with associated smoothness/data terms and alphas as well as
identifiers and output values

Usage:\t OutputFilename DeviceNumber MapFilename DataFilename

OutputFilename: The file root to save the LL term to
DeviceNumber: The device to use
MapFileName: The file containing the GMM
DataFileName: The file containing the image data

//------------------------------------------------------------------------------*/

#include "vtkCudaKSOMProbability.h"
#include "vtkPiecewiseFunction.h"
#include "vtkMetaImageReader.h"
#include "vtkMetaImageWriter.h"
#include "vtkImageThreshold.h"
#include "vtkImageCast.h"

#include <string>
#include <algorithm>
#include <fstream>

#include <vtkVersion.h> //for VTK_MAJOR_VERSION

void showHelpMessage()
{
  std::cerr << "Usage:\t OutputFilename DeviceNumber MapFilename DataFilename" << std::endl;
}

void showLongHelpMessage()
{
  std::cerr <<
            "Description:" << std::endl <<
            "This file is a utility for applying the MAPs as an LL term. It will take the file names " <<
            "of the maps and image data as well as a desired output location." << std::endl <<
            std::endl <<
            "Usage:\t OutputFilename DeviceNumber MapFilename DataFilename" << std::endl <<
            std::endl <<
            "OutputFilename: The file root to save the LL term to "<< std::endl <<
            "DeviceNumber: The device to use " << std::endl <<
            "MapFilename: The file containing the GMM " << std::endl <<
            "DataFilename: The file containing the image data" << std::endl;
}

int main( int argc, char** argv )
{

  if(argc < 2)
  {
    showHelpMessage();
    return 0;
  }

  //check for explicit help message
  std::string FirstArgument = std::string(argv[1]);
  std::transform(FirstArgument.begin(), FirstArgument.end(), FirstArgument.begin(), ::tolower);
  int found = (int) FirstArgument.find(std::string("help"));
  if(found!=std::string::npos)
  {
    showLongHelpMessage();
    return 0;
  }

  //make sure there are enough arguments
  if(argc < 5)
  {
    showHelpMessage();
    return 0;
  }

  //get command line information
  int DeviceNumber = std::atoi(argv[2]);
  std::string OutputFilename = std::string(argv[1]);
  std::string MapFilename = std::string(argv[3]);
  std::string DataFilename = std::string(argv[3]);

  //load image and map
  vtkMetaImageReader* DataReader = vtkMetaImageReader::New();
  DataReader->SetFileName(DataFilename.c_str());
  vtkImageCast* Caster = vtkImageCast::New();
#if (VTK_MAJOR_VERSION < 6)
  Caster->SetInput(DataReader->GetOutput());
#else
  Caster->SetInputConnection(DataReader->GetOutputPort());
#endif
  Caster->SetOutputScalarTypeToFloat();
  Caster->Update();
  vtkMetaImageReader* MapReader = vtkMetaImageReader::New();
  MapReader->SetFileName(MapFilename.c_str());
  MapReader->Update();

  //run applier
  vtkCudaKSOMProbability* Applier = vtkCudaKSOMProbability::New();
  Applier->SetDevice(DeviceNumber);
#if (VTK_MAJOR_VERSION < 6)
  Applier->SetImageInput(Caster->GetOutput());
  Applier->SetMapInput(MapReader->GetOutput());
#else
  Applier->SetImageInputConnection(Caster->GetOutputPort());
  Applier->SetMapInputConnection(MapReader->GetOutputPort());
#endif
  Applier->SetEntropy(true);
  Applier->Update();

  //write output
  std::string OutputFilenameMHD = OutputFilename + ".mhd";
  std::string OutputFilenameRAW = OutputFilename + ".raw";
  vtkMetaImageWriter* Writer = vtkMetaImageWriter::New();
  Writer->SetFileName(OutputFilenameMHD.c_str());
  Writer->SetRAWFileName(OutputFilenameRAW.c_str());
#if (VTK_MAJOR_VERSION < 6)
  Writer->SetInput(Applier->GetOutput());
#else
  Writer->SetInputConnection(Applier->GetOutputPort());
#endif
  Writer->Write();
  Writer->Delete();

  //cleanup
  Applier->Delete();
  Caster->Delete();
  DataReader->Delete();
  MapReader->Delete();
}