/*==========================================================================

Copyright (c) 2015 Uditha L. Jayarathne, ujayarat@robarts.ca

Use, modification and redistribution of the software, in source or
binary forms, are permitted provided that the following terms and
conditions are met:

1) Redistribution of the source code, in verbatim or modified
form, must retain the above copyright notice, this license,
the following disclaimer, and any notices that refer to this
license and/or the following disclaimer.

2) Redistribution in binary form must include the above copyright
notice, a copy of this license and the following disclaimer
in the documentation or with other materials provided with the
distribution.

3) Modified copies of the source code must be clearly marked as such,
and must not be misrepresented as verbatim copies of the source code.

THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGES.
=========================================================================*/

#include <iostream>
#include <string>
// For timing
#include <Windows.h>
#include <stdint.h>

// Use PLUS just to read data from sequence meta file
#include <vtkPlusSequenceIO.h>
#include <vtkPlusTrackedFrameList.h>
#include <vtkPlusVolumeReconstructor.h> // Use PLUS volume reconstructor for comparison purposes.
#include <vtkPlusTransformRepository.h>
#include <PlusTrackedFrame.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <vtkMetaImageWriter.h>
#include <vtkImageFlip.h>

#include <vtkCLVolumeReconstruction.h>

/* Sets output extent and origin from a vtkTrackedFrameList */
int get_extent_from_trackedList(vtkPlusTrackedFrameList*, vtkPlusTransformRepository*,
                                double spacing, int*, double*);

// This is for timing.
int gettimeofday(struct timeval* tp)
{
  // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
  static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

  SYSTEMTIME  system_time;
  FILETIME    file_time;
  uint64_t    time;

  GetSystemTime(&system_time);
  SystemTimeToFileTime(&system_time, &file_time);
  time = ((uint64_t)file_time.dwLowDateTime);
  time += ((uint64_t)file_time.dwHighDateTime) << 32;

  tp->tv_sec = (long)((time - EPOCH) / 10000000L);
  tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
  return 0;
}

int main()
{
  // Read in Data
  vtkSmartPointer< vtkPlusTrackedFrameList > trackedFrameList = vtkSmartPointer< vtkPlusTrackedFrameList >::New();
  // Read Sequence Meta file in the tracked framelist
  vtkPlusSequenceIO::Read(std::string("tracked_us_video.mha"), trackedFrameList);
  PlusTransformName transformName = PlusTransformName("Probe", "Tracker");
  vtkSmartPointer< vtkMatrix4x4 > tFrame2Tracker = vtkSmartPointer< vtkMatrix4x4 >::New();

  vtkSmartPointer< vtkImageFlip > imgFlip = vtkSmartPointer< vtkImageFlip >::New();
  imgFlip->SetFilteredAxis(1);

  // Initialize PLUS VolumeReconstructor
  std::string inputConfigFileName = "config.xml";
  vtkSmartPointer< vtkXMLDataElement> configRootElement = vtkSmartPointer< vtkXMLDataElement >::New();
  if (PlusXmlUtils::ReadDeviceSetConfigurationFromFile(configRootElement, inputConfigFileName.c_str()) == PLUS_FAIL)
  {
    LOG_ERROR("Unable to read configuration from file " << inputConfigFileName.c_str());
    return EXIT_FAILURE;
  }

  vtkSmartPointer< vtkPlusVolumeReconstructor > volumeReconstructor = vtkSmartPointer< vtkPlusVolumeReconstructor >::New();
  vtkSmartPointer< vtkPlusTransformRepository > repository = vtkSmartPointer< vtkPlusTransformRepository >::New();

  /* Read Coordinate system definitions from XML data */
  if (repository->ReadConfiguration(configRootElement) != PLUS_SUCCESS)
  {
    LOG_ERROR("Configuration incorrect for vtkTransformRepository.");
    exit(EXIT_FAILURE);
  }

  /* Read Coordinate system definitions from XML data */
  if (volumeReconstructor->ReadConfiguration(configRootElement) != PLUS_SUCCESS)
  {
    LOG_ERROR("Configuration incorrect for vtkTransformRepository.");
    exit(EXIT_FAILURE);
  }

  std::string errorDetail;
  /* Set output Extent from the input data. During streaming a scout scan may be necessary to set the output extent. */
  if (volumeReconstructor->SetOutputExtentFromFrameList(trackedFrameList, repository, errorDetail) != PLUS_SUCCESS)
  {
    LOG_ERROR("Setting up Output Extent from FrameList:" + errorDetail);
    exit(EXIT_FAILURE);
  }

  // Create vtkCLVolumeReconstructor
  vtkSmartPointer<vtkCLVolumeReconstruction> recon = vtkSmartPointer< vtkCLVolumeReconstruction >::New();
  recon->SetDevice(0);

  // calibration matrix
  float us_cal_mat[12] = { 0.0727f, 0.0076f, -0.0262f, -12.6030f,
                           -0.0030f, 0.0118f, -0.9873f, -7.8930f,
                           -0.0069f, 0.0753f, 0.1568f, 1.0670f
                         };

  recon->SetProgramSourcePath("kernels.cl");
  recon->SetBScanSize(820, 616);
  recon->SetBScanSpacing(0.077, 0.073);

  int extent[6] = { 0, 0, 0, 0, 0, 0 };
  double origin[3] = { 0, 0, 0 };
  double output_spacing = 0.5;

  get_extent_from_trackedList(trackedFrameList, repository, output_spacing, extent, origin);
  recon->SetOutputExtent(extent[0], extent[1], extent[2], extent[3], extent[4], extent[5]);
  recon->SetOutputSpacing(output_spacing);
  recon->SetOutputOrigin(origin[0], origin[1], origin[2]);
  recon->SetCalMatrix(us_cal_mat);
  try
  {
    recon->Initialize();
  }
  catch (const std::exception& e)
  {
    std::cerr << "Unable to initialize OpenCL volume reconstruction. Aborting. Error: " << e.what() << std::endl;
  }

  recon->StartReconstruction();

  std::cout << "vtkCLReconstruction initialized. " << std::endl;

  // Meta image writer
  vtkSmartPointer< vtkMetaImageWriter > writer = vtkSmartPointer< vtkMetaImageWriter >::New();
  writer->SetFileName("3DUS-output.mhd");

  vtkImageData* outputVolume = vtkImageData::New();

  //---------------- Now reconstruct --------------------------------------------------------------------------

  // For timing
  timeval tv;
  long current_time;

  for (unsigned int i = 0; i < trackedFrameList->GetNumberOfTrackedFrames(); i++)
  {
    // Set Image Dada
    PlusTrackedFrame* trackedFrame = trackedFrameList->GetTrackedFrame(i);
    imgFlip->SetInputData(trackedFrame->GetImageData()->GetImage());
    imgFlip->Modified();
    imgFlip->Update();
    recon->SetInputImageData(trackedFrame->GetTimestamp(), imgFlip->GetOutput());

    // Set pose Data
    trackedFrame->GetCustomFrameTransform(transformName, tFrame2Tracker);
    recon->SetInputPoseData(trackedFrame->GetTimestamp(), tFrame2Tracker);

    // Update Reconstruction
    std::cout << "Frame " << i << " : ";
    gettimeofday(&tv);
    current_time = tv.tv_usec;

    recon->UpdateReconstruction();
    recon->GetOutputVolume(outputVolume);
    gettimeofday(&tv);
    std::cout << "Elapsed time : " << tv.tv_usec - current_time << " micro seconds." << std::endl;


    // Use Plus Reconstruction
    // Update transform repository
    /*if ( repository->SetTransforms(*trackedFrame) != PLUS_SUCCESS ){
    LOG_ERROR("Failed to update transform repository with frame"  );
    return -1;
    }

    gettimeofday(&tv);
    current_time = tv.tv_usec;

    // Add this tracked frame to the reconstructor
    if ( volumeReconstructor->AddTrackedFrame(trackedFrame, repository) != PLUS_SUCCESS ){
    LOG_ERROR("Failed to add tracked frame to volume with frame");
    return -1;
    }

    gettimeofday(&tv);
    std::cout << "Elapsed time (PLUS Recon) : " << tv.tv_usec - current_time << " micro seconds." << std::endl;
    */
  }

  recon->GetOutputVolume(outputVolume);
  writer->SetInputData(outputVolume);
  writer->Write();

  std::cout << "Reconstruction done. " << std::endl;

  return 0;
}

int get_extent_from_trackedList(vtkPlusTrackedFrameList* frameList, vtkPlusTransformRepository* repository, double spacing, int* outputExtent, double* origin)
{
  PlusTransformName imageToReferenceTransformName;
  imageToReferenceTransformName = PlusTransformName("Image", "Tracker");

  if (frameList == NULL)
  {
    LOG_ERROR("Failed to set output extent from tracked frame list - input frame list is NULL");
    return -1;
  }

  if (frameList->GetNumberOfTrackedFrames() == 0)
  {

    LOG_ERROR("Failed to set output extent from tracked frame list - input frame list is empty");
    return -1;
  }

  if (repository == NULL)
  {

    LOG_ERROR("Failed to set output extent from tracked frame list - input transform repository is NULL");
    return -1;
  }

  double extent_Ref[6] =
  {
    VTK_DOUBLE_MAX, VTK_DOUBLE_MIN,
    VTK_DOUBLE_MAX, VTK_DOUBLE_MIN,
    VTK_DOUBLE_MAX, VTK_DOUBLE_MIN
  };

  const int numberOfFrames = frameList->GetNumberOfTrackedFrames();
  int numberOfValidFrames = 0;
  for (int frameIndex = 0; frameIndex < numberOfFrames; ++frameIndex)
  {

    PlusTrackedFrame* frame = frameList->GetTrackedFrame(frameIndex);

    if (repository->SetTransforms(*frame) != PLUS_SUCCESS)
    {
      LOG_ERROR("Failed to update transform repository with tracked frame!");
      continue;
    }

    // Get transform
    bool isMatrixValid(false);
    vtkSmartPointer<vtkMatrix4x4> imageToReferenceTransformMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
    if (repository->GetTransform(imageToReferenceTransformName, imageToReferenceTransformMatrix, &isMatrixValid) != PLUS_SUCCESS)
    {
      std::string strImageToReferenceTransformName;
      imageToReferenceTransformName.GetTransformName(strImageToReferenceTransformName);

      LOG_WARNING("Failed to get transform '" << strImageToReferenceTransformName << "' from transform repository!");
      continue;
    }

    if (isMatrixValid)
    {
      numberOfValidFrames++;

      // Get image (only the frame extents will be used)
      vtkImageData* frameImage = frameList->GetTrackedFrame(frameIndex)->GetImageData()->GetImage();

      // Output volume is in the Reference coordinate system.

      // Prepare the four corner points of the input US image.
      int* frameExtent = frameImage->GetExtent();
      std::vector< double* > corners_ImagePix;
      double minX = frameExtent[0];
      double maxX = frameExtent[1];
      double minY = frameExtent[2];
      double maxY = frameExtent[3];

      double c0[4] = { minX, minY, 0, 1 };
      double c1[4] = { minX, maxY, 0, 1 };
      double c2[4] = { maxX, minY, 0, 1 };
      double c3[4] = { maxX, maxY, 0, 1 };

      corners_ImagePix.push_back(c0);
      corners_ImagePix.push_back(c1);
      corners_ImagePix.push_back(c2);
      corners_ImagePix.push_back(c3);

      // Transform the corners to Reference and expand the extent if needed
      for (unsigned int corner = 0; corner < corners_ImagePix.size(); ++corner)
      {
        double corner_Ref[4] = { 0, 0, 0, 1 }; // position of the corner in the Reference coordinate system
        imageToReferenceTransformMatrix->MultiplyPoint(corners_ImagePix[corner], corner_Ref);

        for (int axis = 0; axis < 3; axis++)
        {
          if (corner_Ref[axis] < extent_Ref[axis * 2])
          {
            // min extent along this coord axis has to be decreased
            extent_Ref[axis * 2] = corner_Ref[axis];
          }

          if (corner_Ref[axis] > extent_Ref[axis * 2 + 1])
          {

            // max extent along this coord axis has to be increased
            extent_Ref[axis * 2 + 1] = corner_Ref[axis];
          }
        }
      }
    }
  }

  // Set the output extent from the current min and max values, using the user-defined image resolution.
  outputExtent[1] = int((extent_Ref[1] - extent_Ref[0]) / spacing);
  outputExtent[3] = int((extent_Ref[3] - extent_Ref[2]) / spacing);
  outputExtent[5] = int((extent_Ref[5] - extent_Ref[4]) / spacing);

  origin[0] = extent_Ref[0];
  origin[1] = extent_Ref[2];
  origin[2] = extent_Ref[4];

  return 0;
}