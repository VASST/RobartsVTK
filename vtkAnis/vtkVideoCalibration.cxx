#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include <sstream>

#include "vtkStereoCamera.h"
#include "vtkVideoCalibration.h"
#include "vtkTrackerBuffer.h"
#include "vtkMath.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkImageData.h"
#include "vtkCriticalSection.h"
#include "vtkTimerLog.h"
#include "vtkObjectFactory.h"
#include "vtkUnsignedCharArray.h"
#include "vtkLandmarkTransform.h"
#include "vtkPoints.h"

vtkStandardNewMacro(vtkVideoCalibration);
//vtkCxxSetObjectMacro(vtkVideoCalibration,TrackerTool,vtkTrackerTool);

void cvmat_assign31(CvMat* amat,double a0,double a1,double a2);

//----------------------------------------------------------------------------
vtkVideoCalibration::vtkVideoCalibration() {
  vtkStereoCamera singleCamera = new vtkStereoCamera();
  vtkStereoCamera leftCamera = new vtkStereoCamera();
  vtkStereoCamera rightCamera = new vtkStereoCamera();

  this->ToCheckerboard = vtkMatrix4x4::New();
  this->FromCheckerboard = vtkMatrix4x4::New();

  // Assuming our 14x14 calibration grid for now
  this->chessboard_cornerrows = 5;
  this->chessboard_cornercols = 7;
  this->chessboard_numcorners = chessboard_cornerrows*chessboard_cornercols;

  // Assuming our calibrations grid squares are 5mm x 5mm
  this->chessboard_boxsize = 5.0;

  // Angle treshold in degrees
  //  this->angleThresh = 0.0;

  // Initial framebuffer sync'ing
  this->FrameBufferSync = 0;

  // Tracker latency
  this->TrackerLatency = 0;

  //  this->MaxExpectedDistance = 100.f;
  this->ImageWidth=400;
  this->ImageHeight=400;
}


//----------------------------------------------------------------------------
vtkVideoCalibration::~vtkVideoCalibration() {
  //delete cameras
}


void vtkVideoCalibration::SetCheckerboard(  float ox, float oy, float oz,
                      float xx, float xy, float xz,
                      float yx, float yy, float yz) {

  vtkLandmarkTransform * landmarks = vtkLandmarkTransform::New();
  vtkPoints * source = vtkPoints::New();
  vtkPoints * target = vtkPoints::New();

  source->SetNumberOfPoints(3);
  source->InsertPoint(0,ox,oy,oz);
  source->InsertPoint(1,xx,xy,xz);
  source->InsertPoint(2,yx,yy,yz);


  target->SetNumberOfPoints(3);
  target->InsertPoint(0,0,0,0);
  target->InsertPoint(1,(this->chessboard_cornercols-1)*this->chessboard_boxsize,0,0);
  target->InsertPoint(2,0,(this->chessboard_cornerrows-1)*this->chessboard_boxsize,0);

  landmarks->SetSourceLandmarks(source);
  landmarks->SetTargetLandmarks(target);
  landmarks->Update();

  this->FromCheckerboard->DeepCopy(landmarks->GetMatrix());
  landmarks->Inverse();
  this->ToCheckerboard->DeepCopy(landmarks->GetMatrix());
  
  source->Delete();
  target->Delete();
  landmarks->Delete();
}

void vtkVideoCalibration::AcquireLeftImage() {
  leftCamera.acquireImage();
}

void vtkVideoCalibration::AcquireRightImage() {
  rightCamera.acquireImage();
}

//called every time Acquire Image button is clicked
//this function captures the image, then figures out where the vertices are in it
void vtkVideoCalibration::AcquireCalibrationImage() {
  singleCamera.acquireImage()
}

//this function comes up with calib transform when Calibrate button is clicked
void vtkVideoCalibration::DoCalibration(int apply)
{
  vtkMatrix4x4 matrixToCheckerboard=GetToCheckerboard();
  vtkMatrix4x4 matrixFromCheckerboard=inv(matrixToCheckerboard)
  singleCamera.doCalibration(apply,matrixFromCheckerboard);
}

void vtkVideoCalibration::DoLeftCalibration(int apply)
{
  vtkMatrix4x4 matrixToCheckerboard=GetToCheckerboard();
  vtkMatrix4x4 matrixFromCheckerboard=inv(matrixToCheckerboard)
  leftCamera.doCalibration(apply,matrixFromCheckerboard);
}

void vtkVideoCalibration::DoRightCalibration(int apply)
{
  vtkMatrix4x4 matrixToCheckerboard=GetToCheckerboard();
  vtkMatrix4x4 matrixFromCheckerboard=inv(matrixToCheckerboard)
  rightCamera.doCalibration(apply,matrixFromCheckerboard);
}

void cvmat_assign31(CvMat* amat,double a0,double a1,double a2)
{
 assert(((amat->cols==3)&&(amat->rows==1))||((amat->cols==1)&&(amat->rows==3)));
 if (amat->cols==3)
    {
     cvmSet(amat,0,0,a0);
     cvmSet(amat,0,1,a1);
     cvmSet(amat,0,2,a2);
    }
    else
    {
     cvmSet(amat,0,0,a0);
     cvmSet(amat,1,0,a1);
     cvmSet(amat,2,0,a2);
    }
}

//Save stereo image pairs
void vtkVideoCalibration::SaveImagesStereo(char * directoryL, char * directoryR) {
  for (unsigned int i=0; i < leftCamera.images.size(); i++) {
    std::string imageFilename(directoryL);
    std::ostringstream ossL;
    imageFilename.append("/endoImageLeft-");
    ossL << imageFilename;
    if (i < 10) {
      ossL << 0;
    }
    ossL << i;
    SaveImageL(ossL.str().c_str(), i);
  }

  for (unsigned int i=0; i < rightCamera.images.size(); i++) {
    std::string imageFilename(directoryR);
    std::ostringstream ossR;
    imageFilename.append("/endoImageRight-");
    ossR << imageFilename;
    if (i < 10) {
      ossR << 0;
    }
    ossR << i;
    SaveImageR(ossR.str().c_str(), i);
  }
}

//Called by SaveImagesStereo to complete the saving of the left images
void vtkVideoCalibration::SaveImageL(const char * filename, unsigned int imageNumber) {

  std::string imageFilename(filename);
  std::string transFilename(filename);
  imageFilename.append(".bmp");
  transFilename.append(".transform");

  ofstream outFile(transFilename.c_str());
  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      outFile << leftCamera.toolTransform[imageNumber].GetMatrix().Element[i][j] << " ";
    }
        outFile << endl;
  }

  if (imageNumber >= leftCamera.images.size()) {
    cout << "Image Index to high" << endl;
    return;
  }
    cvSaveImage(imageFilename.c_str(), leftCamera.images[imageNumber]);
}

//Called by SaveImagesStereo to complete the saving of the right images
void vtkVideoCalibration::SaveImageR(const char * filename, unsigned int imageNumber) {

  std::string imageFilename(filename);
  std::string transFilename(filename);
  imageFilename.append(".bmp");
  transFilename.append(".transform");

  ofstream outFile(transFilename.c_str());
  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      outFile << rightCamera.toolTransform[imageNumber].GetMatrix().Element[i][j] << " ";
    }
        outFile << endl;
  }

  if (imageNumber >= rightCamera.images.size()) {
    cout << "Image Index to high" << endl;
    return;
  }
    cvSaveImage(imageFilename.c_str(), rightCamera.images[imageNumber]);

}

//Save images from single camera calibration
void vtkVideoCalibration::SaveImages(char * directory) {
  for (unsigned int i=0; i < singleCamera.images.size(); i++) {
    std::string imageFilename(directory);
    std::ostringstream oss;
    imageFilename.append("/endoImage-");
    oss << imageFilename;
    if (i < 10) {
      oss << 0;
    }
    oss << i;
    this->SaveImage(oss.str().c_str(), i);
  }
}

//Called by SaveImages to complete image save task
void vtkVideoCalibration::SaveImage(const char * filename, unsigned int imageNumber) {

  std::string imageFilename(filename);
  std::string transFilename(filename);
  imageFilename.append(".bmp");
  transFilename.append(".transform");

  ofstream outFile(transFilename.c_str());
  for(int i = 0; i < 4; i++) {
    for(int j = 0; j < 4; j++) {
      outFile << singleCamera.toolTransform[imageNumber].GetMatrix().Element[i][j] << " ";
    }
        outFile << endl;
  }

  if (imageNumber >= singleCamera.images.size()) {
    cout << "Image Index to high" << endl;
    return;
  }
    cvSaveImage(imageFilename.c_str(), singleCamera.images[imageNumber]);

}

//Load images for single camera calibration
void vtkVideoCalibration::LoadImage(const char * filename,  const char * transformFile) {

  if(ifstream(transformFile)) {

  ifstream inFile(transformFile);

  vtkTransform *curTrans = vtkTransform::New();

  for(int i = 0; i < 4; i++)
  for(int j = 0; j < 4; j++)
    inFile >> curTrans->GetMatrix()->Element[i][j];

  IplImage *frame = cvLoadImage(filename,0);

  CvPoint3D32f* tcorners3d=(CvPoint3D32f*)calloc(chessboard_numcorners,sizeof(CvPoint3D32f));
  CvPoint2D32f* tcorners2d=(CvPoint2D32f*)calloc(chessboard_numcorners,sizeof(CvPoint2D32f));

  int value = singleCamera.FindCorners(frame, tcorners3d, tcorners2d);
  
  if (value == chessboard_numcorners) {
  singleCamera.toolTransform.push_back(curTrans);
  singleCamera.images.push_back(frame);
  singleCamera.corners3d.push_back(tcorners3d);
  singleCamera.corners2d.push_back(tcorners2d);
  singleCamera.transvector.push_back(cvCreateMat(1,3,CV_32FC1));
  singleCamera.rotmatrix.push_back(cvCreateMat(3,3,CV_32FC1));
  } else {
    cout << "Only " << value << " of " << chessboard_numcorners << " corners found" << endl;
  }
  cout << "Loaded: " << filename << endl;
  }
  else { cout << "No Transform File Found" << endl; }
}

//Load left images for stereo calibration
void vtkVideoCalibration::LoadLeftImage(const char * filename,  const char * transformFile) {

  if(ifstream(transformFile)) {

  ifstream inFile(transformFile);

  vtkTransform *curTrans = vtkTransform::New();

  for(int i = 0; i < 4; i++)
  for(int j = 0; j < 4; j++)
    inFile >> curTrans->GetMatrix()->Element[i][j];

  IplImage *frame = cvLoadImage(filename,0);

  CvPoint3D32f* tcorners3d=(CvPoint3D32f*)calloc(chessboard_numcorners,sizeof(CvPoint3D32f));
  CvPoint2D32f* tcorners2d=(CvPoint2D32f*)calloc(chessboard_numcorners,sizeof(CvPoint2D32f));

  int value = leftCamera.FindCorners(frame, tcorners3d, tcorners2d);
  
  if (value == chessboard_numcorners) {
  leftCamera.toolTransform.push_back(curTrans);
  leftCamera.images.push_back(frame);
  leftCamera.corners3d.push_back(tcorners3d);
  leftCamera.corners2d.push_back(tcorners2d);
  leftCamera.transvector.push_back(cvCreateMat(1,3,CV_32FC1));
  leftCamera.rotmatrix.push_back(cvCreateMat(3,3,CV_32FC1));
  } else {
    cout << "Only " << value << " of " << chessboard_numcorners << " corners found" << endl;
  }
  cout << "Loaded: " << filename << endl;
  }
  else { cout << "No Transform File Found" << endl; }
}

//Load right images for stereo calibration
void vtkVideoCalibration::LoadRightImage(const char * filename,  const char * transformFile) {

  if(ifstream(transformFile)) {

  ifstream inFile(transformFile);

  vtkTransform *curTrans = vtkTransform::New();

  for(int i = 0; i < 4; i++)
  for(int j = 0; j < 4; j++)
    inFile >> curTrans->GetMatrix()->Element[i][j];

  IplImage *frame = cvLoadImage(filename,0);

  CvPoint3D32f* tcorners3d=(CvPoint3D32f*)calloc(chessboard_numcorners,sizeof(CvPoint3D32f));
  CvPoint2D32f* tcorners2d=(CvPoint2D32f*)calloc(chessboard_numcorners,sizeof(CvPoint2D32f));

  int value = rightCamera.FindCorners(frame, tcorners3d, tcorners2d);
  
  if (value == chessboard_numcorners) {
  rightCamera.toolTransform.push_back(curTrans);
  rightCamera.images.push_back(frame);
  rightCamera.corners3d.push_back(tcorners3d);
  rightCamera.corners2d.push_back(tcorners2d);
  rightCamera.transvector.push_back(cvCreateMat(1,3,CV_32FC1));
  rightCamera.rotmatrix.push_back(cvCreateMat(3,3,CV_32FC1));
  } else {
    cout << "Only " << value << " of " << chessboard_numcorners << " corners found" << endl;
  }
  cout << "Loaded: " << filename << endl;
  }
  else { cout << "No Transform File Found" << endl; }
}