#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>

/*
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif*/

#include <sstream>

#include "vtkStereoCamera.h"

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

vtkStandardNewMacro(vtkStereoCamera);
//vtkCxxSetObjectMacro(vtkVideoCalibration,TrackerTool,vtkTrackerTool);

void cvmat_assign31(CvMat* amat,double a0,double a1,double a2);

vtkStereoCamera::~vtkStereoCamera(){
  Calibration->Delete();
  for (unsigned int i = 0; i < toolTransform.size(); i++){
    toolTransform[i]->Delete();
  }
  for (unsigned int i = 0; i < images.size(); i++){
    cvReleaseImage(&images[i]);
  }

  for (unsigned int i = 0; i < corners2d.size(); i++){
    delete[] this->corners2d[i];
  }
  for (unsigned int i = 0; i < corners3d.size(); i++){
    delete[] this->corners3d[i];
  }
  for (unsigned int i = 0; i < transvector.size(); i++){
    cvReleaseMat(&transvector[i]);
  }
  for (unsigned int i = 0; i < rotmatrix.size(); i++){
    cvReleaseMat(&rotmatrix[i]);
  }

    toolTransform.clear();
  images.clear();
  corners2d.clear();
  corners3d.clear();
  transvector.clear();
  rotmatrix.clear();
}

vtkStereoCamera::vtkStereoCamera() {
  chessboard_cornerrows = 5;
  chessboard_cornercols = 7;
  chessboard_numcorners = chessboard_cornerrows*chessboard_cornercols;
  chessboard_boxsize = 5.0;
  TrackerTool = NULL;
  IsCalibrated = false;
  Calibration = vtkMatrix4x4::New();
  Video = NULL;
  pcamera_matrix = cvCreateMat(3,3,CV_32FC1);
  pcamera_distortion = cvCreateMat(1,4,CV_32FC1);
  ImageWidth=400;
  ImageHeight=400;
}

void vtkStereoCamera::AcquireImage(){
  int ImageWidth=400;
  int ImageHeight=400;
  vtkTransform *curTrans = vtkTransform::New();
  IplImage *frame;

  if(this->TrackerTool->IsMissing()) { printf("Tool missing\n"); fflush(stdout); return;  }
  if(this->TrackerTool->IsOutOfVolume()) { printf("Tool not in volume\n"); fflush(stdout); return; }
  if(this->TrackerTool->IsOutOfView()) { printf("Tool not visible\n"); fflush(stdout); return; }

  // Retrieve the position and orientations
  curTrans->DeepCopy(this->TrackerTool->GetTransform());

  vtkImageData *image = vtkImageData::New();
  image->DeepCopy(this->Video->GetOutput());
  void *ptr = reinterpret_cast<vtkDataArray *>(image->GetScalarPointer());
  
  //create a frame that contains the image data, 3 channels because image is originally RGB
  frame = cvCreateImageHeader(cvSize(ImageWidth,ImageHeight), IPL_DEPTH_8U, this->Video->GetOutput()->GetNumberOfScalarComponents());
  cvSetData(frame, ptr, this->Video->GetOutput()->GetNumberOfScalarComponents()*ImageWidth);  

  //convert RGB image to grayscale image
  if (this->Video->GetOutput()->GetNumberOfScalarComponents() == 3) {
    IplImage *grayframe = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
    cvCvtColor(frame, grayframe, CV_RGB2GRAY);
    frame = grayframe;
  }

  //flip the image so that it is in vtk coordinates instead of opencv coordinates
  cvFlip(frame, NULL, 1);

  //save the image and the tool position it was captured from

  CvPoint3D32f* tcorners3d=(CvPoint3D32f*)calloc(chessboard_numcorners,sizeof(CvPoint3D32f));
  CvPoint2D32f* tcorners2d=(CvPoint2D32f*)calloc(chessboard_numcorners,sizeof(CvPoint2D32f));
  int value = this->FindCorners(frame,tcorners3d,tcorners2d);

  if (value == chessboard_numcorners) {
    this->toolTransform.push_back(curTrans);
    this->images.push_back(frame);
    this->corners3d.push_back(tcorners3d);
    this->corners2d.push_back(tcorners2d);
    this->transvector.push_back(cvCreateMat(1,3,CV_32FC1));
    this->rotmatrix.push_back(cvCreateMat(3,3,CV_32FC1));
  }
  else {
    cout << "Only " << value << " of " << chessboard_numcorners << " corners found" << endl;
  }
}

//this function sets corners3d matrices to be reference grid of points and corners2d matrices to be detected points in images
int vtkStereoCamera::FindCorners(IplImage * image ,CvPoint3D32f * pcorners3d, CvPoint2D32f * pcorners2d)
{

  //sets up pcorners3d to be (0,0,0) (5,0,0) (10,0,0) (15,0,0) (20,0,0) etc
  //              (0,5,0) (5,5,0) etc
  for(int i=0,iy=0;iy<chessboard_cornerrows;iy++) {
    for(int ix=0;ix<chessboard_cornercols;ix++,i++) {
      pcorners3d[i].x= chessboard_boxsize*ix;
      pcorners3d[i].y= chessboard_boxsize*iy;
      pcorners3d[i].z=0;
    }
  }

  int ncorners=chessboard_numcorners;

  //stores the corners that it detects from image in pcorners2d, the number of corners found is ncorners
  flag_cornersfound=cvFindChessboardCorners(image,
                                           cvSize(chessboard_cornercols,chessboard_cornerrows),
                                           pcorners2d,
                                           &ncorners,
                                           CV_CALIB_CB_ADAPTIVE_THRESH);

  if (ncorners!=chessboard_numcorners) flag_cornersfound=0;

  if (flag_cornersfound!=0) {
  //if it's upside down, swap points to flip it over?
    if (hypot(pcorners2d[0].x,pcorners2d[0].y)>hypot(pcorners2d[chessboard_numcorners-1].x,pcorners2d[chessboard_numcorners-1].y)) {
    CvMat* tmp=cvCreateMat(chessboard_numcorners,2,CV_32FC1);
    for (int icorners=0;icorners<chessboard_numcorners;icorners++) {
      cvmSet(tmp,icorners,0,pcorners2d[icorners].x);
      cvmSet(tmp,icorners,1,pcorners2d[icorners].y);
        }
        for (int icorners=0;icorners<chessboard_numcorners;icorners++) {
            pcorners2d[icorners].x=cvmGet(tmp,chessboard_numcorners-icorners-1,0);
            pcorners2d[icorners].y=cvmGet(tmp,chessboard_numcorners-icorners-1,1);
        }
        cvReleaseMat(&tmp);
  }

  //refines the corner locations
    cvFindCornerSubPix(image,
            pcorners2d,
            chessboard_numcorners,
            cvSize(5,5),
            cvSize(-1,-1),
            cvTermCriteria( CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 0.01f ) );
  }
  return ncorners;
}

//this function comes up with calib transform when Calibrate button is clicked
void vtkStereoCamera::DoCalibration(int apply,vtkMatrix4x4 * FromCheckerboard)
{
  //variables used to average all calib matrices to come up with the final one
  std::vector<float> allX, allY, allZ;
  float accX=0, accY=0, accZ=0;
  double ortho[3][3];
  //wxyz stores quaternions, quat is the sum of all quaternions for all images
  double wxyz[4], quat[4] = {0,0,0,0};
  int j;
  unsigned int irec;
  // find all patterns with corner points identified and put them in a separate list
  int numcorners_all= this->images.size()*this->chessboard_cornerrows*this->chessboard_cornercols;

  //stores all 3d corners from all images in a really big array
  CvMat* pcorners3d    =cvCreateMat(numcorners_all,3,CV_32FC1);
  //stores all 2d corners from all images in a really big array
  CvMat* pcorners2d    =cvCreateMat(numcorners_all,2,CV_32FC1);
  CvMat* pnumcorners   =cvCreateMat(this->images.size(),1,CV_32SC1);
  CvMat* ptransvectors =cvCreateMat(this->images.size(),3,CV_32FC1);
  CvMat* protmatrices  =cvCreateMat(this->images.size(),3,CV_32FC1);

  // initialize pnumcorners, pcorners,pcorners_object
  int sumcorners=0;
  for (irec=0;irec < this->images.size();irec++)
    {
    pnumcorners->data.i[irec]  = this->chessboard_numcorners;

    //sumcorners is what decides which position in the really big array stuff goes
    for (int icorners=0;icorners< this->chessboard_numcorners;icorners++,sumcorners++)
    {
      cvmSet(pcorners3d,sumcorners,0,this->corners3d[irec][icorners].x);
        cvmSet(pcorners3d,sumcorners,1,this->corners3d[irec][icorners].y);
        cvmSet(pcorners3d,sumcorners,2,this->corners3d[irec][icorners].z);

        cvmSet(pcorners2d,sumcorners,0,this->ImageWidth-this->corners2d[irec][icorners].x);
        cvmSet(pcorners2d,sumcorners,1,this->ImageHeight-this->corners2d[irec][icorners].y);
    }//end for


    //math to set up the rotation and translation matrices so that they can be used in calibration
    cvmSet(protmatrices,irec,0,0);
    cvmSet(protmatrices,irec,1,-2.22);
    cvmSet(protmatrices,irec,2,-2.22);

    cvmSet(ptransvectors,irec,0,0);
    cvmSet(ptransvectors,irec,1,0);
    cvmSet(ptransvectors,irec,2,0.5);
  }//end for
  // intialize pcamera_matrix (the intrinsic matrix)
  cvmSet(pcamera_matrix,0,0,this->ImageWidth*1.4); cvmSet(pcamera_matrix,0,1,0);                    cvmSet(pcamera_matrix,0,2,this->ImageWidth/2);
  cvmSet(pcamera_matrix,1,0,0);                    cvmSet(pcamera_matrix,1,1,this->ImageWidth*1.4); cvmSet(pcamera_matrix,1,2,this->ImageHeight/2);
  cvmSet(pcamera_matrix,2,0,0);                    cvmSet(pcamera_matrix,2,1,0);                    cvmSet(pcamera_matrix,2,2,1);

  // calibrate camera
  //takes in 3xN 2d and 3d points where N is the total number of points in all images
  //generates 3xM array of rotation and translation vectors where M is the number of images
  cvCalibrateCamera2(pcorners3d,
                    pcorners2d,
                    pnumcorners,
                    cvSize(this->ImageWidth,this->ImageHeight),
                    pcamera_matrix,
                    pcamera_distortion,
                    protmatrices,
                    ptransvectors,0);
  flag_calibrated=true;

  // convert and split extrinsic calibration results to secrecs
  CvMat* tmprow=cvCreateMat(1,3,CV_32FC1);

  for (irec=0;irec< this->images.size();irec++) {
    cvmat_assign31(tmprow,
                     cvmGet(protmatrices,irec,0),
                     cvmGet(protmatrices,irec,1),
                     cvmGet(protmatrices,irec,2));
    cvmat_assign31(this->transvector[irec],
                     cvmGet(ptransvectors,irec,0),
                     cvmGet(ptransvectors,irec,1),
                     cvmGet(ptransvectors,irec,2));
    cvRodrigues2(tmprow,this->rotmatrix[irec]); //unvectorize the rotation matrices
  }//end for
  //result is that this->rotmatrix holds all of the rotation matrices for all images, and this->transvector
  //holds all translation vectors for all images, then they are used to determine checkerboard2camera matrices
  //and eventually average to give final calibration matrix

  vtkMatrix4x4 * checkerboard2camera = vtkMatrix4x4::New();
  vtkMatrix4x4 * sensor2checkerboard = vtkMatrix4x4::New();
  vtkMatrix4x4 * calib = vtkMatrix4x4::New();
  // Now determine the endoscope<->sensor transform

  //do this stuff for each image
  for(unsigned int i = 0; i < this->images.size(); i++) {
    // Load the transform from the checkboard's frame to the camera's frame
    checkerboard2camera->Identity();
    for(j = 0; j < 3; j++) {
      checkerboard2camera->SetElement(j, 0, cvmGet(this->rotmatrix[i], j, 0));
      checkerboard2camera->SetElement(j, 1, cvmGet(this->rotmatrix[i], j, 1));
      checkerboard2camera->SetElement(j, 2, cvmGet(this->rotmatrix[i], j, 2));
      checkerboard2camera->SetElement(j, 3, cvmGet(this->transvector[i],0,j));
    }//end for
    checkerboard2camera->SetElement(2,3,-cvmGet(this->transvector[i],0,2));

    //transforms applied right to left (FromCheckerboard*toolTransform means apply toolTransform, then FromCheckerboard)
    vtkMatrix4x4::Multiply4x4(FromCheckerboard, this->toolTransform[i]->GetMatrix(), sensor2checkerboard);
    
    vtkMatrix4x4::Multiply4x4(checkerboard2camera ,sensor2checkerboard,  calib);

    //invert it so that calib represents transform from camera optical origin to sensor tip
    calib->Invert();
    vtkMatrix4x4 * calibt = vtkMatrix4x4::New();
    calibt->DeepCopy(calib);
    calibs.push_back(calibt);

    //copy the rotation part of calib to ortho
    for (j = 0; j < 3; j++)  {
       ortho[0][j] = calib->GetElement(0,j);
      ortho[1][j] = calib->GetElement(1,j);
      ortho[2][j] = calib->GetElement(2,j);
    }
    //if the determinant of ortho is negative, then take the negative of each element
    if (vtkMath::Determinant3x3(ortho) < 0) {
      for (j = 0; j < 3; j++)  {
        ortho[0][j] = -ortho[0][j];
        ortho[1][j] = -ortho[1][j];
        ortho[2][j] = -ortho[2][j];
      }
    }
    //convert rotation matrix to a quaternion(wxyz) and add it to the quat sum
    vtkMath::Matrix3x3ToQuaternion(ortho, wxyz);
    quat[0] += wxyz[0];
    quat[1] += wxyz[1];
    quat[2] += wxyz[2];
    quat[3] += wxyz[3];

    //Average all of the calibration matrices to get the final matrix calib
    // Accumulate the translations/orientations
    allX.push_back(calib->GetElement(0,3));
    accX += calib->GetElement(0,3);
    allY.push_back(calib->GetElement(1,3));
    accY += calib->GetElement(1,3);
    allZ.push_back(calib->GetElement(2,3));
    accZ += calib->GetElement(2,3);
  }//end for

    // Make sure these all add up
    assert(allX.size() == allY.size());
    assert(allX.size() == allZ.size());

    // Average the accumulated values
    accX /= allX.size(); 
    accY /= allY.size();
    accZ /= allZ.size();

    // Wrap things up here
    double diffX=0, diffY=0, diffZ=0;

    for(unsigned int i = 0; i < allX.size(); i++)
    {
      diffX += pow(allX[i] - accX, 2);
      diffY += pow(allY[i] - accY, 2);
      diffZ += pow(allZ[i] - accZ, 2);
    }
    diffX /= allX.size() - 1;
    diffY /= allX.size() - 1;
    diffZ /= allX.size() - 1;
     
    // Reporting on the stats
    printf("%.2f %.2f %.2f\t+/-\t",       accX, accY, accZ);
    printf("%.2f %.2f %.2f         \n",   sqrt(diffX), sqrt(diffY), sqrt(diffZ) );

  // Average the estimations

  double d = sqrt(quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]);
  quat[0] /= d; quat[1] /= d; quat[2] /= d; quat[3] /= d;

  //turn averaged quat back into ortho, which will then be set as the rotation part of calib
  vtkMath::QuaternionToMatrix3x3(quat, ortho);

  //acc values are set to be the translate part of the transform
  calib->SetElement(0,3,accX);
  calib->SetElement(1,3,accY);
  calib->SetElement(2,3,accZ);

  //ortho is the rotation matrix of the transform
  for (int i = 0; i < 3; i++)  {
    calib->SetElement(0,i,ortho[0][i]);
    calib->SetElement(1,i,ortho[1][i]);
    calib->SetElement(2,i,ortho[2][i]);
  }

  calib->SetElement(0,3,accX);
  calib->SetElement(1,3,accY);
  calib->SetElement(2,3,accZ);

    if(apply && TrackerTool != NULL) {
      TrackerTool->GetCalibrationMatrix()->DeepCopy(calib);
      IsCalibrated = true;
    }
  this->Calibration->DeepCopy(calib);

    // Free the all the related data
    cvReleaseMat(&pcorners3d);
    cvReleaseMat(&pcorners2d);
    cvReleaseMat(&pnumcorners);
    cvReleaseMat(&ptransvectors);
    cvReleaseMat(&protmatrices);
  sensor2checkerboard->Delete();
  checkerboard2camera->Delete();
    calib->Delete();
}

vtkMatrix4x4 * vtkStereoCamera::GetImageMatrix(unsigned int i) {
  if (i >= rotmatrix.size() || i < 0 ) return NULL;
  if (i >= transvector.size() || i < 0 ) return NULL;
  vtkMatrix4x4 *value = vtkMatrix4x4::New();
  for(int j = 0; j < 3; j++) {
        value->SetElement(j, 0, cvmGet(this->rotmatrix[i], j, 0));
        value->SetElement(j, 1, cvmGet(this->rotmatrix[i], j, 1));
        value->SetElement(j, 2, cvmGet(this->rotmatrix[i], j, 2));
        value->SetElement(j, 3, cvmGet(this->transvector[i],0,j));
      }
  value->Invert();
  vtkTransform * trans = vtkTransform::New();
  trans->SetMatrix(value);
  trans->Scale(-1,-1,1);
  return value;
}

/*vtkTransform * vtkStereoCamera::GetToolTransform(unsigned int i) {
  if (i >= toolTransform.size() || i < 0 ) return NULL;
  vtkTransform *t = vtkTransform::New();

    vtkMatrix4x4::Multiply4x4(FromCheckerboard, this->toolTransform[i]->GetMatrix(), t->GetMatrix());
  return t;//this->toolTransform[i];
}*/

vtkMatrix4x4 * vtkStereoCamera::GetCalibMatrix(unsigned int i) {
  if (i >= calibs.size() || i < 0 ) return NULL;
  return this->calibs[i];
}

void vtkStereoCamera::PrintIntrinsic(){
  for (int i=0; i<3; i++){
    for (int j=0; j<3; j++){
      printf("%.4f ",cvmGet(this->pcamera_matrix,i,j));
    }
    printf("\n");
  }
  for (int k=0; k<3; k++)
    printf("%.4f ",cvmGet(this->pcamera_distortion,0,k));
  printf("\n");
}

void vtkStereoCamera::GetIntrinsicMatrix(double values[9]) {
  for (int i=0; i < 3; i++){
    for (int j=0; j < 3; j++){
      values[i*3+j] = cvmGet(this->pcamera_matrix,i,j);
      printf("%.4f ",values[i*3+j]);
    }
    printf("\n");
  }
  printf("\n\n");
}

void vtkStereoCamera::GetDistortionCoefficients(double values[4]) {
    for (int j=0; j < 4; j++){
      values[j] = cvmGet(this->pcamera_distortion,0,j);
      printf("%.4f ",values[j]);
    }
    printf("\n\n");
}

void vtkStereoCamera::GetUndistortedPoint(int pos[2], float newPos[2])
{
  // Create undistortmap
  static CvArr *mapx=cvCreateImage(cvSize(this->ImageWidth, this->ImageHeight),IPL_DEPTH_32F,1), 
          *mapy=cvCreateImage(cvSize(this->ImageWidth, this->ImageHeight),IPL_DEPTH_32F,1);
    cvInitUndistortMap(this->pcamera_matrix, this->pcamera_distortion, mapx, mapy);

    newPos[0] = cvGetReal2D(mapx, pos[1], pos[0]);
    newPos[1] = cvGetReal2D(mapy, pos[1], pos[0]);
}

void vtkStereoCamera::GetUndistortedPoint(int pos[2], double newPos[2])
{
  // Create undistortmap
  static CvArr *mapx=cvCreateImage(cvSize(this->ImageWidth, this->ImageHeight),IPL_DEPTH_32F,1), 
         *mapy=cvCreateImage(cvSize(this->ImageWidth, this->ImageHeight),IPL_DEPTH_32F,1);
    cvInitUndistortMap(this->pcamera_matrix, this->pcamera_distortion, mapx, mapy);

    newPos[0] = cvGetReal2D(mapx, pos[1], pos[0]);
    newPos[1] = cvGetReal2D(mapy, pos[1], pos[0]);
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