/*=========================================================================

Copyright (c) 2005, Anis Ahmad.

=========================================================================*/

#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

// includes for mkdir
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include <mil.h>

#include "mat33.h"

#include "vtkEndoscope.h"
#include "vtkMath.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include "vtkImageData.h"
#include "vtkCriticalSection.h"
#include "vtkTimerLog.h"
#include "vtkObjectFactory.h"
#include "vtkTracker.h"
#include "vtkTrackerTool.h"
#include "vtkTrackerBuffer.h"
#include "vtkDataArray.h"
#include "vtkMultiThreader.h"
#include "vtkMutexLock.h"

//vtkCxxRevisionMacro(vtkEndoscope, "$Revision: 1.1 $");
vtkStandardNewMacro(vtkEndoscope);

vtkCxxSetObjectMacro(vtkEndoscope,TrackerTool,vtkTrackerTool);

enum{
  ENDOSCOPE_INTRINSIC_CALIBRATION=0,
  ENDOSCOPE_EXTRINSIC_CALIBRATION
};

#define QUAT_AVERAGE
//#undef QUAT_AVERAGE

template<class T>
T SwapErase(std::vector<T> &arr, size_t idx)
{
	T ret = arr[idx];
	arr[idx] = arr.back();
	arr.resize(arr.size()-1);
	return(ret);
}

//----------------------------------------------------------------------------
vtkEndoscope::vtkEndoscope() :
  vtkMILVideoSource()
{
  // for running the reconstruction in the background
  this->TrackerTool = NULL;
  this->bCalibrated = false;
  this->ToCheckerboard = NULL;
  this->FromCheckerboard = NULL;

  // Assuming our 14x14 calibration grid for now
  nUCorners = 7;
  nVCorners = 5;

  // Assuming
  divotDepth = 0.0;

  // Assuming our calibrations grid squares are 5mm x 5mm
  cornerDim = 5.0;

  // Angle treshold in degrees
  angleThresh = 5;

  // Distance threshold in millimeters
  distThresh = 50.0;

  // Allocate the calibrated matrices
  Calibration = vtkMatrix4x4::New();

  // Initial framebuffer sync'ing
  FrameBufferSync = 0;

  // Create the initial guess for the intrinsic params
  intrinsic_matrix = cvCreateMat(3, 3, CV_32F);
  cvSetReal2D(intrinsic_matrix, 0, 0, 614.328);
  cvSetReal2D(intrinsic_matrix, 0, 1, 0);
  cvSetReal2D(intrinsic_matrix, 0, 2, 320.805);
  cvSetReal2D(intrinsic_matrix, 1, 0, 0);
  cvSetReal2D(intrinsic_matrix, 1, 1, 617.228);
  cvSetReal2D(intrinsic_matrix, 1, 2, 264.141);
  cvSetReal2D(intrinsic_matrix, 2, 0, 0);
  cvSetReal2D(intrinsic_matrix, 2, 1, 0);
  cvSetReal2D(intrinsic_matrix, 2, 2, 1);
  distortion_coeffs = cvCreateMat(1, 4, CV_32F);
  cvSetReal1D(distortion_coeffs, 0, -0.292244);
  cvSetReal1D(distortion_coeffs, 1, 0.00173742);
  cvSetReal1D(distortion_coeffs, 2, -0.00492392);
  cvSetReal1D(distortion_coeffs, 3, -0.000947305);
  
  // Field information
  bOddField = false;

  // Tracker latency
  TrackerLatency = 0;

  // for threaded capture of transformations
  this->Threader = vtkMultiThreader::New();
  this->ThreadId = -1;

  SetOutputFormat(VTK_RGB);

  lastTrackerIdx = 0;
  PrevMat.clear();
  PrevPos.clear();
  PrevOrient.clear();  
  corners.clear();

  mMaxExpectedDistance = 100.f;

  // Deinterlace or not
  deinterlace = true;
}

//----------------------------------------------------------------------------
vtkEndoscope::~vtkEndoscope()
{
  int i;

  Calibration->Delete();
  this->Threader->Delete();
  PrevPos.clear();
  PrevOrient.clear();  
  for(i = 0; i < corners.size(); i++)
    delete[] corners[i];
  corners.clear();
}

//----------------------------------------------------------------------------
// Override this and provide checks to ensure an appropriate number
// of components was asked for (i.e. 1 for greyscale, 3 for RGB,
// or 4 for RGBA)
void vtkEndoscope::SetOutputFormat(int format)
{
  if (format == this->OutputFormat)
    {
    return;
    }

  this->OutputFormat = format;

  // Ignore the user request and get 24bit colour images
  int numComponents = 3;

  this->NumberOfScalarComponents = numComponents;

  if (this->FrameBufferBitsPerPixel != numComponents*8)
    {
    this->FrameBufferMutex->Lock();
    this->FrameBufferBitsPerPixel = numComponents*8;
    if (this->Initialized)
      {
      this->UpdateFrameBuffer();
      }
    this->FrameBufferMutex->Unlock();
    }

  this->Modified();
}

void vtkEndoscope::SetCheckerboard(float ox, float oy, float oz,
                                   float xx, float xy, float xz,
                                   float yx, float yy, float yz)
{
  int i;
  float orig[3], xend[3], yend[3];
  float xdir[3], ydir[3], zdir[3];

  orig[0] = ox; orig[1] = oy; orig[2] = oz - divotDepth;
  xend[0] = xx; xend[1] = xy; xend[2] = xz - divotDepth;
  yend[0] = yx; yend[1] = yy; yend[2] = yz - divotDepth;

  // Compute directions
  for(i = 0; i < 3; i++)
  {
    xdir[i] = xend[i] - orig[i];
    ydir[i] = yend[i] - orig[i];
  }

  // Normalize directions
  float xlen=0, ylen=0;
  for(i = 0; i < 3; i++)
  {
    xlen += xdir[i]*xdir[i];
    ylen += ydir[i]*ydir[i];
  }
  xlen = sqrt(xlen); ylen = sqrt(ylen);
  for(i = 0; i < 3; i++)
  {
    xdir[i] /= xlen;
    ydir[i] /= ylen;
  }

  // Compute the z direction
  zdir[0] = xdir[1] * ydir[2] - ydir[1] * xdir[2];
  zdir[1] = xdir[2] * ydir[0] - ydir[2] * xdir[0];
  zdir[2] = xdir[0] * ydir[1] - ydir[0] * xdir[1];

  ToCheckerboard = vtkMatrix4x4::New();
  ToCheckerboard->Identity();
  // Load the transformation matrix
  for(i = 0; i < 3; i++)
  {
    ToCheckerboard->Element[i][0] = xdir[i];
    ToCheckerboard->Element[i][1] = ydir[i];
    ToCheckerboard->Element[i][2] = zdir[i];
    ToCheckerboard->Element[i][3] = orig[i];
  }

  // Preserve the transform from checkerboard space
  FromCheckerboard = vtkMatrix4x4::New();
  FromCheckerboard->DeepCopy(ToCheckerboard);

  // Invert this transform
  ToCheckerboard->Invert();
}

//----------------------------------------------------------------------------
// Sleep until the specified absolute time has arrived.
// You must pass a handle to the current thread.  
// If '0' is returned, then the thread was aborted before or during the wait.
static int vtkThreadSleep(vtkMultiThreader::ThreadInfo *data, double time)
{
/*
  for (int i = 0;; i++)
    {
    double remaining = time - vtkTimerLog::GetCurrentTime();

    // check to see if we have reached the specified time
    if (remaining <= 0)
      {
      if (i == 0)
        {
        vtkGenericWarningMacro("Dropped a video frame.");
        }
      return 1;
      }
    // check the ActiveFlag at least every 0.1 seconds
    if (remaining > 0.1)
      {
      remaining = 0.1;
      }
*/
    // check to see if we are being told to quit 
    data->ActiveFlagLock->Lock();
    int activeFlag = *(data->ActiveFlag);
    data->ActiveFlagLock->Unlock();

    if (activeFlag == 0)
      {
      return 0;
      }

    return 1;
/*
    vtkSleep(remaining);
    }
*/
}

//----------------------------------------------------------------------------
// this function runs in an alternate thread to asyncronously grab frames
static void *vtkEndoscopeRecordThread(vtkMultiThreader::ThreadInfo *data)
{
  vtkEndoscope *self = (vtkEndoscope *)(data->UserData);
  
  double startTime = vtkTimerLog::GetCurrentTime();
  double rate = self->GetFrameRate();
  int frame = 0;

  do
    {
    self->GrabBegin();
    self->GrabEnd();
    frame++;
    }
  while (vtkThreadSleep(data, startTime + frame/rate));

  return NULL;
}

//----------------------------------------------------------------------------
// Set the source to grab frames continuously.
// You should override this as appropriate for your device.  
void vtkEndoscope::Record()
{
  if(deinterlace)
  {
    if (this->Playing)
      {
      this->Stop();
      }

    if (!this->Recording)
      {
      this->Initialize();

      this->Recording = 1;
      this->FrameCount = 0;
      this->Modified();
      this->ThreadId = 
        this->Threader->SpawnThread((vtkThreadFunctionType)\
                                  &vtkEndoscopeRecordThread,this);
      }
  }
  else
  {
    vtkMILVideoSource::Record();
  }
}
 
//----------------------------------------------------------------------------
// Stop continuous grabbing or playback.  You will have to override this
// if your class overrides Play() and Record()
void vtkEndoscope::Stop()
{
  if(deinterlace)
  {
    if (this->Recording)
      {
      this->Threader->TerminateThread(this->ThreadId);
      this->ThreadId = -1;
      this->Recording = 0;
      this->Modified();
      }
  }
  else
  {
    vtkMILVideoSource::Stop();
  }
} 

void vtkEndoscope::GrabBegin()
{
  // ensure that the hardware is initialized.
  if (!this->Initialized)
    {
    this->Initialize();
    return;
    }
  MdigControl(this->MILDigID,M_GRAB_FIELD_NUM,1);
  if(bOddField) MdigControl(this->MILDigID, M_GRAB_START_MODE, M_FIELD_START_ODD);
  else MdigControl(this->MILDigID, M_GRAB_START_MODE, M_FIELD_START_EVEN);

  this->FrameBufferMutex->Lock();
  MdigGrab(this->MILDigID,this->MILBufID);
}

void vtkEndoscope::GrabEnd()
{
  MdigGrabWait(this->MILDigID,M_GRAB_END);
  this->InternalGrab();
  bOddField = !bOddField;
  
  this->FrameBufferMutex->Unlock();
}

void vtkEndoscope::AcquireCalibrationImage()
{
  static vtkMatrix4x4 *mat = vtkMatrix4x4::New();
  static vtkTransform *curTrans = vtkTransform::New();

  int index = (this->FrameBufferIndex + this->FrameBufferSync) % this->FrameBufferSize;
  void *ptr = ((reinterpret_cast<vtkDataArray *>( \
                       this->FrameBuffer[index]))->GetVoidPointer(0));

  Tracker->Update();
  if(TrackerTool->IsMissing())
  {
    printf("Tool missing\n"); fflush(stdout);
    return;
  }
  if(TrackerTool->IsOutOfVolume())
  {
    printf("Tool not in volume\n"); fflush(stdout);
    return;
  }
  if(TrackerTool->IsOutOfView())
  {
    printf("Tool not visible\n"); fflush(stdout);
    return;
  }
 
  TrackerTool->GetBuffer()->Lock();
  TrackerTool->GetBuffer()->GetFlagsAndMatrixFromTime(mat,this->FrameBufferTimeStamps[index]+TrackerLatency);
  TrackerTool->GetBuffer()->Unlock();
  curTrans->SetMatrix(mat);

  // Retrieve the position and orientations
  float *pos = new float[3], *orient = new float[3];
  curTrans->GetPosition(pos);
  curTrans->GetOrientation(orient);

  if(!PrevPos.size())
  {
    // If there are no previous views, add this one
    PrevPos.push_back(pos);
    PrevOrient.push_back(orient);
  }
  else
  {
    bool validView = true;

    // Compare current position and orientation to previous ones
    for(int i = 0; i < PrevPos.size(); i++)
    {
/* Zhang's approach is concerned with orientation differences
    if((pow(pos[0]-PrevPos[i][0],2)+pow(pos[1]-PrevPos[i][1],2)+pow(pos[1]-PrevPos[i][1],2)) < (distThresh*distThresh))
    {
      validView = false;
      break;
    }
*/
      double maxAngle = fabs(orient[0] - PrevOrient[i][0]);
      if(maxAngle < fabs(orient[1] - PrevOrient[i][1])) maxAngle = fabs(orient[1] - PrevOrient[i][1]);
      if(maxAngle < fabs(orient[2] - PrevOrient[i][2])) maxAngle = fabs(orient[2] - PrevOrient[i][2]);

      if(maxAngle < angleThresh)
      {
printf("orientation hasn't changed enough, only %f \r", maxAngle);
        validView = false;
      }
    }

    // Only add this view if it is sufficiently different from previous views
    if( validView )
    {
      PrevPos.push_back(pos);
      PrevOrient.push_back(orient);
    }
    else
    {
      // This view won't be used, delete it
      delete[] pos;
      delete[] orient;
      return;
    }
  }

  // Only consider finding and using the calibration pattern if
  // the endoscope has moved substantially
  if(lastTrackerIdx != PrevPos.size())
  {
    int curIdx = PrevPos.size();
    int cbRes, nDetectedCorners;
    IplImage *frame = cvCreateImageHeader(cvSize(640,480), IPL_DEPTH_8U, 3);
    cvSetData(frame, ptr, 3*640);

    // Find the internal corners in the image
    if(corners.size() < PrevPos.size())
      corners.push_back(new CvPoint2D32f[nUCorners*nVCorners]);

    cbRes = cvFindChessboardCorners(frame,
                                    cvSize(nUCorners,nVCorners),
                                    corners[curIdx-1],
                                    &nDetectedCorners);

    // Only process the image if all points are seen
    if(cbRes)
    {
printf("                                  \r");
      // Create a new image which is a grayscale version of the video feed
      IplImage *grayframe = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
      cvCvtColor(frame, grayframe, CV_RGB2GRAY);

      // Refine the corner positions
      cvFindCornerSubPix(grayframe, corners[curIdx-1], nDetectedCorners, cvSize(3,3), 
                         cvSize(-1,-1), cvTermCriteria(CV_TERMCRIT_ITER,128,0));

      // Delete the image
      cvReleaseImage(&grayframe);

      // Ensure the orientation of the recorded points is correct
      int uorient=0, vorient=0;
      if(corners[curIdx-1][0].x < corners[curIdx-1][nUCorners-1].x) uorient++;
      if(corners[curIdx-1][0].x < corners[curIdx-1][nUCorners*(nVCorners-1)].x) uorient++;
      if(corners[curIdx-1][0].x < corners[curIdx-1][nUCorners*(nVCorners-1)+nUCorners-1].x) uorient++;
      if(corners[curIdx-1][0].y < corners[curIdx-1][nUCorners-1].y) vorient++;
      if(corners[curIdx-1][0].y < corners[curIdx-1][nUCorners*(nVCorners-1)].y) vorient++;
      if(corners[curIdx-1][0].y < corners[curIdx-1][nUCorners*(nVCorners-1)+nUCorners-1].y) vorient++;
      if(uorient < 2)
      {
        int uu, vv;
        for(vv = 0; vv < nVCorners; vv++)
          for(uu = 0; uu < nUCorners; uu++)
          {
            float xx, yy;
            xx = corners[curIdx-1][vv*nUCorners + uu].x;
            yy = corners[curIdx-1][vv*nUCorners + uu].y;
            corners[curIdx-1][vv*nUCorners + uu].x = corners[curIdx-1][vv*nUCorners + (nUCorners-1 - uu)].x;
            corners[curIdx-1][vv*nUCorners + uu].y = corners[curIdx-1][vv*nUCorners + (nUCorners-1 - uu)].y;
            corners[curIdx-1][vv*nUCorners + (nUCorners-1 - uu)].x = xx;
            corners[curIdx-1][vv*nUCorners + (nUCorners-1 - uu)].y = yy;
          }
      }
      if(vorient < 2)
      {
        int uu, vv;
        for(vv = 0; vv < nVCorners; vv++)
          for(uu = 0; uu < nUCorners; uu++)
          {
            float xx, yy;
            xx = corners[curIdx-1][vv*nUCorners + uu].x;
            yy = corners[curIdx-1][vv*nUCorners + uu].y;
            corners[curIdx-1][vv*nUCorners + uu].x = corners[curIdx-1][(nVCorners-1 - vv)*nUCorners + uu].x;
            corners[curIdx-1][vv*nUCorners + uu].y = corners[curIdx-1][(nVCorners-1 - vv)*nUCorners + uu].y;
            corners[curIdx-1][(nVCorners-1 - vv)*nUCorners + uu].x = xx;
            corners[curIdx-1][(nVCorners-1 - vv)*nUCorners + uu].y = yy;
          }
      }

      PrevMat.push_back(vtkMatrix4x4::New());
      PrevMat[curIdx-1]->DeepCopy(mat);
printf("%d images acquired\n", curIdx);

      // Perform a mock calibration. It might remove an acquired image if it
      // produces a bogus estimate.
      DoCalibration(0);

      // Update the last TrackerIdx used
      lastTrackerIdx = PrevPos.size();
    }
    else
    {
printf("could not find whole checkerboard \r");
      // We didn't use this view, remove it from the previous views
      PrevPos.pop_back(); 
      PrevOrient.pop_back();
	  delete[] corners.back();
	  corners.pop_back();
    }

    cvReleaseImageHeader(&frame);
  }
}

void vtkEndoscope::DoCalibration(int bSetMatrix)
{
    int i, j, u, v;
    std::vector<float> allX, allY, allZ;
    float accX=0, accY=0, accZ=0;
    double wxyz[4], quat[4] = {0,0,0,0};
    vtkMatrix4x4 *sensor2checkerboard = vtkMatrix4x4::New(),
                 *checkerboard2camera = vtkMatrix4x4::New(),
                 *calib = vtkMatrix4x4::New();
    double (*matrix)[4];
    double ortho[3][3];
    mat33 aveMat;
    int nActualCalibrationViews = corners.size();

    // One cannot perform an acceptable calibration with a single image
    if(nActualCalibrationViews == 1)
	{
	  return;
	}

	  ///////////////////////////////////////////////////
	  // Make some important assertions that we rely on
	  assert(corners.size() == PrevMat.size());

    //////////////////////////////////////////
    // First we do the intrinsic calibration

    // Allocate the matrices used for the calibration
    object_points = cvCreateMat(1, nActualCalibrationViews*nUCorners*nVCorners, CV_32FC3);
    image_points = cvCreateMat(1, nActualCalibrationViews*nUCorners*nVCorners, CV_32FC2);
    point_counts = cvCreateMat(1, nActualCalibrationViews, CV_32SC1);
    rotation_vectors = cvCreateMat(1, nActualCalibrationViews, CV_32FC3);
    translation_vectors = cvCreateMat(1, nActualCalibrationViews, CV_32FC3);

    // Initialize the allocated data
    for(i = 0; i < nActualCalibrationViews; i++)
    {
      // Set the number of points
      cvSet2D(point_counts, 0, i, cvScalar(nUCorners*nVCorners));

      // Set the real-world coordinate of these points
      for(v = 0; v < nVCorners; v++)
        for(u = 0; u < nUCorners; u++)
        {
          int baseIdx = i * nUCorners * nVCorners;

          // Assume the checkboard is on the XY plane with one
          // corner being the origin
          cvSet1D(object_points, 
                  baseIdx + v*nUCorners + u, 
                  cvScalar(u*cornerDim, v*cornerDim, 0.0)
                 );
        }
  
      // Add this view to what we will feed the calibrate function
      int baseIdx = i * nUCorners*nVCorners;
      for(j = 0; j < nUCorners*nVCorners; j++)
      {
        cvSet1D(image_points, baseIdx+j, cvScalar(corners[i][j].x, corners[i][j].y));
      }
    }
        
    // Calibrate the endoscope
    cvCalibrateCamera2(object_points, image_points, point_counts, cvSize(640,480),
                       intrinsic_matrix, distortion_coeffs, rotation_vectors, 
                       translation_vectors);

    // Now determine the endoscope<->sensor transform
    for(i = 1; i < nActualCalibrationViews; i++)
    {
      CvMat *rvec = cvCreateMat(1, 1, CV_32FC3);
      CvMat *tvec = cvCreateMat(1, 1, CV_32FC3);

      for(j = 0; j < nUCorners*nVCorners; j++)
      {
        cvSet1D(image_points, j, cvScalar(corners[i][j].x, corners[i][j].y));
      }

      CvMat *rotmat = cvCreateMat(3,3,CV_32FC1);
      cvSet1D(rvec, 0, cvGet1D(rotation_vectors, i)); 

      CvScalar val = cvGet1D(translation_vectors, i);

      cvRodrigues2(rvec, rotmat);

      // Compute the transform from the sensors frame to the checkerboard's frame
      vtkMatrix4x4::Multiply4x4(ToCheckerboard,
                                PrevMat[i], 
                                sensor2checkerboard
                               );

      // Load the transform from the checkboard's frame to the camera's frame
      checkerboard2camera->Identity();
      for(j = 0; j < 3; j++)
      {
        checkerboard2camera->SetElement(j, 0, cvGetReal2D(rotmat, j, 0));
        checkerboard2camera->SetElement(j, 1, cvGetReal2D(rotmat, j, 1));
        checkerboard2camera->SetElement(j, 2, cvGetReal2D(rotmat, j, 2));
        checkerboard2camera->SetElement(j, 3, val.val[j]);
      }

      // Concatenate these two transforms to go from sensor space to camera space
      vtkMatrix4x4::Multiply4x4(checkerboard2camera,
                                sensor2checkerboard,
                                calib);

      // Invert the calibration matrix to go from camera space to sensor space
      calib->Invert();

      // convenient access to matrix
#ifdef QUAT_AVERAGE
      matrix = calib->Element;
      for (j = 0; j < 3; j++)
        {
        ortho[0][j] = matrix[0][j];
        ortho[1][j] = matrix[1][j];
        ortho[2][j] = matrix[2][j];
        }
      if (vtkMath::Determinant3x3(ortho) < 0)
          for (j = 0; j < 3; j++)
          {
          ortho[0][j] = -ortho[0][j];
          ortho[1][j] = -ortho[1][j];
          ortho[2][j] = -ortho[2][j];
          }
#else
      // Average in the current matrix
      if(i == 0)
      {
        aveMat.FromMatrix4x4(calib);
      }
      else
      {
        aveMat.Average(i, mat33(calib));
      }
#endif

      // Throw away this acquisition if it's estimates are ridiculous (ie. any of
      // the translation components are greater than a meter)
      if( (fabs(calib->GetElement(0,3)) > mMaxExpectedDistance) || 
          (fabs(calib->GetElement(1,3)) > mMaxExpectedDistance) || 
          (fabs(calib->GetElement(2,3)) > mMaxExpectedDistance) )
      {
printf("Nonsensical result... omitting.\n");
        lastTrackerIdx--;
        PrevMat.pop_back();
        PrevPos.pop_back();
        PrevOrient.pop_back();
		delete[] corners.back();
        corners.pop_back();

        // Free the all the related data
        cvReleaseMat(&point_counts);
        cvReleaseMat(&rotation_vectors);
        cvReleaseMat(&translation_vectors);
        cvReleaseMat(&object_points);
        cvReleaseMat(&image_points);

        return;
      }

      // Accumulate the translations/orientations
#ifdef QUAT_AVERAGE
      vtkMath::Matrix3x3ToQuaternion(ortho, wxyz);
      quat[0] += wxyz[0];
      quat[1] += wxyz[1];
      quat[2] += wxyz[2];
      quat[3] += wxyz[3];
#endif
      allX.push_back(calib->GetElement(0,3));
      accX += calib->GetElement(0,3);
      allY.push_back(calib->GetElement(1,3));
      accY += calib->GetElement(1,3);
      allZ.push_back(calib->GetElement(2,3));
      accZ += calib->GetElement(2,3);
    }

    // Make sure these all add up
    assert(allX.size() == allY.size());
    assert(allX.size() == allZ.size());

    // Average the accumulated values
    accX /= allX.size(); 
    accY /= allY.size();
    accZ /= allZ.size();

    // Wrap things up here
    double diffX=0, diffY=0, diffZ=0;

    for(i = 0; i < allX.size(); i++)
    {
      diffX += pow(allX[i] - accX, 2);
      diffY += pow(allY[i] - accY, 2);
      diffZ += pow(allZ[i] - accZ, 2);
    }
    diffX /= allX.size() - 1;
    diffY /= allX.size() - 1;
    diffZ /= allX.size() - 1;
	   
	  // Reporting on the stats
    printf("%.2f %.2f %.2f\t+/-\t",
       accX, accY, accZ);
    printf("%.2f %.2f %.2f         \n", 
         sqrt(diffX), sqrt(diffY), sqrt(diffZ) );

	  // Find the worst outlier
	  float deviation = 0.f;
	  for(i = 1; i < nActualCalibrationViews; i++)
	  {
		  float curDev = (allX[i] - accX)*(allX[i] - accX) + 
					     (allY[i] - accY)*(allY[i] - accY) +
					     (allZ[i] - accZ)*(allZ[i] - accZ);

		  // Record this, if it's the worst outlier
		  if(curDev > deviation)
		  {
			  deviation = curDev;
			  worstOutlier = i;
		  }
	  }

#ifdef QUAT_AVERAGE
    // Average the estimations
    double d = sqrt(quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]);
    quat[0] /= d; quat[1] /= d; quat[2] /= d; quat[3] /= d;
    vtkMath::QuaternionToMatrix3x3(quat, ortho);
    calib->SetElement(0,3,accX);
    calib->SetElement(1,3,accY);
    calib->SetElement(2,3,accZ);
    for (i = 0; i < 3; i++)
    {
      calib->SetElement(0,i,ortho[0][i]);
      calib->SetElement(1,i,ortho[1][i]);
      calib->SetElement(2,i,ortho[2][i]);
    }
#else
      // Set the orientation elements according to the average 
    for (i = 0; i < 3; i++)
    {
      calib->SetElement(0,i,aveMat.m[0][i]);
      calib->SetElement(1,i,aveMat.m[1][i]);
      calib->SetElement(2,i,aveMat.m[2][i]);
    }
#endif

    if(bSetMatrix)
    {
      TrackerTool->SetCalibrationMatrix(calib);
      bCalibrated = true;
    }

    // Free the all the related data
    cvReleaseMat(&point_counts);
    cvReleaseMat(&rotation_vectors);
    cvReleaseMat(&translation_vectors);
    cvReleaseMat(&object_points);
    cvReleaseMat(&image_points);
}

void vtkEndoscope::DeleteOutlier(void)
{
	// First, do a mock calibration
	DoCalibration(0);

	// Now remove the worst outlier
	delete[] SwapErase(corners, worstOutlier);
	delete[] SwapErase(PrevPos, worstOutlier);
	SwapErase(PrevMat, worstOutlier)->Delete();
	delete[] SwapErase(PrevOrient, worstOutlier);
}

void vtkEndoscope::InternalGrab()
{
  static vtkMatrix4x4 *curMat = vtkMatrix4x4::New();
  static vtkTransform *curTrans = vtkTransform::New();

  if (this->AutoAdvance)
    {
    this->AdvanceFrameBuffer(1);
    if (this->FrameIndex + 1 < this->FrameBufferSize)
      {
      this->FrameIndex++;
      }
    }

  int index = this->FrameBufferIndex;

  this->FrameBufferTimeStamps[index] = 
    this->CreateTimeStampForFrame(this->LastFrameCount + 1);
  if (this->FrameCount++ == 0)
    {
    this->StartTimeStamp = this->FrameBufferTimeStamps[index];
    }

  void *ptr = ((reinterpret_cast<vtkDataArray *>( \
                       this->FrameBuffer[index]))->GetVoidPointer(0));

  int depth = this->FrameBufferBitsPerPixel/8;

  int offsetX = this->FrameBufferExtent[0];
  int offsetY = this->FrameBufferExtent[2];

  int sizeX = this->FrameBufferExtent[1] - this->FrameBufferExtent[0] + 1;
  int sizeY = this->FrameBufferExtent[3] - this->FrameBufferExtent[2] + 1;

  if (sizeX > 0 && sizeY > 0)
    {
      MbufGetColor2d(this->MILBufID,M_RGB24+M_PACKED,M_ALL_BAND,
                     offsetX,offsetY,sizeX,sizeY,ptr);
    }

  // Deinterlace
  if(deinterlace)
  {
    void *ptrLast = ((reinterpret_cast<vtkDataArray *>( \
                         this->FrameBuffer[(index+1)%this->FrameBufferSize]))->GetVoidPointer(0));
    unsigned char *cptr = (unsigned char*)ptr;
    unsigned char *cptrLast = (unsigned char*)ptrLast;
    if(!bOddField)
    {
      int i, j;

      // First copy the image down to the right place
      for(j = 1; j < 479; j+=2)
      {
        for(i = 0; i < 640*3; i++) cptr[j*640*3+i] = cptr[(j-1)*640*3+i];
      }

      // Bob
      for(j = 2; j < 479; j+=2)
      {
        for(i = 0; i < 640*3; i++) cptr[j*640*3+i] = (int(cptr[(j-1)*640*3+i])+int(cptr[(j+1)*640*3+i])) >> 1;
      }
    }
    else
    {
      int i, j;

      // Bob
      for(j = 1; j < 479; j+=2)
      {
        for(i = 0; i < 640*3; i++) cptr[j*640*3+i] = (int(cptr[(j-1)*640*3+i])+int(cptr[(j+1)*640*3+i])) >> 1;
      }
    }
  }

  this->Modified();
}

// TODO: Move these somewhere proper
#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480
void vtkEndoscope::GetImageParams(float &fx, float &fy, float &cx, float &cy)
{
  fx = SCREEN_WIDTH / cvGetReal2D(intrinsic_matrix, 0, 0);
  fy = SCREEN_WIDTH / cvGetReal2D(intrinsic_matrix, 1, 1);  // Screen width again, 'cause that's how OpenCV seems to do things
  cx = ((SCREEN_WIDTH/2) - cvGetReal2D(intrinsic_matrix, 0, 2));// / SCREEN_WIDTH;
  cy = (cvGetReal2D(intrinsic_matrix, 1, 2) - (SCREEN_HEIGHT/2));// / SCREEN_HEIGHT;
}

void vtkEndoscope::Load(const char *filename)
{
  int i, j;
  ifstream inFile(filename);
  vtkMatrix4x4 *mat = vtkMatrix4x4::New();

  for(j = 0; j < 4; j++)
    for(i = 0; i < 4; i++)
      inFile >> mat->Element[j][i];

  TrackerTool->SetCalibrationMatrix(mat);

  for(j = 0; j < 3; j++)
    for(i = 0; i < 3; i++)
    {
      double val;
      inFile >> val;
      cvSetReal2D(intrinsic_matrix, i, j, val);
    }

  for(i = 0; i < 4; i++)
  {
    double val;
    inFile >> val;
    cvSetReal1D(distortion_coeffs, i, val);
  }

  // Set the real-world coordinate of these points
  object_points = cvCreateMat(1, nUCorners*nVCorners, CV_32FC3);

  // We're basically calibrated now
  bCalibrated = true;
}

void vtkEndoscope::Save(const char *filename)
{
  int i, j;

  DoCalibration();

  ofstream outFile(filename);
  vtkMatrix4x4 *mat = TrackerTool->GetCalibrationMatrix();

  for(j = 0; j < 4; j++)
  {
    for(i = 0; i < 4; i++)
      outFile << mat->Element[j][i] << " ";

    outFile << endl;
  }

  for(j = 0; j < 3; j++)
  {
    for(i = 0; i < 3; i++)
      outFile << cvGetReal2D(intrinsic_matrix, i, j) << " ";

    outFile << endl;
  }

  for(i = 0; i < 4; i++)
    outFile << cvGetReal1D(distortion_coeffs, i) << " ";
}

void vtkEndoscope::GetRawImageAndTransform(void ** ptr, vtkMatrix4x4 *mat)
{
  // Grab the frame that we believe is sync'd up with the tracking
  this->FrameBufferMutex->Lock();

  int index = (this->FrameBufferIndex + this->FrameBufferSync) % this->FrameBufferSize;
  *ptr = ((reinterpret_cast<vtkDataArray *>( \
                  this->FrameBuffer[index]))->GetVoidPointer(0));

  this->FrameBufferMutex->Unlock();

  TrackerTool->GetBuffer()->Lock();
  TrackerTool->GetBuffer()->GetFlagsAndMatrixFromTime(mat,this->FrameBufferTimeStamps[this->FrameBufferIndex]+TrackerLatency);
  TrackerTool->GetBuffer()->Unlock();
}

void vtkEndoscope::GetRawImageAndTransform(int i, void ** ptr, vtkMatrix4x4 *mat)
{
  // Grab the frame that we believe is sync'd up with the tracking
  this->FrameBufferMutex->Lock();

  int index = (this->FrameBufferIndex + this->FrameBufferSync + i) % this->FrameBufferSize;
  *ptr = ((reinterpret_cast<vtkDataArray *>( \
                  this->FrameBuffer[index]))->GetVoidPointer(0));
  this->FrameBufferMutex->Unlock();

  TrackerTool->GetBuffer()->Lock();
  TrackerTool->GetBuffer()->GetFlagsAndMatrixFromTime(mat,this->FrameBufferTimeStamps[(this->FrameBufferIndex+i) % 
                                                                                       this->FrameBufferSize]+TrackerLatency);
  TrackerTool->GetBuffer()->Unlock();
}

void vtkEndoscope::SetIntrinsicInitialGuess(double intrinsic[3][3], double distortion[4])
{
  // TODO
}

void vtkEndoscope::CreateUndistortMap(int w, int h, float *&pb)
{
  // Create undistortmap
  CvArr *mapx=cvCreateImage(cvSize(640,480),IPL_DEPTH_32F,1), 
        *mapy=cvCreateImage(cvSize(640,480),IPL_DEPTH_32F,1);
  cvInitUndistortMap(intrinsic_matrix, distortion_coeffs, mapx, mapy);

  // Convert this to a texture and free the memory
  int i = 0;
  pb = new float[w*h*4];
  for(int y = 0; y < h; y++)
    for(int x = 0; x < w; x++)
    {
        double U = cvGetReal2D(mapx, y, x);
        double V = cvGetReal2D(mapy, y, x);

        pb[i++] = U;
        pb[i++] = V;
        i += 2;
    }
}

//----------------------------------------------------------------------------
// The Execute method is fairly complex, so I would not recommend overriding
// it unless you have to.  Override the UnpackRasterLine() method instead.
// You should only have to override it if you are using something other 
// than 8-bit vtkUnsignedCharArray for the frame buffer.
void vtkEndoscope::ExecuteData(vtkDataObject *output)
{
  vtkImageData *data = this->AllocateOutputData(output);
  int i,j;

  int outputExtent[6];     // will later be clipped in Z to a single frame
  int saveOutputExtent[6]; // will possibly contain multiple frames
  data->GetExtent(outputExtent);
  for (i = 0; i < 6; i++)
    {
    saveOutputExtent[i] = outputExtent[i];
    }
  // clip to extent to the Z size of one frame  
  outputExtent[4] = this->FrameOutputExtent[4]; 
  outputExtent[5] = this->FrameOutputExtent[5]; 

  int frameExtentX = this->FrameBufferExtent[1]-this->FrameBufferExtent[0]+1;
  int frameExtentY = this->FrameBufferExtent[3]-this->FrameBufferExtent[2]+1;
  int frameExtentZ = this->FrameBufferExtent[5]-this->FrameBufferExtent[4]+1;

  int extentX = outputExtent[1]-outputExtent[0]+1;
  int extentY = outputExtent[3]-outputExtent[2]+1;
  int extentZ = outputExtent[5]-outputExtent[4]+1;

  // if the output is more than a single frame,
  // then the output will cover a partial or full first frame,
  // several full frames, and a partial or full last frame

  // index and Z size of the first frame in the output extent
  int firstFrame = (saveOutputExtent[4]-outputExtent[4])/extentZ;
  int firstOutputExtent4 = saveOutputExtent[4] - extentZ*firstFrame;

  // index and Z size of the final frame in the output extent
  int finalFrame = (saveOutputExtent[5]-outputExtent[4])/extentZ;
  int finalOutputExtent5 = saveOutputExtent[5] - extentZ*finalFrame;

  char *outPtr = (char *)data->GetScalarPointer();
  char *outPtrTmp;

  int inIncY = (frameExtentX*this->FrameBufferBitsPerPixel + 7)/8;
  inIncY = ((inIncY + this->FrameBufferRowAlignment - 1)/
            this->FrameBufferRowAlignment)*this->FrameBufferRowAlignment;
  int inIncZ = inIncY*frameExtentY;

  int outIncX = this->NumberOfScalarComponents;
  int outIncY = outIncX*extentX;
  int outIncZ = outIncY*extentY;

  int inPadX = 0;
  int inPadY = 0;
  int inPadZ; // do inPadZ later

  int outPadX = -outputExtent[0];
  int outPadY = -outputExtent[2];
  int outPadZ;  // do outPadZ later

  if (outPadX < 0)
    {
    inPadX -= outPadX;
    outPadX = 0;
    }

  if (outPadY < 0)
    {
    inPadY -= outPadY;
    outPadY = 0;
    }

  int outX = frameExtentX - inPadX; 
  int outY = frameExtentY - inPadY; 
  int outZ; // do outZ later

  if (outX > extentX - outPadX)
    {
    outX = extentX - outPadX;
    }
  if (outY > extentY - outPadY)
    {
    outY = extentY - outPadY;
    }

  // if output extent has changed, need to initialize output to black
  for (i = 0; i < 3; i++)
    {
    if (saveOutputExtent[i] != this->LastOutputExtent[i])
      {
      this->LastOutputExtent[i] = saveOutputExtent[i];
      this->OutputNeedsInitialization = 1;
      }
    }

  // ditto for number of scalar components
  if (data->GetNumberOfScalarComponents() != 
      this->LastNumberOfScalarComponents)
    {
    this->LastNumberOfScalarComponents = data->GetNumberOfScalarComponents();
    this->OutputNeedsInitialization = 1;
    }

  // initialize output to zero only when necessary
  if (this->OutputNeedsInitialization)
    {
    memset(outPtr,0,
           (saveOutputExtent[1]-saveOutputExtent[0]+1)*
           (saveOutputExtent[3]-saveOutputExtent[2]+1)*
           (saveOutputExtent[5]-saveOutputExtent[4]+1)*outIncX);
    this->OutputNeedsInitialization = 0;
    } 

  // we have to modify the outputExtent of the first frame,
  // because it might be complete (it will be restored after
  // the first frame has been copied to the output)
  int saveOutputExtent4 = outputExtent[4];
  outputExtent[4] = firstOutputExtent4;

  this->FrameBufferMutex->Lock();

  // ANIS: This next line is my only change
  int index = this->FrameBufferIndex + this->FrameBufferSync;
  this->FrameTimeStamp = 
    this->FrameBufferTimeStamps[index % this->FrameBufferSize];

  int frame;
  for (frame = firstFrame; frame <= finalFrame; frame++)
    {
    if (frame == finalFrame)
      {
      outputExtent[5] = finalOutputExtent5;
      } 
    
    vtkDataArray *frameBuffer = reinterpret_cast<vtkDataArray *>(this->FrameBuffer[(index + frame) % this->FrameBufferSize]);

    char *inPtr = reinterpret_cast<char*>(frameBuffer->GetVoidPointer(0));
    char *inPtrTmp ;

    extentZ = outputExtent[5]-outputExtent[4]+1;
    inPadZ = 0;
    outPadZ = -outputExtent[4];
    
    if (outPadZ < 0)
      {
      inPadZ -= outPadZ;
      outPadZ = 0;
      }

    outZ = frameExtentZ - inPadZ;

    if (outZ > extentZ - outPadZ)
      {
      outZ = extentZ - outPadZ;
      }

    if (this->FlipFrames)
      { // apply a vertical flip while copying to output
      outPtr += outIncZ*outPadZ+outIncY*outPadY+outIncX*outPadX;
      inPtr += inIncZ*inPadZ+inIncY*(frameExtentY-inPadY-outY);

      for (i = 0; i < outZ; i++)
        {
        inPtrTmp = inPtr;
        outPtrTmp = outPtr + outIncY*outY;
        for (j = 0; j < outY; j++)
          {
          outPtrTmp -= outIncY;
          if (outX > 0)
            {
            this->UnpackRasterLine(outPtrTmp,inPtrTmp,inPadX,outX);
            }
          inPtrTmp += inIncY;
          }
        outPtr += outIncZ;
        inPtr += inIncZ;
        }
      }
    else
      { // don't apply a vertical flip
      outPtr += outIncZ*outPadZ+outIncY*outPadY+outIncX*outPadX;
      inPtr += inIncZ*inPadZ+inIncY*inPadY;

      for (i = 0; i < outZ; i++)
        {
        inPtrTmp = inPtr;
        outPtrTmp = outPtr;
        for (j = 0; j < outY; j++)
          {
          if (outX > 0) 
            {
            this->UnpackRasterLine(outPtrTmp,inPtrTmp,inPadX,outX);
            }
          outPtrTmp += outIncY;
          inPtrTmp += inIncY;
          }
        outPtr += outIncZ;
        inPtr += inIncZ;
        }
      }
    // restore the output extent once the first frame is done
    outputExtent[4] = saveOutputExtent4;
    }

  this->FrameBufferMutex->Unlock();
}


