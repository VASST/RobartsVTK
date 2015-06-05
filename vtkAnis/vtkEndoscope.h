// .NAME vtkEndoscope - interfaces VTK to an endoscope
// .SECTION Description
// The vtkEndoscope provides an interface between a tracked endoscope
// .SECTION see also
// vtkTracker vtkMicroBirdTracker vtkVideoSource

#ifndef __vtkEndoscope_h
#define __vtkEndoscope_h

#include "vtkMILVideoSource.h"

#include <vector>
#include <cv.h>
#include <cxcore.h>
#include <vtkVersion.h> //for VTK_MAJOR_VERSION

class vtkTransform;
class vtkMatrix4x4;
class vtkTracker;
class vtkTrackerTool;
class vtkMultiThreader;
class vtkCriticalSection;

class VTK_EXPORT vtkEndoscope : public vtkMILVideoSource
{
public:
  static vtkEndoscope *New();
#if (VTK_MAJOR_VERSION <= 5)
  vtkTypeRevisionMacro(vtkEndoscope,vtkMILVideoSource);
#else
  vtkTypeMacro(vtkEndoscope,vtkMILVideoSource);
#endif

  // Description:
  // Set the tracker tool to input transforms from.
  void SetTracker(vtkTracker *t)
  {
    Tracker = t;
  }

  virtual void SetTrackerTool(vtkTrackerTool *);
  vtkGetObjectMacro(TrackerTool,vtkTrackerTool);

  void SetTrackerLatency(double t)
  {
    TrackerLatency = t;
  }

  virtual void SetFrameBufferSync(int i) { this->FrameBufferSync = i; };

  // Description:
  // Standard VCR functionality: Record incoming video.
  void Record();

  // Description:
  // Standard VCR functionality: Stop incoming video.
  void Stop();

  // Description:
  // Standard VCR functionality: Stop incoming video.
  void GrabEnd();

  // Description:
  // Grab a single video frame.
  void GrabBegin();

  // Description:
  // Start calibrating the endoscope. Computes the barrel distortion
  // Description:
  // Determine if the endoscope has been satisfactorily calibrated.
  bool Calibrated() { return bCalibrated; }

  // Description:
  // Sets the checkerboard space
  void SetCheckerboard(float ox, float oy, float oz,
                       float xx, float xy, float xz,
                       float yx, float yy, float yz);

  // Description:
  // Acquire an image for calibration purposes
  void AcquireCalibrationImage(void);

  // Description:
  // Complete calibration
  void DoCalibration(int bSetMatrix=1);

  // Description:
  // Removes an outliner from the set of collected images used in calibration
  void DeleteOutlier(void);

  // Description:
  // Sets the checkerboard square dimension
  void SetSquareDim_inch(double dim) { cornerDim = dim * 25.4; }
  void SetSquareDim_mm(double dim) { cornerDim = dim; }

  // Sets the number of internal corners in the U direction
  void SetNumberOfUCorners(int u) { nUCorners = u; }

  // Sets the number of internal corners in the V direction
  void SetNumberOfVCorners(int v) { nVCorners = v; }

  // Sets the number of internal corners in the V direction
  void SetDivotDepth(double d) { divotDepth = d; }

  // Load/Save the calibration
  void Load(const char *);
  void Save(const char *);

  // Sets the minimum amount of orientation difference between views, in degrees
  void SetMinimumAngularThreshold(double thresh) { angleThresh = thresh; }

  // Sets the minimum amount of translational difference in mm
  void SetMinimumTranslationalThreshold(double thresh) { distThresh = thresh; }

  // Overriden functions
  void SetOutputFormat(int format);
  void InternalGrab();

  // Grab this frame and camera position
  void GetRawImageAndTransform(void ** ptr, vtkMatrix4x4 *mat);

  // Grab this frame and camera position
  void GetRawImageAndTransform(int i, void ** ptr, vtkMatrix4x4 *mat);

  // Specify intrinsic initial guesses
  void SetIntrinsicInitialGuess(double intrinsic[3][3], double distortion[4]);

  // Create an opengl texture for the undistortion
  void CreateUndistortMap(int, int, float*&);

  // Get the center of the image as reported by the intrinsic coords
  void GetImageParams(float &fx, float &fy, float &cx, float &cy);

  // Get the transform from checkerboard space to tracker space
  vtkGetObjectMacro(FromCheckerboard,vtkMatrix4x4);

  int GetNumUCorners(void)
  {
    return nUCorners;
  }

  int GetNumVCorners(void)
  {
    return nVCorners;
  }

  int GetCornerDim(void)
  {
    return cornerDim;
  }

  CvMat *GetIntrinsicMatrix(void)
  {
    return intrinsic_matrix;
  }

protected:
  vtkEndoscope();
  ~vtkEndoscope();

  bool bCalibrated;  // Is the endoscope calibrated?

  bool bOddField;

  int nUCorners;
  int nVCorners;
  double divotDepth;
  double cornerDim;
  double angleThresh;
  double distThresh;
  vtkTracker *Tracker;
  vtkTrackerTool *TrackerTool;
  vtkMatrix4x4 *ToCheckerboard; // Transformation from tracker coords to checkboard
  vtkMatrix4x4 *FromCheckerboard; // Transformation from checkerboard to tracker coords
  vtkMatrix4x4 *Calibration;    // Transformation from sensor to camera-space
  int FrameBufferSync;          // Number of frames to buffer in order to sync with tracking
  double TrackerLatency;
  int lastTrackerIdx;
  int deinterlace;
  float mMaxExpectedDistance;
  size_t worstOutlier;

  //BTX
  int nAcqs;
  std::vector<CvPoint2D32f*> corners;
  CvMat *object_points, *image_points, *point_counts;
  CvMat *intrinsic_matrix, *distortion_coeffs,
        *rotation_vectors, *translation_vectors;
  IplImage *distorted_image, *undistorted_image;

  std::vector<vtkMatrix4x4*> PrevMat;
  std::vector<float*> PrevPos;
  std::vector<float*> PrevOrient;
  //ETX

  vtkMultiThreader *Threader;
  int ThreadId;

  virtual void ExecuteData(vtkDataObject *data);
};

#endif
