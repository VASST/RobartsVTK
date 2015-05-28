#ifndef __vtkStereoCamera_h
#define __vtkStereoCamera_h

#include <vector>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

#include "vtkTrackerTool.h"
#include "vtkVideoSource.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include <fstream>

class vtkObject;

class VTK_EXPORT vtkStereoCamera: public vtkObject {

  public:
    static vtkStereoCamera *New();

    int chessboard_cornerrows;
    int chessboard_cornercols;
    int chessboard_numcorners;
    float chessboard_boxsize;

    vtkSetMacro(ImageWidth, int);
    vtkGetMacro(ImageWidth, int);

    vtkSetMacro(ImageHeight, int);
    vtkGetMacro(ImageHeight, int);

    bool Calibrated() { return IsCalibrated; }

    void SetVideoSource(vtkVideoSource *v) {this->Video = v;}
    vtkGetObjectMacro(Video, vtkVideoSource);

    void SetTrackerTool(vtkTrackerTool *t) {this->TrackerTool = t;}
    vtkGetObjectMacro(TrackerTool, vtkTrackerTool);

    void vtkStereoCamera::AcquireImage(void);
    void vtkStereoCamera::DoCalibration(int apply, vtkMatrix4x4 * ToCheckerboard);
    int vtkStereoCamera::FindCorners(IplImage * image ,CvPoint3D32f * pcorners3d, CvPoint2D32f * pcorners2d);

    vtkMatrix4x4 * vtkStereoCamera::GetImageMatrix(unsigned int i);
    vtkTransform * vtkStereoCamera::GetToolTransform(unsigned int i);
    vtkMatrix4x4 * vtkStereoCamera::GetCalibMatrix(unsigned int i);
    void vtkStereoCamera::PrintIntrinsic();
    void vtkStereoCamera::GetIntrinsicMatrix(double values[9]);
    void vtkStereoCamera::GetDistortionCoefficients(double values[4]);
    void vtkStereoCamera::GetUndistortedPoint(int pos[2], float newPos[2]);
    void vtkStereoCamera::GetUndistortedPoint(int pos[2], double newPos[2]);

    int GetNumberOfImages() {return this->images.size();}
    vtkMatrix4x4 * GetCalibration() {return this->Calibration;}
    /*void SaveImage(const char * filename, unsigned int imageNumber);
    void SaveImages(char * directory);
    void LoadImage(const char * filename , const char * transformFile);*/

  protected:
    vtkStereoCamera();
    ~vtkStereoCamera();

  private:
    vtkTrackerTool *TrackerTool;        //The tracker tool that represents this camera
    bool IsCalibrated;              //Whether or not the camera is calibrated
    vtkMatrix4x4 *Calibration;          //The camera calibration matrix (final objective)
    vtkVideoSource *Video;            //Video source for this camera
    CvMat* pcamera_matrix;            //Intrinsic parameters
    CvMat* pcamera_distortion;          //Distortion coefficiennts
    std::vector<vtkTransform *> toolTransform;  //Tool transform from which each image was taken
    std::vector<vtkMatrix4x4 *> calibs;      //Calculated calib matrix for each image
    std::vector<IplImage *> images;        //Image
    std::vector<CvPoint2D32f *> corners2d;    //2D coordinates of where the vertices are in the image
    std::vector<CvPoint3D32f *> corners3d;    //3D coordinates of target
    std::vector<CvMat*> transvector;      //Stores translation vectors returned from calibration
    std::vector<CvMat*> rotmatrix;        //Stores rotation matrices returned from calibration

    int ImageWidth;
    int ImageHeight;

    int flag_cornersfound;
    int flag_calibrated;
};
#endif