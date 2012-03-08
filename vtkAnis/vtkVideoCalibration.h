// .NAME vtkVideoCalibration - interfaces VTK to an endoscope
// .SECTION Description
// The vtkVideoCalibration provides an interface between a tracked endoscope
// .SECTION see also
// vtkTracker vtkMicroBirdTracker vtkVideoSource
/*Carling: Functionality to calibrate either a single camera or a stereo camera
	For stereo calibration, left and right images are captured simultaneously.
	Functions have three versions: normal, left and right. Eventually, the normal functions
	should be removed so that to calibrate a single camera, set up the system to use only the left
	or only the right camera.
*/
#ifndef __vtkVideoCalibration_h
#define __vtkVideoCalibration_h

#include <vector>
#include "cv.h"
#include "cxcore.h"
//#include "highgui.h"

#include "vtkTrackerTool.h"

#include "vtkVideoSource.h"
#include "vtkMatrix4x4.h"
#include "vtkTransform.h"
#include <fstream>

class vtkObject;

class VTK_EXPORT vtkVideoCalibration: public vtkObject
{
public:
	static vtkVideoCalibration *New();
	vtkTypeMacro(vtkVideoCalibration,vtkVideoSource);

	void SetTrackerLatency(double t) { TrackerLatency = t;  }
	virtual void SetFrameBufferSync(int i) { this->FrameBufferSync = i; };
	bool Calibrated() { return IsCalibrated; }

	void SaveImageL(const char * filename, unsigned int imageNumber);
	void SaveImageR(const char * filename, unsigned int imageNumber);
	void SaveImagesStereo(char * directoryL, char * directoryR);

	// Description:
	// Sets the checkerboard space
	void SetCheckerboard(float ox, float oy, float oz, float xx, float xy, float xz, float yx, float yy, float yz);

	// Description:
	// Acquire an image for calibration purposes
	void AcquireCalibrationImage(void);
	void AcquireLeftImage();
	void AcquireRightImage();

	// Description:
	// Complete calibration
	void DoCalibration(int apply=1,vtkMatrix4x4 * matrixToCheckerboard);
	void DoLeftCalibration(int apply=1, vtkMatrix4x4 *matrixToCheckerboard);
	void DoRightCalibration(int apply=1, vtkMatrix4x4 *matrixToCheckerboard);

	// Specify intrinsic initial guesses
	void SetIntrinsicInitialGuess(double intrinsic[3][3], double distortion[4]);

	// Create an opengl texture for the undistortion
	void CreateUndistortMap(int, int, float*&);

	// Get the center of the image as reported by the intrinsic coords
	int GetImageParams(float &fx, float &fy, float &cx, float &cy);

	// Get the transform from checkerboard space to tracker space
	vtkGetObjectMacro(FromCheckerboard,vtkMatrix4x4);

	vtkSetMacro(ImageWidth, int);
	vtkGetMacro(ImageWidth, int);

	vtkSetMacro(ImageHeight, int);
	vtkGetMacro(ImageHeight, int);

	// Get/Set the number of internal corners in the U direction
	void SetRows(int u)	{ chessboard_cornerrows = u; }
	int GetRows(void)			{ return chessboard_cornerrows; }

	// Get/Set the number of internal corners in the V direction
	void SetColumns(int v)	{ chessboard_cornercols = v; }
	int GetColumns(void)			{ return chessboard_cornercols; }

	// Get/Set the checkerboard square dimension
	void SetSquareDim_inch(double dim) { chessboard_boxsize = dim * 25.4; }
	void SetSquareDim_mm(double dim) { chessboard_boxsize = dim; }

	int GetCornerDim(void) { return chessboard_boxsize; }

	void SetAverageQuaternionsOn() { this->AverageQuaternions = true;}
	void SetAverageQuaternionsOff() { this->AverageQuaternions = false;}
	void SetAverageQuaternions(bool value) { this->AverageQuaternions = value;}
	bool IsAveraging() { return this->AverageQuaternions;}

	vtkMatrix4x4 * GetToCheckerboard() { return this->ToCheckerboard; }
	void SaveImage(const char * filename, unsigned int imageNumber);
	void SaveImages(char * directory);
	void LoadImage(const char * filename , const char * transformFile);
	void LoadLeftImage(const char * filename , const char * transformFile);
	void LoadRightImage(const char * filename , const char * transformFile);
	vtkMatrix4x4 * GetImageMatrix(unsigned int i);
	vtkMatrix4x4 * GetLeftImageMatrix(unsigned int i);
	vtkMatrix4x4 * GetRightImageMatrix(unsigned int i);
	int GetNumberOfImages() {return singleCamera.images.size();}
	int GetNumberOfLeftImages() {return leftCamera.images.size();}
	int GetNumberOfRightImages() {return rightCamera.images.size();}
	vtkMatrix4x4 * GetCalibration() {return singleCamera->Calibration;}
	vtkMatrix4x4 * GetLeftCalibration() {return leftCamera.Calibration;}
	vtkMatrix4x4 * GetRightCalibration() {return rightCamera.Calibration;}

protected:
	vtkVideoCalibration();
	~vtkVideoCalibration();
  
	// cwedlake - accessers methods required
	bool AverageQuaternions; // Should Quaternions be averaged 

	int FrameBufferSync;          // Number of frames to buffer in order to sync with tracking
	double TrackerLatency;
	int lastTrackerIdx;


private:
	vtkMatrix4x4 *ToCheckerboard; // Transformation from tracker coords to checkboard 
	vtkMatrix4x4 *FromCheckerboard;

	int ImageWidth;
	int ImageHeight;

	//BTX
	int flag_cornersfound;
	int flag_calibrated;

	int chessboard_cornerrows;
	int chessboard_cornercols;
	float chessboard_boxsize;
	int chessboard_numcorners;
	//ETX

};

#endif