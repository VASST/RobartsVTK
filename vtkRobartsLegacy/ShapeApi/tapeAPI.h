#ifndef TAPEAPI_H
#define TAPEAPI_H

// The data class collects data from the tape and converts it to usable 
// calibrated data.  The base function of this class is getCurvatureData
// which returns bend and twist from the raw data of the tape.  From bend 
// and twist, vector cartesian data can be calculated.
// Written by Jordan Lutes
// for Measurand Inc. 30/06/2000 
// Added bend only and uneven sensor length support. 30/08/2001
// Added new tape model.  19/11/2001

#include <cmath>
#include "calibrate.h"
#include "tapeAPI.h"

#define FALSE 0
#define TRUE 1

// magic numbers...
const int new_way = 0; 
const double build_angle = 45;

class SHAPEAPI_API tapeAPI:public calibrate
{
public:
	tapeAPI(int num_sense, int baud_rate,
			  char* cal_file, double length, int interval); // Constructor.
	tapeAPI(int num_sense, int baud_rate,
			  char* cal_file, double length, int interval,
			  bool only_bend, int num_regions, double region_length[48]); // Constructor w/ bend only and
																		  // uneven sensor length support.
	tapeAPI(char* config_file, char *rootfolder); // Constructor that reads settings in from a text file.
	~tapeAPI(); // Destructor.

	// The following two functions allow the user to set the base position and 
	// orientation of the tape.
	setPosition(double x, double y, double z); // Set x,y,z.
	setOrientation(double baseU[3], double baseB[3]); // Set the base U and B vectors.
	// Where U is a unit vector along length of tape(twist about U), and 
	// B is a unit vector along width of tape. (bend about B)
	setOrientation(double yaw, double pitch, double roll); // Set the base yaw, pitch and roll.
	// Where the yaw is about x, then pitch about y' and finally,
	// roll about z''

	// Gets the various settings for use in the program.  This function is normally used 
	// when the mst file constructor was used.
	getTapeSettings(char *cal_file, int &num_sense, int &interp_interv, double &length, int &num_regions, double region_lengths[]);

	// The following four functions poll the tape for raw data and then convert it 
	// to the desired format for use.  In all cases the dimension of the arrays 
	// passed in should be of size = number of sensor regions - num_region.
	bool getCurvatureData(double bend[], double twist[]); // Returns bend and twist in rad/mm.
	getVectorData(double u[][3], double n[][3], double b[][3], double r[][3]); // Returns 
	// u,n,b,r vector set.  The vectors u,n and b are unit vectors describing the 
	// orientation of the tape.  u is along the length, n is normal to the surface, and
	// b is along the width of the tape.  Therefore the tape is bent about its b vector 
	// and twisted about its u vector.  The r vector contain the vector from the origin  
	// to the current sensors position.  ie. It contains the x,y, and z coordinates of the
	// sensors.
	bool getCartesianData(double x[], double y[], double z[],
				     double q1[], double q2[], double q3[], double q4[]); // Returns the 
	// x,y,z coordinates of the sensors and the quaternion describing their orientations. 

	// Saves the calibration file and references the mstFile for the tape taper twist corrections.
	void saveAdvancedCalibration(char *filename, char *mstfile);
	// Not rocommended for use at this time.
	void saveNormalCalibration(char *filename);
	// Saves a new flat pose in the calibration file.  
	void saveFlatCalibration(char *filename);
	// Adds a helical pose to an older file which does not have one.  
	void saveHelicalCalibration(char *filename);

protected:
	char *cal_file_name; // Calibration file name.
	int interval; // Intpolation interval.
	double Uo[3]; // Base vector along length of tape. (twist about U)
	double Bo[3]; // Base vector along width of tape. (bend about B)
	double Xo, Yo, Zo; // Base position of the tape.
	void bt2cart(double bend[], double twist[], double u[][3],
			double n[][3], double b[][3], double r[][3]);
	normalize(double vector[3]); // normalize the vector array.
	cross(double A[3], double B[3], double C[3]); // A cross B = C

private:
	void interpolate(double y[], double yi[]);
	convertToVectorData(double u[][3], double n[][3], double b[][3], double r[][3], double interp_bend[], double interp_twist[]);
	void quat2cart(double q1[], double q2[], double q3[], double q4[], 
				   double x[], double y[], double z[]);
	void rotline(double* fpL1, double* fpLA,	double fAlpha);	
	void rotatePoint(double x, double y, double z, 
					 double q_w, double q_x, double q_y, double q_z);
	double magnitude(double vector[3]); // return the magnitude of the vector.
	
	void checkMSTFile(char *settings_file); // error checking on the mst file.
	void getCartData(double x[], double y[], double z[], 
					 double q1[], double q2[], double q3[], double q4[]);
	// Same as getCartesianData above only uses quaternions for everything beyond getCurvatureData function.
	void vectoralign(double vector1[], double vector2[], double rotaxis[], double &rotangle);
	void reftotip(double bend[], double twist[]);


private:
};

#endif