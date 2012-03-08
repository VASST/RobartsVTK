#ifndef CALIBRATE_H
#define CALIBRATE_H

// Calibrate class inherits the collect class to read raw
// data from the tape.  The purpose of this class is to 
// store the calibration poses in memory and to file and to 
// calculate the bend and twist coefficients from this raw 
// data.  
// Written by Jordan Lutes
// for Measurand Inc. 30/06/2000 

// Convert the standard four pose calibration to a seven pose 
// helical calibration method.  The end closer poses...
// Feb. 28 2001
// Changed tape model to the new helical model.

#include <stdio.h>
//#include <fstream>
#include "collect.h"
#include "shapeAPI.h"

class SHAPEAPI_API calibrate:public collect
{
public:
	double * getTempfactor();
	calibrate(int num_sense, int baud_rate, double length); // Constructor.
	calibrate(int num_sense, int baud_rate, double length, bool only_bend, int num_regions, double region_length[48]); 
	calibrate(char* config_file); // 
	~calibrate(); // Destructor.

	// The following functions record five sets of data, average them,
	// and store them in there appriate class member.
	flat(); // Stores data in flat_data.
	bend_positive_tiny(); //Stores data in bend_up_tiny.
	bend_positive_small(); // Stores data in bend_up_small.
	bend_positive_large(); // Stores data in bend_up_large.
	bend_negative_tiny(); //Stores data in bend_down_tiny.
	bend_negative_small(); // Stores data in bend_down_small.
	bend_negative_large(); // Stores data in bend_down_large.
	twist_positive_small(); // Stores data in twist_ccw_small.
	twist_positive_large(); // Stores data in twist_ccw_large.
	twist_negative_small(); // Stores data in twist_cw_small.
	twist_negative_large(); // Stores data in twist_cw_large.
	offsets(); // Stores data in offset.

	helix_down_cw(); // Helical pose for new helical model.

	// The calculate function takes the raw data from the five poses 
	// and calculates the four quadrant calibration coefficients for 
	// bend and twist.  See documentation for details.
	calculate_advanced(double bend_top[][4], double bend_bottom[][4], double twist_top[][4], double twist_bottom[][4],
		char *mstfile);
	calculate_normal(double bend_top[][4], double bend_bottom[][4], double twist_top[][4], double twist_bottom[][4]);
	
	// Load pose data in from file.
	loadFile(char* cal_file);
	loadNewData(char* raw_data_file, unsigned short ser_num);
	//loadOldFile(char* old_cal_file);

	// Save the current class member pose data to file. 
	saveFile(char* cal_file);

	// Reads in a set of raw data from the tape, then subtracts the flat 
	// pose data from it.  ie. pure data = raw data - offsets (flat data) 
	getPureData(int data[]);
	// Same as above except passes frame number and time stamp information.
	getPureData(int data[], int frame, unsigned int time);

	// Sets/gets the proper amount of bend and twist for the poses.
	setPoseCurvature(double tiny_bend_radius, double small_bend_radius, double large_bend_radius,
					 double small_twist_angle, double large_twist_angle);
	getPoseCurvature(double &tiny_bend_radius, double &small_bend_radius, double &large_bend_radius,
					 double &small_twist_angle, double &large_twist_angle);	

	// Sets/gets the helical pose radius.
	setHelixRadius(double helix_pose_radius);
	getHelixRadius(double &helix_pose_radius);

	// 2 Functions used in special pose.
	void getoffsets(int *offsets);
	void UpdateOffsets(int newoffsets[]);
	
	//function used to update offsets according to temperature changes seen by the 1st sensor,
	//the tape type (i.e. arm, leg, or head), and the cutoff point at which the tape's 
	//sensors come into close contact with the wearer
	void UpdateOffsets(int first_sensor,int tapetype,int cutoff); 

	// Sets the measured twist values into the twist_offset member variable.
	void setTwistOffset(double twist[]);
	
private:
	// averages five sets of pose data.  Currently not in use.
	average(int data[][5], int ave_data[]);
	// collects five sets of data.  Currently not in use.
	collectSamples(int data[][5]);
	// subtracts the offset pose information from the data signal and returns a pure voltage.
	subtractOffsets(int data[], int pure_data[]);
	// subtracts the flat pose information from the data signal and returns a pure voltage.
	subtractFlat(int data[], int pure_data[]);
	// subtracts the second array from the first and stores it in the result array.
	subtract(int first[], int second[], int result[]);
	// finds the determinate of the double matrix [[a11, a12], [a21, a22]].
	double findDeterminate(double a11, double a12, double a21, double a22);
	// solves a set of 2 equations 2 unknowns using Kramers rule.
	solve(double a[2], double b[2], double c[2], double &x, double &y);
	// finds the y intercept of a linear function.
	double findYIntercept(double delta_curvature, int delta_counts, int small_counts, double small_curvature);
	// calculates the radius which each sensor sees when it is wrapped around a calibration fixture 
	// and overlaps itself.
	double *calculateSpiral(double calib_radius, double tape_thickness, double gap_distance);
	// puts a linear twist correction on the calibration pose to account for the taper.
	void linearTwistCorrection(double twist[], double twist_amount, char *textfile);
	void linearTwistCorrection(double twist[], double twist_amount);
	// find the bend for a given voltage reading from a tape.  Used in the calculation of the huge_bend coefficients.
	void FindHugeBend(int V[],double *hugebend);
	initializePoseData(); // initializes the pose data to zeras.

private:
	int* flat_data; // Pose voltages for the various poses.
	int* bend_up_large;
	int* bend_down_large;
	int* bend_up_small;
	int* bend_down_small;
	int* bend_up_tiny;
	int* bend_down_tiny;
	int* twist_cw_large;
	int* twist_ccw_large;
	int* twist_cw_small;
	int* twist_ccw_small;
	int* offset;

	int* helical_pose;
	
	double tiny_radius_of_bend, small_radius_of_bend, large_radius_of_bend,
		   small_angle_of_twist, large_angle_of_twist; // calibration parameters.

protected:
	int num_sensors; // the number of sensors on the tape.
	double tape_length; // the total length of the tape (in mm).  Soon to be replaced 
						// by summing region_length array.
	int num_region; // the number of sensor regions on the tape.
	double region_length[48]; // the length of each sensor region (paired or unpaired).
	bool bend_only; // true if bend_only tape, false if bend/twist.
	double twist_offset[24]; // The twist offset to be applied
	double bend_top[24][4], bend_bottom[24][4]; // Bend coefficients.
	double twist_top[24][4], twist_bottom[24][4]; // Twist coefficients.
	double ratio_top[24][4], ratio_bottom[24][4]; // Twist to Bend Ratios.
	double hibend_up[24], hibend_down[24]; //High bend correction coefficients
	int y_bend_up_top[24], y_bend_down_top[24]; // Y inercepts...
	int y_bend_up_bottom[24], y_bend_down_bottom[24];
	int y_twist_ccw_top[24], y_twist_ccw_bottom[24];
	int y_twist_cw_top[24], y_twist_cw_bottom[24];
	double* k1_down; // k values - gains in helical model.
	double* k1_up;
	double* k2_down;
	double* k2_up;
	double* phase_angle; // phase angle in the bend term of the model.
	double helix_radius; // radius of helical pose.
	void findKValues(); // finds the k values and phase angle for the new model using the calibration 
						// coefficients from the old model and the new helical pose. 
	double *tempfactor; //temperature correction factors for each sensor
};
#endif // calibrate.h