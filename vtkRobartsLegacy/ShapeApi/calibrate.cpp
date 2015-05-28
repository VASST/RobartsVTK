#include "stdafx.h"
#include "calibrate.h"
#include "filedata.h"
#include <cmath>
#include <cstdio>

using namespace std;

const double pi = 3.14159;

/*
Name:    calibrate
Purpose:  constructor
Accepts:  num_sense = number of sensors.
      baud_rate = baud_rate of serial connection.
      length    = overall length of sensorized region of tape.
Returns:    void
*/
calibrate::calibrate(int num_sense, int baud_rate, double length) // Constructor.
      :collect(num_sense, baud_rate)
{
  flat_data = new int[num_sense+1];
  bend_up_large = new int[num_sense+1];
  bend_up_small = new int[num_sense+1];
  bend_up_tiny = new int[num_sense+1];
  bend_down_large = new int[num_sense+1];
  bend_down_small = new int[num_sense+1];
  bend_down_tiny = new int[num_sense+1];

  twist_cw_large = new int[num_sense+1];
  twist_cw_small = new int[num_sense+1];
  twist_ccw_large = new int[num_sense+1];
  twist_ccw_small = new int[num_sense+1];
  offset = new int[num_sense+1];
  tempfactor=new double[num_sense+1];

  helical_pose = new int[num_sense+1];
  
  tape_length = length;
  num_sensors = num_sense;
  large_radius_of_bend = 300; // always in mm
  large_angle_of_twist = 360;
  small_radius_of_bend = 115;
  small_angle_of_twist = 180;
  tiny_radius_of_bend = 60;

  initializePoseData();

  for (int i = 0;i < 48;i++)
    region_length[i] = 0.0;

  // assume it is a bend/twist tape...
  bend_only = false;
  num_region = num_sensors/2;
  for (i = 0; i < num_region; i++)
    region_length[i] = tape_length/num_region;

  k1_down = new double[num_region+1];
  k1_up = new double[num_region+1];
  k2_down = new double[num_region+1];
  k2_up = new double[num_region+1];

  phase_angle = new double[num_region+1];
}

/*
Name:    calibrate
Purpose:  constructor - with new features. bend only and uneven sensor size support.
Accepts:  num_sense      = number of sensors.
      baud_rate      = baud_rate of serial connection.
      length         = overall length of sensorized region of tape.
      bend_only      = true if bend only, flase if not.
      num_regions    = the number of sensorized regions along the tape.
      region_lengths = the length of each region.
Returns:    void
*/
calibrate::calibrate(int num_sense, int baud_rate, double length, 
           bool only_bend, int num_regions, double region_lengths[48])
          :collect(num_sense, baud_rate)
{
  flat_data = new int[num_sense+1];
  bend_up_large = new int[num_sense+1];
  bend_up_small = new int[num_sense+1];
  bend_up_tiny = new int[num_sense+1];
  bend_down_large = new int[num_sense+1];
  bend_down_small = new int[num_sense+1];
  bend_down_tiny = new int[num_sense+1];

  twist_cw_large = new int[num_sense+1];
  twist_cw_small = new int[num_sense+1];
  twist_ccw_large = new int[num_sense+1];
  twist_ccw_small = new int[num_sense+1];
  offset = new int[num_sense+1];

  helical_pose = new int[num_sense+1];

  tempfactor=new double[num_sense+1];

  initializePoseData();

  tape_length = length;
  num_sensors = num_sense;
  large_radius_of_bend = 300; // always in mm
  large_angle_of_twist = 360;
  small_radius_of_bend = 115;
  small_angle_of_twist = 180;
  tiny_radius_of_bend = 60;

  bend_only = only_bend;
  num_region = num_regions;

  for(int i = 0;i < 48;i++)
    region_length[i] = region_lengths[i];

  k1_down = new double[num_region+1];
  k1_up = new double[num_region+1];
  k2_down = new double[num_region+1];
  k2_up = new double[num_region+1];

  phase_angle = new double[num_region+1];
}

/*
Name:    calibrate
Purpose:  constructor - with new features. bend only and uneven sensor size support.
Accepts:  settings_file = file name for the tape setting file. Contains all the variables of the
      above constructor in the filedata file format.
Returns:    void
*/
calibrate::calibrate(char* settings_file)
      :collect(settings_file)
{
  int temp_bend_only;

  filedata file(settings_file);
  num_sensors = file.getInteger("[settings]","number of sensors");
  tape_length = file.getDouble("[settings]","length (mm)");
  temp_bend_only = file.getInteger("[settings]", "bend only");
  num_region = file.getInteger("[settings]", "num regions");
  file.getDouble("[settings]","region length",48,region_length);

  flat_data = new int[num_sensors+1];
  bend_up_large = new int[num_sensors+1];
  bend_up_small = new int[num_sensors+1];
  bend_up_tiny = new int[num_sensors+1];
  bend_down_large = new int[num_sensors+1];
  bend_down_small = new int[num_sensors+1];
  bend_down_tiny = new int[num_sensors+1];

  twist_cw_large = new int[num_sensors+1];
  twist_cw_small = new int[num_sensors+1];
  twist_ccw_large = new int[num_sensors+1];
  twist_ccw_small = new int[num_sensors+1];
  offset = new int[num_sensors+1];

  helical_pose = new int[num_sensors+1];
  
  tempfactor=new double[num_sensors+1];

  k1_down = new double[num_region+1];
  k1_up = new double[num_region+1];
  k2_down = new double[num_region+1];
  k2_up = new double[num_region+1];

  phase_angle = new double[num_region+1];

  initializePoseData();

  large_radius_of_bend = 300; // always in mm
  large_angle_of_twist = 360;
  small_radius_of_bend = 115;
  small_angle_of_twist = 180;
  tiny_radius_of_bend = 60;

  if (temp_bend_only == 0)
    bend_only = false;
  else if (temp_bend_only == 1)
    bend_only = true;
  else
    bend_only = false;

}

/*
Name:    ~calibrate
Purpose:  destructor
Accepts:  void
Returns:    void
*/
calibrate::~calibrate() // Destructor.
{
  if (flat_data != NULL)
    delete [] flat_data;
  if (bend_up_tiny !=NULL) 
    delete [] bend_up_tiny;
  if (bend_down_tiny !=NULL)
    delete [] bend_down_tiny;
  if (bend_up_large != NULL)
    delete [] bend_up_large;
  if (bend_up_small != NULL)
    delete [] bend_up_small;
  if (bend_down_large != NULL)
    delete [] bend_down_large;
  if (bend_down_small != NULL)
    delete [] bend_down_small;
  if (twist_cw_large != NULL)
    delete [] twist_cw_large;
  if (twist_cw_small != NULL)
    delete [] twist_cw_small;
  if (twist_ccw_large != NULL)
    delete [] twist_ccw_large;
  if (twist_ccw_small != NULL)
    delete [] twist_ccw_small;
  if (offset != NULL)
    delete [] offset;
  if (k1_down != NULL)
    delete [] k1_down;
  if (k1_up != NULL)
    delete [] k1_up;
  if (k2_down != NULL)
    delete [] k2_down;
  if (k2_up != NULL)
    delete [] k2_up;
  if (helical_pose != NULL)
    delete [] helical_pose;
  if (phase_angle !=NULL)
    delete [] phase_angle;
  if (tempfactor !=NULL)
    delete []tempfactor;
}

/*
Name:    flat
Purpose:  collects a set of raw tape data for the flat pose.
Accepts:  void
Returns:    void
*/
calibrate::flat()
{
  pollTape(flat_data);
  for (int i = 0; i < num_sensors; i ++)
  {
    offset[i] = flat_data[i];
  }
  // set the offsets equal to flat until offsets is called.
}

/*
Name:    bend_positive_small
Purpose:  collects a set of raw tape data for the large radius bend up pose.
Accepts:  void
Returns:    void
*/
calibrate::bend_positive_small()
{
  pollTape(bend_up_small);
}

/*
Name:    bend_positive_tiny
Purpose:  collects a set of raw tape data for the huge radius bend up pose.
Accepts:  void
Returns:    void
*/
calibrate::bend_positive_tiny()
{
  pollTape(bend_up_tiny);
}

/*
Name:    bend_positive_large
Purpose:  collects a set of raw tape data for the small radius bend up pose.
Accepts:  void
Returns:    void
*/
calibrate::bend_positive_large()
{
  pollTape(bend_up_large);
}

/*
Name:    bend_negative_small
Purpose:  collects a set of raw tape data for the large radius bend down pose.
Accepts:  void
Returns:    void
*/
calibrate::bend_negative_small()
{
  pollTape(bend_down_small);
}

/*
Name:    bend_negative_tiny
Purpose:  collects a set of raw tape data for the huge radius bend down pose.
Accepts:  void
Returns:    void
*/
calibrate::bend_negative_tiny()
{
  pollTape(bend_down_tiny);
}

/*
Name:    bend_negative_large
Purpose:  collects a set of raw tape data for the small radius bend down pose.
Accepts:  void
Returns:    void
*/
calibrate::bend_negative_large()
{
  pollTape(bend_down_large);
}

/*
Name:    twist_negative_small
Purpose:  collects a set of raw tape data for the small angle twist cw pose.
Accepts:  void
Returns:    void
*/
calibrate::twist_negative_small()
{
  pollTape(twist_cw_small);
}

/*
Name:    twist_negative_large
Purpose:  collects a set of raw tape data for the large angle twist cw pose.
Accepts:  void
Returns:    void
*/
calibrate::twist_negative_large()
{
  pollTape(twist_cw_large);
}

/*
Name:    twist_positive_small
Purpose:  collects a set of raw tape data for the small angle twist ccw pose.
Accepts:  void
Returns:    void
*/
calibrate::twist_positive_small()
{
  pollTape(twist_ccw_small);
}

/*
Name:    twist_positive_large
Purpose:  collects a set of raw tape data for the large angle twist ccw pose.
Accepts:  void
Returns:    void
*/
calibrate::twist_positive_large()
{
  pollTape(twist_ccw_large);
}

/*
Name:    offsets
Purpose:  collects a set of raw tape data for the offset pose.
Accepts:  void
Returns:    void
*/
calibrate::offsets()
{
  pollTape(offset);
}

/*
Name:    helix_down_cw
Purpose:  collects a set of raw tape data for the helix pose (negaive bend, negative twist).
Accepts:  void
Returns:    void
*/
calibrate::helix_down_cw()
{
  // This collection function is different from the others in that the flat pose is subtracted.
  pollTape(helical_pose);
  subtract(helical_pose, flat_data, helical_pose); 
}

/*
Name:    calculate_advanced
Purpose:  Calculates the bend and twist coeficients using the various pose information stored in this class.  
      The size of the  calibration coefficients are num_region * 4 (quadrants).  
Accepts:  bend_top     = bend coefficient for top sensor (of a pair).
      bend_bottom  = bend coefficient for bottom sensor.
      twist_top    = twist coefficient for top sensor. 
      twist_bottom = twist coefficient for bottom sensor.
      mstfile      = file containing twist taper information.
Returns:    void
*/
calibrate::calculate_advanced(double bend_top[][4], double bend_bottom[][4], double twist_top[][4], double twist_bottom[][4],
                char *mstfile)
{
  // OLD METHOD
  // These coefficients are from the equation 
  // Bend = bend_top * Voltage1 + bend_bottom * Voltage2
  // Twist = twist_top * Voltage1 + twist_bottom * Voltage2
  // A single sensor is used two different poses.
  // The two poses gives two equations / two unknowns, allowing us to 
  // solve for bend_top, bend_bottom, twist_top, and twist_bottom...  The calibration coefficients.
  // Where each of the bend/twist factors can be positive or negative.
  // This gives rise to a four quadrant calibraion coefficient.
  // One coefficient per pair.  Each having values pertaining to
  // all the bend/twist sign posibilities -- -+ +- and ++.
  // The assignment of the coefficients is in the above order.
  // Using the right hand rule, an upward bend is positive as well 
  // as a counter clockwise twist.
  // The twist / bend ratios are calculated for all quadrants.
  // These values are used later in the normal calibration to calculate the 
  // twist coefficients once the bend ones are found.

  // NEW METHOD
  // see math_model.doc for details

  double a[2], b[2], c[2];
  double* calib_radius;
  double* huge_bend;
  double* large_bend;
  double* small_bend;
  double* large_twist;
  double* small_twist;
  double* sensor_bend;
  double* sensor_twist;

  int* pure_bend_up_tiny;
  int* pure_bend_up_small;
  int* pure_bend_up_large;
  int* pure_bend_down_tiny;
  int* pure_bend_down_small;
  int* pure_bend_down_large;
  int* pure_twist_ccw_small;
  int* pure_twist_ccw_large;
  int* pure_twist_cw_small;
  int* pure_twist_cw_large;
  int* bend_up;
  int* bend_down;
  int* twist_ccw;
  int* twist_cw;

  pure_bend_up_tiny = new int[num_sensors+1];
  pure_bend_down_tiny = new int[num_sensors+1];
  pure_bend_up_small = new int[num_sensors+1];
  pure_bend_up_large = new int[num_sensors+1];
  pure_bend_down_small = new int[num_sensors+1];
  pure_bend_down_large = new int[num_sensors+1];
  pure_twist_ccw_small = new int[num_sensors+1];
  pure_twist_ccw_large = new int[num_sensors+1];
  pure_twist_cw_small = new int[num_sensors+1];
  pure_twist_cw_large = new int[num_sensors+1];

  bend_up = new int[num_sensors+1];
  bend_down = new int[num_sensors+1];
  twist_ccw = new int[num_sensors+1];
  twist_cw = new int[num_sensors+1];

  huge_bend = new double[num_region +1];
  large_bend = new double[num_region +1];
  small_bend = new double[num_region +1];
  large_twist = new double[num_region +1];
  small_twist = new double[num_region +1];

  sensor_bend = new double[num_region +1];
  sensor_twist = new double[num_region +1];

  // subtract flat from all sensor counts.
  subtractFlat(bend_up_tiny,pure_bend_up_tiny);
  subtractFlat(bend_up_small, pure_bend_up_small);
  subtractFlat(bend_up_large, pure_bend_up_large);
  subtractFlat(bend_down_tiny,pure_bend_down_tiny);
  subtractFlat(bend_down_small, pure_bend_down_small);
  subtractFlat(bend_down_large, pure_bend_down_large);
  subtractFlat(twist_ccw_small, pure_twist_ccw_small);
  subtractFlat(twist_ccw_large, pure_twist_ccw_large);
  subtractFlat(twist_cw_small, pure_twist_cw_small);
  subtractFlat(twist_cw_large, pure_twist_cw_large);

  // find the change in voltage between the large and small poses.
  subtract(bend_up_small,bend_up_large,bend_up);
  subtract(bend_down_small,bend_down_large,bend_down);
  subtract(twist_ccw_large, twist_ccw_small, twist_ccw);
  subtract(twist_cw_large, twist_cw_small, twist_cw);

// Always
  // figure out the bend on each sensor region.
  calib_radius = calculateSpiral(small_radius_of_bend, 1.5, 80);
  for (int x = 0; x < num_region; x++)
  {
    large_bend[x] = region_length[x]/calib_radius[x];
  }

  delete [] calib_radius;
  calib_radius = calculateSpiral(large_radius_of_bend, 1.5, 80);
  for (x = 0; x < num_region; x++)
  {
    small_bend[x] = region_length[x]/calib_radius[x];
  }

  delete [] calib_radius;

  for (x = 0; x < num_region; x++)
  {
    sensor_bend[x] = large_bend[x] - small_bend[x];
  }
// end Always

// Bend / Twist tapes
  if (bend_only == false)
  {
    // figure out the twist on each sensor pair.
    for (x = 0; x < num_region; x++)
    {
      large_twist[x] = 0.0;
      small_twist[x] = 0.0;
    }
    linearTwistCorrection(large_twist, large_angle_of_twist, mstfile);
    linearTwistCorrection(small_twist, small_angle_of_twist, mstfile);
    for (x = 0; x < num_region; x++)
    {
      sensor_twist[x] = large_twist[x] - small_twist[x];
    }
  }
// end Bend / Twist tapes
  
  // main cal loop
  for (int i = 0;i < num_region;i++)
  {
    // find the y-intercept between the sets of poses.
    y_bend_up_top[i] = -(int)findYIntercept(sensor_bend[i], bend_up[2*i], pure_bend_up_small[2*i],large_bend[i]);
    y_bend_down_top[i] = -(int)findYIntercept(-sensor_bend[i], bend_down[2*i], pure_bend_down_small[2*i],-large_bend[i]);
    y_bend_up_bottom[i] = -(int)findYIntercept(sensor_bend[i], bend_up[2*i+1], pure_bend_up_small[2*i+1],large_bend[i]);
    y_bend_down_bottom[i] = -(int)findYIntercept(-sensor_bend[i], bend_down[2*i+1], pure_bend_down_small[2*i+1],-large_bend[i]);
    
    // Pairs
    if (num_region == num_sensors / 2)
    {  
      // Bend only pairs
      if (bend_only == true)
      {
        // Negative bend, negative twist.
        bend_bottom[i][0] = 0.5 / bend_down[2*i];
        bend_top[i][0] = 0.5 / bend_down[2*i+1];
        
        // Negative bend, positive twist.
        bend_bottom[i][1] = 0.5 / bend_down[2*i];
        bend_top[i][1] = 0.5 / bend_down[2*i+1];
        
        // Positive bend, negative twist.
        bend_bottom[i][2] = 0.5 / bend_up[2*i];
        bend_top[i][2] = 0.5 / bend_up[2*i+1];
        
        // Positive bend, positive twist.
        bend_bottom[i][3] = 0.5 / bend_up[2*i];
        bend_top[i][3] = 0.5 / bend_up[2*i+1];
      }
      // Bend / Twist pairs
      else
      {
        y_twist_ccw_top[i] = -(int)findYIntercept(sensor_twist[i], twist_ccw[2*i], pure_twist_ccw_small[2*i],small_twist[i]);
        y_twist_cw_top[i] = -(int)findYIntercept(-sensor_twist[i], twist_cw[2*i], pure_twist_cw_small[2*i],-small_twist[i]);
        y_twist_ccw_bottom[i] = -(int)findYIntercept(sensor_twist[i], twist_ccw[2*i+1], pure_twist_ccw_small[2*i+1],small_twist[i]);
        y_twist_cw_bottom[i] = -(int)findYIntercept(-sensor_twist[i], twist_cw[2*i+1], pure_twist_cw_small[2*i+1],-small_twist[i]);
        
        // calculate the bend and twist coefficients.
        // Negative bend, negative twist.
        a[0] = bend_down[2*i];
        a[1] = twist_cw[2*i];
        b[0] = bend_down[2*i+1];
        b[1] = twist_cw[2*i+1];
        c[0] = -sensor_bend[i];
        c[1] = 0;
        // Solve for bend coefficients.
        solve(a,b,c,bend_top[i][0],bend_bottom[i][0]);
        c[0] = 0;
        c[1] = -sensor_twist[i];
        // Solve for twist coefficients.
        solve(a,b,c,twist_top[i][0],twist_bottom[i][0]);
        // Calculate the twist to bend ratios.
        ratio_top[i][0] = twist_top[i][0]/bend_top[i][0];
        ratio_bottom[i][0] = twist_bottom[i][0]/bend_bottom[i][0];
        
        // Negative bend, positive twist.
        a[0] = bend_down[2*i];
        a[1] = twist_ccw[2*i];
        b[0] = bend_down[2*i+1];
        b[1] = twist_ccw[2*i+1];
        c[0] = -sensor_bend[i];
        c[1] = 0;
        // Solve for bend coefficients.
        solve(a,b,c,bend_top[i][1],bend_bottom[i][1]);
        c[0] = 0;
        c[1] = sensor_twist[i];
        // Solve for twist coefficients.
        solve(a,b,c,twist_top[i][1],twist_bottom[i][1]);
        // Calculate the twist to bend ratios.
        ratio_top[i][1] = twist_top[i][1]/bend_top[i][1];
        ratio_bottom[i][1] = twist_bottom[i][1]/bend_bottom[i][1];
        
        // Positive bend, negative twist.
        a[0] = bend_up[2*i];
        a[1] = twist_cw[2*i];
        b[0] = bend_up[2*i+1];
        b[1] = twist_cw[2*i+1];
        c[0] = sensor_bend[i];
        c[1] = 0;
        // Solve for bend coefficients.
        solve(a,b,c,bend_top[i][2],bend_bottom[i][2]);
        c[0] = 0;
        c[1] = -sensor_twist[i];
        // Solve for twist coefficients.
        solve(a,b,c,twist_top[i][2],twist_bottom[i][2]);
        // Calculate the twist to bend ratios.
        ratio_top[i][2] = twist_top[i][2]/bend_top[i][2];
        ratio_bottom[i][2] = twist_bottom[i][2]/bend_bottom[i][2];
        
        // Positive bend, positive twist.
        a[0] = bend_up[2*i];
        a[1] = twist_ccw[2*i];
        b[0] = bend_up[2*i+1];
        b[1] = twist_ccw[2*i+1];
        c[0] = sensor_bend[i];
        c[1] = 0;
        // Solve for bend coefficients.
        solve(a,b,c,bend_top[i][3],bend_bottom[i][3]);
        c[0] = 0;
        c[1] = sensor_twist[i];
        // Solve for twist coefficients.
        solve(a,b,c,twist_top[i][3],twist_bottom[i][3]);
        // Calculate the twist to bend ratios.
        ratio_top[i][3] = twist_top[i][3]/bend_top[i][3];
        ratio_bottom[i][3] = twist_bottom[i][3]/bend_bottom[i][3];
      }
      // end Bend / Twist pairs
    }
    // end Pairs
    // Singles
    else 
    {
      // Negative bend, negative twist.
      bend_bottom[i][0] = 1.0 / bend_down[2*i];
      bend_top[i][0] = 1.0 / bend_down[2*i+1];
      
      // Negative bend, positive twist.
      bend_bottom[i][1] = 1.0 / bend_down[2*i];
      bend_top[i][1] = 1.0 / bend_down[2*i+1];
      
      // Positive bend, negative twist.
      bend_bottom[i][2] = 1.0 / bend_up[2*i];
      bend_top[i][2] = 1.0 / bend_up[2*i+1];
      
      // Positive bend, positive twist.
      bend_bottom[i][3] = 1.0 / bend_up[2*i];
      bend_top[i][3] = 1.0 / bend_up[2*i+1];
    }
    // end Singles
  }

// All 
  //now calculate high bend correction factors
  calib_radius = calculateSpiral(tiny_radius_of_bend, 1.5, 80);
  FindHugeBend(pure_bend_up_tiny,huge_bend);
  for (i=0; i<num_region; i++)
  {
    double theoretical_bend = region_length[i]/calib_radius[i];
    hibend_up[i] = theoretical_bend / huge_bend[i];
  }
  FindHugeBend(pure_bend_down_tiny,huge_bend);
  for (i=0; i<num_region; i++)
  {
    double theoretical_bend = -region_length[i]/calib_radius[i];
    hibend_down[i] = theoretical_bend / huge_bend[i];
  }
// end All

  findKValues(); // implements new model useing the old one as a base. 

  delete [] huge_bend;
  delete [] pure_bend_up_tiny;
  delete [] pure_bend_down_tiny;
  delete [] pure_bend_up_small;
  delete [] pure_bend_up_large;
  delete [] pure_bend_down_small;
  delete [] pure_bend_down_large;
  delete [] pure_twist_cw_small;
  delete [] pure_twist_cw_large;
  delete [] pure_twist_ccw_small;
  delete [] pure_twist_ccw_large;
  delete [] bend_up;
  delete [] bend_down;
  delete [] twist_ccw;
  delete [] twist_cw;
  delete [] small_bend;
  delete [] large_bend;
  delete [] small_twist;
  delete [] large_twist;
  delete [] sensor_bend;
  delete [] sensor_twist;
  delete [] calib_radius;
}

/*
Name:    calculate_advanced
Purpose:  Calculates the bend and twist coeficients using the various pose information stored in this class.  
      The size of the  calibration coefficients are num_region * 4 (quadrants).  This function is not tested
      properly and is NOT RECOMMENDED FOR USE.
Accepts:  bend_top     = bend coefficient for top sensor (of a pair).
      bend_bottom  = bend coefficient for bottom sensor.
      twist_top    = twist coefficient for top sensor. 
      twist_bottom = twist coefficient for bottom sensor.
Returns:    void
*/
calibrate::calculate_normal(double bend_top[][4], double bend_bottom[][4], double twist_top[][4], double twist_bottom[][4])
{
  int* pure_bend_up_tiny;
  int* pure_bend_up_small;
  int* pure_bend_up_large;
  int* pure_bend_down_tiny;
  int* pure_bend_down_small;
  int* pure_bend_down_large;
  int* bend_up;
  int* bend_down;
  double* huge_bend;
  double* calib_radius;
  double* large_bend;
  double* small_bend;
  double* sensor_bend;

  pure_bend_up_tiny = new int[num_sensors+1];
  pure_bend_up_small = new int[num_sensors+1];
  pure_bend_up_large = new int[num_sensors+1];
  pure_bend_down_tiny = new int[num_sensors+1];
  pure_bend_down_small = new int[num_sensors+1];
  pure_bend_down_large = new int[num_sensors+1];
  bend_up = new int[num_sensors +1];
  bend_down = new int[num_sensors +1];
  huge_bend = new double[num_region +1];
  large_bend = new double[num_region +1];
  small_bend = new double[num_region +1];
  sensor_bend = new double[num_region +1];
  
  // subtract flat from all sensor counts.
  subtractFlat(bend_up_tiny,pure_bend_up_tiny);
  subtractFlat(bend_down_tiny,pure_bend_down_tiny);
  subtractFlat(bend_up_small, pure_bend_up_small);
  subtractFlat(bend_up_large, pure_bend_up_large);
  subtractFlat(bend_down_small, pure_bend_down_small);
  subtractFlat(bend_down_large, pure_bend_down_large);

  // find the change in voltage between the large and small poses.
  subtract(bend_up_small,bend_up_large,bend_up);
  subtract(bend_down_small,bend_down_large,bend_down);

  // figure out the bend on each sensor pair.
  calib_radius = calculateSpiral(small_radius_of_bend, 1.5, 80);
  for (int x = 0; x < num_region; x++)
  {
    large_bend[x] = region_length[x]/calib_radius[x];
  }

  delete [] calib_radius;
  calib_radius = calculateSpiral(large_radius_of_bend, 1.5, 80);
  for (x = 0; x < num_sensors / 2; x++)
  {
    small_bend[x] = region_length[x]/calib_radius[x];
  }
  delete [] calib_radius;

  for (x = 0; x < num_sensors / 2; x++)
  {
    sensor_bend[x] = large_bend[x] - small_bend[x];
  }

  // main cal loop
  for (int i = 0;i < num_region;i=i+1)
  {   
    // find the y intercept between the sets of poses.
    y_bend_up_top[i] = -(int)findYIntercept(sensor_bend[i], bend_up[2*i], pure_bend_up_small[2*i],large_bend[i]);
    y_bend_down_top[i] = -(int)findYIntercept(-sensor_bend[i], bend_down[2*i], pure_bend_down_small[2*i],-large_bend[i]);
    y_bend_up_bottom[i] = -(int)findYIntercept(sensor_bend[i], bend_up[2*i+1], pure_bend_up_small[2*i+1],large_bend[i]);
    y_bend_down_bottom[i] = -(int)findYIntercept(-sensor_bend[i], bend_down[2*i+1], pure_bend_down_small[2*i+1],-large_bend[i]);
    
    // Pairs
    if (num_region == num_sensors/2)
    {
      // Bend only pairs
      if (bend_only == true)
      {
        // Negative bend, negative twist.
        bend_bottom[i][0] = 0.5 / bend_down[2*i];
        bend_top[i][0] = 0.5 / bend_down[2*i+1];
        
        // Negative bend, positive twist.
        bend_bottom[i][1] = 0.5 / bend_down[2*i];
        bend_top[i][1] = 0.5 / bend_down[2*i+1];
        
        // Positive bend, negative twist.
        bend_bottom[i][2] = 0.5 / bend_up[2*i];
        bend_top[i][2] = 0.5 / bend_up[2*i+1];
        
        // Positive bend, positive twist.
        bend_bottom[i][3] = 0.5 / bend_up[2*i];
        bend_top[i][3] = 0.5 / bend_up[2*i+1];
      }
      // Bend / Twist pairs
      else
      {
        // Negative bend, negative twist.
        bend_top[i][0] = -sensor_bend[i]/(bend_down[2*i]-(bend_down[2*i]*ratio_top[i][0]/ratio_bottom[i][0]));
        bend_bottom[i][0] = -sensor_bend[i]/(bend_down[2*i+1]-(bend_down[2*i+1]*ratio_bottom[i][0]/ratio_top[i][0]));
        twist_top[i][0] = ratio_top[i][0] * bend_top[i][0];
        twist_bottom[i][0] = ratio_bottom[i][0] * bend_bottom[i][0];
        
        // Negative bend, positive twist.
        bend_top[i][1] = -sensor_bend[i]/(bend_down[2*i]-(bend_down[2*i]*ratio_top[i][1]/ratio_bottom[i][1]));
        bend_bottom[i][1] = -sensor_bend[i]/(bend_down[2*i+1]-(bend_down[2*i+1]*ratio_bottom[i][1]/ratio_top[i][1]));
        twist_top[i][1] = ratio_top[i][1] * bend_top[i][1];
        twist_bottom[i][1] = ratio_bottom[i][1] * bend_bottom[i][1];
        
        // Positive bend, negative twist.
        bend_top[i][2] = sensor_bend[i]/(bend_up[2*i]-(bend_up[2*i]*ratio_top[i][2]/ratio_bottom[i][2]));
        bend_bottom[i][2] = sensor_bend[i]/(bend_up[2*i+1]-(bend_up[2*i+1]*ratio_bottom[i][2]/ratio_top[i][2]));
        twist_top[i][2] = ratio_top[i][2] * bend_top[i][2];
        twist_bottom[i][2] = ratio_bottom[i][2] * bend_bottom[i][2];
        
        // Positive bend, positive twist.
        bend_top[i][3] = sensor_bend[i]/(bend_up[2*i]-(bend_up[2*i]*ratio_top[i][3]/ratio_bottom[i][3]));
        bend_bottom[i][3] = sensor_bend[i]/(bend_up[2*i+1]-(bend_up[2*i+1]*ratio_bottom[i][3]/ratio_top[i][3]));
        twist_top[i][3] = ratio_top[i][3] * bend_top[i][3];
        twist_bottom[i][3] = ratio_bottom[i][3] * bend_bottom[i][3];
      }
    }
    // Singles
    else
    {
      // Negative bend, negative twist.
      bend_bottom[i][0] = 1.0 / bend_down[2*i];
      bend_top[i][0] = 1.0 / bend_down[2*i+1];
      
      // Negative bend, positive twist.
      bend_bottom[i][1] = 1.0 / bend_down[2*i];
      bend_top[i][1] = 1.0 / bend_down[2*i+1];
      
      // Positive bend, negative twist.
      bend_bottom[i][2] = 1.0 / bend_up[2*i];
      bend_top[i][2] = 1.0 / bend_up[2*i+1];
      
      // Positive bend, positive twist.
      bend_bottom[i][3] = 1.0 / bend_up[2*i];
      bend_top[i][3] = 1.0 / bend_up[2*i+1];
    }
  }

// All
  //now calculate high bend correction factors
  calib_radius = calculateSpiral(tiny_radius_of_bend, 1.5, 80);
  FindHugeBend(pure_bend_up_tiny,huge_bend);
  for (i=0;i<(num_region);i++)
  {
    double theoretical_bend = region_length[x]/calib_radius[x];
    hibend_up[i] = theoretical_bend / huge_bend[i];
  }
  FindHugeBend(pure_bend_down_tiny,huge_bend);
  for (i=0;i<(num_region);i++)
  {
    double theoretical_bend = -region_length[x]/calib_radius[x];
    hibend_down[i] = theoretical_bend / huge_bend[i];
  }
// end All
  findKValues();

  delete [] pure_bend_up_tiny;
  delete [] pure_bend_up_small;
  delete [] pure_bend_up_large;
  delete [] pure_bend_down_tiny;
  delete [] pure_bend_down_small;
  delete [] pure_bend_down_large;
  delete [] bend_up;
  delete [] bend_down;
  delete [] small_bend;
  delete [] large_bend;
  delete [] huge_bend;
  delete [] sensor_bend;
  delete [] calib_radius;
}

/*
Name:    loadNewData
Purpose:  Used by the raw data server to load the calibration coefficients during a 
      raw data playback.
Accepts:  raw_data_file = name of the raw data file.
      ser_num       = serial number of the tape. 
Returns:    void
*/
calibrate::loadNewData(char *raw_data_file, unsigned short ser_num)
{
  char serstring[7];
  sprintf(serstring, "[%i]", ser_num); 
  
  filedata file(raw_data_file);
  double bend_top_nn[24],
       bend_top_np[24],
       bend_top_pn[24],
       bend_top_pp[24],
       bend_bottom_nn[24],
       bend_bottom_np[24],
       bend_bottom_pn[24],
       bend_bottom_pp[24];
  double twist_top_nn[24],
       twist_top_np[24],
       twist_top_pn[24],
       twist_top_pp[24],
       twist_bottom_nn[24],
       twist_bottom_np[24],
       twist_bottom_pn[24],
       twist_bottom_pp[24];
  double ratio_top_nn[24],
       ratio_top_np[24],
       ratio_top_pn[24],
       ratio_top_pp[24],
       ratio_bottom_nn[24],
       ratio_bottom_np[24],
       ratio_bottom_pn[24],
       ratio_bottom_pp[24];

  file.getInteger(serstring, "flat", num_sensors, flat_data);
  for (int i=0;i<num_sensors;i++) offset[i]=flat_data[i];
  file.getDouble(serstring, "bend top nn", num_region, bend_top_nn);
  file.getDouble(serstring, "bend top np", num_region, bend_top_np);
  file.getDouble(serstring, "bend top pn", num_region, bend_top_pn);
  file.getDouble(serstring, "bend top pp", num_region, bend_top_pp);
  file.getDouble(serstring, "bend bottom nn", num_region, bend_bottom_nn);
  file.getDouble(serstring, "bend bottom np", num_region, bend_bottom_np);
  file.getDouble(serstring, "bend bottom pn", num_region, bend_bottom_pn);
  file.getDouble(serstring, "bend bottom pp", num_region, bend_bottom_pp);
  file.getDouble(serstring, "twist top nn", num_region, twist_top_nn);
  file.getDouble(serstring, "twist top np", num_region, twist_top_np);
  file.getDouble(serstring, "twist top pn", num_region, twist_top_pn);
  file.getDouble(serstring, "twist top pp", num_region, twist_top_pp);
  file.getDouble(serstring, "twist bottom nn", num_region, twist_bottom_nn);
  file.getDouble(serstring, "twist bottom np", num_region, twist_bottom_np);
  file.getDouble(serstring, "twist bottom pn", num_region, twist_bottom_pn);
  file.getDouble(serstring, "twist bottom pp", num_region, twist_bottom_pp);
  file.getDouble(serstring, "ratio top nn", num_region, ratio_top_nn);
  file.getDouble(serstring, "ratio top np", num_region, ratio_top_np);
  file.getDouble(serstring, "ratio top pn", num_region, ratio_top_pn);
  file.getDouble(serstring, "ratio top pp", num_region, ratio_top_pp);
  file.getDouble(serstring, "ratio bottom nn", num_region, ratio_bottom_nn);
  file.getDouble(serstring, "ratio bottom np", num_region, ratio_bottom_np);
  file.getDouble(serstring, "ratio bottom pn", num_region, ratio_bottom_pn);
  file.getDouble(serstring, "ratio bottom pp", num_region, ratio_bottom_pp);  

  file.getInteger(serstring, "helical_pose", num_sensors, helical_pose);
  helix_radius = file.getInteger(serstring, "helix_radius");
  file.getDouble(serstring,"tempfactor",num_sensors,tempfactor);

  for (i = 0; i < num_sensors / 2; i++)
  {
    // Move values into bend_top
    bend_top[i][0] = bend_top_nn[i];
    bend_top[i][1] = bend_top_np[i];
    bend_top[i][2] = bend_top_pn[i];
    bend_top[i][3] = bend_top_pp[i];

    // Move values into bend_bottom
    bend_bottom[i][0] = bend_bottom_nn[i];
    bend_bottom[i][1] = bend_bottom_np[i];
    bend_bottom[i][2] = bend_bottom_pn[i];
    bend_bottom[i][3] = bend_bottom_pp[i];

    // Move values into twist_top
    twist_top[i][0] = twist_top_nn[i];
    twist_top[i][1] = twist_top_np[i];
    twist_top[i][2] = twist_top_pn[i];
    twist_top[i][3] = twist_top_pp[i];

    // Move values into twist_bottom
    twist_bottom[i][0] = twist_bottom_nn[i];
    twist_bottom[i][1] = twist_bottom_np[i];
    twist_bottom[i][2] = twist_bottom_pn[i];
    twist_bottom[i][3] = twist_bottom_pp[i];

    // Move values into ratio_top
    ratio_top[i][0] = ratio_top_nn[i];
    ratio_top[i][1] = ratio_top_np[i];
    ratio_top[i][2] = ratio_top_pn[i];
    ratio_top[i][3] = ratio_top_pp[i];

    // Move values into ratio_bottom
    ratio_bottom[i][0] = ratio_bottom_nn[i];
    ratio_bottom[i][1] = ratio_bottom_np[i];
    ratio_bottom[i][2] = ratio_bottom_pn[i];
    ratio_bottom[i][3] = ratio_bottom_pp[i];

  }

  small_radius_of_bend = file.getDouble(serstring, "small bend radius");
  large_radius_of_bend = file.getDouble(serstring, "large bend radius");
  small_angle_of_twist = file.getDouble(serstring, "small twist angle");
  large_angle_of_twist = file.getDouble(serstring, "large twist angle");
  file.getInteger(serstring, "y bend up top", num_region, y_bend_up_top);
  file.getInteger(serstring, "y bend down top", num_region, y_bend_down_top);
  file.getInteger(serstring, "y bend up bottom", num_region, y_bend_up_bottom);
  file.getInteger(serstring, "y bend down bottom", num_region, y_bend_down_bottom);
  file.getInteger(serstring, "y twist ccw top", num_region, y_twist_ccw_top);
  file.getInteger(serstring, "y twist cw top", num_region, y_twist_cw_top);
  file.getInteger(serstring, "y twist ccw bottom", num_region, y_twist_ccw_bottom);
  file.getInteger(serstring, "y twist cw bottom", num_region, y_twist_cw_bottom);
  file.getDouble(serstring, "hibend_up", num_region,hibend_up);
  file.getDouble(serstring, "hibend_down", num_region,hibend_down);

  file.getInteger(serstring, "helical_pose", num_sensors, helical_pose);
  helix_radius = file.getDouble(serstring, "helix_radius");

  findKValues();
/*
  file.getInteger(serstring, "flat", num_sensors, flat_data);
  file.getInteger(serstring, "positive bend", num_sensors, bend_up);
  file.getInteger(serstring, "negative bend", num_sensors, bend_down);
  file.getInteger(serstring, "positive twist", num_sensors, twist_ccw);
  file.getInteger(serstring, "negative twist", num_sensors, twist_cw);
  file.getInteger(serstring, "offset", num_sensors, offset);
  radius_of_bend = file.getDouble(serstring, "bend radius");
  angle_of_twist = file.getDouble(serstring, "twist angle");
*/
}

/*
Name:    loadFile
Purpose:  reads in the calibration coefficients from a calibration file.
Accepts:  cal_file = name of the calibration file.
Returns:    void
*/
calibrate::loadFile(char* cal_file)
{
  filedata file(cal_file);

  double bend_top_nn[24],
       bend_top_np[24],
       bend_top_pn[24],
       bend_top_pp[24],
       bend_bottom_nn[24],
       bend_bottom_np[24],
       bend_bottom_pn[24],
       bend_bottom_pp[24];
  double twist_top_nn[24],
       twist_top_np[24],
       twist_top_pn[24],
       twist_top_pp[24],
       twist_bottom_nn[24],
       twist_bottom_np[24],
       twist_bottom_pn[24],
       twist_bottom_pp[24];
  double ratio_top_nn[24],
       ratio_top_np[24],
       ratio_top_pn[24],
       ratio_top_pp[24],
       ratio_bottom_nn[24],
       ratio_bottom_np[24],
       ratio_bottom_pn[24],
       ratio_bottom_pp[24];

  file.getInteger("[calibration]", "flat", num_sensors, flat_data);
  for (int i=0;i<num_sensors;i++) offset[i]=flat_data[i];
  file.getDouble("[calibration]", "bend top nn", num_region, bend_top_nn);
  file.getDouble("[calibration]", "bend top np", num_region, bend_top_np);
  file.getDouble("[calibration]", "bend top pn", num_region, bend_top_pn);
  file.getDouble("[calibration]", "bend top pp", num_region, bend_top_pp);
  file.getDouble("[calibration]", "bend bottom nn", num_region, bend_bottom_nn);
  file.getDouble("[calibration]", "bend bottom np", num_region, bend_bottom_np);
  file.getDouble("[calibration]", "bend bottom pn", num_region, bend_bottom_pn);
  file.getDouble("[calibration]", "bend bottom pp", num_region, bend_bottom_pp);
  file.getDouble("[calibration]", "twist top nn", num_region, twist_top_nn);
  file.getDouble("[calibration]", "twist top np", num_region, twist_top_np);
  file.getDouble("[calibration]", "twist top pn", num_region, twist_top_pn);
  file.getDouble("[calibration]", "twist top pp", num_region, twist_top_pp);
  file.getDouble("[calibration]", "twist bottom nn", num_region, twist_bottom_nn);
  file.getDouble("[calibration]", "twist bottom np", num_region, twist_bottom_np);
  file.getDouble("[calibration]", "twist bottom pn", num_region, twist_bottom_pn);
  file.getDouble("[calibration]", "twist bottom pp", num_region, twist_bottom_pp);
  file.getDouble("[calibration]", "ratio top nn", num_region, ratio_top_nn);
  file.getDouble("[calibration]", "ratio top np", num_region, ratio_top_np);
  file.getDouble("[calibration]", "ratio top pn", num_region, ratio_top_pn);
  file.getDouble("[calibration]", "ratio top pp", num_region, ratio_top_pp);
  file.getDouble("[calibration]", "ratio bottom nn", num_region, ratio_bottom_nn);
  file.getDouble("[calibration]", "ratio bottom np", num_region, ratio_bottom_np);
  file.getDouble("[calibration]", "ratio bottom pn", num_region, ratio_bottom_pn);
  file.getDouble("[calibration]", "ratio bottom pp", num_region, ratio_bottom_pp);
  tiny_radius_of_bend = file.getDouble("[calibration]","tiny bend radius");
  if (tiny_radius_of_bend==0) {
    //use the default value
    tiny_radius_of_bend=60;
  }
  small_radius_of_bend = file.getDouble("[calibration]", "small bend radius");
  if (small_radius_of_bend==0) {
    //use default
    small_radius_of_bend=112.5;
  }
  large_radius_of_bend = file.getDouble("[calibration]", "large bend radius");
  if (large_radius_of_bend==0) {
    //use default
    large_radius_of_bend=297.5;
  }
  small_angle_of_twist = file.getDouble("[calibration]", "small twist angle");
  if (small_angle_of_twist==0) {
    //use default, depending on tape length
    if (tape_length > 1500) small_angle_of_twist=360;
    else small_angle_of_twist=180;
  }
  large_angle_of_twist = file.getDouble("[calibration]", "large twist angle");
  if (large_angle_of_twist==0) {
    //use default, depending on tape length
    if (tape_length>1500) large_angle_of_twist=720;
    else large_angle_of_twist=360;
  }

  file.getInteger("[calibration]", "y bend up top", num_region, y_bend_up_top);
  file.getInteger("[calibration]", "y bend down top", num_region, y_bend_down_top);
  file.getInteger("[calibration]", "y bend up bottom", num_region, y_bend_up_bottom);
  file.getInteger("[calibration]", "y bend down bottom", num_region, y_bend_down_bottom);
  file.getInteger("[calibration]", "y twist ccw top", num_region, y_twist_ccw_top);
  file.getInteger("[calibration]", "y twist cw top", num_region, y_twist_cw_top);
  file.getInteger("[calibration]", "y twist ccw bottom", num_region, y_twist_ccw_bottom);
  file.getInteger("[calibration]", "y twist cw bottom", num_region, y_twist_cw_bottom);
  file.getDouble("[calibration]", "hibend_up", num_region,hibend_up);
  file.getDouble("[calibration]", "hibend_down",num_region,hibend_down);
  file.getDouble("[calibration]", "tempfactor",num_sensors,tempfactor);

  file.getInteger("[calibration]", "helical_pose", num_sensors, helical_pose);
  helix_radius = file.getDouble("[calibration]", "helix_radius");

  for (i = 0; i < num_sensors / 2; i++)
  {
    // Move values into bend_top
    bend_top[i][0] = bend_top_nn[i];
    bend_top[i][1] = bend_top_np[i];
    bend_top[i][2] = bend_top_pn[i];
    bend_top[i][3] = bend_top_pp[i];

    // Move values into bend_bottom
    bend_bottom[i][0] = bend_bottom_nn[i];
    bend_bottom[i][1] = bend_bottom_np[i];
    bend_bottom[i][2] = bend_bottom_pn[i];
    bend_bottom[i][3] = bend_bottom_pp[i];

    // Move values into twist_top
    twist_top[i][0] = twist_top_nn[i];
    twist_top[i][1] = twist_top_np[i];
    twist_top[i][2] = twist_top_pn[i];
    twist_top[i][3] = twist_top_pp[i];

    // Move values into twist_bottom
    twist_bottom[i][0] = twist_bottom_nn[i];
    twist_bottom[i][1] = twist_bottom_np[i];
    twist_bottom[i][2] = twist_bottom_pn[i];
    twist_bottom[i][3] = twist_bottom_pp[i];

    // Move values into ratio_top
    ratio_top[i][0] = ratio_top_nn[i];
    ratio_top[i][1] = ratio_top_np[i];
    ratio_top[i][2] = ratio_top_pn[i];
    ratio_top[i][3] = ratio_top_pp[i];

    // Move values into ratio_bottom
    ratio_bottom[i][0] = ratio_bottom_nn[i];
    ratio_bottom[i][1] = ratio_bottom_np[i];
    ratio_bottom[i][2] = ratio_bottom_pn[i];
    ratio_bottom[i][3] = ratio_bottom_pp[i];
  }

  findKValues();

/*
  file.getInteger("[calibration]", "flat", num_sensors, flat_data);
  file.getInteger("[calibration]", "positive bend", num_sensors, bend_up);
  file.getInteger("[calibration]", "negative bend", num_sensors, bend_down);
  file.getInteger("[calibration]", "positive twist", num_sensors, twist_ccw);
  file.getInteger("[calibration]", "negative twist", num_sensors, twist_cw);
  file.getInteger("[calibration]", "offset", num_sensors, offset);
  radius_of_bend = file.getDouble("[calibration]", "bend radius");
  angle_of_twist = file.getDouble("[calibration]", "twist angle");
*/
  //delete file;
}

/*
Name:    saveFile
Purpose:  Saves the calculated calibration coefficients to file.
Accepts:  cal_file = name of the calibration file.
Returns:    void
*/
calibrate::saveFile(char *cal_file)
{
  filedata file(cal_file);

  double *bend_top_nn = new double[num_region];
  double *bend_top_np = new double[num_region];
  double *bend_top_pn = new double[num_region];
  double *bend_top_pp = new double[num_region];
  double *bend_bottom_nn = new double[num_region];
  double *bend_bottom_np = new double[num_region];
  double *bend_bottom_pn = new double[num_region];
  double *bend_bottom_pp = new double[num_region];
  double *twist_top_nn = new double[num_region];
  double *twist_top_np = new double[num_region];
  double *twist_top_pn = new double[num_region];
  double *twist_top_pp = new double[num_region];
  double *twist_bottom_nn = new double[num_region];
  double *twist_bottom_np = new double[num_region];
  double *twist_bottom_pn = new double[num_region];
  double *twist_bottom_pp = new double[num_region];
  double *ratio_top_nn = new double[num_region];
  double *ratio_top_np = new double[num_region];
  double *ratio_top_pn = new double[num_region];
  double *ratio_top_pp = new double[num_region];
  double *ratio_bottom_nn = new double[num_region];
  double *ratio_bottom_np = new double[num_region];
  double *ratio_bottom_pn = new double[num_region];
  double *ratio_bottom_pp = new double[num_region];

  for (int i = 0; i < num_sensors / 2; i++)
  {
    // Move values into bend_top
    bend_top_nn[i] = bend_top[i][0];
    bend_top_np[i] = bend_top[i][1];
    bend_top_pn[i] = bend_top[i][2];
    bend_top_pp[i] = bend_top[i][3];

    // Move values into bend_bottom
    bend_bottom_nn[i] = bend_bottom[i][0];
    bend_bottom_np[i] = bend_bottom[i][1];
    bend_bottom_pn[i] = bend_bottom[i][2];
    bend_bottom_pp[i] = bend_bottom[i][3];

    // Move values into twist_top
    twist_top_nn[i] = twist_top[i][0];
    twist_top_np[i] = twist_top[i][1];
    twist_top_pn[i] = twist_top[i][2];
    twist_top_pp[i] = twist_top[i][3];

    // Move values into twist_bottom
    twist_bottom_nn[i] = twist_bottom[i][0];
    twist_bottom_np[i] = twist_bottom[i][1];
    twist_bottom_pn[i] = twist_bottom[i][2];
    twist_bottom_pp[i] = twist_bottom[i][3];

    // Move values into ratio_top
    ratio_top_nn[i] = ratio_top[i][0];
    ratio_top_np[i] = ratio_top[i][1];
    ratio_top_pn[i] = ratio_top[i][2];
    ratio_top_pp[i] = ratio_top[i][3];

    // Move values into ratio_bottom
    ratio_bottom_nn[i] = ratio_bottom[i][0];
    ratio_bottom_np[i] = ratio_bottom[i][1];
    ratio_bottom_pn[i] = ratio_bottom[i][2];
    ratio_bottom_pp[i] = ratio_bottom[i][3];
  }

  file.writeData("[calibration]", "flat", num_sensors, flat_data);
  file.writeData("[calibration]", "bend top nn", num_region, bend_top_nn);
  file.writeData("[calibration]", "bend top np", num_region, bend_top_np);
  file.writeData("[calibration]", "bend top pn", num_region, bend_top_pn);
  file.writeData("[calibration]", "bend top pp", num_region, bend_top_pp);
  file.writeData("[calibration]", "bend bottom nn", num_region, bend_bottom_nn);
  file.writeData("[calibration]", "bend bottom np", num_region, bend_bottom_np);
  file.writeData("[calibration]", "bend bottom pn", num_region, bend_bottom_pn);
  file.writeData("[calibration]", "bend bottom pp", num_region, bend_bottom_pp);
  file.writeData("[calibration]", "twist top nn", num_region, twist_top_nn);
  file.writeData("[calibration]", "twist top np", num_region, twist_top_np);
  file.writeData("[calibration]", "twist top pn", num_region, twist_top_pn);
  file.writeData("[calibration]", "twist top pp", num_region, twist_top_pp);
  file.writeData("[calibration]", "twist bottom nn", num_region, twist_bottom_nn);
  file.writeData("[calibration]", "twist bottom np", num_region, twist_bottom_np);
  file.writeData("[calibration]", "twist bottom pn", num_region, twist_bottom_pn);
  file.writeData("[calibration]", "twist bottom pp", num_region, twist_bottom_pp);
  file.writeData("[calibration]", "ratio top nn", num_region, ratio_top_nn);
  file.writeData("[calibration]", "ratio top np", num_region, ratio_top_np);
  file.writeData("[calibration]", "ratio top pn", num_region, ratio_top_pn);
  file.writeData("[calibration]", "ratio top pp", num_region, ratio_top_pp);
  file.writeData("[calibration]", "ratio bottom nn", num_region, ratio_bottom_nn);
  file.writeData("[calibration]", "ratio bottom np", num_region, ratio_bottom_np);
  file.writeData("[calibration]", "ratio bottom pn", num_region, ratio_bottom_pn);
  file.writeData("[calibration]", "ratio bottom pp", num_region, ratio_bottom_pp);
  file.writeData("[calibration]", "tiny bend radius",tiny_radius_of_bend);
  file.writeData("[calibration]", "small bend radius",small_radius_of_bend);
  file.writeData("[calibration]", "large bend radius",large_radius_of_bend);
  file.writeData("[calibration]", "small twist angle",small_angle_of_twist);
  file.writeData("[calibration]", "large twist angle",large_angle_of_twist);
  file.writeData("[calibration]", "y bend up top", num_region ,y_bend_up_top);
  file.writeData("[calibration]", "y bend down top", num_region ,y_bend_down_top);
  file.writeData("[calibration]", "y bend up bottom", num_region ,y_bend_up_bottom);
  file.writeData("[calibration]", "y bend down bottom", num_region ,y_bend_down_bottom);
  file.writeData("[calibration]", "y twist ccw top", num_region ,y_twist_ccw_top);
  file.writeData("[calibration]", "y twist cw top", num_region ,y_twist_cw_top);
  file.writeData("[calibration]", "y twist ccw bottom", num_region ,y_twist_ccw_bottom);
  file.writeData("[calibration]", "y twist cw bottom", num_region ,y_twist_cw_bottom);
  file.writeData("[calibration]", "hibend_up",num_region,hibend_up);
  file.writeData("[calibration]", "hibend_down",num_region,hibend_down);

  file.writeData("[calibration]", "helical_pose", num_sensors, helical_pose);
  file.writeData("[calibration]", "helix_radius", helix_radius);
  //file.writeData("[calibration]", "tempfactor",num_sensors,tempfactor);

  delete [] bend_top_nn;
  delete [] bend_top_np;
  delete [] bend_top_pn;
  delete [] bend_top_pp;
  delete [] bend_bottom_nn;
  delete [] bend_bottom_np;
  delete [] bend_bottom_pn;
  delete [] bend_bottom_pp;
  delete [] twist_top_nn;
  delete [] twist_top_np;
  delete [] twist_top_pn;
  delete [] twist_top_pp;
  delete [] twist_bottom_nn;
  delete [] twist_bottom_np;
  delete [] twist_bottom_pn;
  delete [] twist_bottom_pp;
  delete [] ratio_top_nn;
  delete [] ratio_top_np;
  delete [] ratio_top_pn;
  delete [] ratio_top_pp;
  delete [] ratio_bottom_nn;
  delete [] ratio_bottom_np;
  delete [] ratio_bottom_pn;
  delete [] ratio_bottom_pp;

  findKValues();

/*
  file.writeData("[calibration]", "flat", num_sensors, flat_data);
  file.writeData("[calibration]", "positive bend", num_sensors, bend_up);
  file.writeData("[calibration]", "negative bend", num_sensors, bend_down);
  file.writeData("[calibration]", "positive twist", num_sensors, twist_ccw);
  file.writeData("[calibration]", "negative twist", num_sensors, twist_cw);
  file.writeData("[calibration]", "offset", num_sensors, offset);
  file.writeData("[calibration]", "bend radius", radius_of_bend);
  file.writeData("[calibration]", "twist angle", angle_of_twist);
*/
  //delete file;
}

/* Loads an old format calibration file. */
/* obsolete
calibrate::loadOldFile(char* old_cal_file)
{
  filedata file(old_cal_file);

  file.getInteger("[calibration]", "flat", num_sensors, flat_data);
  file.getInteger("[calibration]", "bend up", num_sensors, bend_up);
  file.getInteger("[calibration]", "bend down", num_sensors, bend_down);
  file.getInteger("[calibration]", "twist ccw", num_sensors, twist_ccw);
  file.getInteger("[calibration]", "twist cw", num_sensors, twist_cw);
  file.getInteger("[calibration]", "offset", num_sensors, offset);
  radius_of_bend = file.getDouble("[calibration]", "bend radius");
  angle_of_twist = file.getDouble("[calibration]", "twist angle");
}
*/

/*
Name:    getPureData
Purpose:  Subtracts flat from the tape data set, essentially applying an offset to the signal.
Accepts:  data = raw tape data set.
Returns:    void
*/
calibrate::getPureData(int data[])
{
  // Collect a set of raw data and subtract the offset from it.
  pollTape(data);
  subtractFlat(data,data);
}

/*
Name:    getPureData
Purpose:  Subtracts flat from the tape data set, essentially applying an offset to the signal.
      New version which also gets a time stamp and frame number for the raw data.
Accepts:  data  = raw tape data set.
      frame = the frame number of the current frame.
      time  = the time stamp of the current frame.
Returns:    void
*/
calibrate::getPureData(int data[], int frame, unsigned int time)
{
  pollTape(data, frame, time);
  subtractFlat(data, data);
}

/*
Name:    setPoseCurvature
Purpose:  Sets the curvature parameters for the pure bend and twist poses.
Accepts:  tiny_bend_radius  = the radius used in the huge bend calibration poses.
      small_bend_radius = the radius used in the large bend calibration poses.
      large_bend_radius = the radius used in the small bend calibration poses.
      small_twist_angle = small twist angle for twist calibration poses.
      large_twist_angle = large twist angle for twist calibration poses.
Returns:    void
*/
calibrate::setPoseCurvature(double tiny_bend_radius, double small_bend_radius, double large_bend_radius,
              double small_twist_angle, double large_twist_angle)
{
  tiny_radius_of_bend = tiny_bend_radius;
  small_radius_of_bend = small_bend_radius;
  large_radius_of_bend = large_bend_radius;
  small_angle_of_twist = small_twist_angle;
  large_angle_of_twist = large_twist_angle;
}

/*
Name:    setHelixRadius
Purpose:  Sets the radius of the helical pose.
Accepts:  helix_pose_radius = radius of the cylinder that the tape is wrapped around. 
Returns:    void
*/
calibrate::setHelixRadius(double helix_pose_radius)
{
  helix_radius = helix_pose_radius;
}

/*
Name:    getPoseCurvature
Purpose:  Gets the curvature parameters for the pure bend and twist poses.
Accepts:  tiny_bend_radius  = the radius used in the huge bend calibration poses.
      small_bend_radius = the radius used in the large bend calibration poses.
      large_bend_radius = the radius used in the small bend calibration poses.
      small_twist_angle = small twist angle for twist calibration poses.
      large_twist_angle = large twist angle for twist calibration poses.
Returns:    void
*/
calibrate::getPoseCurvature(double &tiny_bend_radius, double &small_bend_radius, double &large_bend_radius,
              double &small_twist_angle, double &large_twist_angle)
{
  tiny_bend_radius = tiny_radius_of_bend;
  small_bend_radius = small_radius_of_bend;
  large_bend_radius = large_radius_of_bend;
  small_twist_angle = small_angle_of_twist;
  large_twist_angle = large_angle_of_twist;
}

/*
Name:    getHelixRadius
Purpose:  Gets the radius of the helical pose.
Accepts:  helix_pose_radius = radius of the cylinder that the tape is wrapped around. 
Returns:    void
*/
calibrate::getHelixRadius(double &helix_pose_radius)
{
  helix_pose_radius = helix_radius;
}

/*
Name:    average
Purpose:  Averages five raw tape data sets into one.  Not in use.
Accepts:  data     = array of five data sets.
      ave_data = averaged data set. 
Returns:    void
*/
calibrate::average(int data[][5], int ave_data[])
{
  // Recieves an array of five sets of tape data 
  //int cumulative_data[48];
  int *cumulative_data = new int[num_sensors];

  for (int i = 0;i < num_sensors;i++)
  {
    cumulative_data[i] = 0;
    ave_data[i] = 0;
    for (int j = 0;j < 5;j++)
    {
      cumulative_data[i] = cumulative_data[i] + data[i][j];
    }
    ave_data[i] = cumulative_data[i] / 5;
  }

  delete cumulative_data;
}

/*
Name:    collectSamples
Purpose:  Collects five raw tape data sets.  Not in use.
Accepts:  data = array of five data sets.
Returns:    void
*/
calibrate::collectSamples(int data[][5])
{
  // Collects five sets of tape data and stores them in the data variable.
  int *single_poll = new int[num_sensors];

  pollTape(single_poll);// moved this
  for (int i = 0;i < 5;i++)
  {
    
    for(int j = 0;j < num_sensors;j++)
    {
      data[j][i] = single_poll[j];
    }
  }

  delete single_poll;
}

/*
Name:    subtractOffsets
Purpose:  Subtracts the offset values from a raw tape data set.  The offset variable is 
            used to implement special shape.
Accepts:  data      = raw tape data set.
      pure_data = data set with offsets subtracted from them. 
Returns:    void
*/
calibrate::subtractOffsets(int data[], int pure_data[])
{
  // Subtract offsets from the bend/twist signals to 
  // get pure bend/twist signals.
  
  for (int i = 0; i < num_sensors; i++)
  {
    pure_data[i] = data[i] - offset[i];
  }
}

/*
Name:    UpdateOffsets (1st instance)
Purpose:  Updates the current offset values to the new values passed to this function.  The offset 
      variable is used to implement special shape.
Accepts:  n[5]=0.8037;
    twist_correction[6]=0.8212;
    twist_correction[7]=0.8386;
    twist_cor                 member variables.
Returns:    void
*/
void calibrate::UpdateOffsets(int newoffsets[])
{
  for (int i = 0; i < num_sensors; i++)
  {
    offset[i] = newoffsets[i];
    flat_data[i] = newoffsets[i];
  }
}

/*
Name:    UpdateOffsets (2nd instance)
Purpose:  Updates the current offset values based on the change in offset of the 1st sensor.
      Acts as a temperature compensation method.
Accepts:  first_sensor: the raw data for the first sensor
      tapetype: integer from 0 to 4 describing how the tape is worn or -1 if not worn
      cutoff: sensor index beyond which the tape is in close contact with the wearer
Returns:    void
*/
void calibrate::UpdateOffsets(int first_sensor,int tapetype,int cutoff)
{
  const int RIGHTARM=0,LEFTARM=1,RIGHTLEG=2,LEFTLEG=3,HEAD=4;
  for (int i = 0; i < num_sensors; i++)
  {
    if ((tapetype==RIGHTARM)||(tapetype==LEFTARM)) { 
      //arm tape
      if (i>cutoff) //double the temperature effect for points beyond cutoff, 
              //i.e. double the effect for all points in close contact with wearer
        flat_data[i] = (int)(2*tempfactor[i]*(first_sensor+flat_data[0]-offset[0]))+offset[i];
      else
        flat_data[i] = (int)(tempfactor[i]*(first_sensor+flat_data[0]-offset[0]))+offset[i];
    }
    else {
      flat_data[i] = (int)(tempfactor[i]*(first_sensor+flat_data[0]-offset[0]))+offset[i];
    }
  }
}

/*
Name:    getoffsets
Purpose:  Retreives the stored values for the offset variable.  The offset variable is 
            used to implement special shape.
Accepts:  offsets = array of integer to store the offset raw tape data set. 
Returns:    void
*/
void calibrate::getoffsets(int *offsets)
{
  for (int i = 0; i < num_sensors; i++)
    offsets[i]=this->offset[i];
}

/*
Name:    subtractFlat
Purpose:  Subtracts the flat values from a raw tape data set. 
Accepts:  data      = raw tape data set.
      pure_data = data set with flat values subtracted from them. 
Returns:    void
*/
calibrate::subtractFlat(int data[], int pure_data[])
{
  // Subtract flat from the bend/twist signals to 
  // get pure bend/twist signals.
  
  for (int i = 0; i < num_sensors; i++)
  {
    pure_data[i] = data[i] - flat_data[i];
  }
}

/*
Name:    subtract
Purpose:  Subtracts the second raw tape data set from the first and stores them in result.
Accepts:  first     = raw tape data set.
      second    = raw tape data set.
      pure_data = first - second raw tape data set. 
Returns:    void
*/
calibrate::subtract(int first[], int second[], int result[])
{
  // Subtract the second array from the first and store
  // it in the result array.

  for (int i = 0; i < num_sensors; i++)
  {
    result[i] = first[i] - second[i];
  }
}

/*
Name:    findDeterminate
Purpose:  Finds the determinate of the array [[a11,a12],[a21,a22]].
Accepts:  a11 = array value.
      a12 = array value.
      a21 = array value.
      a22 = array value.
Returns:    The determinate of the array.
*/
double calibrate::findDeterminate(double a11, double a12, double a21, double a22)
{
  return a11*a22-a12*a21;
}

/*
Name:    solve
Purpose:  Solves a set of two equations, two unknowns using Cramer's rule.  The two equations are:
      a[0] * x + b[0] * y = c[0] and a[1] * x + b[1] * y = c[1]
Accepts:  a = array of a values. (see equations above)
      b = array of b values.
      c = array of c values.
      x = x value.
      y = y value.
Returns:    void
*/
calibrate::solve(double a[2], double b[2], double c[2], double &x, double &y)
{
  // Solves a linear system with two equations, two unknowns using Cramer's rule.
  x = findDeterminate(c[0],b[0],c[1],b[1])/
    findDeterminate(a[0],b[0],a[1],b[1]);
  y = findDeterminate(a[0],c[0],a[1],c[1])/
    findDeterminate(a[0],b[0],a[1],b[1]);
}

/*
Name:    findYIntercept
Purpose:  Finds the y intercept of the counts vs. curvature function for a given function. 
Accepts:  delta_curvature = the change in curvature (bend or twist) between two poses.
      delta_counts    = the change in voltage between the same two poses.
      small_counts    = a counts value that falls on the line.
      small_curvature = a curvature value that falls on the line.
Returns:    The y intercept.
*/
double calibrate::findYIntercept(double delta_curvature, int delta_counts, int small_counts, double small_curvature)
{
  //small_curvature actually corresponds to a small radius of curvature, i.e. the larger bend.
  double slope = delta_counts / delta_curvature;
  return (small_counts - (slope * small_curvature));
}

/*
Name:    calculateSpiral
Purpose:  Calculate the radius for each sensor region along the tape when it is wrapped in multiple 
      layers on a calibration fixture.  
Accepts:  calib_radius   = the radius of the calibration fixture.
      tape_thickness = the thickness of the tape.
      gap_distance   = the distance the gap makes along the circumference (guess) when it transitions between layers.
Returns:    an array filled with the radii of each sensor region..
*/
double *calibrate::calculateSpiral(double calib_radius, double tape_thickness, double gap_distance)
{
  // Find the various parameters needed later.
  double current_s; // length along space curve.
  double s[10]; // array of s values. 10 is magic number.
  int turns; // the number of revolutions around the manderal.
  double rev_radius[10]; // radius of the current revolution.
  double circumference; // circumference of the current revolution.
  double *radius = new double [tape_length];
  double *effective_radius = new double [num_sensors / 2];
  double *inverted_radius = new double [num_sensors /2];
  double new_calib_radius;
  double gap_rise;
//  double sensor_length;
  int current_distance;
  double sum;
  
  current_s = 0.0; // start at the base of the tape.
  turns = 0; 
  s[turns] = 0.0;
  rev_radius[turns] = 0.0;
  circumference = 0.0;
  current_distance = 0;
  new_calib_radius = calib_radius + (0.5 * tape_thickness);
//  sensor_length = tape_length / (num_sensors / 2); // assumes even sensor spacing and pairs.

  while (current_s < tape_length) // in mm.
  {
    turns ++;
    rev_radius[turns] = new_calib_radius + (turns * tape_thickness); 
    circumference = 2 * pi * rev_radius[turns];
    current_s = current_s + circumference;
    s[turns] = current_s;  
  }
  
  // Figure out the tape bend profile in mm.
  turns = 1;
  for (int mm = 0; mm < int(tape_length); mm++)
  {
    if (mm > s[turns])
    { // if the current value is greater than the turn limit then increment.
      turns ++;
    }
    if (mm <= (s[turns] - gap_distance))
    { // if the current value is in a turn and not in the gap...
      radius[mm] = new_calib_radius + (turns-1) * tape_thickness; 
    }
    if (mm > (s[turns] - gap_distance) && mm <= s[turns])
    { // if the current value is in a gap.height = (mm - (smm(turn)-inairmm)) * (thickmm / inairmm);
      gap_rise = (mm - (s[turns] - gap_distance)) * (tape_thickness / gap_distance);
      radius[mm] = new_calib_radius + (turns-1) * tape_thickness + gap_rise;
    }
  }  

  // Map the curvatures onto the sensor regions.
  for (int region = num_region - 1; region >= 0; region--)
  {
    sum = 0;
    for (mm = 0; mm < region_length[region]; mm++)
    {
      sum = sum + radius[(int)current_distance + mm]; // adds up the radii for a given pair.
    }
    inverted_radius[region] = sum / region_length[region];
    current_distance = current_distance + region_length[region];
  }
  
  // Invert the radius vector because the above calculations are from the tip instead of the base.
  for (int x = num_region; x > 0; x--)
  {
    effective_radius[num_region - x] = inverted_radius[x-1];
  }

  // clean up.
  delete [] radius;
  delete [] inverted_radius;

  return effective_radius;
}

/*
Name:    linearTwistCorrection
Purpose:  This function calculates corrects the twist values based on the amount of taper that 
      a tape has.  It is only used during calibration.
Accepts:  twist        = an array of twist values.
      twist_amount = the total amount of twist places on the entire tape during a calibration pose.
      textfile     = Settings file for this tape.  Usually *.mst
Returns:    void
*/
void calibrate::linearTwistCorrection(double twist[], double twist_amount, char *textfile)
{
  //convert twist_amount to radians
  double twist_radians=twist_amount*3.14159265359/180;
  //average twist per sensor.
  double avg_twist = twist_radians/(num_region);
  
  double twist_correction[24]; //values which scale the curvature to match what happens in real life.
  
  // set the corection equal to one in case there is none to be had.
  for (int i = 0;i < 24;i++)
    twist_correction[i] = 1.0;

  //get settings to determine taper type
  filedata settings(textfile);
  
  //check to see if taper information is stored in text file
  int nTaperavailable = settings.getDouble("[taper]","available");

  if (nTaperavailable>0)
    settings.getDouble("[taper]","data",num_region,twist_correction);
   else if (tape_length>1800 && tape_length< 2000) //assume some default values for helical arm tape
   {
   /*
   This is the new coeffs changed by the wire stiffeners added Aug 4,2001.
   Not implamented yet...
   Columns 1 through 7 
   0.7165    0.7339    0.7514    0.7688    0.7863    0.8037    0.8212
   Columns 8 through 14 
   0.8386    0.8561    0.9033    1.0264    1.1506    1.2747    1.3988
   Columns 15 through 16 
     1.5229    1.6470*/
     
     /*Columns 1 through 7 
     0.7885    0.7885    0.7885    0.7885    0.7885    0.7885    0.7885
     Columns 8 through 14 
     0.7885    0.7897    0.9045    1.0269    1.1493    1.2717    1.3941
     Columns 15 through 16 
     1.5165    1.6390*/
     
     twist_correction[0]=0.7165;
     twist_correction[1]=0.7339;
     twist_correction[2]=0.7514;
     twist_correction[3]=0.7688;
     twist_correction[4]=0.7863;
     twist_correction[5]=0.8037;
     twist_correction[6]=0.8212;
     twist_correction[7]=0.8386;
     twist_correction[8]=0.8561;
     twist_correction[9]=0.9033;
     twist_correction[10]=1.0264;
     twist_correction[11]=1.1506;
     twist_correction[12]=1.2747;
     twist_correction[13]=1.3988;
     twist_correction[14]=1.5229;
     twist_correction[15]=1.6470;
   }
  else if (tape_length > 940 && tape_length < 980)  
    // assume tape is regular 96 cm tape (32 sensors) without taper values in
    // settings file
  {
    /*Columns 1 through 7 
    0.6618    0.7069    0.7520    0.7971    0.8422    0.8873    0.9324
    Columns 8 through 14 
    0.9775    1.0225    1.0676    1.1127    1.1578    1.2029    1.2480
    Columns 15 through 16 
    1.2931    1.3382
    */

    twist_correction[0]=0.6618;
    for (i = 1;i < 16; i++)
      twist_correction[i] = twist_correction[i-1] + 0.0451;
  }

  for (i=0;i<num_region;i++)
  {
    twist[i] = avg_twist * twist_correction[i];
  }
}

/*
Name:    linearTwistCorrection
Purpose:  This function calculates corrects the twist values based on the amount of taper that 
      a tape has.  It is only used during calibration.
Accepts:  twist        = an array of twist values.
      twist_amount = the total amount of twist places on the entire tape during a calibration pose.
Returns:    void
*/
void calibrate::linearTwistCorrection(double twist[], double twist_amount)
{
  //convert twist_amount to radians
  double twist_radians=twist_amount*3.14159265359/180;
  //average twist per sensor.
  double avg_twist = twist_radians/(num_region);
  
  double twist_correction[24]; //values which scale the curvature to match what happens in real life.
  
  // set the corection equal to one in case there is none to be had.
  for (int i = 0;i < 24;i++)
    twist_correction[i] = 1.0;

  if (tape_length>1800 && tape_length< 2000) //assume some default values for helical arm tape
  {
    /*
    This is the new coeffs changed by the wire stiffeners added Aug 4,2001.
    Not implamented yet...
    Columns 1 through 7 
    0.7165    0.7339    0.7514    0.7688    0.7863    0.8037    0.8212
    Columns 8 through 14 
    0.8386    0.8561    0.9033    1.0264    1.1506    1.2747    1.3988
    Columns 15 through 16 
      1.5229    1.6470*/
 
      /*Columns 1 through 7 
    0.7885    0.7885    0.7885    0.7885    0.7885    0.7885    0.7885
    Columns 8 through 14 
    0.7885    0.7897    0.9045    1.0269    1.1493    1.2717    1.3941
    Columns 15 through 16 
    1.5165    1.6390*/
 
    twist_correction[0]=0.7165;
    twist_correction[1]=0.7339;
    twist_correction[2]=0.7514;
    twist_correction[3]=0.7688;
    twist_correction[4]=0.7863;
    twist_correction[5]=0.8037;
    twist_correction[6]=0.8212;
    twist_correction[7]=0.8386;
    twist_correction[8]=0.8561;
    twist_correction[9]=0.9033;
    twist_correction[10]=1.0264;
    twist_correction[11]=1.1506;
    twist_correction[12]=1.2747;
    twist_correction[13]=1.3988;
    twist_correction[14]=1.5229;
    twist_correction[15]=1.6470;
  }
  else if (tape_length > 940 && tape_length < 980)  
    // assume tape is regular 96 cm tape (32 sensors) without taper values in
    // settings file
  {
    /*Columns 1 through 7 
    0.6618    0.7069    0.7520    0.7971    0.8422    0.8873    0.9324
    Columns 8 through 14 
    0.9775    1.0225    1.0676    1.1127    1.1578    1.2029    1.2480
    Columns 15 through 16 
    1.2931    1.3382
    */

    twist_correction[0]=0.6618;
    for (i = 1;i < 16; i++)
      twist_correction[i] = twist_correction[i-1] + 0.0451;
  }

  for (i=0;i<num_region;i++)
  {
    twist[i] = avg_twist * twist_correction[i];
  }
}

/*
Name:    FindHugeBend
Purpose:  This function is used during calibration to calculate bend correction coefficents for high bends.
Accepts:  V            = an array of pure bend values (corresponding to the raw data that was collected 
               during the tiny circle pose.  All offsets should be subtracted before passing 
               this data.
      hugebend     = an array of bend data which is calculated from the pure_bend inputs.
Returns:    void
*/
void calibrate::FindHugeBend(int V[], double *hugebend)
{
  int voltage_top, voltage_bottom;
  double bend_edge_q0, bend_edge_q1, bend_edge_q2, bend_edge_q3;
  double bend_multiplier;
  double temp_bend, temp_twist, bend;
  double filter;


  for (int i = 0; i < num_region; i++)
  {
    voltage_top = V[2*i];
    voltage_bottom = V[2*i+1];
    
    ///////////Calculate bends and twists at the edges of the zero crossing zone
    bend_multiplier = 1.2;    
    //The mlt constant sets the bend criteria  higher than
    //the curvature corresponding to -count_intercept. This is because the
    //extension of the straight line through the cal points crosses the bend axis
    //at zero counts, and we know the knee occurs at a bend corresponding to larger
    //than zero counts.

    bend_edge_q0 = bend_top[i][0]*(double)y_bend_down_top[i]+bend_bottom[i][0]*(double)y_bend_down_bottom[i];
    bend_edge_q1 = bend_top[i][1]*(double)y_bend_down_top[i]+bend_bottom[i][1]*(double)y_bend_down_bottom[i];
    bend_edge_q2 = bend_top[i][2]*(double)y_bend_up_top[i]+bend_bottom[i][2]*(double)y_bend_up_bottom[i];
    bend_edge_q3 = bend_top[i][3]*(double)y_bend_up_top[i]+bend_bottom[i][3]*(double)y_bend_up_bottom[i];
    
    //Apply the mltb correction:
    bend_edge_q0 *= bend_multiplier;
    bend_edge_q1 *= bend_multiplier;
    bend_edge_q2 *= bend_multiplier;
    bend_edge_q3 *= bend_multiplier;
    
    //**************************APPLY INITIAL CAL CONSTANTS
    //This first application uses bend_top and twist_top from the first quadrant.
    //The cal constants bend_top and twist_top come from the calibration file opened near the beginning.
    temp_bend = bend_top[i][0]*voltage_top + bend_bottom[i][0]*voltage_bottom;  
    temp_twist = twist_top[i][0]*voltage_top + twist_bottom[i][0]*voltage_bottom;
    
    //APPLY ALL CAL CONSTANTS
    //see notes above for first quadrant
    //first quadrant results from above are used to detect signs of bend and twist
    //for this next stage. Depending on signs, calibrations are refined, using
    //cal constants for each of the FOUR quadrants:
    //NOTE that twist is positive here for a cw right hand screw facing from base to tip
    
    // Negative bend, negative twist.
    if ((temp_bend<0)&&(temp_twist<0))
    {
      bend = bend_top[i][0]*voltage_top + bend_bottom[i][0]*voltage_bottom;     
      if (bend<=bend_edge_q0)
        bend+=bend_edge_q0;
      else if (bend>bend_edge_q0)
        if (bend_edge_q0!=0) bend=fabs(bend/bend_edge_q0)*bend;
    }
    // Negative bend, positive twist.
    else if ((temp_bend<0)&&(temp_twist>=0))
    {
      bend = bend_top[i][1]*voltage_top + bend_bottom[i][1]*voltage_bottom;     
      if (bend<=bend_edge_q1)
        bend+=bend_edge_q1;
      else if (bend>bend_edge_q1)
        if (bend_edge_q1!=0) bend=fabs(bend/bend_edge_q1)*bend;
    }
    // Positive bend, negative twist.
    else if ((temp_bend>=0)&&(temp_twist<0))
    {
      bend = bend_top[i][2]*voltage_top + bend_bottom[i][2]*voltage_bottom;     
      if (bend>=bend_edge_q2)
        bend+=bend_edge_q2;
      else if (bend<bend_edge_q2)
      {
        if (bend_edge_q2!=0) bend=fabs(bend/bend_edge_q2)*bend;
      }
    }
    // Positive bend, positive twist.
    else if ((temp_bend>=0)&&(temp_twist>=0))
    {
      bend = bend_top[i][3]*voltage_top + bend_bottom[i][3]*voltage_bottom;     
      if (bend>=bend_edge_q3)
        bend+=bend_edge_q3;
      else if (bend<bend_edge_q3)
      {
        //For bends inside the zc zone, make them continuous with the edge value but
        //diminish their importance as bend approaches zero:
        if (bend_edge_q3!=0) bend=fabs(bend/bend_edge_q3)*bend;
      }
    }
      
      //Tail off the very small bends, continuous with filter edge:
      filter = .001*tape_length/(num_region);  //a bend of 1 m radius on any tape
      if (fabs(bend)<filter)
    {
      bend=bend*(pow(bend/filter,2));
      }
    hugebend[i] = bend;  
  }
}

/*
    //record raw data for test purposes
  FILE *testFile;
  testFile = fopen("test.txt","wt");
  char caldata[1024];
  sprintf(caldata,"pure_bend_up_tiny_top: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_up_tiny[0],pure_bend_up_tiny[2],
    pure_bend_up_tiny[4],pure_bend_up_tiny[6],pure_bend_up_tiny[8],pure_bend_up_tiny[10],pure_bend_up_tiny[12],
    pure_bend_up_tiny[14],pure_bend_up_tiny[16],pure_bend_up_tiny[18],pure_bend_up_tiny[20],pure_bend_up_tiny[22],
    pure_bend_up_tiny[24],pure_bend_up_tiny[26],pure_bend_up_tiny[28],pure_bend_up_tiny[30]);
  
  fwrite(caldata,1,strlen(caldata),testFile);
  sprintf(caldata,"pure_bend_up_small_top: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_up_small[0],pure_bend_up_small[2],
    pure_bend_up_small[4],pure_bend_up_small[6],pure_bend_up_small[8],pure_bend_up_small[10],pure_bend_up_small[12],
    pure_bend_up_small[14],pure_bend_up_small[16],pure_bend_up_small[18],pure_bend_up_small[20],pure_bend_up_small[22],
    pure_bend_up_small[24],pure_bend_up_small[26],pure_bend_up_small[28],pure_bend_up_small[30]);
  
  fwrite(caldata,1,strlen(caldata),testFile);

  sprintf(caldata,"pure_bend_up_large_top: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_up_large[0],pure_bend_up_large[2],
    pure_bend_up_large[4],pure_bend_up_large[6],pure_bend_up_large[8],pure_bend_up_large[10],pure_bend_up_large[12],
    pure_bend_up_large[14],pure_bend_up_large[16],pure_bend_up_large[18],pure_bend_up_large[20],pure_bend_up_large[22],
    pure_bend_up_large[24],pure_bend_up_large[26],pure_bend_up_large[28],pure_bend_up_large[30]);
  
  fwrite(caldata,1,strlen(caldata),testFile);

  sprintf(caldata,"pure_bend_down_tiny_top: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_down_tiny[0],pure_bend_down_tiny[2],
    pure_bend_down_tiny[4],pure_bend_down_tiny[6],pure_bend_down_tiny[8],pure_bend_down_tiny[10],pure_bend_down_tiny[12],
    pure_bend_down_tiny[14],pure_bend_down_tiny[16],pure_bend_down_tiny[18],pure_bend_down_tiny[20],pure_bend_down_tiny[22],
    pure_bend_down_tiny[24],pure_bend_down_tiny[26],pure_bend_down_tiny[28],pure_bend_down_tiny[30]);
  
  fwrite(caldata,1,strlen(caldata),testFile);
  sprintf(caldata,"pure_bend_down_small_top: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_down_small[0],pure_bend_down_small[2],
    pure_bend_down_small[4],pure_bend_down_small[6],pure_bend_down_small[8],pure_bend_down_small[10],pure_bend_down_small[12],
    pure_bend_down_small[14],pure_bend_down_small[16],pure_bend_down_small[18],pure_bend_down_small[20],pure_bend_down_small[22],
    pure_bend_down_small[24],pure_bend_down_small[26],pure_bend_down_small[28],pure_bend_down_small[30]);
  
  fwrite(caldata,1,strlen(caldata),testFile);

  sprintf(caldata,"pure_bend_down_large_top: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_down_large[0],pure_bend_down_large[2],
    pure_bend_down_large[4],pure_bend_down_large[6],pure_bend_down_large[8],pure_bend_down_large[10],pure_bend_down_large[12],
    pure_bend_down_large[14],pure_bend_down_large[16],pure_bend_down_large[18],pure_bend_down_large[20],pure_bend_down_large[22],
    pure_bend_down_large[24],pure_bend_down_large[26],pure_bend_down_large[28],pure_bend_down_large[30]);
  
  fwrite(caldata,1,strlen(caldata),testFile);

  sprintf(caldata,"pure_bend_up_tiny_bottom: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_up_tiny[1],pure_bend_up_tiny[3],
    pure_bend_up_tiny[5],pure_bend_up_tiny[7],pure_bend_up_tiny[9],pure_bend_up_tiny[11],pure_bend_up_tiny[13],
    pure_bend_up_tiny[15],pure_bend_up_tiny[17],pure_bend_up_tiny[19],pure_bend_up_tiny[21],pure_bend_up_tiny[23],
    pure_bend_up_tiny[25],pure_bend_up_tiny[27],pure_bend_up_tiny[29],pure_bend_up_tiny[31]);
  
  fwrite(caldata,1,strlen(caldata),testFile);
  sprintf(caldata,"pure_bend_up_small_bottom: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_up_small[1],pure_bend_up_small[3],
    pure_bend_up_small[5],pure_bend_up_small[7],pure_bend_up_small[9],pure_bend_up_small[11],pure_bend_up_small[13],
    pure_bend_up_small[15],pure_bend_up_small[17],pure_bend_up_small[19],pure_bend_up_small[21],pure_bend_up_small[23],
    pure_bend_up_small[25],pure_bend_up_small[27],pure_bend_up_small[29],pure_bend_up_small[31]);
  
  fwrite(caldata,1,strlen(caldata),testFile);

  sprintf(caldata,"pure_bend_up_large_bottom: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_up_large[1],pure_bend_up_large[3],
    pure_bend_up_large[5],pure_bend_up_large[7],pure_bend_up_large[9],pure_bend_up_large[11],pure_bend_up_large[13],
    pure_bend_up_large[15],pure_bend_up_large[17],pure_bend_up_large[19],pure_bend_up_large[21],pure_bend_up_large[23],
    pure_bend_up_large[25],pure_bend_up_large[27],pure_bend_up_large[29],pure_bend_up_large[31]);
  
  fwrite(caldata,1,strlen(caldata),testFile);

  sprintf(caldata,"pure_bend_down_tiny_bottom: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_down_tiny[1],pure_bend_down_tiny[3],
    pure_bend_down_tiny[5],pure_bend_down_tiny[7],pure_bend_down_tiny[9],pure_bend_down_tiny[11],pure_bend_down_tiny[13],
    pure_bend_down_tiny[15],pure_bend_down_tiny[17],pure_bend_down_tiny[19],pure_bend_down_tiny[21],pure_bend_down_tiny[23],
    pure_bend_down_tiny[25],pure_bend_down_tiny[27],pure_bend_down_tiny[29],pure_bend_down_tiny[31]);
  
  fwrite(caldata,1,strlen(caldata),testFile);
  sprintf(caldata,"pure_bend_down_small_bottom: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_down_small[1],pure_bend_down_small[3],
    pure_bend_down_small[5],pure_bend_down_small[7],pure_bend_down_small[9],pure_bend_down_small[11],pure_bend_down_small[13],
    pure_bend_down_small[15],pure_bend_down_small[17],pure_bend_down_small[19],pure_bend_down_small[21],pure_bend_down_small[23],
    pure_bend_down_small[25],pure_bend_down_small[27],pure_bend_down_small[29],pure_bend_down_small[31]);
  
  fwrite(caldata,1,strlen(caldata),testFile);

  sprintf(caldata,"pure_bend_down_large_bottom: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",pure_bend_down_large[1],pure_bend_down_large[3],
    pure_bend_down_large[5],pure_bend_down_large[7],pure_bend_down_large[9],pure_bend_down_large[11],pure_bend_down_large[13],
    pure_bend_down_large[15],pure_bend_down_large[17],pure_bend_down_large[19],pure_bend_down_large[21],pure_bend_down_large[23],
    pure_bend_down_large[25],pure_bend_down_large[27],pure_bend_down_large[29],pure_bend_down_large[31]);
  
  fwrite(caldata,1,strlen(caldata),testFile);

  fclose(testFile);
  */

/*
Name:    initializePoseData
Purpose:  This function sets all the pose starage variables initially to zero.
Accepts:  void
Returns:    void
*/
calibrate::initializePoseData()
{
  for (int i = 0; i < num_sensors; i++)
  {
    flat_data[i] = 0;
    bend_up_large[i] = 0;
    bend_up_small[i] = 0;
    bend_up_tiny[i] = 0;
    bend_down_large[i] = 0;
    bend_down_small[i] = 0;
    bend_down_tiny[i] = 0;

    twist_cw_large[i] = 0;
    twist_cw_small[i] = 0;
    twist_ccw_large[i] = 0;
    twist_ccw_small[i] = 0;
    offset[i] = 0;
    helical_pose[i] = 0;
    twist_offset[i]=0;
    tempfactor[i] = 0;
  }
}

/*
Name:    findKValues
Purpose:  This function finds the k_values and phase_angles for each sensorized region.
Accepts:  void
Returns:    void
*/
void calibrate::findKValues()
{
  double a[2], b[2], c[2];
  double* calib_radius;
  double* large_bend;
  double* small_bend;
  double* large_twist;
  double* small_twist;
  double* sensor_bend;
  double* sensor_twist;
  double* bend_up;
  double* bend_down;
  double* twist_ccw;
  double* twist_cw;
  double radius;
  int V1, V2;
  double build_angle;
  double C1, C2;
  double M1, M2;
  double factor, theta;

  large_bend = new double[num_region +1];
  small_bend = new double[num_region +1];
  large_twist = new double[num_region +1];
  small_twist = new double[num_region +1];

  sensor_bend = new double[num_region +1];
  sensor_twist = new double[num_region +1];

  bend_up = new double[num_sensors+1];
  bend_down = new double[num_sensors+1];
  twist_ccw = new double[num_sensors+1];
  twist_cw = new double[num_sensors+1];

  build_angle = 45;

  // figure out the bend on each sensor region.
  calib_radius = calculateSpiral(small_radius_of_bend, 1.5, 80);

  for (int x = 0; x < num_region; x++)
  {
    large_bend[x] = region_length[x]/calib_radius[x];
  }

  delete [] calib_radius;
  calib_radius = calculateSpiral(large_radius_of_bend, 1.5, 80);
  for (x = 0; x < num_region; x++)
  {
    small_bend[x] = region_length[x]/calib_radius[x];
  }
  
  delete [] calib_radius;
  
  for (x = 0; x < num_region; x++)
  {
    sensor_bend[x] = large_bend[x] - small_bend[x];
  }

  // figure out the twist on each sensor pair.
  for (x = 0; x < num_region; x++)
  {
    large_twist[x] = 0.0;
    small_twist[x] = 0.0;
  }
  linearTwistCorrection(large_twist, large_angle_of_twist);
  linearTwistCorrection(small_twist, small_angle_of_twist);

  for (x = 0; x < num_region; x++)
  {
    sensor_twist[x] = large_twist[x] - small_twist[x];
  }

  for (int i = 0; i < num_region; i++)
  {
    // Operate on the first quadrant to get the pure bend down data.
    // Negative bend, negative twist.
    a[0] = bend_top[i][0];
    a[1] = twist_top[i][0];
    b[0] = bend_bottom[i][0];
    b[1] = twist_bottom[i][0];
    c[0] = -sensor_bend[i];
    c[1] = 0;//-sensor_twist[i];
    // Solve for bend coefficients.
    solve(a,b,c,bend_down[2*i],bend_down[2*i+1]);

    // Operate on the third quadrant to get the pure bend up data.
    // positive bend, negative twist.
    a[0] = bend_top[i][3];
    a[1] = twist_top[i][3];
    b[0] = bend_bottom[i][3];
    b[1] = twist_bottom[i][3];
    c[0] = sensor_bend[i];
    c[1] = 0;//sensor_twist[i];
    // Solve for bend coefficients.
    solve(a,b,c,bend_up[2*i],bend_up[2*i+1]);  
  
    radius = region_length[i] / sensor_bend[i];
    
    k1_down[i] = 1.414 * fabs(bend_down[2*i]) * radius;
    k1_up[i] = -1.414 * fabs(bend_up[2*i]) * radius;
    k2_down[i] = -1.414 * fabs(bend_down[2*i+1]) * radius;
    k2_up[i] = 1.414 * fabs(bend_up[2*i+1]) * radius;
    
    if (helical_pose[2*i] != 0)
    {
      V1 = helical_pose[2*i];
      V2 = helical_pose[2*i+1];
      
      // Make sure factor calculation does not divid by zero.
      if (V1 == 0)
        V1 = 1;
      if (V2 == 0)
        V2 = 1;
      
      // Calculate the bend and twist using the new helical model.
      if (V1 >= V2)
      {
        factor = -(V2 * fabs(k1_down[i]))/(V1 * fabs(k2_down[i]));
      }
      else if (V1 < V2)
      {
        factor = -(V2 * fabs(k1_up[i]))/(V1 * fabs(k2_up[i]));
      }
      
      // Calculate the bend and twist coefficients.
      C1 = cos(build_angle*pi/180); // Bend coefficients.
      C2 = cos(build_angle*pi/180);
      M1 = cos(2.0 * (pi/4 - build_angle*pi/180)); // Twist coefficients.
      M2 = -cos(2.0 * (pi/4 - build_angle*pi/180));
      
      // Calculate theta.
      if (fabs(M2 - factor * M1) > 0.01)
        theta = atan((C1 * factor - C2) / (M2 - factor * M1)); 
      else 
        theta = atan((C1 * factor - C2) / 0.01);
      
      // Iterate through the various phase correction angles until the best one is found.
      // Start with a phase angle of 0.1 and increment until 0.4 radians or until the best 
      // value for radius is found for the helical pose.
      double temp_phase_angle = 0.0;
      double this_error, error_limit;
      error_limit = 0.1;
      double previous_error;
      
      // Do the first radius calculation to get the iteration process.
      // Calculate radius.
      //phase_angle = 0.2; // magic number
      if (((V1>=V2 && V1>0) || theta>1.1) && theta>=0) // Bend down and theta is positive. 
      {
        radius = -(k1_down[i]/V1) * (C1*pow(cos(theta-temp_phase_angle),2)+
          (M1*cos(theta)*sin(theta))); 
      }
      else if ((V1>=V2 && V2<0) && theta<=0) // Bend down and theta is negative.
      {
        radius = -(k2_down[i]/V2) * (C2*pow(cos(theta+temp_phase_angle),2)+
          (M2*cos(theta)*sin(theta)));
      }
      else if (((V1<V2 && V2>0) || theta<-1.1) && theta<=0) // Bend up and theta is negative. 
      {
        radius = (k2_up[i]/V2) * (C2*pow(cos(theta+temp_phase_angle),2)+
          (M2*cos(theta)*sin(theta)));
      }
      else if ((V1<V2 && V1<0) && theta>=0) // Bend up and theta is positive.
      {
        radius = (k1_up[i]/V1) * (C1*pow(cos(theta-temp_phase_angle),2)+
          (M1*cos(theta)*sin(theta)));
      }
      
      previous_error = 2147483647; //big number - max. long
      this_error = fabs(fabs(radius) - fabs(helix_radius));
      
      while (this_error <= previous_error && temp_phase_angle <= 0.78)
      {
        
        previous_error = this_error;
        temp_phase_angle += 0.01;
        
        // Calculate radius.
        //phase_angle = 0.2; // magic number
        if (((V1>=V2 && V1>0) || theta>1.1) && theta>=0) // Bend down and theta is positive. 
        {
          radius = -(k1_down[i]/V1) * (C1*pow(cos(theta-temp_phase_angle),2)+
            (M1*cos(theta)*sin(theta))); 
        }
        else if ((V1>=V2 && V2<=0) && theta<=0) // Bend down and theta is negative.
        {
          radius = -(k2_down[i]/V2) * (C2*pow(cos(theta+temp_phase_angle),2)+
            (M2*cos(theta)*sin(theta)));
        }
        else if (((V1<V2 && V2>0) || theta<-1.1) && theta<=0) // Bend up and theta is negative. 
        {
          radius = (k2_up[i]/V2) * (C2*pow(cos(theta+temp_phase_angle),2)+
            (M2*cos(theta)*sin(theta)));
        }
        else if ((V1<V2 && V1<0) && theta>=0) // Bend up and theta is positive.
        {
          radius = (k1_up[i]/V1) * (C1*pow(cos(theta-temp_phase_angle),2)+
            (M1*cos(theta)*sin(theta)));
        }
        
        this_error = fabs(fabs(radius) - fabs(helix_radius));  
      }
      //phase_angle[i] = 0.5;
      
      phase_angle[i] = temp_phase_angle;
    }
    else
    {
      phase_angle[i] = 0.25;
    }
  }

  filedata file2("phase.txt");
  file2.writeData("[test]", "phase", num_region, phase_angle);
  /*
  // Negative bend, positive twist.
  a[0] = bend_down[2*i];
  a[1] = twist_ccw[2*i];
  b[0] = bend_down[2*i+1];
  b[1] = twist_ccw[2*i+1];
  c[0] = -sensor_bend[i];
  c[1] = 0;
  // Solve for bend coefficients.
  solve(a,b,c,bend_top[i][1],bend_bottom[i][1]);
  c[0] = 0;
  c[1] = sensor_twist[i];
  // Solve for twist coefficients.
  solve(a,b,c,twist_top[i][1],twist_bottom[i][1]);
  // Calculate the twist to bend ratios.
  ratio_top[i][1] = twist_top[i][1]/bend_top[i][1];
  ratio_bottom[i][1] = twist_bottom[i][1]/bend_bottom[i][1];
        
        
  %Operate in first quadrant to get pure bent up data:
  
  DEN1=(sbend.*stwist)./(T2(:,:,1).*B1(:,:,1)-T1(:,:,1).*B2(:,:,1));
  %v11 is v bent top, up in this quadrant
  v11u=(T2(:,:,1)./stwist).*DEN1;
  v12u=-(T1(:,:,1)./stwist).*DEN1;
  
  %Operate in third quadrant to get pure bent down data:
  DEN3=(-sbend.*stwist)./(T2(:,:,3).*B1(:,:,3)-T1(:,:,3).*B2(:,:,3));
  %v11 is v bent top, down in this quadrant
  v11d=(T2(:,:,3)./stwist).*DEN3;
  v12d=-(T1(:,:,3)./stwist).*DEN3;
    
  nnd=length(v11d);
  x=1:nnd;
      
  bdowntop=v11d;
  bdownbot=v12d;
  buptop=v11u;
  bupbot=v12u;
    
        
  if (max(lengthvec)==1 & min(lengthvec)==1)  %Equal sensor lengths 
    radvec=10*(lengthcm/end_active)./sbend; %vector of radii along tape, in mm
  else
    radvec=10*lengthvec./sbend;
  end
  k1d=+1.414*abs(v11d).*radvec;  
  k1u=-1.414*abs(v11u).*radvec;
  k2d=-1.414*abs(v12d).*radvec;
  k2u=+1.414*abs(v12u).*radvec;
  */

  delete [] small_bend;
  delete [] large_bend;
  delete [] small_twist;
  delete [] large_twist;
  delete [] sensor_bend;
  delete [] sensor_twist;
  delete [] bend_up;
  delete [] bend_down;
  delete [] twist_ccw;
  delete [] twist_cw;
}

double * calibrate::getTempfactor()
{
  return tempfactor;
}

/*
Name:  setTwistOffset
Purpose: Set the twist offset required to counteract the effects of temperature on the sensor readings.
   This offset is usually calculated by subtracting the twist curvatures of two like poses to
   determine the 
Accepts: filename = the name of the calibration file.
Returns:    void
*/
void calibrate::setTwistOffset(double twist[24])
{
  for (int i = 0;i < num_region;i++)
  {
    twist_offset[i] = twist[i]; 
  }
}
