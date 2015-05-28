#include "stdafx.h"
//#include "windows.h"
#include "tapeAPI.h"
#include "quaternion.h"
#include "filedata.h"
#include "shapeAPI_error.h"

const double pi = 3.14159;

/*
Name:    tapeAPI
Purpose:  constructor - this constructor assumes that the tape has even spacing and is 
      a bend/twist paired tape.
Accepts:  num_sense = number of sensors on the tape.
      buad_rate = serial speed of tape - 57600 or 115200.
      cal_file  = calibration file name.
      length    = the overall length of the tape.
      interp_interval = the number of interpolation steps between sensor boundaries.
Returns:    void
*/
tapeAPI::tapeAPI(int num_sense, int baud_rate, 
           char* cal_file, double length, int interp_interval) // Constructor.
      :calibrate(num_sense, baud_rate, length)
{
  interval = interp_interval;
  loadFile(cal_file);
  Xo = 0; // Set the base orientation and position of the tape 
  Yo = 0; // to 0,0,0 with no rotation.
  Zo = 0;
  Uo[0] = 1;
  Uo[1] = 0;
  Uo[2] = 0;
  Bo[0] = 0;
  Bo[1] = 0;
  Bo[2] = 1;
}

/*
Name:    tapeAPI
Purpose:  constructor
Accepts:  num_sense = number of sensors on the tape.
      buad_rate = serial speed of tape - 57600 or 115200.
      cal_file  = calibration file name.
      length    = the overall length of the tape.
      interp_interval = the number of interpolation steps between sensor boundaries.
      bend_only = boolean value, one for bend only tapes and zero for bend/twist tapes.
      num_regions = the number of sensor regions on the tape.
      region_length = array of sensor region lengths.  Allows for variable sensor lengths.
Returns:    void
*/
tapeAPI::tapeAPI(int num_sense, int baud_rate, char* cal_file, double length, 
         int interp_interval, bool only_bend, int num_regions, double region_length[48])
         :calibrate(num_sense, baud_rate, length, only_bend, num_regions, region_length)
{
  interval = interp_interval;
  loadFile(cal_file);
  Xo = 0; // Set the base orientation and position of the tape 
  Yo = 0; // to 0,0,0 with no rotation.
  Zo = 0;
  Uo[0] = 1;
  Uo[1] = 0;
  Uo[2] = 0;
  Bo[0] = 0;
  Bo[1] = 0;
  Bo[2] = 1;

}

/*
Name:    tapeAPI
Purpose:  constructor
Accepts:  settings_file = file name for the tape. Contains all the variables of the
      above constructor in the filedata file format.
Returns:    void
*/
tapeAPI::tapeAPI(char* settings_file, char *rootfolder)
      :calibrate(settings_file)
{
  SetCurrentDirectory(rootfolder);
  filedata file(settings_file);
  interval = file.getInteger("[settings]","interp interval");
  file.getString("[settings]","calibration file", &cal_file_name);
  // Error checking.  Checks to ensure all the variables in the mst file are reasonable.
  checkMSTFile(settings_file);

  SetCurrentDirectory(rootfolder);
  loadFile(cal_file_name);

  //assign base position for tape
  Xo = file.getDouble("[settings]","base x"); 
  Yo = file.getDouble("[settings]","base y"); 
  Zo = file.getDouble("[settings]","base z"); 

  //assign starting orientation
  double roll = file.getDouble("[settings]","roll");
  double pitch = file.getDouble("[settings]","pitch");
  double yaw = file.getDouble("[settings]","yaw");
  setOrientation(yaw,pitch,roll);
}

/*
Name:    ~tapeAPI
Purpose:  destructor
Accepts:  void
Returns:    void
*/
tapeAPI::~tapeAPI()
{
  if (cal_file_name) delete []cal_file_name;
}

/*
Name:    setPosition
Purpose:  Sets the base position of the tape.
Accepts:  x = x position.
      y = y position.
      z = z position.
Returns:    void
*/
tapeAPI::setPosition(double x, double y, double z)
{
  // Set the base position of the tape.
  Xo = x;
  Yo = y;
  Zo = z;
}

/*
Name:    setOrientation
Purpose:  Sets the base orientation of the tape.
Accepts:  baseU = base U vector.
      baseB = base B vector.
Returns:    void
*/
tapeAPI::setOrientation(double baseU[3], double baseB[3])
{
  // Set the base orientation of the tape.
  normalize(baseU); // Make sure the passed vectors are unit length.
  normalize(baseB);
  for(int i = 0;i < 3;i++)
  {
    Uo[i] = baseU[i];
    Bo[i] = baseB[i];
  }
}

/*
Name:    setOrientation
Purpose:  Sets the base orientation of the tape.
Accepts:  yaw   = yaw about y - these are in the yaw, pitch, roll ordering.
      pitch = pitch about z.
      roll  = roll about x.
Returns:    void
*/
tapeAPI::setOrientation(double yaw, double pitch, double roll)
{
  // Set the base orientation of the tape.
  double U[3] = {1,0,0}; 
  double N[3] = {0,1,0};
  double B[3] = {0,0,1}; // Temporary orientation vectors. 

     yaw = yaw * (pi/180);
  rotline(U,N,yaw);
  rotline(B,N,yaw);

  pitch = pitch * (pi/180);
  rotline(U,B,pitch);
  rotline(N,B,pitch);

  roll = roll * (pi/180);
  rotline(N,U,roll);
  rotline(B,U,roll);

  normalize(U); // Make sure the passed vectors are unit length.
  normalize(B);
  
  for(int i = 0;i < 3;i++)
  {
    Uo[i] = U[i];
    Bo[i] = B[i];
  }
}

/*
Name:    getTapeSettings
Purpose:  Gets a sub set of the available settings so that a tape can be flattened, drawn etc.
Accepts:  cal_file = calibration file name.
      num_sense = the number of sensors on the tape.
      interp_interv = the interpolation interval.
      length = the length of the sensorized region on the tape in mm.
      num_regions = the number of sensorized region on the tape (= num_sense/2 for bend/twist tapes).
      region_length = the length of each region (in mm).
Returns:    void
*/
tapeAPI::getTapeSettings(char *cal_file, int &num_sense, int &interp_interv, double &length, int &num_regions, double region_lengths[]) 
{
  strcpy(cal_file,cal_file_name);
  num_sense = num_sensors;
  interp_interv = interval;
  length = tape_length;
  num_regions = num_region;
  for (int i = 0;i < 48;i++)
    region_lengths[i] = region_length[i]; 
}

/*
Name:    checkMSTFile
Purpose:  Performs some error checking on the mst settings file.
Accepts:  settings_file = the *.mst settings file. 
Returns:    void
*/
void tapeAPI::checkMSTFile(char *settings_file)
{
  // Create temp variables for the mst file values;
  char *temp_cal_file;
  char *temp_com_port;
  int temp_serial_number;
  int temp_direct_read;
  int temp_interv;
  int temp_num_regions;
  int temp_bend_only;
  double temp_tape_length;
  int temp_num_sensors;
  int temp_baud_rate;
  double temp_region_length[48];
  double region_length_sum = 0.0;

  // Initialize the temp_region_length array.
  for (int i = 0;i < 48; i++)
    temp_region_length[i] = 0.0;

  // Read in the settings values from the file.
  filedata file(settings_file);
  file.getString("[settings]","calibration file", &temp_cal_file);
  file.getString("[settings]","com port", &temp_com_port);
  temp_serial_number = file.getInteger("[settings]","serial number");
  temp_direct_read = file.getInteger("[settings]","direct read");
  temp_interv = file.getInteger("[settings]","interp interval");
  temp_num_regions = file.getInteger("[settings]", "num regions");
  temp_bend_only = file.getInteger("[settings]", "bend only");
  temp_tape_length = file.getInteger("[settings]","length (mm)");
  temp_num_sensors = file.getInteger("[settings]","number of sensors");
  temp_baud_rate = file.getInteger("[settings]","baud rate");
  file.getDouble("[settings]","region length",48,temp_region_length);

  // Test each variable and setLastError if there is a problem.

  // Test the calibration file...

  // Test the com port setting...

  // Test the serial number.
  // Make sure it is a positive 16 bit number. (0 - 65535)
  if (temp_serial_number < 0 || temp_serial_number > 65535)
    SetLastError(INVALID_SERIAL_NUMBER);

  // Test the direct read variable.
  // Make sure that it is a boolean value. (0 or 1)
  if (temp_direct_read != 0 && temp_direct_read != 1)
    SetLastError(INVALID_BOOLEAN_VALUE);

  // Test the interpolation interval value.
  // Make sure that it is greater than zero.
  if (temp_interv <= 0)
    SetLastError(NEGATIVE_INTERPOLATION_VALUE);

  // Test the num_regions variable.
  // Ensure that it is between 0 and num_sensors.
  if (temp_num_regions < 0 || temp_num_regions > temp_num_sensors)
    SetLastError(INVALID_NUM_REGIONS_VARIABLE);

  // Test the bend only variable.
  // Make sure that it is a boolean value. (0 or 1)
  if (temp_bend_only != 0 && temp_bend_only != 1)
    SetLastError(INVALID_BOOLEAN_VALUE);

  // Test the tape length variable.
  // Make sure it is positive.
  if (temp_tape_length <= 0)
    SetLastError(NEGATIVE_TAPE_LENGTH);

  // Test the num sensors variable.
  // Make sure it is greater than zero and less than 48.
  if (temp_num_sensors <= 0 || temp_num_sensors > 48)
    SetLastError(INVALID_NUMBER_OF_SENSORS);

  // Test the buad rate.
  // The baud rate gets tested in the serial class.

  // Test the region lengths.
  // Make sure that each region length is not less then zero length and
  // the sum of the region lengths is not greater then the tape length.
  for (i = 0; i < temp_num_regions ; i++)
  {
    region_length_sum += temp_region_length[i];
    if (temp_region_length[i] < 0)
      SetLastError(INVALID_REGION_LENGTH);
  }
  if (region_length_sum > temp_tape_length)
    SetLastError(INVALID_REGION_LENGTH);
}

/*
Name:    getCurvatureData
Purpose:  Uses the current raw tape values and the calibration coefficients 
      to calculate the bend and twist for each sensorized region.
Accepts:  bend  = empty array of bend values to be filled (length of array = num_region).
      twist = empty array of twist values to be filled.
Returns:    success variable 1 if true 0 if false.
*/
bool tapeAPI::getCurvatureData(double bend[], double twist[])
{
  int* V = new int[num_sensors+1];
  int voltage_top, voltage_bottom;
  double bend_multiplier;
  double filter, start_hibend, full_hibend;
  double bend_edge_q0, bend_edge_q1, bend_edge_q2, bend_edge_q3;
  double temp_bend, temp_twist;
  double factor;

  int V1, V2;
  double C1, C2, M1, M2;  
  double theta, radius;//, bend_correction;

  getPureData(V); // Grab the raw voltages from the tape w/ offsets subtracted.
  for (int i = 0; i < num_region; i++)
  {
    filter=.001*region_length[i];  //a bend of 1 m radius on any tape
    // Start Pairs
    if (num_region == num_sensors / 2)
    {
      voltage_top = V[2*i];
      voltage_bottom = V[2*i+1];
      
      ///////////Calculate bends at the edges of the zero crossing zone
      bend_multiplier=1.2;    
      //The multiplier constant sets the bend and twist criteria  higher than
      //the curvature corresponding to -count_intercept. This is because 
      //(for example for bend), the
      //extension of the straight line through the cal points crosses the bend axis
      //at zero counts, and we know the knee occurs at a bend corresponding to larger
      //than zero counts.
      
      bend_edge_q0=bend_top[i][0]*(double)y_bend_down_top[i]+bend_bottom[i][0]*(double)y_bend_down_bottom[i];
      bend_edge_q1=bend_top[i][1]*(double)y_bend_down_top[i]+bend_bottom[i][1]*(double)y_bend_down_bottom[i];
      bend_edge_q2=bend_top[i][2]*(double)y_bend_up_top[i]+bend_bottom[i][2]*(double)y_bend_up_bottom[i];
      bend_edge_q3=bend_top[i][3]*(double)y_bend_up_top[i]+bend_bottom[i][3]*(double)y_bend_up_bottom[i];
      
      //Apply the bend_multiplier.
      bend_edge_q0*=bend_multiplier;
      bend_edge_q1*=bend_multiplier;
      bend_edge_q2*=bend_multiplier;
      bend_edge_q3*=bend_multiplier;

      //make sure abs(bend_edge) is <=0.9 * filter
      if (fabs(bend_edge_q0)>(0.9*filter)) 
        bend_edge_q0 *= 0.9*filter/fabs(bend_edge_q0);
      if (fabs(bend_edge_q1)>(0.9*filter)) 
        bend_edge_q1 *= 0.9*filter/fabs(bend_edge_q1);
      if (fabs(bend_edge_q2)>(0.9*filter)) 
        bend_edge_q2 *= 0.9*filter/fabs(bend_edge_q2);
      if (fabs(bend_edge_q3)>(0.9*filter)) 
        bend_edge_q3 *= 0.9*filter/fabs(bend_edge_q3);

      if (bend_only == true)
      {
        temp_bend = bend_top[i][0]*voltage_top + bend_bottom[i][0]*voltage_bottom;
          if (temp_bend<0)
        {
          bend[i] = bend_top[i][0]*voltage_top + bend_bottom[i][0]*voltage_bottom;
          //if (bend[i]<=bend_edge_q0)
          // bend[i]+=bend_edge_q0;
          //else if (bend[i]>bend_edge_q0)
          // if (bend_edge_q0!=0) bend[i]=fabs(bend[i]/bend_edge_q0)*bend[i];
          if (bend[i]<=-fabs(bend_edge_q0))
          bend[i]+=bend_edge_q0;
          else 
          {
            if (bend_edge_q0!=0) 
            bend[i]+=bend_edge_q0*fabs(bend[i]/bend_edge_q0);
          }
        }
        else if (temp_bend>=0)
        {
          bend[i] = bend_top[i][2]*voltage_top + bend_bottom[i][2]*voltage_bottom;
          //if (bend[i]>=bend_edge_q2)
          // bend[i]+=bend_edge_q2;
          //else if (bend[i]<bend_edge_q2)
          // if (bend_edge_q2!=0) bend[i]=fabs(bend[i]/bend_edge_q2)*bend[i];
          if (bend[i]>=fabs(bend_edge_q2))
          bend[i]+=bend_edge_q2;
          else 
          {
            if (bend_edge_q2!=0) 
            bend[i]+=bend_edge_q2*fabs(bend[i]/bend_edge_q2);
          }
        }
          twist[i] = 0.0;
      }
       else
      {
        if  (new_way)
        {
          V1 = V[2*i];
          V2 = V[2*i+1];

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
           
          // Calculate radius.
          //phase_angle = 0.2; // magic number
          if (((V1>=V2 && V1>0) || theta>1.1) && theta>=0) // Bend down and theta is positive. 
          {
            radius = -(k1_down[i]/V1) * (C1*pow(cos(theta-phase_angle[i]),2)+
              (M1*cos(theta)*sin(theta))); 
          }
          else if ((V1>=V2 && V2<=0) && theta<=0) // Bend down and theta is negative.
          {
            radius = -(k2_down[i]/V2) * (C2*pow(cos(theta+phase_angle[i]),2)+
              (M2*cos(theta)*sin(theta)));
          }
          else if (((V1<V2 && V2>0) || theta<-1.1) && theta<=0) // Bend up and theta is negative. 
          {
            radius = (k2_up[i]/V2) * (C2*pow(cos(theta+phase_angle[i]),2)+
              (M2*cos(theta)*sin(theta)));
          }
          else if ((V1<V2 && V1<0) && theta>=0) // Bend up and theta is positive.
          {
            radius = (k1_up[i]/V1) * (C1*pow(cos(theta-phase_angle[i]),2)+
              (M1*cos(theta)*sin(theta)));
          }

          // Calculate bend and twist values from theta and radius.
          bend[i] = region_length[i] * (1/radius) * pow(cos(theta),2);

          if (fabs(theta) < (85 * pi/180))
            twist[i] = region_length[i] * (1/radius) * cos(theta) * sin(theta);
          else // calculate pure twist the old way.
          {
            /*  
            if rdata_top(i)>=0 & rdata_bot(i)>=0 %clockwise, or NEG twist
              twista(i)=T1(:,i,2).*rdata_top(i)+T2(:,i,2)*rdata_bot(i);
            else
              twista(i)=T1(:,i,1).*rdata_top(i)+T2(:,i,1)*rdata_bot(i);
            end
            */
            if (V1 >= 0 && V2 >= 0)
              twist[i] = twist_top[i][0]*V1 + twist_bottom[i][0]*V2;
            else
              twist[i] = twist_top[i][1]*V1 + twist_bottom[i][1]*V2; 
          }
        }
        else if (!new_way)
        {
          // Start bend / twist
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
            bend[i] = bend_top[i][0]*voltage_top + bend_bottom[i][0]*voltage_bottom;     
            twist[i] = twist_top[i][0]*voltage_top + twist_bottom[i][0]*voltage_bottom;
            if (bend[i]<=-fabs(bend_edge_q0))
              bend[i]+=bend_edge_q0;
            else //if (bend[i]>bend_edge_q0)
              //if (bend_edge_q0!=0) bend[i]=fabs(bend[i]/bend_edge_q0)*bend[i];
              if (bend_edge_q0!=0) bend[i]+=bend_edge_q0*fabs(bend[i]/bend_edge_q0);
          }
          // Negative bend, positive twist.
          else if ((temp_bend<0)&&(temp_twist>=0))
          {
            bend[i] = bend_top[i][1]*voltage_top + bend_bottom[i][1]*voltage_bottom;     
            twist[i] = twist_top[i][1]*voltage_top + twist_bottom[i][1]*voltage_bottom;
            if (bend[i]<=-fabs(bend_edge_q1))
              bend[i]+=bend_edge_q1;
            else //if (bend[i]>bend_edge_q1)
              //if (bend_edge_q1!=0) bend[i]=fabs(bend[i]/bend_edge_q1)*bend[i];
              if (bend_edge_q1!=0) bend[i]+=bend_edge_q1*fabs(bend[i]/bend_edge_q1);
          }
          // Positive bend, negative twist.
          else if ((temp_bend>=0)&&(temp_twist<0))
          {
            bend[i] = bend_top[i][2]*voltage_top + bend_bottom[i][2]*voltage_bottom;     
            twist[i] = twist_top[i][2]*voltage_top + twist_bottom[i][2]*voltage_bottom;
            if (bend[i]>=fabs(bend_edge_q2))
              bend[i]+=bend_edge_q2;
            else //if (bend[i]<bend_edge_q2)
            {
              //if (bend_edge_q2!=0) bend[i]=fabs(bend[i]/bend_edge_q2)*bend[i];
              if (bend_edge_q2!=0) bend[i]+=bend_edge_q2*fabs(bend[i]/bend_edge_q2);
            }
          }
          // Positive bend, positive twist.
          else if ((temp_bend>=0)&&(temp_twist>=0))
          {
            bend[i] = bend_top[i][3]*voltage_top + bend_bottom[i][3]*voltage_bottom;     
            twist[i] = twist_top[i][3]*voltage_top + twist_bottom[i][3]*voltage_bottom;
            if (bend[i]>=fabs(bend_edge_q3))
              bend[i]+=bend_edge_q3;
            else //if (bend[i]<bend_edge_q3)
            {
              //For bends inside the zc zone, make them continuous with the edge value but
              //diminish their importance as bend approaches zero:
              if (bend_edge_q3!=0) //bend[i]=fabs(bend[i]/bend_edge_q3)*bend[i];
                bend[i]+=bend_edge_q3*fabs(bend[i]/bend_edge_q3);
            }
          }
          //The twist edges are too small; it is better not to do a correction for twist
          
        }
      }
    }
    // End pairs
    
    // Singles
    else 
    {
      ///////////Calculate bends at the edges of the zero crossing zone
      bend_multiplier=1.2;    
      //The multiplier constant sets the bend and twist criteria  higher than
      //the curvature corresponding to -count_intercept. This is because 
      //(for example for bend), the
      //extension of the straight line through the cal points crosses the bend axis
      //at zero counts, and we know the knee occurs at a bend corresponding to larger
      //than zero counts.
      
      bend_edge_q0=bend_top[i][0]*V[i];
      bend_edge_q1=bend_top[i][1]*V[i];
      bend_edge_q2=bend_top[i][2]*V[i];
      bend_edge_q3=bend_top[i][3]*V[i];

      bend_edge_q0 *= bend_multiplier;
      bend_edge_q1 *= bend_multiplier;
      bend_edge_q2 *= bend_multiplier;
      bend_edge_q3 *= bend_multiplier;

      temp_bend = V[i] * bend_top[i][0];
      if (temp_bend  >= 0)
      {
        bend[i] = V[i] * bend_top[i][0];
      }
      else if (temp_bend < 0)
      {
        bend[i] = V[i] * bend_top[i][2];
      }

      twist[i] = 0.0;
    }

    //Tail off the very small bends, continuous with filter edge:
    if (fabs(bend[i])<filter)
    {
      //bend[i]=bend[i]*(pow(bend[i]/filter,2));
      bend[i]=fabs(bend[i])*bend[i]/filter;
    }
        
    //apply high bend correction (for bends of radii less than 5 cm)
    //start_hibend, full_hibend;
    full_hibend = region_length[i]/50;
    start_hibend = region_length[i]/75;

    if ((bend[i]>start_hibend)&&(bend[i]<full_hibend)) 
    {
      bend[i]*=(((1-hibend_up[i])/(start_hibend-full_hibend))*(bend[i]-full_hibend)+hibend_up[i]);
    }
    else if (bend[i]>=full_hibend) 
    {
      bend[i]*=hibend_up[i];
    }
    else if ((bend[i]<-start_hibend)&&(bend[i]>-full_hibend))
    {
      bend[i]*=(((1-hibend_down[i])/(-start_hibend+full_hibend))*(bend[i]+full_hibend)+hibend_down[i]);
    }
    else if (bend[i]<=-full_hibend)
    {
      bend[i]*=hibend_down[i];
    }

    //subtract twist offset
    twist[i]-=twist_offset[i];
  }
  delete [] V;
  return true;
}

/*
Name:    convertToVectorData
Purpose:  Gets a U,N,B,R vector set for the current raw data set.  The size of the 
      vectors is interp_interval * num_region.  This function was created for the
      getCartesianData function.  Without this function, the API would call the 
      getCurvatureData function twice during a getCartesianData call.  Once in the 
      getCurvatureData function and once in the getVactorData cal from within the 
      getCartesianData function.
Accepts:  u = vector along the tape.
      n = vector normal to the tape.
      b = vector normal to both the u and b vectors (u * n = b).
      r = vector from base to the current index along the tape.
Returns:    void
*/
tapeAPI::convertToVectorData(double u[][3], double n[][3], double b[][3], double r[][3], double bend[], double twist[])
{
  // This function returns u,n,b,r for the tape as well as the bend and twist.  This function can only 
  // be called by getCartesianData and getVectorData.  The purpose is to return bend and twist to 
  // getCartesianData so the tape does not need to be polled twice for one data read.
  double *temp_bend = new double[num_region]; // Create new curvature variables 
  double *temp_twist = new double[num_region];// to include interpolation.
  int base;
  
  getCurvatureData(temp_bend, temp_twist); // Convert the raw data into curvature data.
  
  // Interpolate by linearly interpolating the curvature data.
  for (int i = 0;i < num_region;i++)
  {
    base = interval * i; // Get the index for a given sensor.
    for (int j = 0;j < interval;j++)
    {  
      bend[base+j] = temp_bend[i]/interval; // Divide each sensor value by the 
      // interpolation interval and place it into the correct interp_bend value.
      twist[base+j] = temp_twist[i]/interval;
    }
  }

  // Convert the bend/twist data to cartesian, in this case u,n,b,r vector set.
  bt2cart(bend,twist,u,n,b,r);

  delete [] temp_bend;
  delete [] temp_twist;
}

/*
Name:    getVectorData
Purpose:  Gets a U,N,B,R vector set for the current raw data set.  The size of the 
      vectors is interp_interval * num_region.  
Accepts:  u = vector along the tape. 
      n = vector normal to the tape.
      b = vector normal to both the u and b vectors (u * n = b).
      r = vector from base to the current index along the tape.
Returns:    void
*/
tapeAPI::getVectorData(double u[][3], double n[][3], double b[][3], double r[][3])
{
  // Public getVectorData function.  Uses functionality of private getVectorData
  // only it discards the unwanted bend and twist values.
  double *temp_bend = new double[num_region*interval]; 
  double *temp_twist = new double[num_region*interval];

  convertToVectorData(u, n, b, r, temp_bend, temp_twist);

  delete [] temp_bend;
  delete [] temp_twist;
}

/*
Name:    getCartesianData
Purpose:  Gets x,y,z,and quaternion inforamation for the current raw data set.  The size 
      of the vectors is interp_interval * num_region.  
Accepts:  x  = x position along the tape. 
      y  = y position along the tape.
      z  = z position along the tape.
      q1 = quaternion orientation value - equivalent to w in most references.
      q2 = quaternion orientation value - equivalent to x in most references.
      q3 = quaternion orientation value - equivalent to y in most references.
      q4 = quaternion orientation value - equivalent to z in most references.
Returns:    void
*/
bool tapeAPI::getCartesianData(double x[], double y[], double z[],
                  double q1[], double q2[], double q3[], double q4[])
{
  double (*temp_u)[3] = new double[num_region*interval+1][3];
  double (*temp_n)[3] = new double[num_region*interval+1][3];
  double (*temp_b)[3] = new double[num_region*interval+1][3];
  double (*temp_r)[3] = new double[num_region*interval+1][3];
  double *temp_bend = new double[num_region*interval]; 
  double *temp_twist = new double[num_region*interval];

  double U[3],B[3];
  quaternion quat, accum_quat;
  
  convertToVectorData(temp_u,temp_n,temp_b,temp_r,temp_bend,temp_twist);
  
  for (int i = 0;i < num_region*interval +1;i++)
  {
    x[i] = temp_r[i][0];
    y[i] = temp_r[i][1];
    z[i] = temp_r[i][2];
    if (i == 0) // base position...
    {
      U[0] = temp_u[i][0];
      U[1] = temp_u[i][1];
      U[2] = temp_u[i][2];
      
      B[0] = temp_b[i][0];
      B[1] = temp_b[i][1];
      B[2] = temp_b[i][2];
      
      quat.vector2Quaternion(U,B);
    }
    else
    {
      U[0] = temp_u[i][0];
      U[1] = temp_u[i][1];
      U[2] = temp_u[i][2];
      
      B[0] = temp_b[i][0];
      B[1] = temp_b[i][1];
      B[2] = temp_b[i][2];

      quat.bt2Quaternion(temp_bend[i-1], temp_twist[i-1]);//, U, B);
    }

    accum_quat = accum_quat * quat; 
    accum_quat.normalize();

    q1[i] = accum_quat.m_w;
    q2[i] = accum_quat.m_x;
    q3[i] = accum_quat.m_y;
    q4[i] = accum_quat.m_z;
  }

  delete [] temp_u;
  delete [] temp_n;
  delete [] temp_b;
  delete [] temp_r;
  delete [] temp_bend;
  delete [] temp_twist;
  return TRUE;
}

// Not in use...
void tapeAPI::getCartData(double x[], double y[], double z[], 
              double q1[], double q2[], double q3[], double q4[])
{
  double *temp_bend = new double[num_sensors/2]; // Create new curvature variables 
  double *temp_twist = new double[num_sensors/2];// to include interpolation.
  quaternion quat;
  quaternion accum_quat;
  quaternion *temp_quat = new quaternion[2];

  getCurvatureData(temp_bend, temp_twist);

  for (int i = 0; i < num_region +1; i++)
  {
    if (i == 0) // base position...
    {
      quat.vector2Quaternion(Uo,Bo);
    }
    else
    {
      quat.bt2Quaternion(temp_bend[i-1], temp_twist[i-1]);
    }
  
    accum_quat = accum_quat * quat; 
    accum_quat.normalize();

    q1[i] = accum_quat.m_w;
    q2[i] = accum_quat.m_x;
    q3[i] = accum_quat.m_y;
    q4[i] = accum_quat.m_z;
  }
  
  // Interpolate the quaternion data.
  

  // Find the cartesian position data with quat2cart function.
  quat2cart(q1, q2, q3, q4, x, y, z);

  delete [] temp_bend;
  delete [] temp_twist;
}

/*
Name:    bt2cart
Purpose:  Converts bend and twist curvature values into Cartesian U,N,B and R vectors.
Accepts:  bend  = bend values along the tape.
      twist = twist values along the tape.
      U     = vector along the tape.
      N     = vector normal to the tape.
      B     = vector normal to both the u and b vectors (u * n = b).
      R     = vector from base to the current index along the tape.
Returns:    void
*/
void tapeAPI::bt2cart(double bend[], double twist[], double U[][3], double N[][3], 
        double B[][3], double R[][3])
{
  double No[3];
    
  U[0][0] = Uo[0]; // Set the base orientation for the calculations
  U[0][1] = Uo[1];
  U[0][2] = Uo[2];
    
  B[0][0] = Bo[0];
  B[0][1] = Bo[1];
  B[0][2] = Bo[2];
  
  cross(Bo,Uo,No);// Find No by performing the cross product of Bo and Uo.
  normalize(No);  // The Normal vector No will always be perpendicular to 
          // the Bo and Uo vectors.
  N[0][0] = No[0];
  N[0][1] = No[1];
  N[0][2] = No[2];
  
  R[0][0] = 0.0; // The base  position is handled later (when the R is converted 
  R[0][1] = 0.0; // engineering units.
  R[0][2] = 0.0;
  
  for (int i=0; i<num_region*interval; i++)
  {
    B[i+1][0] = B[i][0]; // Set the next set of vectors equal to these ones.
    B[i+1][1] = B[i][1];
    B[i+1][2] = B[i][2];
    N[i+1][0] = N[i][0];
    N[i+1][1] = N[i][1];
    N[i+1][2] = N[i][2];
    U[i+1][0] = U[i][0];
    U[i+1][1] = U[i][1];
    U[i+1][2] = U[i][2];
    R[i+1][0] = 0.0; // Initial R for all sensors.  
    R[i+1][1] = 0.0;
    R[i+1][2] = 0.0;

    rotline(B[i+1], U[i], twist[i]); // Rotate next B about this U by twist.
    rotline(U[i+1], B[i+1], bend[i]);// Rotate next U around the new B by bend.    
  }

  for (int j=1; j<(num_region*interval)+1; j++)
  {
    // perform cross product to find N
    cross(B[j],U[j],N[j]);
    
    // accumulate U's to find R
    R[j][0] = R[j-1][0] + U[j][0];
    R[j][1] = R[j-1][1] + U[j][1];
    R[j][2] = R[j-1][2] + U[j][2];
  }

  R[0][0] = Xo; // Set the base positions.
  R[0][1] = Yo;
  R[0][2] = Zo;

  for (int k=1; k<(num_region*interval)+1; k++)
  {
    // convert R to mm and add the base position (offset)
    R[k][0] = R[k][0]*tape_length/(num_region*interval)+Xo; 
    R[k][1] = R[k][1]*tape_length/(num_region*interval)+Yo;
    R[k][2] = R[k][2]*tape_length/(num_region*interval)+Zo;      
  }
}


// Not in use...
void tapeAPI::quat2cart(double q1[], double q2[], double q3[], double q4[], 
            double x[], double y[], double z[])
{
  // set the initial position to (0,0,0)
  x[0] = 0.0;
  y[0] = 0.0;
  z[0] = 0.0;

  // find the length of each segment of the tape.
  double segment;
  segment = tape_length/(num_region*interval);

  // rotate each section to its new position.
  for (int i=0; i<num_region*interval; i++)
  {
    // rotate each point from a base of (1, 0, 0) by the accumulated
    // quaternion. (starting U, can change to N or B)
    x[i+1] = 1.0;
    y[i+1] = 0.0;
    z[i+1] = 0.0;

    // rotate the point with the quaternion
    rotatePoint(x[i+1],y[i+1],z[i+1],q1[i],q2[i],q3[i],q4[i]);
    
    // translate the rotated point out to meet the end of the previous. 
    x[i+1] += x[i];
    y[i+1] += y[i];
    z[i+1] += z[i];
  }

  // scale the segments from a length of 1 to the real length
  // and offset the data set by the initial position
  for (i=0; i < (num_region*interval)+1; i++)
  {
    x[i] = x[i]*segment + Xo;
    y[i] = y[i]*segment + Yo;
    z[i] = z[i]*segment + Zo;
  }
}

/*
Name:    rotline
Purpose:  Rotates one vector about an other by the angle fAlpha.
Accepts:  fpL1   = rotated vector.
      fplA   = base vector.
      fAlpha = angle of rotation.
Returns:    void  
*/
void tapeAPI::rotline(double* fpL1, double* fpLA, double fAlpha)
{
  // The center of rotation is (0,0,0) 
  // and all lines must start at (0,0,0).
  // alpha is the rotation angle in radians
  // fAlpha = alpha*pi/180;
  double cosa, sina, vera;
  double x, y, z;
  double rot[3][3];
  double oldxyz[2][3];
  double newxyz[2][3];
  double length1;
  double norm1;
  cosa = cos(fAlpha);
  sina = sin(fAlpha);
  vera = 1.0f - cosa;
  
  //The line is rotated about a rotation axis [x y z]=fpLA
  x = fpLA[0];
  y = fpLA[1];
  z = fpLA[2];
  rot[0][0] = cosa + x*x*vera;
  rot[1][0] = x*y*vera - z*sina;
  rot[2][0] = x*z*vera + y*sina;
  rot[0][1] = x*y*vera + z*sina;
  rot[1][1] = cosa + y*y*vera;
  rot[2][1] = y*z*vera - x*sina;
  rot[0][2] = x*z*vera - y*sina;
  rot[1][2] = y*z*vera + x*sina;
  rot[2][2] = cosa + z*z*vera;
  oldxyz[0][0] = fpL1[0];
  oldxyz[0][1] = fpL1[1];
  oldxyz[0][2] = fpL1[2];
  
  // Rotate fpL1 and fpL2 about fpLA by -fAlpha.
  // Matrix multiplication the tedious way
  newxyz[0][0] = oldxyz[0][0]*rot[0][0] + oldxyz[0][1]*rot[1][0] + oldxyz[0][2]*rot[2][0];
  newxyz[0][1] = oldxyz[0][0]*rot[0][1] + oldxyz[0][1]*rot[1][1] + oldxyz[0][2]*rot[2][1];
  newxyz[0][2] = oldxyz[0][0]*rot[0][2] + oldxyz[0][1]*rot[1][2] + oldxyz[0][2]*rot[2][2];
  
  // now we ensure that the new vectors have the same length as the originals
  // lengths 1 & 2 and norms 1 & 2 are my additions - L. Malloch
  length1 = sqrt(fpL1[0]*fpL1[0] + fpL1[1]*fpL1[1] + fpL1[2]*fpL1[2]);
  norm1 = sqrt(newxyz[0][0]*newxyz[0][0]+newxyz[0][1]*newxyz[0][1]+newxyz[0][2]*newxyz[0][2]);
  fpL1[0] = newxyz[0][0] * length1/norm1;
  fpL1[1] = newxyz[0][1] * length1/norm1;
  fpL1[2] = newxyz[0][2] * length1/norm1;
}

/*
Name:    cross
Purpose:  Calculates the cross product of two vectors.
Accepts:  A = first vector.
      B = second vector.
      C = cross product of A and B (A*B).
Returns:    void
*/
tapeAPI::cross(double A[3], double B[3], double C[3])
{
  // A cross B = C
  C[0] = A[1]*B[2]-A[2]*B[1];
  C[1] = A[2]*B[0]-A[0]*B[2];
  C[2] = A[0]*B[1]-A[1]*B[0];
}

/*
Name:    normalize
Purpose:  Normalizes the vector.
Accepts:  vector = the vector to be normalized.
Returns:    void
*/
tapeAPI::normalize(double vector[3])
{
  // Normalize the vector. ie make the length of the vector 1.
  double length;
  length = magnitude(vector);

  if (length != 0.0)
  {
    vector[0] = vector[0] / length;
    vector[1] = vector[1] / length;
    vector[2] = vector[2] / length;
  }
}

/*
Name:    magnitude
Purpose:  Find the magnitude of a vector.
Accepts:  vector = the vector you wish to find the magnitude of.
Returns:    the magnitude of the vector.
*/
double tapeAPI::magnitude(double vector[3])
{
  // Find the length of the vector.
  double result;

  result = sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2]);
  return result;
}

// Not used...
void tapeAPI::rotatePoint(double x, double y, double z, 
              double q_w, double q_x, double q_y, double q_z)
{
  // to rotate a point, convert the point into quaterion form.
  // multiply it in quaternion space by the rotation quaternion
  // rotatedPoint = q * Point * q'
  quaternion point(0.0,x,y,z);
  quaternion inverse;
  quaternion rotation(q_w,q_x,q_y,q_z);
  quaternion result;

  inverse = rotation;
  inverse.invert();

  result = rotation*point;
  result = result*inverse;

  x = result.m_x;
  y = result.m_y;
  z = result.m_z;
}

/*
Name:    saveAdvancedCalibration
Purpose:  Calculates the calibration coefficients and stores them away using both 
      bend and twist poses. 
Accepts:  filename = the name of the calibration file.
      mstfile  = the *.mst file for the tape (settings file) This file may or 
      may not contain a twist normalization vector.  If it does, it is read in 
      and used to correct for tape taper. 
Returns:    void
*/
void tapeAPI::saveAdvancedCalibration(char *filename,char *mstfile)
{
  calculate_advanced(bend_top,bend_bottom,twist_top,twist_bottom,mstfile);
  saveFile(filename);
}

/*
Name:    saveNormalCalibration
Purpose:  Calculates the calibration coefficients using only the bend poses.
      NOT RECOMMENDED...  Use the advenced calibration until this is 
      further tested.
Accepts:  filename = the name of the calibration file.
Returns:    void
*/
void tapeAPI::saveNormalCalibration(char *filename)
{
  calculate_normal(bend_top,bend_bottom,twist_top,twist_bottom);
  saveFile(filename);
}

/*
Name:    saveFlatCalibration
Purpose:  Save away the new flat pose data in the calibration file.
Accepts:  filename = the name of the calibration file.
Returns:    void
*/
void tapeAPI::saveFlatCalibration(char *filename)
{
  saveFile(filename);
}

/*
Name:    saveHelicalCalibration
Purpose:  Save away the helical pose data in the calibration file and recalculate.
      the k values for the tape.  See the calibration class for a description of 
      the k values. 
Accepts:  filename = the name of the calibration file.
Returns:    void
*/
void tapeAPI::saveHelicalCalibration(char *filename)
{
  saveFile(filename);
  findKValues();
}

// Not used...
void tapeAPI::interpolate(double y[], double yi[])
{
  // Pass the y and yi:
  // y = coordinates of the current function (equally spaced)
  // yi = an empty array of size 
  // Based on "Cubic Convolution Interpolation for Digital Image
  // Processing", Robert G. Keys, IEEE Trans. on Acoustics, Speech, and
  // Signal Processing, Vol. 29, No. 6, Dec. 1981, pp. 1153-1160.
  // This method is used for two reasons;
  // 1. the function fits through the control points.
  // 2. it is a computationally efficient algorithm.
  // For more info, see the interp1 function in Matlab.

  int u, s;
  int count = 0;
  for (s = 1;s < num_region; s++)
  {
    for (u = 0;u <= 1 - 1/interval;u = u + 1/interval)
    {
      count ++;
      yi[count] = (y[u] * (-pow(s,3)+2 * pow(s,2) - s))
        + (y[u+1] * (3 * pow(s,3) - 5 * pow(s,2) + 2)) 
        + (y[u+2] * (-3 * pow(s,3) + 4 * pow(s,2) + s))
        + (y[u+3] * (pow(s,3) - pow(s,2)));
      yi[count] /= 2;
    }
  }
  count ++;
  u = 1;
  yi[count] = (y[u] * (-pow(s,3)+2 * pow(s,2) - s))
    + (y[u+1] * (3 * pow(s,3) - 5 * pow(s,2) + 2))
    + (y[u+2] * (-3 * pow(s,3) + 4 * pow(s,2) + s))
    + (y[u+3] * (pow(s,3) - pow(s,2)));
  yi[count] /= 2;
} 

/*
Name:    vectoralign
Purpose:  Finds what type of rotation is required to transform one 3D vector (vector1) into
      a second 3D vector (vector2).
Accepts:  vector1 = double array of 3 elements, 1st 3D vector
      vector2 = double array of 3 elements, 2nd 3D vector
      rotaxis = double array of 3 elements, the axis about which vector1 is rotated in order 
            to match up with vector2
      rotangle = the angle through which vector1 must be rotated (about rotaxis) in order to 
             line up with vector2.
Returns:    void (rotaxis and rotangle are set by this function).
*/
void tapeAPI::vectoralign(double vector1[], double vector2[], double rotaxis[], double &rotangle)
{
  //Input: vector 1 to be aligned with the orientation of vector 2.
  //Output: rotaxis, the axis about which to rotate vector1 by angle.
  //      rot_angle, the rotation angle
  
  //normalize vector1 and vector2
  normalize(vector1);
  normalize(vector2);
  
  //rot_angle = arccos(vector1 dot vector 2) 
  rotangle=acos(vector1[0]*vector2[0]+vector1[1]*vector2[1]+vector1[2]*vector2[2]);
  //rot_axis = vector1 X vector2
  cross(vector1, vector2, rotaxis);
  
  //special case, check to see if cross-product is zero (i.e. vectors are already aligned)
  if ((rotaxis[0]==0)&&(rotaxis[1]==0)&&(rotaxis[2]==0))
  {
    //need a rotation of either zero or pi, therefore need to choose a rotation
    //axis which is perpendicular to either vector1 or vector2
    rotline(rotaxis,vector1,3.14159265359/2);
  }
  
  //normalize axis vector
  normalize(rotaxis);
}


/*
Name:    reftotip
Purpose:  Reverses the order of the bend and twist arrays for the tape, in effect assuming that the 
      tape reference point (or base) is at the physical tip of the tape, and the tip of the tape
      is actually at the physical base of the tape's sensorized region.
Accepts:  bend = array of bend values for each of the tape's sensorized regions, starting at the 
      physical base of the tape and extending to the physical tip of the tip.
      twist = array of twist values for each of the tape's sensorized regions, starting at the 
      physical base of the tape and extending to the physical tip of the tip.
Returns:    void (values in bend and twist arrays are reversed, i.e. last element switched with first, 
      etc.)
*/
void tapeAPI::reftotip(double bend[], double twist[])
{
  int last = num_region-1;
  int first = 0;
  double temp;

  while (last-first > 1)
  {
    temp = bend[first];
    bend[first] = bend[last];
    bend[last] = temp;
    temp = twist[first];
    twist[first] = twist[last];
    twist[last] = temp;
    first++;
    last--;
  }  
}