// Back.cpp: implementation of the Back class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Back.h"
#include "filedata.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

/*
Name:    Back
Purpose:  constructor - this constructor builds a Back object with settings defined in the 
      Measurand ShapeTape file: configfile
Accepts:  config_file = Measurand ShapeTape (.mst) file that contains all of the settings 
      for this particular ShapeTape. 
      
Returns:    void
*/
Back::Back(char *configfile, char *rootfolder)
   :BodyPart(configfile, rootfolder) 
{
  m_bDoBackCalc=false;
  m_bFlatoffsets=false;
  m_bSpinecaptured=false;
  for (int i=0;i<MAXREGIONS;i++) m_defSpine[i]=0.0;
  GetBackConfig();
}

/*
Name:    ~Back
Purpose:  destructor - the Back object gets destroyed here.
Accepts:  void
      
Returns:    void
*/
Back::~Back()
{

}

/*
Name:    FindLimbAnglesUsingBT
Purpose:  Just calls the default BodyPart implemenation of this function for computing the 
      link-model bend and twist for the tape.
Accepts:  void
Returns:  void  
*/
void Back::FindLimbAnglesUsingBT() {
  BodyPart::FindLimbAnglesUsingBT();
}

/*
Name:    ReCalculateBendTwist
Purpose:  Takes the normally calculated bend and twist for the ShapeTape, and re-calculates
      (or re-distributes it) based on the assumptions of the back link-model.
Accepts:  bend = the normally calculated ShapeTape bend for this tape (array of num_regions * 
          interval + 1 points).
      twist = the normally calculated ShapeTape twist for this tape (array of num_regions * 
          interval + 1 points).
      numToAvg = number of consecutive samples to average. Default is 1.
Returns:  void  
*/
void Back::ReCalculateBendTwist(double bend[], double twist[], int numToAvg/*=1*/) {
  BodyPart::ReCalculateBendTwist(bend,twist);
  int numVertices = num_region*interval+1;
  //set bend / twist to zero
  for (int i=0;i<numVertices;i++) {
    m_recalc_bend[i]=0;
    m_recalc_twist[i]=0;
    m_avgbend[m_avgindex][i]=0;
    m_avgtwist[m_avgindex][i]=0;
  }
  //First, collect and apply in the standard way (TWIST ONLY):
  for (i=0;i<MAXITEMS;i++) {
    //Calculate twist (applied evenly along bones):
    if (tapply_start[i]>0) {
      double jointtwist=0.0;
      for (int j=tcollect_start[i];j<=tcollect_end[i];j++) jointtwist+=twist[j];
      int tsegs = tapply_end[i] - tapply_start[i] + 1;
            double jointtwist_avg=jointtwist/tsegs;
      //spread the average twist along the subsegments:
      for (j=tapply_start[i];j<=tapply_end[i];j++) m_avgtwist[m_avgindex][j]=jointtwist_avg;
    }  
  }    
     
  //Apply any special rules
  int base=bapply_start[0];
  int first_active=base;
  int mid=bapply_start[2];
  int th1=bapply_start[3];
  int top=80;
  int neck1=0,neck2=0,head1=0,head2=0;
  if (tape_length>530) {
    top=bapply_start[4];
    neck1=top+1;
    neck2=bapply_start[5];
    head1=neck2+1;
    head2=numVertices-1;
  }

  if (!m_bSpinecaptured) {
  
    //model the back with bend offsets:
    //lumbar bend offset:
    int blen=mid-base+1;
    const double pi=3.14159265359;
    double lumbang=70*pi/180;
    double seg_bendoffset = lumbang / blen;
    for (i=base;i<=mid;i++) m_bendoffsets[i]=seg_bendoffset;
    
    //lower thoracic bend offset:
    blen=th1-mid;
    double thorang=-20*pi/180;
    seg_bendoffset=thorang/blen;
    for (i=mid+1;i<=th1;i++) m_bendoffsets[i]=seg_bendoffset;

    //upper thoracic bend offset:
    blen=top-th1;
    thorang=-40*pi/180;
    seg_bendoffset=thorang/blen;
    for (i=th1+1;i<=top;i++) m_bendoffsets[i]=seg_bendoffset;
    m_bendoffsets[top]=25*pi/180; //orientation correction to make person lie flat in bed
    
    //neck bend offset 
    blen=neck2-neck1+1;
    double neckang=30*pi/180;
    seg_bendoffset=neckang/blen;
    for (i=neck1;i<=neck2;i++) m_bendoffsets[i]=seg_bendoffset;
    
    //round head (visual only)
    blen=head2-head1+1;
    double headang=-360*pi/180;
    seg_bendoffset=headang/blen;
    for (i=head1;i<=head2;i++) m_bendoffsets[i]=seg_bendoffset;
    
    //coil up the base:
    //coil up the null link and turn off its twist:
    double jointbend=-40*pi;
    if (first_active>0) {
      seg_bendoffset=jointbend/first_active;
      for (i=0;i<first_active;i++) {
        m_avgbend[m_avgindex][i]=seg_bendoffset;
        m_avgtwist[m_avgindex][i]=0.0;
      }
    }
  } //end if for 'spine not captured'
  else {
    //round head (visual only)
    int blen=head2-head1+1;
    double headang=-360*12*3.14159/180;
    double seg_bendoffset=headang/blen;
    for (i=head1;i<=head2;i++) m_bendoffsets[i]=seg_bendoffset;
  }
  //apply bends to entire back in shapetape fashion (no links):
  int uplimit=neck2;
  for (i=base;i<=uplimit;i++) m_avgbend[m_avgindex][i]=bend[i];
  
  //kill any twist beyond head:
  for (i=head1;i<=interval*num_region;i++) m_avgtwist[m_avgindex][i]=0.0;
  //end of special rules
  
  //add on offsets (if we're not using flat offsets)
  if (!m_bFlatoffsets) {
    for (i=0;i<numVertices;i++) {
      m_avgbend[m_avgindex][i]+=m_bendoffsets[i];
      m_avgtwist[m_avgindex][i]+=m_twistoffsets[i];
    }
  }
  AverageBendTwist(numToAvg);
  
  //set joint readouts
  double lumbar_bend=0.0, lumbar_twist=0.0, thor_bend=0.0, thor_twist=0.0;
  for (i=base;i<mid;i++) {
    lumbar_bend+=m_recalc_bend[i];
    lumbar_twist+=m_recalc_twist[i];
  }
  for (i=mid;i<top;i++) {
    thor_bend+=m_recalc_bend[i];
    thor_twist+=m_recalc_twist[i];
  }
  const double dRadtoDeg = 57.295779513;
  m_jbangles[0]=lumbar_bend*dRadtoDeg;
  m_jtangles[0]=lumbar_twist*dRadtoDeg;
  m_jbangles[1]=thor_bend*dRadtoDeg;
  m_jtangles[1]=thor_twist*dRadtoDeg;
}
  
/*
Name:    AssignCapturedOffsets
Purpose:  Assigns values to the link-model bend offsets (m_bendoffsets) based on the spine 
      capture that was most recently performed.
Accepts:  void  
Returns:  void
*/
void Back::AssignCapturedOffsets() {
  //zero all bend and twist offsets
  int numVertices=interval*num_region+1;
  for (int i=0;i<numVertices;i++) {
    m_bendoffsets[i]=0.0;
    m_twistoffsets[i]=0.0;
  }
  int first_active=bapply_start[0];
  int neck2=bapply_start[5];
  int head1=neck2+1;
  for (i=first_active;i<=head1;i++) m_bendoffsets[i]=m_defSpine[i/interval]/interval;
  m_bSpinecaptured=true;
}

/*
Name:    GetThoracicIndex
Purpose:  Gets the zero-based index of the tape where the thoracic section of the spine should
      be located. Typically useful for someone who wants to estimate chest orientation based
      on the thoracic section of the spine tape.
Accepts:  void  
Returns:  Zero-based index along tape where the thoracic section of the spine should be located.
*/
int Back::GetThoracicIndex() {
  //returns the index of the back tape that is used for computing torso orientation
  return bapply_start[3];
}

/*
Name:    CaptureSpine
Purpose:  Captures the shape of the ShapeTape on the wearer's back. The region of interest that is
      captured lies between the base of the tape and the neck. This is a critical
      region for accurate back measurements, which is why it is recommended that this procedure
      be performed once for each different person wearing the back tape. It is necessary that the
      ShapeTape first be placed on a flat surface and offsets collected before performing this 
      procedure. It is also necessary that the wearer stand up straight when performing this 
      procedure. Once the procedure is completed, tape offsets should be collected again in the 
      same standing up straight pose.
Accepts:  void
Returns:  void
*/
void Back::CaptureSpine() {
  double temp_bend[MAXREGIONS], temp_twist[MAXREGIONS];
  //initialize
  for (int i=0;i<MAXREGIONS;i++) {
    temp_bend[i]=0.0;
    temp_twist[i]=0.0;
  }
  getCurvatureData(temp_bend,temp_twist);
  for (i=0;i<MAXREGIONS;i++) m_defSpine[i]=temp_bend[i];
  AssignCapturedOffsets();
  m_bSpinecaptured=true;
  //save offsets to file
  filedata mstfile(m_configfile);
  mstfile.writeData("[back]","spinepose",MAXREGIONS,m_defSpine);
}

/*
Name:    GetBackData
Purpose:  This is the general purpose function that most users will call to get back data for the 
      tape. It takes care of acquiring the serial data, and performing all of the necessary
      back tape calculations.
Accepts:  r = two dimensional double array of ShapeTape positions. Array must be num_regions * 
        interval + 1 long by 3 wide.
      roll = double array of ShapeTape roll angles (in deg) along length. Array must be 
          num_regions *interval + 1 long.
      pitch = double array of ShapeTape pitch angles (in deg) along length. Array must be 
          num_regions *interval + 1 long.
      yaw = double array of ShapeTape yaw angles (in deg) along length. Array must be 
          num_regions *interval + 1 long.
      numToAvg = number of consecutive samples to average. Default is 1.
Returns:  void (r, roll, pitch, yaw arrays are filled).
*/
void Back::GetBackData(double r[][3], double roll[], double pitch[], double yaw[], int numToAvg/*=1*/) {
  //gets positions and orientations of interpolated spine segments
  double temp_bend[MAXREGIONS], temp_twist[MAXREGIONS];
  double bend[MAXSUBREGIONS], twist[MAXSUBREGIONS];
  getCurvatureData(temp_bend,temp_twist);
  DistribSensorData(temp_bend,bend);
  DistribSensorData(temp_twist,twist);
  ReCalculateBendTwist(bend,twist,numToAvg);
  FindLimbAnglesUsingBT();
  int numVertices = num_region*interval+1;
  const double dRadtoDeg = 57.295779513;
  for (int i=0;i<numVertices;i++) {
    r[i][0]=m_r[i][0]; r[i][1]=m_r[i][1]; r[i][2]=m_r[i][2];
    double seg_roll=0.0, seg_pitch=0.0, seg_yaw=0.0;
    unb2rpy(m_u[i],m_n[i],m_b[i],seg_roll,seg_pitch,seg_yaw);
    roll[i]=seg_roll*dRadtoDeg;
    pitch[i]=seg_pitch*dRadtoDeg;
    yaw[i]=seg_yaw*dRadtoDeg;
  }
}

/*
Name:    GetBackConfig
Purpose:  Gets all of the configuration parameters (stored in the .mst and .lin files)
      for this back tape.
Accepts:  void
Returns:  true if the configuration for the back tape was valid, false otherwise.
*/
bool Back::GetBackConfig() {
  if (!GetBodyConfig()) return false;
  if (m_nTapetype!=BACKTAPE) return false;
  filedata mstfile(m_configfile);
  if (mstfile.getFileLength(m_configfile)<10) return false; //check for invalid mst file
    
  //get link file settings
  char *linfilename = NULL;
  mstfile.getString("[settings]","linfile",&linfilename);
  if (!linfilename) return false;
  SetCurrentDirectory(m_rootfolder);
  if (!LoadLinFile(linfilename)) {
    delete []linfilename;
    return false;
  }
  delete []linfilename;

  mstfile.getDouble("[back]","spinepose",MAXREGIONS,m_defSpine);
  //check to see if the spine pose has been previously captured
  double dSum=0.0;
  for (int i=0;i<MAXREGIONS;i++) dSum+=m_defSpine[i];
  if (dSum!=0.0) {
    m_bSpinecaptured=true;
    AssignCapturedOffsets();
  }
  else m_bSpinecaptured=false;
  return true;
}

/*
Name:    ComputeOffsets
Purpose:  Determines the rotational offsets for the back tape so that it shows up 
      properly in the homing pose (person standing up straight)
Accepts:  void
Returns:  void
*/
void Back::ComputeOffsets() {
  double r[MAXSUBREGIONS][3], roll[MAXSUBREGIONS], pitch[MAXSUBREGIONS], yaw[MAXSUBREGIONS];
  ZeroTransforms();
  ResetAverages();
  GetBackData(r,roll,pitch,yaw,1);
  BodyPart::ComputeOffsets();
}

/*
Name:    UpdateLinkFile
Purpose:  Updates the link file for a back tape, based on the contents of a given subject file.
      Subject files can be created using Measurand's ShapeRecorder software.
Accepts:  subjectFile = null terminated string specifying the filename of the subject file.
      nClampindex = the index of the back tape sensing region which coincides with the 
              bottom of the person's lumbar.
Returns:  true if link file was updated successfully, false otherwise
*/
bool Back::UpdateLinkFile(char *subjectFile, int &nClampindex) {
  nClampindex=-1;
  filedata subfile(subjectFile);
  //tape to bonelength ratio for back
  const double back_ratio=1.00;
  double dBacklength = subfile.getDouble("[linklengths]","backlength");  
  
  //get spine capture info from subject file
  double dTemp[24];
  subfile.getDouble("[spine_capture]","spine",24,dTemp);
  //check to see if there is any valid spine pose data
  bool bValidspine = false;
  for (int i=0;i<24;i++) {
    if (dTemp[i]!=0.0) {
      bValidspine = true;
      break;
    }
  }

  filedata mstfile(m_configfile);
  //get name of link file
  char *linkfilename = NULL;
  mstfile.getString("[settings]","linfile",&linkfilename);
  if (!linkfilename) return false; //no linkfile to update!
  filedata linfile(linkfilename);
  //assume linkfile based on the spine_tip_C1 model
  double tape_bonelength_mm[7];
  linfile.getDouble("bonelabel_string","tape_bonelength_mm",7,tape_bonelength_mm);
  double dBacktapelength = back_ratio * dBacklength;
  double dRegionlength = region_length[0]; //assume tape to be constructed
                                    //from regions of uniform length
  const double dNecklength = 200.0; //assume neck & head region of 200 mm
  double dRequired_tapelength = dBacktapelength + dNecklength;

  dBacktapelength += num_region*dRegionlength - dRequired_tapelength;
  dRequired_tapelength = dBacktapelength + dNecklength;

  //check to make sure that there is enough tape:
  if (dRequired_tapelength>tape_length) {
    //uh oh... not enough tape!
    return false;
  }
  
  //fill in tape_bonelength_mm
  tape_bonelength_mm[0] = tape_length - dRequired_tapelength;
  tape_bonelength_mm[1] = dBacktapelength*4.0/15; //2 lumbar regions
  tape_bonelength_mm[2] = dBacktapelength/15.0; 
  tape_bonelength_mm[3] = dBacktapelength*3.0/15; //2 thoracic regions
  tape_bonelength_mm[4] = dBacktapelength*7.0/15.0; 
  tape_bonelength_mm[5] = 150.0;
  tape_bonelength_mm[6] = 50.0;
      
  //write changes to link file
  linfile.writeData("bonelabel_string","tape_bonelength_mm",7,tape_bonelength_mm);
  nClampindex = num_sensors/2 - num_region;

  if (nClampindex < 0) nClampindex = 0;

  linfile.closeDataFile();
  //reload link file with new settings
  LoadLinFile(linkfilename);
  if (m_bSpinecaptured&&!bValidspine) {
    //reassign offsets now that link file has changed
    AssignCapturedOffsets();
  }
  else if (bValidspine) {
    //copy spine offsets
    for (i=0;i<24;i++) m_defSpine[i] = dTemp[i];
    m_bSpinecaptured=true;
    AssignCapturedOffsets();
  }  
  return true;
}

