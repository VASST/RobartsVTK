// Head.cpp: implementation of the Head class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Head.h"
#include "filedata.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

/*
Name:		Head
Purpose:	constructor - this constructor builds a Head object with settings defined in the 
			Measurand ShapeTape file: configfile
Accepts:	config_file = Measurand ShapeTape (.mst) file that contains all of the settings 
			for this particular ShapeTape. 
			
Returns:    void
*/
Head::Head(char *configfile, char *rootfolder)
	 :BodyPart(configfile, rootfolder) 

{
	m_bSpinecaptured=false;
	for (int i=0;i<MAXREGIONS;i++) m_defSpine[i]=0.0;
	m_bIncludeback=false;
	headinfo.dHeadpitch=0.0;
	headinfo.dHeadroll=0.0;
	headinfo.dHeadyaw=0.0;
	GetHeadConfig();
}

/*
Name:		~Head
Purpose:	destructor - the Head object gets destroyed here.
Accepts:	void
			
Returns:    void
*/
Head::~Head()
{
}

/*
Name:		FindLimbAnglesUsingBT
Purpose:	Finds all of the head's orientation, based on the recalculated bend
			and twist of the tape. Fills the headinfo structure with the head orientation information.
Accepts:	void
Returns:	void  (headinfo member variable structure is updated with new values by this function).
*/
void Head::FindLimbAnglesUsingBT() {
	const double dRadtoDeg = 57.2957795;
	BodyPart::FindLimbAnglesUsingBT();

	double head_u[3], head_n[3], head_b[3];
	//head section of tape's orientation
	double cosroll=cos(m_bone_orient[11][0]);
	double sinroll=sin(m_bone_orient[11][0]);
	double cospitch=cos(m_bone_orient[11][1]);
	double sinpitch=sin(m_bone_orient[11][1]);
	double cosyaw=cos(m_bone_orient[11][2]);
	double sinyaw=sin(m_bone_orient[11][2]);
	
	head_u[0]=cosyaw*cospitch;
	head_u[1]=sinpitch;
	head_u[2]=sinyaw*cospitch;
	head_n[0]=-cosyaw*sinpitch*cosroll + sinyaw*sinroll;
	head_n[1]=cospitch*cosroll;
	head_n[2]=-sinyaw*sinpitch*cosroll-cosyaw*sinroll;
	head_b[0]=-cosyaw*sinpitch*sinroll-sinyaw*cosroll;
	head_b[1]=cospitch*sinroll;
	head_b[2]=-sinyaw*sinpitch*sinroll+cosyaw*cosroll;
	
	//apply inverse transformations to account for backwards and upside down orientation of the head
	//part of the tape
	for (int i=0;i<3;i++) {
		head_u[i]=-head_u[i];
		head_n[i]=-head_n[i];
	}
	//find actual orientation of head
	unb2rpy(head_u,head_n,head_b,headinfo.dHeadroll,headinfo.dHeadpitch,headinfo.dHeadyaw);
	//convert to deg
	headinfo.dHeadroll*=dRadtoDeg;
	headinfo.dHeadpitch*=dRadtoDeg;
	headinfo.dHeadyaw*=dRadtoDeg;
}

/*
Name:		ReCalculateBendTwist
Purpose:	Takes the normally calculated bend and twist for the ShapeTape, and re-calculates
			(or re-distributes it) based on the assumptions of the head (spine) link-model.
Accepts:	bend = the normally calculated ShapeTape bend for this tape (array of num_regions * 
					interval + 1 points).
			twist = the normally calculated ShapeTape twist for this tape (array of num_regions * 
					interval + 1 points).
			numToAvg = number of consecutive samples to average. Default is 1.
Returns:	void  
*/
void Head::ReCalculateBendTwist(double bend[], double twist[], int numToAvg/*=1*/) {
	const double pi=3.14159265359;
	BodyPart::ReCalculateBendTwist(bend,twist);
	//set bend / twist to zero
	int numVertices = interval*num_region+1;
	for (int i=0;i<numVertices;i++) {
		m_recalc_bend[i]=0;
		m_recalc_twist[i]=0;
		m_avgbend[m_avgindex][i]=0;
		m_avgtwist[m_avgindex][i]=0;
	}

	//useful in following the special rules:
	int base=bapply_start[0];
	int first_active=base;
	int mid=bapply_start[4];
	int th1=bapply_start[6];
	int top=bapply_start[8]; //top of spine

	//describes the 4-link zigzag at the tip of the tape:
	int	s2=bapply_start[m_nNumsegs-3];
	int	s3=bapply_start[m_nNumsegs-2];
	
	
	if (!m_bIncludeback) {
		//standard part (if spine is not being calculated) 
		for (i=0;i<m_nNumsegs-4;i++) {
			//sum bend & twist data over collection region
			double sumbend=0.0, sumtwist=0.0;
			for (int j=bcollect_start[i];j<=bcollect_end[i];j++) sumbend+=bend[j]; 
			for (j=tcollect_start[i];j<=tcollect_end[i];j++) sumtwist+=twist[j];
			//compute average values
			double dAvgbend = sumbend / (bapply_end[i]-bapply_start[i]+1);
			double dAvgtwist = sumtwist / (tapply_end[i]-tapply_start[i]+1);
			//apply average values to application region
			for (j=bapply_start[i];j<=bapply_end[i];j++) {
				m_avgbend[m_avgindex][j]=dAvgbend;
			}
			for (j=tapply_start[i];j<=tapply_end[i];j++) {
				m_avgtwist[m_avgindex][j]=dAvgtwist;
			}
		}
	}
	else {
		//head tape is also being used to compute spine info
		//First, collect and apply in the standard way (TWIST ONLY):
		for (i=0;i<m_nNumsegs-4;i++) {
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
		if (!m_bSpinecaptured) {
			
			//model the back with bend offsets:
			//lumbar bend offset:
			int blen=mid-base+1;
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
			thorang=-20*pi/180;
			seg_bendoffset=thorang/blen;
			for (i=th1+1;i<=top;i++) m_bendoffsets[i]=seg_bendoffset;
			//m_bendoffsets[top]=25*pi/180; //orientation correction to make person lie flat in bed
		}
				
		//apply bends to entire back in shapetape fashion (no links):
		for (i=base;i<=top;i++) m_avgbend[m_avgindex][i]=bend[i];

		//coil up the base:
		//coil up the null link and turn off its twist:
		double jointbend=-40*pi;
		if (first_active>0) {
			double seg_bendoffset=jointbend/first_active;
			for (i=0;i<first_active;i++) {
				m_avgbend[m_avgindex][i]=seg_bendoffset;
				m_avgtwist[m_avgindex][i]=0.0;
			}
		}
	} 

	
	//fore-aft bend of head:
	double head_bend=0.0;
	int numPoints = interval*num_region+1;
	for (i=numPoints-10;i<numPoints;i++) head_bend+=bend[i];
	//twist of neck:
	double twist_secondlast = 0.0;
	double twist_last = 0.0;
	for (i=numPoints-10;i<numPoints-5;i++) twist_secondlast+=twist[i];
	for (i=numPoints-5;i<numPoints;i++) twist_last+=twist[i];
	//left-right sway of head:
	double head_sway=-twist_last+twist_secondlast;
	double head_twist = twist_last+twist_secondlast;
	
	m_avgtwist[m_avgindex][s3]=head_sway;
	m_avgbend[m_avgindex][s2-1]=head_bend;
	m_avgtwist[m_avgindex][s2-1]=head_twist;
		
	//add on offsets
	for (i=0;i<numVertices;i++) {
		m_avgbend[m_avgindex][i]+=m_bendoffsets[i];
		m_avgtwist[m_avgindex][i]+=m_twistoffsets[i];
	}
	AverageBendTwist(numToAvg);

	//set spine joint readouts
	double lumbar_bend=0.0, lumbar_twist=0.0, thor_bend=0.0, thor_twist=0.0;
	for (i=base;i<mid;i++) {
		lumbar_bend+=m_recalc_bend[i];
		lumbar_twist+=m_recalc_twist[i];
	}
	for (i=mid;i<top;i++) {
		thor_bend+=m_recalc_bend[i];
		thor_twist+=m_recalc_twist[i];
	}

	//set joint angles
	m_jbangles[0]=lumbar_bend+thor_bend; //total back bend
	m_jbangles[1]=m_recalc_bend[s2-1]; //head bend
	m_jbangles[2]=m_recalc_twist[s2-1]; //head twist
	m_jbangles[3]=m_recalc_twist[s3]; //head sway
	m_jbangles[4]=lumbar_bend; //lumbar bend
	m_jbangles[5]=thor_bend; //thoracic bend

	m_jtangles[0]=0.0;
	for (i=bapply_start[0];i<s2-1;i++) m_jtangles[0]+=m_recalc_twist[i];
	m_jtangles[1]=m_recalc_twist[s2-1];
	m_jtangles[2]=m_recalc_twist[s3];
	m_jtangles[3]=lumbar_twist;
	m_jtangles[4]=thor_twist;

	//convert to degrees
	const double dDegtoRad = 57.2958;
	for (i=0;i<6;i++) {
		m_jbangles[i]*=dDegtoRad;
		m_jtangles[i]*=dDegtoRad;
	}
}

/*
Name:		AssignCapturedOffsets
Purpose:	Assigns values to the link-model bend offsets (m_bendoffsets) based on the spine 
			capture that was most recently performed.
Accepts:	void	
Returns:	void
*/
void Head::AssignCapturedOffsets() {
	//zero bend and twist offsets
	int first_active=bapply_start[0];
	int neck=bapply_start[9]-1;
	for (int i=0;i<=neck;i++) {
		m_bendoffsets[i]=0.0;
		m_twistoffsets[i]=0.0;
	}
	for (i=first_active;i<=neck;i++) m_bendoffsets[i]=m_defSpine[i/interval]/interval;
}

/*
Name:		GetThoracicIndex
Purpose:	Gets the zero-based index of the tape where the thoracic section of the spine should
			be located. Typically useful for someone who wants to estimate chest orientation based
			on the thoracic section of the head (spine) tape.
Accepts:	void	
Returns:	Zero-based index along tape where the thoracic section of the spine should be located.
*/
int Head::GetThoracicIndex() {
	//returns the index of the back tape that is used for computing torso orientation
	return bapply_start[6];
}


/*
Name:		GetHeadData
Purpose:	This is the general purpose function that most users will call to get head data for the 
			tape. It takes care of acquiring the serial data, and performing all of the necessary
			calculations before filling the headinfo structure and returning a pointer to this 
			structure.
Accepts:	numToAvg = number of consecutive samples to average. Default is 1.
Returns:	pointer to headinfo member variable structure
*/
HEADINFO * Head::GetHeadData(int numToAvg/*=1*/) {
	double temp_bend[MAXREGIONS], temp_twist[MAXREGIONS];
	double bend[MAXSUBREGIONS], twist[MAXSUBREGIONS];
	getCurvatureData(temp_bend,temp_twist);
	DistribSensorData(temp_bend,bend);
	DistribSensorData(temp_twist,twist);
	ReCalculateBendTwist(bend,twist,numToAvg);
	FindLimbAnglesUsingBT();
	return &headinfo;
}

/*
Name:		GetBackData
Purpose:	This is the function that users would call to get only back data from the head (spine)
			tape. It takes care of acquiring the serial data, and performing all of the necessary
			spine tape calculations.
Accepts:	r = two dimensional double array of ShapeTape positions. Array must be num_regions * 
				interval + 1 long by 3 wide.
			roll = double array of ShapeTape roll angles (in deg) along length. Array must be 
					num_regions *interval + 1 long.
			pitch = double array of ShapeTape pitch angles (in deg) along length. Array must be 
					num_regions *interval + 1 long.
			yaw = double array of ShapeTape yaw angles (in deg) along length. Array must be 
					num_regions *interval + 1 long.
			numToAvg = number of consecutive samples to average. Default is 1.
Returns:	void (r, roll, pitch, yaw arrays are filled).
*/
int Head::GetSpineData(double r[][3], double roll[], double pitch[], double yaw[], int numToAvg/*=1*/) {
	//gets positions and orientations of interpolated spine segments
	double temp_bend[MAXREGIONS], temp_twist[MAXREGIONS];
	double bend[MAXSUBREGIONS], twist[MAXSUBREGIONS];
	m_bIncludeback=true;
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
	return bapply_start[8]; //index defining the end of the spine part of the tape
}

/*
Name:		GetSpineAndHeadData
Purpose:	This function is used to acquire both head and spine data from the head (spine) 
			ShapeTape. It finds the orientation of the head (returns three Euler angles in the
			headinfo structure) and finds the interpolated positions and Euler angle orientations
			of all of the interpolated points along the spine.
Accepts:	r = two dimensional double array of ShapeTape positions. Array must be num_regions * 
				interval + 1 long by 3 wide.
			roll = double array of ShapeTape roll angles (in deg) along length. Array must be 
					num_regions *interval + 1 long.
			pitch = double array of ShapeTape pitch angles (in deg) along length. Array must be 
					num_regions *interval + 1 long.
			yaw = double array of ShapeTape yaw angles (in deg) along length. Array must be 
					num_regions *interval + 1 long.
			hi = structure of type HEADINFO which receives the three Euler angles corresponding to the
			orientation of the head.
			numToAvg = number of consecutive samples to average. Default is 1.
Returns:	void (r, roll, pitch, yaw arrays are filled, and hi structure is filled).
*/
int Head::GetSpineAndHeadData(double r[][3], double roll[], double pitch[], double yaw[], HEADINFO *hi, 
						int numToAvg/*=1*/) {
	//gets positions and orientations of interpolated spine segments
	double temp_bend[MAXREGIONS], temp_twist[MAXREGIONS];
	double bend[MAXSUBREGIONS], twist[MAXSUBREGIONS];
	m_bIncludeback=true;
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
	hi->dHeadpitch=headinfo.dHeadpitch;
	hi->dHeadroll=headinfo.dHeadroll;
	hi->dHeadyaw=headinfo.dHeadyaw;
	return bapply_start[8]; //index defining the end of the spine part of the tape
}

/*
Name:		GetHeadConfig
Purpose:	Gets all of the configuration parameters (stored in the .mst and .lin files)
			for this head tape.
Accepts:	void
Returns:	true if the configuration for the head tape was valid, false otherwise.
*/
bool Head::GetHeadConfig() {
	if (!GetBodyConfig()) return false;
	if (m_nTapetype!=HEADTAPE) return false;
	filedata mstfile(m_configfile);
	if (mstfile.getFileLength(m_configfile)<10) return false;//invalid configuration file
	char *linfilename=NULL;
	mstfile.getString("[settings]","linfile",&linfilename);
	if (!linfilename) return false;
	SetCurrentDirectory(m_rootfolder);
	filedata linfile(linfilename);
	if (linfile.getFileLength(linfilename)<10) return false; //invalid link file
	
	//get other link file settings
	linfile.closeDataFile();
	if (!LoadLinFile(linfilename)) {
		delete []linfilename;
		return false;
	}
	delete []linfilename;

	//load head parameters from file
	m_bIncludeback=(bool)mstfile.getInteger("[head]","includeback");
	mstfile.getDouble("[head]","spinepose",24,m_defSpine);
	//check to see if the spine pose has been previously captured
	double dSum=0.0;
	for (int i=0;i<24;i++) dSum+=m_defSpine[i];
	if (dSum!=0.0) {
		m_bSpinecaptured=true;
		AssignCapturedOffsets();
	}
	else m_bSpinecaptured=false;
	return true;
}

/*
Name:		ComputeOffsets
Purpose:	Determines the rotational offsets for the head tape so that it shows up 
			properly in the homing pose (person standing with head straight, eyes facing forwards).
Accepts:	void
Returns:	void
*/
void Head::ComputeOffsets() {
	ZeroTransforms();
	ResetAverages();
	GetHeadData(1);
	BodyPart::ComputeOffsets();
}

/*
Name:		CaptureSpine
Purpose:	Captures the shape of the ShapeTape on the wearer's back. The region of interest that is
			captured lies between the base of the tape and the neck. This is a critical
			region for accurate back measurements, which is why it is recommended that this procedure
			be performed once for each different person wearing the head (spine) tape. It is necessary 
			that the ShapeTape first be placed on a flat surface and offsets collected before performing 
			this procedure. It is also necessary that the wearer stand up straight when performing this 
			procedure. Once the procedure is completed, tape offsets should be collected again in the 
			same standing up straight pose.
Accepts:	void
Returns:	void
*/
void Head::CaptureSpine() {
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
	mstfile.writeData("[head]","spinepose",MAXREGIONS,m_defSpine);
}

/*
Name:		UpdateLinkFile
Purpose:	Updates the link file for a head tape, based on the contents of a given subject file.
			Subject files can be created using Measurand's ShapeRecorder software.
Accepts:	subjectFile = null terminated string specifying the filename of the subject file.
			nClampindex = the index of the head / spine tape sensing region which coincides with the 
						  bottom of the person's lumbar.
Returns:	true if link file was updated successfully, false otherwise
*/
bool Head::UpdateLinkFile(char *subjectFile, int &nClampindex) {
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
	double tape_bonelength_mm[13];
	linfile.getDouble("bonelabel_string","tape_bonelength_mm",13,tape_bonelength_mm);
	double dBacktapelength = back_ratio * dBacklength;
	double dRegionlength = region_length[0]; //assume tape to be constructed from regions of uniform length
	int nHeadregions = 0;
	//need to have a total head length (including neck) of 300 mm
	double dHeadlength = 300.0;
	while (nHeadregions*dRegionlength<dHeadlength) nHeadregions++;
	dHeadlength=nHeadregions*dRegionlength;
	
	int nBackregions = 0;
	while (nBackregions*dRegionlength<dBacktapelength) nBackregions++;
	dBacktapelength = nBackregions*dRegionlength;
	
	//check to make sure that there is enough tape:
	double dRequired_tapelength = dBacktapelength + dHeadlength;
	if (dRequired_tapelength>tape_length) {
		//uh oh... not enough tape!
		return false;
	}
	
	//fill in tape_bonelength_mm
	tape_bonelength_mm[0] = tape_length - dRequired_tapelength;
	tape_bonelength_mm[1] = dBacktapelength/11.0; //four lumbar regions
	tape_bonelength_mm[2] = dBacktapelength/11.0; 
	tape_bonelength_mm[3] = dBacktapelength/11.0; 
	tape_bonelength_mm[4] = dBacktapelength/11.0; 
	tape_bonelength_mm[5] = dBacktapelength*3.0/22.0; //four plus one thoracic regions
	tape_bonelength_mm[6] = dBacktapelength*3.0/22.0; 
	tape_bonelength_mm[7] = dBacktapelength*3.0/22.0;
	tape_bonelength_mm[8] = dBacktapelength*3.0/22.0;
	tape_bonelength_mm[9] = dBacktapelength/11.0; 
	tape_bonelength_mm[10] = dHeadlength/6.0; //2 neck regions
	tape_bonelength_mm[11] = dHeadlength/2.0;
	tape_bonelength_mm[12] = dHeadlength/3.0; //1 head region
		
		
	//write changes to link file
	linfile.writeData("bonelabel_string","tape_bonelength_mm",13,tape_bonelength_mm);
	//exit if 'include back' option is not enabled
	if (!m_bIncludeback) return true;
	//send message to user to let them know where to position tape on back
	
	nClampindex = num_sensors/2 - nHeadregions-nBackregions;
	
	
	if (nClampindex < 0) nClampindex = 0;
		
	linfile.closeDataFile();
	//reload link file with new settings
	LoadLinFile(linkfilename);
	delete []linkfilename;
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