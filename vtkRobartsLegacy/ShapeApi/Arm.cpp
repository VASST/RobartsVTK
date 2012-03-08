// Arm.cpp: implementation of the Arm class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Arm.h"
#include "filedata.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

/*
Name:		Arm
Purpose:	constructor - this constructor builds an Arm object with settings defined in the 
			Measurand ShapeTape file: configfile
Accepts:	config_file = Measurand ShapeTape (.mst) file that contains all of the settings 
			for this particular ShapeTape. 
			
Returns:    void
*/
Arm::Arm(char *configfile, char *rootfolder):BodyPart(configfile,rootfolder)
{
	//default values (in cm) for typical arm dimensions
	m_dShoulderwidth=400;
	m_dShoulderheight=500;
	m_dUpperarmlength=400;
	m_dForearmlength=400;
	for (int i=0;i<3;i++) m_dShoulderoffsets[i]=0;
	for (i=0;i<MAXREGIONS;i++) {
		m_defShoulderBend[i]=0.0;
		m_defShoulderTwist[i]=0.0;
	}
	m_bUsecapturedshoulder=false;
	GetArmConfig();
}

/*
Name:		~Arm
Purpose:	destructor - the Arm object gets destroyed here.
Accepts:	void
			
Returns:    void
*/
Arm::~Arm()
{
	
}

/*
Name:		CalculateShoulderOffsets
Purpose:	Figures out what the shoulder offsets should be based on dimensions defined in the
			.mst file, the .lin file, and the current position of the shoulder.
Accepts:	shoulder = vector describing the current xyz position of the shoulder without any 
			offsets added to it.
Returns:	void (m_dShoulderoffsets member variable array is set by function).
*/
void Arm::CalculateShoulderOffsets(double *shoulder) {
	//compute desired shoulder height based on bone dimensions
	m_dShoulderheight = 500*m_anat_bonelengths[0]/350.0;
	m_dShoulderoffsets[0]=-shoulder[0]; //x coordinate of shoulder should be zero
	m_dShoulderoffsets[1]=m_dShoulderheight - shoulder[1]; //y coordinate of shoulder 
														//should be m_dShoulderheight
	if (m_nTapetype==0) m_dShoulderoffsets[2] = m_dShoulderwidth/2 - shoulder[2]; //right arm
	else m_dShoulderoffsets[2] = -m_dShoulderwidth/2 - shoulder[2]; //left arm
	
	//save to file
	filedata mstfile(m_configfile);
	mstfile.writeData("[arm]","shldr_offsets2",3,m_dShoulderoffsets);
}

/*
Name:		FindLimbAnglesUsingBT
Purpose:	Finds all of the arm's limb angles and joint positions, based on the recalculated bend
			and twist of the tape. Fills the arminfo structure with all of the arm's limb
			segment information.
Accepts:	void
Returns:	void  (arminfo member variable structure is updated with new values by this function).
*/
void Arm::FindLimbAnglesUsingBT() {
	const double dRadtoDeg = 57.2957795;
	BodyPart::FindLimbAnglesUsingBT();
	
	//add on shoulder offsets
	for (int i=0;i<3;i++) {
			for (int j=0;j<MAXJOINTS;j++) {
				m_jointpos[j][i]+=m_dShoulderoffsets[i];
			}
			arminfo.dElbowposition[i] = m_jointpos[2][i];
	}

	arminfo.dUpperarm_roll=m_bone_orient[4][0]*dRadtoDeg;
	arminfo.dUpperarm_pitch=m_bone_orient[4][1]*dRadtoDeg;
	arminfo.dUpperarm_yaw=m_bone_orient[4][2]*dRadtoDeg;
	arminfo.dForearm_roll=m_bone_orient[5][0]*dRadtoDeg;
	arminfo.dForearm_pitch=m_bone_orient[5][1]*dRadtoDeg;
	arminfo.dForearm_yaw=m_bone_orient[5][2]*dRadtoDeg;
	arminfo.dHand_roll=m_bone_orient[7][0]*dRadtoDeg;
	arminfo.dHand_pitch=m_bone_orient[7][1]*dRadtoDeg;
	arminfo.dHand_yaw=m_bone_orient[7][2]*dRadtoDeg;
	arminfo.dFinger_roll=m_bone_orient[8][0]*dRadtoDeg;
	arminfo.dFinger_pitch=m_bone_orient[8][1]*dRadtoDeg;
	arminfo.dFinger_yaw=m_bone_orient[8][2]*dRadtoDeg;
	
	//find shoulder, wrist, and knuckle positions based on orientations of upper-arm,forearm, and hand respectively
	//shoulder
	double pos[3];
	for (i=0;i<3;i++) pos[i]=arminfo.dElbowposition[i];
	GetPosFromOrientation(pos,arminfo.dUpperarm_yaw,arminfo.dUpperarm_pitch,-m_anat_bonelengths[1]);
	for (i=0;i<3;i++) arminfo.dShoulderposition[i]=pos[i];

	//wrist
	for (i=0;i<3;i++) pos[i]=arminfo.dElbowposition[i];
	GetPosFromOrientation(pos,arminfo.dForearm_yaw,arminfo.dForearm_pitch,m_anat_bonelengths[2]);
	for (i=0;i<3;i++) arminfo.dWrist[i]=pos[i];
	//knuckle
	GetPosFromOrientation(pos,arminfo.dHand_yaw,arminfo.dHand_pitch,m_anat_bonelengths[3]);
	for (i=0;i<3;i++) arminfo.dKnuckleposition[i]=pos[i];
}


/*
Name:		CalculateShoulderOffsets
Purpose:	Finds the correct shoulder offsets for this arm tape. First resets offsets to
			zero, resets data averages, and then calculates all of the data for the arm. Then
			passes the current shoulder position to the private CalculateShoulderOffsets function
			which in turn fills the m_dShoulderoffsets member variable array.
Accepts:	void
Returns:	void  
*/
void Arm::CalculateShoulderOffsets()
{
	double shoulder[3];
	m_dShoulderoffsets[0]=0.0; m_dShoulderoffsets[1]=0.0; m_dShoulderoffsets[2]=0.0;
	ResetAverages();
	GetArmData(1);
	shoulder[0]=arminfo.dShoulderposition[0];
	shoulder[1]=arminfo.dShoulderposition[1];
	shoulder[2]=arminfo.dShoulderposition[2];
	CalculateShoulderOffsets(shoulder);
}
/*
Name:		ReCalculateBendTwist
Purpose:	Takes the normally calculated bend and twist for the ShapeTape, and re-calculates
			(or re-distributes it) based on the assumptions of the arm link-model.
Accepts:	bend = the normally calculated ShapeTape bend for this tape (array of num_regions * 
					interval + 1 points).
			twist = the normally calculated ShapeTape twist for this tape (array of num_regions * 
					interval + 1 points).
			numToAvg = number of consecutive samples to average. Default is 1.
Returns:	void  
*/
void Arm::ReCalculateBendTwist(double bend[], double twist[], int numToAvg/*=1*/) {
	BodyPart::ReCalculateBendTwist(bend,twist);
	//These make the special rules section easier to read:
	int first_active=bapply_start[0];
	int shoulder1=bapply_start[1];
	int shoulder2=bapply_start[2];
	int shoulder3=bapply_start[3];
	int uarm1=bapply_start[4];
	int uarm2=uarm1+5;

	int elbow=bapply_start[5];
	int wrist1=bapply_start[6];
	int wrist2=bapply_start[7];
	int finger=bapply_start[8];

	const int slen=2; //constant used for finding key points along the tape
	int numVertices = interval*num_region+1;

	//set bend / twist to zero
	for (int i=0;i<numVertices;i++) {
		m_recalc_bend[i]=0;
		m_recalc_twist[i]=0;
		m_avgbend[m_avgindex][i]=0;
		m_avgtwist[m_avgindex][i]=0;
	}
	//collect and apply normally 
	for (i=5;i<MAXITEMS;i++) {
		if (m_tape_bonelengths[i]==0) continue;
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
	
	const double dPI = 3.14159265359;
	if (!m_bUsecapturedshoulder) {
		//cancel bend offsets coming from linksegs, except for wrist and hand
		for (i=first_active;i<=elbow;i++) m_bendoffsets[i]=0.0;

		//sharper, earlier shoulder bend; overbend at shoulder, cancelled by underbend at elbow;
		if (m_nTapetype==RIGHTARM) {
			for (i=shoulder1-1;i<=shoulder1;i++) m_bendoffsets[i]=-dPI*.3;
			m_bendoffsets[elbow-1]=dPI*.1;
			m_bendoffsets[uarm1]=-.5*dPI;
			m_bendoffsets[uarm1+1]=.5*dPI;
		}
		else {
			for (i=shoulder1-1;i<=shoulder1;i++) m_bendoffsets[i]=dPI*.3;
			m_bendoffsets[elbow-1]=-dPI*.1;
			m_bendoffsets[uarm1]=.5*dPI;
			m_bendoffsets[uarm1+1]=-.5*dPI;
		}
	}

	//apply twist and bend in shapetape fashion up to the wrist
	for (i=first_active;i<=wrist1-slen;i++) {
		m_avgtwist[m_avgindex][i]=twist[i];
		m_avgbend[m_avgindex][i]=bend[i];
	}

			
	//Hand/finger
	//The hand has two sublinks: boffset=-90, and boffset=90.
	//The first of these is the non-axial sublink; Twist applied to it
	//will cause the second sublink to move in ad/ab fashion (in plane of tape);
	//So concentrate the twist
	//on the non_axial sublink:
	//%First, delete twist and bend from hand/finger:
	int vertseg=m_nNumsegs-3; //the non_axial subseg
	int fingend=m_nNumsegs-1;
	for (i=vertseg;i<=fingend;i++) {
		for (int j=bapply_start[i];j<=bapply_end[i];j++) m_avgbend[m_avgindex][j]=0.0;
		for (j=tcollect_start[i];j<=tcollect_end[i];j++) m_avgtwist[m_avgindex][j]=0.0;
	}

	//collect from the finger only; any more, and ad/ab gets polluted by forearm rots:
	double fingertwist=0.0;
	for (i=tcollect_start[fingend-1];i<=tcollect_end[fingend];i++) fingertwist+=twist[i];
	
	//minus sign added so it goes the correct way; Alternatively, one could have 
	//a vertical subseg with the opposite bend, but I wanted a -90 vertseg so the
	//hand would look like it 'steps down' when laid flat on a table:
	m_avgtwist[m_avgindex][bapply_start[vertseg+1]]=-fingertwist;

	//get wrist bend
	double wrist_bend=0.0;
	for (i=tcollect_start[vertseg]-1;i<=tcollect_start[fingend]-3;i++) wrist_bend+=bend[i];
	m_avgbend[m_avgindex][bapply_start[vertseg]]=wrist_bend;

	//get finger bend
	double finger_bend=0.0;
	for (i=tcollect_start[fingend]-2;i<=bcollect_end[fingend];i++) finger_bend+=bend[i];
	m_avgbend[m_avgindex][bapply_start[fingend]]=finger_bend;
	
	
	//add on offsets
	for (i=0;i<numVertices;i++) {
		m_avgbend[m_avgindex][i]+=m_bendoffsets[i];
		m_avgtwist[m_avgindex][i]+=m_twistoffsets[i];
	}
	
	AverageBendTwist(numToAvg);

	//set joint readouts
	m_jbangles[0]=0.0;
	for (i=first_active;i<=elbow-1;i++) m_jbangles[0]+=m_recalc_bend[i];
	double elbow_bend=0.0;
	for (i=elbow;i<=wrist1-slen;i++) elbow_bend+=m_recalc_bend[i];
	m_jbangles[1]=elbow_bend;
	m_jbangles[2]=0.0;
	for (i=wrist1;i<=wrist2;i++) m_jbangles[2]+=m_recalc_bend[i];
	m_jbangles[3]=m_recalc_bend[finger];

	for (i=0;i<4;i++) m_jtangles[i]=0.0;
	for (i=first_active;i<shoulder2;i++) m_jtangles[0]+=m_recalc_twist[i];
	for (i=shoulder2;i<elbow;i++) m_jtangles[1]+=m_recalc_twist[i];
	for (i=elbow;i<wrist1;i++) m_jtangles[2]+=m_recalc_twist[i];
	for (i=wrist1;i<numVertices;i++) m_jtangles[3]+=m_recalc_twist[i];
	//convert to degrees
	const double dRadtoDeg = 57.2958;
	for (i=0;i<4;i++) {
		m_jbangles[i]*=dRadtoDeg;
		m_jtangles[i]*=dRadtoDeg;
	}
}

/*
Name:		AssignCapturedOffsets
Purpose:	Assigns values to the link-model bend offsets (m_bendoffsets) based on the shoulder 
			capture that was most recently performed.
Accepts:	void	
Returns:	void
*/
void Arm::AssignCapturedOffsets() {
	//zero all bend and twist offsets
	int numVertices = num_region*interval+1;
	for (int i=0;i<numVertices;i++) {
		m_bendoffsets[i]=0.0;
		m_twistoffsets[i]=0.0;
	}
	int first_active=bapply_start[0];
	int elbow=bapply_start[5];
	for (i=first_active;i<=elbow;i++) {
		m_bendoffsets[i]=m_defShoulderBend[i/interval]/interval;
		m_twistoffsets[i]=m_defShoulderTwist[i/interval]/interval;
	}
}

/*
Name:		GetArmData
Purpose:	This is the general purpose function that most users will call to get arm data for the 
			tape. It takes care of acquiring the serial data, and performing all of the necessary
			calculations before filling the arminfo structure and returning a pointer to this 
			structure.
Accepts:	numToAvg = number of consecutive samples to average. Default is 1.
Returns:	pointer to arminfo member variable structure
*/
ARMINFO *Arm::GetArmData(int numToAvg/*=1*/) {
	double temp_bend[MAXREGIONS], temp_twist[MAXREGIONS];
	double bend[MAXSUBREGIONS], twist[MAXSUBREGIONS];
	getCurvatureData(temp_bend,temp_twist);
	DistribSensorData(temp_bend,bend);
	DistribSensorData(temp_twist,twist);
	ReCalculateBendTwist(bend,twist,numToAvg);
	FindLimbAnglesUsingBT();
	return &arminfo;
}

/*
Name:		CaptureShoulder
Purpose:	Captures the shape of the ShapeTape on the wearer's arm. The region of interest that is
			captured lies between the base of the tape and just before the elbow. This is a critical
			region for accurate arm measurements, which is why it is recommended that this procedure
			be performed once for each different person wearing the arm tape. It is necessary that the
			ShapeTape first be placed on a flat surface and offsets collected before performing this 
			procedure. It is also necessary that the wearer stand up straight with his / her arm 
			pointing horizontally outwards, palm facing down when performing this procedure. Once the 
			procedure is completed, tape offsets should be collected again in the same pose.
Accepts:	void
Returns:	void
*/
void Arm::CaptureShoulder() {
	double temp_bend[MAXREGIONS], temp_twist[MAXREGIONS];
	//initialize
	for (int i=0;i<MAXREGIONS;i++) {
		temp_bend[i]=0.0;
		temp_twist[i]=0.0;
	}
	getCurvatureData(temp_bend,temp_twist);
	for (i=0;i<MAXREGIONS;i++) {
		m_defShoulderBend[i]=temp_bend[i];
		m_defShoulderTwist[i]=temp_twist[i];
	}
	AssignCapturedOffsets();
	m_bUsecapturedshoulder=true;
	//save offsets to file
	filedata mstfile(m_configfile);
	mstfile.writeData("[arm]","shoulderpose",MAXREGIONS,m_defShoulderBend);
	mstfile.writeData("[arm]","shouldertwistpose",MAXREGIONS,m_defShoulderTwist);
}


/*
Name:		GetArmConfig
Purpose:	Gets all of the configuration parameters (stored in the .mst and .lin files)
			for this arm tape.
Accepts:	void
Returns:	true if the configuration for the arm tape was valid, false otherwise.
*/
bool Arm::GetArmConfig() {
	if (!GetBodyConfig()) return false;
	if (m_nTapetype!=LEFTARM&&m_nTapetype!=RIGHTARM) return false; //not an arm tape
	filedata mstfile(m_configfile);
	if (mstfile.getFileLength(m_configfile)<10) return false;//invalid configuration file
	char *linfilename=NULL;
	mstfile.getString("[settings]","linfile",&linfilename);
	if (!linfilename) return false;
	SetCurrentDirectory(m_rootfolder);
	filedata linfile(linfilename);
	if (linfile.getFileLength(linfilename)<10) return false; //invalid link file
	mstfile.getDouble("[settings]","shoulder_offsets",3,m_dShoulderoffsets);
	m_dShoulderwidth = mstfile.getDouble("[arm]","shoulderwidth");
	double bonelengths[MAXBONES];
	linfile.getDouble("bonelabel_string","anat_bonelength_mm",MAXBONES,bonelengths);
	m_dShoulderheight = 500*bonelengths[0]/350.0;
	m_dUpperarmlength = bonelengths[1];
	m_dForearmlength = bonelengths[2];
	
	//get other link file settings
	linfile.closeDataFile();
	if (!LoadLinFile(linfilename)) {
		delete []linfilename;
		return false;
	}
	delete []linfilename;

	mstfile.getDouble("[arm]","shoulderpose",MAXREGIONS,m_defShoulderBend);
	mstfile.getDouble("[arm]","shouldertwistpose",MAXREGIONS,m_defShoulderTwist);
	//check to see if the shoulder pose has been previously captured
	double dSum=0.0;
	for (int i=0;i<MAXREGIONS;i++) {
		if (m_defShoulderBend[i]!=0.0||m_defShoulderTwist[i]!=0.0) {
			m_bUsecapturedshoulder=true;
			AssignCapturedOffsets();
			return true;
		}
	}
	m_bUsecapturedshoulder=false;
	return true;
}

/*
Name:		ComputeOffsets
Purpose:	Determines the positional and rotational offsets for the arm tape so that it shows up 
			properly in the homing pose (person standing up straight, arm facing horizontally 
			outwards in front of the chest, palm facing down).
Accepts:	void
Returns:	void
*/
void Arm::ComputeOffsets() {
	ZeroTransforms();
	ResetAverages();
	GetArmData(1);
	BodyPart::ComputeOffsets();
	CalculateShoulderOffsets();
}

/*
Name:		GetShoulderOffsets
Purpose:	Gets the current shoulder offsets for this arm tape.
Accepts:	shoulder = 3 element double array corresponding to the xyz components of the shoulder
					 offsets.
Returns:	void
*/
void Arm::GetShoulderOffsets(double *shoulder) {
	shoulder[0]=m_dShoulderoffsets[0];
	shoulder[1]=m_dShoulderoffsets[1];
	shoulder[2]=m_dShoulderoffsets[2];
}

/*
Name:		UpdateLinkFile
Purpose:	Updates the link file for an arm tape, based on the info contained within the subject file.
Accepts:	subjectFile = null terminated string specifying the subject file to be used. (Subject files
						  can be created using Measurand's ShapeRecorder software.
			nClampindex = index of tape sensing region where the tape should be clamped in an arm 
						  clamp at the small of the back (set by this function).
Returns:	true if link file was updated successfully, false otherwise.
*/
bool Arm::UpdateLinkFile(char *subjectFile, int &nClampindex) {
	filedata subfile(subjectFile);
	nClampindex=-1;	
	//tape to bonelength ratios for back, upperarm, forearm, and hand
	const double back_ratio=1.00, upperarm_ratio=1.10, forearm_ratio=1.10, hand_ratio=1.00;
	double dBacklength = subfile.getDouble("[linklengths]","backlength");	
	double dArmtapeonback = subfile.getDouble("[linklengths]","armtapeonback");
	double dHandlength = subfile.getDouble("[linklengths]","handlength");
	double dFingerlength = subfile.getDouble("[linklengths]","fingerlength");
	if (dHandlength==0.0) dHandlength=115.0; //default value
	if (dFingerlength==0.0) dFingerlength=115.0; //default value
	if (dArmtapeonback==0.0) {
		//probably an old-style subject file
		double dBackpackheight = subfile.getDouble("[linklengths]","backpackheight");
		double dClamplength = subfile.getDouble("[linklengths]","clamplength");
		dArmtapeonback = dBacklength - dBackpackheight - dClamplength;
	}
	double dUpperarmlength = subfile.getDouble("[linklengths]","upperarmlength");
	double dForearmlength = subfile.getDouble("[linklengths]","forearmlength");
		
	//get shoulder capture info from subject file
	double dTempbend[24],dTemptwist[24];
	if (m_nTapetype==RIGHTARM) {
		subfile.getDouble("[shldr_capture]","rightarm",24,dTempbend);
		subfile.getDouble("[shldr_capture]","ratwist",24,dTemptwist);
	}
	else {//left arm
		subfile.getDouble("[shldr_capture]","leftarm",24,dTempbend);
		subfile.getDouble("[shldr_capture]","latwist",24,dTemptwist);
	}
	//check to see if there is any valid shoulder pose data
	bool bValidshoulderpose = false;
	for (int i=0;i<24;i++) {
		if (dTempbend[i]!=0.0||dTemptwist[i]!=0.0) {
			bValidshoulderpose = true;
			break;
		}
	}

	//get shoulder separation from subject file
	double dTempshldrsep = subfile.getDouble("[linklengths]","shouldersep");
	if (dTempshldrsep>0.0) m_dShoulderwidth = dTempshldrsep;
	filedata mstfile(m_configfile);
	//get name of link file
	char *linkfilename = NULL;
	mstfile.getString("[settings]","linfile",&linkfilename);
	//save shoulder pose info to mst file 
	if (bValidshoulderpose) {
		mstfile.writeData("[arm]","shoulderpose",24,dTempbend);
		mstfile.writeData("[arm]","shouldertwistpose",24,dTemptwist);
	}
	//save new shoulder height
	m_dShoulderheight = dBacklength*.8;
	mstfile.writeData("[arm]","shoulderheight",m_dShoulderheight);
	if (!linkfilename) return false; //no linkfile to update!
	filedata linfile(linkfilename);
	double tape_bonelength_mm[10];
	double anat_bonelength_mm[5];
	linfile.getDouble("bonelabel_string","tape_bonelength_mm",10,tape_bonelength_mm);
	linfile.getDouble("bonelabel_string","anat_bonelength_mm",5,anat_bonelength_mm);	
	anat_bonelength_mm[0] = dArmtapeonback;
	anat_bonelength_mm[1] = dUpperarmlength*.65;
	anat_bonelength_mm[2] = dForearmlength;
	anat_bonelength_mm[3] = dHandlength;
	anat_bonelength_mm[4] = dFingerlength;
	m_dUpperarmlength=anat_bonelength_mm[1];
	m_dForearmlength=anat_bonelength_mm[2];
	double dBacktapelength = back_ratio * dArmtapeonback;
	double dUpperarmtapelength = upperarm_ratio * dUpperarmlength;
	double dForearmtapelength = forearm_ratio * dForearmlength;
	double dRegionlength = region_length[0]; //assume tape to be constructed
																		//from regions of uniform length
	
	//known quantities for tape_bonelength_mm:
	tape_bonelength_mm[9]=dFingerlength; //finger
	tape_bonelength_mm[8]=dHandlength/2; //2nd part of hand (closest to finger)
	tape_bonelength_mm[7]=dHandlength/2; //1st part of hand (closest to wrist)

	double dHandtapelength = tape_bonelength_mm[9] + tape_bonelength_mm[8] + tape_bonelength_mm[7];

	//check to make sure that there is enough tape:
	double dRequired_tapelength = dBacktapelength + dUpperarmtapelength + dForearmtapelength + 
		dHandtapelength;
		
	if (dRequired_tapelength>tape_length) {
		//uh oh... not enough tape!
		return false;
	}
	
	//try to end null region on a sensor boundary
	dRequired_tapelength = dBacktapelength + dUpperarmtapelength + dForearmtapelength + 
		dHandtapelength;
	int nNum_armsensors=0;
	while (nNum_armsensors*dRegionlength<dRequired_tapelength) nNum_armsensors++;
	//add any extra length onto back region
	dBacktapelength+=nNum_armsensors*dRegionlength - dRequired_tapelength;
	if (dBacktapelength<0.0) dBacktapelength=0.0; //make sure back tape length is not negative
	//recheck to make sure that there is still enough tape length available
	dRequired_tapelength = dBacktapelength + dUpperarmtapelength + dForearmtapelength + 
		dHandtapelength;
	if (dRequired_tapelength>tape_length) {
		//uh oh... not enough tape!
		return false;
	}
	//fill in remaining parts of tape_bonelength_mm
	tape_bonelength_mm[0] = tape_length - dRequired_tapelength;
	tape_bonelength_mm[1] = dBacktapelength*.85; //two back regions
	tape_bonelength_mm[2] = dBacktapelength*.15; 
	tape_bonelength_mm[3] = dUpperarmtapelength*.3; //3 upperarm regions
	tape_bonelength_mm[4] = dUpperarmtapelength*.15;
	tape_bonelength_mm[5] = dUpperarmtapelength*.55;
	tape_bonelength_mm[6] = dForearmtapelength;
		
	//write changes to link file
	linfile.writeData("bonelabel_string","tape_bonelength_mm",10,tape_bonelength_mm);
	linfile.writeData("bonelabel_string","anat_bonelength_mm",5,anat_bonelength_mm);

	nClampindex = num_sensors/2 - nNum_armsensors;
	if (nClampindex < 0) nClampindex = 0;

	linfile.closeDataFile();
	//reload link file with new settings
	LoadLinFile(linkfilename);
	delete []linkfilename;
	
	if (!bValidshoulderpose&&m_bUsecapturedshoulder) {
		//recalculate shoulder offsets (in case elbow is at different position along tape)
		AssignCapturedOffsets();
	}
	else if (bValidshoulderpose) {
		//copy shoulder offsets
		for (i=0;i<24;i++) {
			m_defShoulderBend[i] = dTempbend[i];
			m_defShoulderTwist[i] = dTemptwist[i];
		}
		m_bUsecapturedshoulder = true;
		AssignCapturedOffsets();
	}
	return true;
}
