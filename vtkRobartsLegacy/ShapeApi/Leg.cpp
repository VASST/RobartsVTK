// Leg.cpp: implementation of the Leg class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Leg.h"
#include "filedata.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

/*
Name:		Leg
Purpose:	constructor - this constructor builds a Leg object with settings defined in the 
			Measurand ShapeTape file: configfile
Accepts:	config_file = Measurand ShapeTape (.mst) file that contains all of the settings 
			for this particular ShapeTape. 
			
Returns:    void
*/
Leg::Leg(char *configfile, char *rootfolder)
	:BodyPart(configfile, rootfolder)
{
	//default values (in mm) for typical leg dimensions
	m_dFootlength=260.0;
	m_dHipwidth=350.0;
	m_dThighlength=440.0;
	m_dShinlength=460.0;
	m_dThighwidth=80.0;
	m_dFootlength=140.0;
	m_dToelength=60.0;
	m_dLegseparation=180.0;
	m_hippos[0]=m_hippos[1]=m_hippos[2]=0.0;
	m_dHipoffsets[0]=m_dHipoffsets[1]=m_dHipoffsets[2]=0.0;
	GetLegConfig();
}

/*
Name:		~Leg
Purpose:	destructor - the Leg object gets destroyed here.
Accepts:	void
			
Returns:    void
*/
Leg::~Leg()
{
	
}

/*
Name:		CalculateHipOffsets
Purpose:	Finds the correct hip offsets for this leg tape. 
Accepts:	void
Returns:	void  
*/
void Leg::CalculateHipOffsets()
{
	//desired location of hip
	double desired_hip[3]={0,0,0};
	if (m_nTapetype==RIGHTLEG) desired_hip[2]=m_dHipwidth/2-m_dThighwidth/2;
	else desired_hip[2]=-m_dHipwidth/2+m_dThighwidth/2;
	for (int i=0;i<3;i++) {
		m_dHipoffsets[i]=desired_hip[i]-m_hippos[i];
	}
	//save hip offsets to file
	filedata mstfile(m_configfile);
	mstfile.writeData("[leg]","hipoffsets",3,m_dHipoffsets);
}

/*
Name:		FindLimbAnglesUsingBT
Purpose:	Finds all of the leg's limb angles and joint positions, based on the recalculated bend
			and twist of the tape. Fills the leginfo structure with all of the leg's limb
			segment information.
Accepts:	void
Returns:	void  (leginfo member variable structure is updated with new values by this function).
*/
void Leg::FindLimbAnglesUsingBT() {
	const double dRadtoDeg = 57.2957795;
	BodyPart::FindLimbAnglesUsingBT();
	//add on hip offsets
	for (int i=0;i<3;i++) {
		for (int j=0;j<MAXJOINTS;j++) {
			m_jointpos[j][i]+=m_dHipoffsets[i];
		}
		m_hippos[i]=m_jointpos[0][i];
	}

	leginfo.dThigh_roll=m_bone_orient[4][0]*dRadtoDeg;
	leginfo.dThigh_pitch=m_bone_orient[4][1]*dRadtoDeg;
	leginfo.dThigh_yaw=m_bone_orient[4][2]*dRadtoDeg;
	leginfo.dShin_roll=m_bone_orient[5][0]*dRadtoDeg;
	leginfo.dShin_pitch=m_bone_orient[5][1]*dRadtoDeg;
	leginfo.dShin_yaw=m_bone_orient[5][2]*dRadtoDeg;
	leginfo.dFoot_roll=m_bone_orient[6][0]*dRadtoDeg;
	leginfo.dFoot_pitch=m_bone_orient[6][1]*dRadtoDeg;
	leginfo.dFoot_yaw=m_bone_orient[6][2]*dRadtoDeg;
	leginfo.dToe_roll=m_bone_orient[7][0]*dRadtoDeg;
	leginfo.dToe_pitch=m_bone_orient[7][1]*dRadtoDeg;
	leginfo.dToe_yaw=m_bone_orient[7][2]*dRadtoDeg;
	for (i=0;i<3;i++) {
		leginfo.dUpperthighposition[i]=m_jointpos[0][i];
		leginfo.dKneeposition[i]=m_jointpos[1][i];
		leginfo.dAnkle[i]=m_jointpos[2][i];
		leginfo.dKnuckle[i]=m_jointpos[3][i];
	}

	//calculate knee, ankle, and knuckle positions based on upper thigh and shin orientations
	//knee
	double pos[3];
	for (i=0;i<3;i++) pos[i]=m_hippos[i];
	GetPosFromOrientation(pos,leginfo.dThigh_yaw,leginfo.dThigh_pitch,m_anat_bonelengths[0]);
	for (i=0;i<3;i++) leginfo.dKneeposition[i]=pos[i];
	//ankle
	GetPosFromOrientation(pos,leginfo.dShin_yaw,leginfo.dShin_pitch,m_anat_bonelengths[1]);
	for (i=0;i<3;i++) leginfo.dAnkle[i]=pos[i];
	//knuckle
	GetPosFromOrientation(pos,leginfo.dFoot_yaw,leginfo.dFoot_pitch,m_anat_bonelengths[2]);
	for (i=0;i<3;i++) leginfo.dKnuckle[i]=pos[i];
}


/*
Name:		ReCalculateBendTwist
Purpose:	Takes the normally calculated bend and twist for the ShapeTape, and re-calculates
			(or re-distributes it) based on the assumptions of the leg link-model.
Accepts:	bend = the normally calculated ShapeTape bend for this tape (array of num_regions * 
					interval + 1 points).
			twist = the normally calculated ShapeTape twist for this tape (array of num_regions * 
					interval + 1 points).
			numToAvg = number of consecutive samples to average. Default is 1.
Returns:	void  
*/
void Leg::ReCalculateBendTwist(double bend[], double twist[], int numToAvg/*=1*/) {
	BodyPart::ReCalculateBendTwist(bend,twist);
	int numVertices = interval*num_region+1;
	//set bend / twist to zero
	for (int i=0;i<numVertices;i++) {
		m_recalc_bend[i]=0;
		m_recalc_twist[i]=0;
		m_avgbend[m_avgindex][i]=0;
		m_avgtwist[m_avgindex][i]=0;
	}
	for (i=0;i<MAXITEMS;i++) {
		//sum bend & twist data over collection region
		if (m_tape_bonelengths[i]==0) continue;
		double sumbend=0.0, sumtwist=0.0;
		for (int j=bcollect_start[i];j<=bcollect_end[i];j++) sumbend+=bend[j]; 
		for (j=tcollect_start[i];j<=tcollect_end[i];j++) sumtwist+=twist[j];
		//compute average values
		double dAvgbend = sumbend / (bapply_end[i]-bapply_start[i]+1);
		double dAvgtwist = sumtwist / (tapply_end[i]-tapply_start[i]+1);
		//apply average values to application region and add on offsets
		for (j=bapply_start[i];j<=bapply_end[i];j++) {
			m_avgbend[m_avgindex][j]=dAvgbend+m_bendoffsets[j];
			//m_recalc_bend[j]=dAvgbend+m_bendoffsets[j];
		}
		for (j=tapply_start[i];j<=tapply_end[i];j++) {
			m_avgtwist[m_avgindex][j]=dAvgtwist+m_twistoffsets[j];
			//m_recalc_twist[j]=dAvgtwist+m_twistoffsets[j];
		}
	}
	AverageBendTwist(numToAvg);
	//joint angles
	m_jbangles[0]=0.0; 
	for (i=0;i<4;i++) m_jtangles[i]=0.0;
	for (i=0;i<=bapply_start[4];i++) m_jbangles[0]+=m_recalc_bend[i];
	for (i=bapply_start[4];i<bapply_start[5];i++) m_jtangles[0]+=m_recalc_twist[i];
	for (i=bapply_start[5];i<bapply_start[6];i++) m_jtangles[1]+=m_recalc_twist[i];
	for (i=bapply_start[6];i<bapply_start[7];i++) m_jtangles[2]+=m_recalc_twist[i];
	for (i=bapply_start[7];i<numVertices;i++) m_jtangles[3]+=m_recalc_twist[i];
	m_jbangles[1]=m_recalc_bend[bapply_start[5]];
	m_jbangles[2]=m_recalc_bend[bapply_start[6]];
	m_jbangles[3]=m_recalc_bend[bapply_start[7]];

	//convert to degrees
	const double dDegtoRad = 57.2958;
	for (i=0;i<4;i++) {
		m_jbangles[i]*=dDegtoRad;
		m_jtangles[i]*=dDegtoRad;
	}
}


/*
Name:		GetLegData
Purpose:	This is the general purpose function that most users will call to get leg data for the 
			tape. It takes care of acquiring the serial data, and performing all of the necessary
			calculations before filling the leginfo structure and returning a pointer to this 
			structure.
Accepts:	numToAvg = number of consecutive samples to average. Default is 1.
Returns:	pointer to leginfo member variable structure
*/
LEGINFO *Leg::GetLegData(int numToAvg/*=1*/) {
	double temp_bend[MAXREGIONS], temp_twist[MAXREGIONS];
	double bend[MAXSUBREGIONS], twist[MAXSUBREGIONS];
	getCurvatureData(temp_bend,temp_twist);
	DistribSensorData(temp_bend,bend);
	DistribSensorData(temp_twist,twist);
	ReCalculateBendTwist(bend,twist,numToAvg);
	FindLimbAnglesUsingBT();
	return &leginfo;
}


/*
Name:		GetLegConfig
Purpose:	Gets all of the configuration parameters (stored in the .mst and .lin files)
			for this leg tape.
Accepts:	void
Returns:	true if the configuration for the leg tape was valid, false otherwise.
*/
bool Leg::GetLegConfig() {
	if (!GetBodyConfig()) return false;
	if (m_nTapetype!=LEFTLEG&&m_nTapetype!=RIGHTLEG) return false;
	filedata mstfile(m_configfile);
	if (mstfile.getFileLength(m_configfile)<10) return false;//invalid configuration file
	char *linfilename=NULL;
	mstfile.getString("[settings]","linfile",&linfilename);
	if (!linfilename) return false;
	SetCurrentDirectory(m_rootfolder);
	filedata linfile(linfilename);
	if (linfile.getFileLength(linfilename)<10) return false; //invalid link file
	
	//get bone lengths
	double legbones[MAXBONES];
	linfile.getDouble("bonelabel_string","anat_bonelength_mm",MAXBONES,legbones);
	m_dThighlength=legbones[0];
	m_dShinlength=legbones[1];
	m_dFootlength=legbones[2];
	m_dToelength=legbones[3];
	m_dThighwidth = mstfile.getDouble("[leg]","thighwidth");

	//get other link file settings
	linfile.closeDataFile();
	if (!LoadLinFile(linfilename)) {
		delete []linfilename;
		return false;
	}
	delete []linfilename;

	//get hip offsets
	mstfile.getDouble("[leg]","hipoffsets",3,m_dHipoffsets);
	return true;
}

/*
Name:		ComputeOffsets
Purpose:	Determines the positional and rotational offsets for the leg tape so that it shows up 
			properly in the homing pose (person standing up straight, feet slightly less than shoulder-
			width apart and toes pointing forwards; feet flat on the floor.
Accepts:	void
Returns:	void
*/
void Leg::ComputeOffsets() {
	ZeroTransforms();
	ResetAverages();
	m_dHipoffsets[0]=m_dHipoffsets[1]=m_dHipoffsets[2]=0.0;
	GetLegData(1);
	BodyPart::ComputeOffsets();
	GetLegData(1);
	CalculateHipOffsets();
}

/*
Name:		GetHipOffsets
Purpose:	Gets the current hip offsets for this leg tape.
Accepts:	hip = 3 element double array corresponding to the xyz components of the hip offsets.
Returns:	void
*/
void Leg::GetHipOffsets(double *hip) {
	hip[0]=m_dHipoffsets[0];
	hip[1]=m_dHipoffsets[1];
	hip[2]=m_dHipoffsets[2];
}

/*
Name:		UpdateLinkFile
Purpose:	Updates the link file for a leg tape, based on the info contained within the subject file.
Accepts:	subjectFile = null terminated string specifying the subject file to be used. (Subject files
						  can be created using Measurand's ShapeRecorder software.
			nHipclampindex = integer index of tape sensing region where the tape should be clamped
						     at the hip.
			dKneetapeindex = index of tape sensing region where the tape should coincide with the 
							 wearer's knee joint. dKneetapeindex can be a fractional value, ex: 9.5
Returns:	true if link file was updated successfully, false otherwise.
*/
bool Leg::UpdateLinkFile(char *subjectFile, int &nHipclampindex, double &dKneetapeindex) {
	nHipclampindex=-1;
	dKneetapeindex=-1.0;
	filedata subfile(subjectFile);
	const double shin_ratio=1.03; //typical ratio of tape length to bone length on shin
	double dShinlength = subfile.getDouble("[linklengths]","shinlength");	
	double dThighlength = subfile.getDouble("[linklengths]","thighlength");
	double thigh_ratio = subfile.getDouble("[linklengths]","thighfactor");
	double dFootlength = subfile.getDouble("[linklengths]","footlength");
	if (dFootlength==0) dFootlength=230.0; //default value
	m_dLegseparation = subfile.getDouble("[linklengths]","legseparation");
	if (m_dLegseparation==0.0) m_dLegseparation = 180.0; //use default
	filedata mstfile(m_configfile);
	//save leg separation to mstfile
	mstfile.writeData("[leg]","legseparation",m_dLegseparation);
	//get name of link file
	char *linkfilename = NULL;
	mstfile.getString("[settings]","linfile",&linkfilename);
	if (!linkfilename) return false; //no linkfile to update!
	filedata linfile(linkfilename);
	double tape_bonelength_mm[9];
	double anat_bonelength_mm[4];
	linfile.getDouble("bonelabel_string","tape_bonelength_mm",9,tape_bonelength_mm);
	linfile.getDouble("bonelabel_string","anat_bonelength_mm",4,anat_bonelength_mm);	
	anat_bonelength_mm[1]=dShinlength;
	anat_bonelength_mm[0]=dThighlength;
	m_dThighlength=anat_bonelength_mm[0];
	anat_bonelength_mm[2]=.75*dFootlength;
	anat_bonelength_mm[3]=.25*dFootlength;
	double dThightapelength = dThighlength * thigh_ratio;
	double dRegionlength = region_length[0];
	if (dThightapelength<2*dRegionlength) {
		//uh oh... thigh is too small! (need to have at least 2 region lengths for thigh)
		return false;
	}
	//fill in known quantities
	tape_bonelength_mm[8]=region_length[num_sensors/2-1]; //use last region for toe of foot
	
	//calculate region of tape used for ankle
	const double ankle_factor = 1.00;
	tape_bonelength_mm[7]=ankle_factor*dFootlength-tape_bonelength_mm[8];
	//round to nearest region length
	tape_bonelength_mm[7]=((int)(tape_bonelength_mm[7]/dRegionlength+.5))*dRegionlength;
	tape_bonelength_mm[5]=2*dRegionlength;
	double dRemaining_tapelength = tape_length - tape_bonelength_mm[7] - tape_bonelength_mm[8];
	double dShintapelength = dShinlength * shin_ratio;
	double dRequired_tapelength = dShintapelength + dThightapelength;
	if (dRequired_tapelength>dRemaining_tapelength) {
		//need longer leg tape
		return false;
	}
	//the following code assumes that all region lengths for this leg tape are of uniform length
	//figure out how much of tape should be applied to shin and how much should be applied to thigh regions
	int nRequired_sensors = (int)(dRequired_tapelength / dRegionlength + .9); //anything less than x.1 
		//is rounded down to x, anything greater or equal to x.1 is rounded up to (x+1).
	//add extra length onto thigh region
	dThightapelength+=nRequired_sensors*dRegionlength - dRequired_tapelength;
	
	//find number of null regions (i.e. find the index of the tape at the hip clamp)
	dRequired_tapelength = dThightapelength + dShintapelength+tape_bonelength_mm[7]
		+tape_bonelength_mm[8];
	nHipclampindex = (int)((tape_length-dRequired_tapelength)/dRegionlength+.1);
		
	//fill in remaining values of tape_bonelength_mm
	tape_bonelength_mm[1] = (dThightapelength-2*dRegionlength)/3.0;
	tape_bonelength_mm[2] = (dThightapelength-2*dRegionlength)/3.0;
	tape_bonelength_mm[3] = (dThightapelength-2*dRegionlength)/6.0;
	tape_bonelength_mm[4] = (dThightapelength-2*dRegionlength)/6.0;
	tape_bonelength_mm[6] = dShintapelength;
	tape_bonelength_mm[0] = nHipclampindex*dRegionlength;

	//test - replace 90 deg bend angle at ankle with 80 deg bend (necessary to prevent unwanted 
	//"bending up" of ankle whenever additional weight is applied to the foot.
	double b_offsets[9]; 
	linfile.getDouble("bonelabel_string","b_offset_deg",9,b_offsets);
	if (m_nTapetype==RIGHTLEG) b_offsets[7]=80.0; 
	else b_offsets[7]=-80.0;
	linfile.writeData("bonelabel_string","b_offset_deg",9,b_offsets);

	//write changes to link file
	linfile.writeData("bonelabel_string","tape_bonelength_mm",9,tape_bonelength_mm);
	linfile.writeData("bonelabel_string","anat_bonelength_mm",4,anat_bonelength_mm);

	if (nHipclampindex < 0) nHipclampindex = 0;
	dKneetapeindex = nHipclampindex+dThightapelength/dRegionlength;
	linfile.closeDataFile();
	//reload link file with new settings
	LoadLinFile(linkfilename);
	delete []linkfilename;
	return true;
}
