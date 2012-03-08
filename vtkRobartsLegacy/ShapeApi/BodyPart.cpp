// BodyPart.cpp: implementation of the BodyPart class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "BodyPart.h"
#include "filedata.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////


/*
Name:		BodyPart
Purpose:	constructor - this constructor builds a BodyPart object with settings defined in the 
			Measurand ShapeTape file: configfile
Accepts:	config_file = Measurand ShapeTape (.mst) file that contains all of the settings 
			for this particular ShapeTape. 
			
Returns:    void
*/
BodyPart::BodyPart(char *config_file, char *rootfolder)
		 :tapeAPI(config_file,rootfolder)
{
	m_configfile = new char[strlen(config_file)+1];
	m_rootfolder = new char[strlen(rootfolder)+1];
	strcpy(m_rootfolder,rootfolder);
	strcpy(m_configfile,config_file);
	for (int i=0;i<MAXSUBREGIONS;i++) {
		m_bendoffsets[i]=0.0;
		m_twistoffsets[i]=0.0;
		for (int j=0;j<MAXAVGBUFFER;j++) {
			m_avgbend[j][i]=0.0;
			m_avgtwist[j][i]=0.0;
		}
	}
	for (i=0;i<MAXSUBREGIONS;i++) {
		m_recalc_bend[i]=0.0;
		m_recalc_twist[i]=0.0;
		for (int j=0;j<3;j++) {
			m_u[i][j]=0.0;
			m_n[i][j]=0.0;
			m_b[i][j]=0.0;
			m_r[i][j]=0.0;
			m_stU[i][j]=0.0;
			m_stN[i][j]=0.0;
			m_stB[i][j]=0.0;
			m_stR[i][j]=0.0;
		}
	}
	for (i=0;i<MAXJOINTS;i++) {
		m_jbangles[i]=0.0;
		m_jtangles[i]=0.0;
		for (int j=0;j<3;j++) {
			m_jointpos[i][j]=0.0;
		}
	}
	double m_tape_bonelengths[MAXITEMS];
	int sensor_indices[MAXITEMS];
	int bcollect_start[MAXITEMS];
	int bcollect_end[MAXITEMS];
	int tcollect_start[MAXITEMS];
	int tcollect_end[MAXITEMS];
	int bapply_start[MAXITEMS];
	int bapply_end[MAXITEMS];
	int tapply_start[MAXITEMS];
	int tapply_end[MAXITEMS];
	double boffset_rad[MAXITEMS];
	double toffset_rad[MAXITEMS];
	double homingpose_boffset[MAXITEMS];
	double homingpose_toffset[MAXITEMS];
	
	for (i=0;i<MAXITEMS;i++) {
		m_tape_bonelengths[i]=0.0;
		sensor_indices[i]=0;
		bcollect_start[i]=0;
		bcollect_end[i]=0;
		tcollect_start[i]=0;
		tcollect_end[i]=0;
		bapply_start[i]=0;
		bapply_end[i]=0;
		tapply_start[i]=0;
		tapply_end[i]=0;
		boffset_rad[i]=0.0;
		toffset_rad[i]=0.0;
		homingpose_boffset[i]=0.0;
		homingpose_toffset[i]=0.0;
		m_majorJoint[i]=FALSE;
		for (int j=0;j<3;j++) m_bone_orient[i][j]=0.0;
	}

	for (i=0;i<MAXBONES;i++) m_anat_bonelengths[i]=0.0;
	m_nTapetype=RIGHTARM; //assume right arm by default
	m_limbparts=NULL;
	m_jointnames=NULL;
	m_numAveraged=0;
	m_avgindex=0;
	m_nNumsegs=0;
	m_dOrient_rotax_deg=0.0;
	m_nOrientBoneNum=0;
	m_orientXYZ[0]=1.0; m_orientXYZ[1]=0.0; m_orientXYZ[2]=0.0; //assume default pointing along x-axis
	m_limbtransform[0][0]=1.0; m_limbtransform[0][1]=0.0; m_limbtransform[0][2]=0.0;
	m_limbtransform[1][0]=0.0; m_limbtransform[1][1]=1.0; m_limbtransform[1][2]=0.0;
	m_limbtransform[2][0]=0.0; m_limbtransform[2][1]=0.0; m_limbtransform[2][2]=1.0;
}

/*
Name:		~BodyPart
Purpose:	destructor - the BodyPart object gets destroyed here.
Accepts:	void
			
Returns:    void
*/
BodyPart::~BodyPart()
{
	if (m_configfile) delete []m_configfile;
	if (m_limbparts) delete []m_limbparts;
	if (m_jointnames) delete []m_jointnames;
	if (m_rootfolder) delete []m_rootfolder;
}

/*Name:		GetPosFromOrientation
Purpose:	Recalculates the position of a joint segment based on the length of the preceeding 
			bone segment and the orientation of that preceeding bone segment. The orientation is 
			described by the 2 Euler angles yaw and pitch which follow the conventions described
			in the unb2rpy function.
Accepts:	pos = 3 element double array corresponding to the returned xyz location of the joint centre.
			yaw = yaw angle in degrees of the preceeding bone segment (see unb2rpy function).
			pitch = pitch angle in degrees of the preceeding bone segment (see unb2rpy function)
			length = length in mm of the preceeding bone segment.
Returns:	void (fills the pos array with the xyz position of the joint centre).
*/
void BodyPart::GetPosFromOrientation(double pos[], double yaw, double pitch, double length) {
	//Function takes an initial base position in Cartesian space in the pos[] array, and calculates
	//what the end point position of a vector of a certain length would be if it started at 
	//pos and was rotated through the given yaw and pitch angles.  The result of the function is 
	//returned in the pos[] array.
	const double degToRad = 0.017453292519944;
	double yaw_rad=yaw*degToRad;
	double pitch_rad=pitch*degToRad;
	pos[0]+=cos(yaw_rad)*cos(pitch_rad)*length;
	pos[1]+=sin(pitch_rad)*length;
	pos[2]+=sin(yaw_rad)*cos(pitch_rad)*length;
}


/*Name:		GetTransforms
Purpose:	To find the rotational transform that is necessary to orient the ShapeTape so that a predefined
			section of ShapeTape link-model (defined by the m_nOrientBoneNum variable) is oriented in a 
			particular direction (defined by the m_orientXYZ axis and the m_dOrient_rotax_deg angle about
			this axis. The rotational transform is stored in 3X3 matrix form in the m_limbtransform 
			two dimensional array.
Accepts:	void
Returns:	void (fills m_limbtransform member variable array).
*/
void BodyPart::GetTransforms() {
	//computes m_limbtransform matrix based on orientation of key bone segment	
	double r_desired[3][3]; //desired rotation matrix
	double r_actual[3][3]; //actual rotation matrix
	double r_actual_inv[3][3]; //inverted actual rotation matrix

	const double dDegtoRad=0.01745329251994;
	
	double desired_yaw=0,desired_pitch=0,desired_roll=0;
	desired_pitch=asin(m_orientXYZ[1]);
	double dTest=m_orientXYZ[0]/(cos(desired_pitch));
	if (dTest>1) dTest=1;
	else if (dTest<-1) dTest=-1;
	if (dTest==0&&(m_orientXYZ[1]==-1||m_orientXYZ[1]==1)) desired_yaw=0; //pole correction
	else desired_yaw=acos(dTest);
	dTest=sin(desired_yaw)*cos(desired_pitch)*m_orientXYZ[2];
	if (dTest<0) desired_yaw=-desired_yaw;
	desired_roll=m_dOrient_rotax_deg*dDegtoRad;

	double cosyaw=cos(desired_yaw);
	double sinyaw=sin(desired_yaw);
	double cospitch=cos(desired_pitch);
	double sinpitch=sin(desired_pitch);
	double cosroll=cos(desired_roll);
	double sinroll=sin(desired_roll);

	r_desired[0][0]=cosyaw*cospitch;
	r_desired[0][1]=-cosyaw*sinpitch*cosroll+sinyaw*sinroll;
	r_desired[0][2]=-cosyaw*sinpitch*sinroll-sinyaw*cosroll;
	r_desired[1][0]=sinpitch;
	r_desired[1][1]=cospitch*cosroll;
	r_desired[1][2]=cospitch*sinroll;
	r_desired[2][0]=sinyaw*cospitch;
	r_desired[2][1]=-sinyaw*sinpitch*cosroll-cosyaw*sinroll;
	r_desired[2][2]=-sinyaw*sinpitch*sinroll+cosyaw*cosroll;
	
	double actual_roll=m_bone_orient[m_nOrientBoneNum][0];
	double actual_pitch=m_bone_orient[m_nOrientBoneNum][1];
	double actual_yaw=m_bone_orient[m_nOrientBoneNum][2];

	cosyaw=cos(actual_yaw);
	sinyaw=sin(actual_yaw);
	cospitch=cos(actual_pitch);
	sinpitch=sin(actual_pitch);
	cosroll=cos(actual_roll);
	sinroll=sin(actual_roll);

	r_actual[0][0]=cosyaw*cospitch;
	r_actual[0][1]=-cosyaw*sinpitch*cosroll+sinyaw*sinroll;
	r_actual[0][2]=-cosyaw*sinpitch*sinroll-sinyaw*cosroll;
	r_actual[1][0]=sinpitch;
	r_actual[1][1]=cospitch*cosroll;
	r_actual[1][2]=cospitch*sinroll;
	r_actual[2][0]=sinyaw*cospitch;
	r_actual[2][1]=-sinyaw*sinpitch*cosroll-cosyaw*sinroll;
	r_actual[2][2]=-sinyaw*sinpitch*sinroll+cosyaw*cosroll;

	//since R = Rtranspose for a rotation matrix, we get the following for r_actual_inv
	for (int i=0;i<3;i++) {
		for (int j=0;j<3;j++) r_actual_inv[i][j]=r_actual[j][i];
	}

	//compute m_limbtransform = r_desired * r_actual_inv
	for (i=0;i<3;i++) {
		for (int j=0;j<3;j++) {
			m_limbtransform[i][j] = r_desired[i][0]*r_actual_inv[0][j] + r_desired[i][1]*
				r_actual_inv[1][j] + r_desired[i][2]*r_actual_inv[2][j];
		}
	}
	//save limb transform to configuration file
	filedata mstfile(m_configfile);
	mstfile.writeData("[settings]","limbtransform",9,(double *)m_limbtransform);
}

/*Name:		ResetAverages
Purpose:	Resets quantities used for averaging to zero, so that averageing will begin completely anew. 
Accepts:	void
Returns:	void
*/
void BodyPart::ResetAverages() {
	m_numAveraged=0;
	m_avgindex=0;
}

/*Name:		GetJointBendAngle
Purpose:	Returns the joint bend angle in degrees at the joint identified by the zero-based index 
			jointnum.
Accepts:	jointnum = zero-based index of joint whose bend angle will be returned.
Returns:	Joint bend angle in degrees at the joint identified by the zero-based index 
			jointnum.
*/
double BodyPart::GetJointBendAngle(int jointnum) {
	return m_jbangles[jointnum];
}

/*Name:		GetJointTwistAngle
Purpose:	Returns the joint twist angle in degrees for the bone segment that lies between the major 
			joints (jointnum) and (jointnum+1)
			jointnum.
Accepts:	jointnum = zero-based index of joint preceeding the bone segment whose twist is to be 
						returned.
Returns:	Joint twist angle in degrees for the bone segment that lies between the major 
			joints (jointnum) and (jointnum+1).
*/
double BodyPart::GetJointTwistAngle(int jointnum) {
	return m_jtangles[jointnum];
}

/*Name:		GetLimbTransform
Purpose:	To retrieve the orientation settings for this ShapeTape from its configuration file. The 
			m_limbtransform 3X3 matrix is filled with the desired rotational transform.
Accepts:	void
Returns:	true if the orientations settings were successfully retrieved from the configuration file, 
			false otherwise.
*/
bool BodyPart::GetLimbTransform() {
	if (!m_configfile) return false;
	filedata mstfile(m_configfile);
	if (mstfile.getFileLength(m_configfile)<10) return false; //check for invalid file
	//get limb transform from .mst configuration file
	double dVals[9];
	mstfile.getDouble("[settings]","limbtransform",9,dVals);
	//check for case in which limbtransform entry did not exist
	if (dVals[0]==0&&dVals[1]==0&&dVals[2]==0&&dVals[3]==0&&dVals[4]==0&&dVals[5]==0&&
		dVals[6]==0&&dVals[7]==0&&dVals[8]==0) {
		dVals[0]=1.0; dVals[4]=1.0; dVals[8]=1.0;
	}
	for (int i=0;i<9;i++) m_limbtransform[i/3][i%3]=dVals[i];
	return true;
}

/*Name:		AverageBendTwist
Purpose:	Averages the bend and twist values for the interpolated points along the ShapeTape. The average
			is computed in a gradient manner as follows:  the first 20% of the tape nearest the base is 
			averaged 5*numToAvg times (using the 5*numToAvg most recent samples), the next 20% of the tape
			nearest the base is averaged 4*numToAvg times, the next 20% is averaged 3*numToAvg times, the
			next 20% is averaged 2*numToAvg times, and the last 20% is averaged numToAvg times. Averaging
			is done in this fashion to avoid excessive positional jitter at the tip of the tape. It is 
			possible to average more near the base of the tape because bend/twist events typically occur
			over a much slower timescale near the base of the tape for typical body-type motions. Ex: a 
			person can bend his/her finger back and forth much more quickly than they can raise and lower 
			their arm at the shoulder joint.
Accepts:	numToAvg = the number of samples to average at the tip of the tape. The number of samples 
						averaged at higher points on the tape are described in the above paragraph.
Returns:	none
*/
void BodyPart::AverageBendTwist(int numToAvg) {
	int numPoints = num_region*interval+1;
	m_avgindex++; if (m_avgindex==5*numToAvg) m_avgindex=0;
	if (m_numAveraged<5*numToAvg) m_numAveraged++;
	int numsamples[5];
	numsamples[0]=min(5*numToAvg,m_numAveraged);
	numsamples[1]=min(4*numToAvg,m_numAveraged);
	numsamples[2]=min(3*numToAvg,m_numAveraged);
	numsamples[3]=min(2*numToAvg,m_numAveraged);
	numsamples[4]=min(numToAvg,m_numAveraged);

	for (int i=0;i<numPoints;i++) {	
		int ns_index=4;
		if (i<.2*numPoints) ns_index=0;
		else if (i<.4*numPoints) ns_index=1;
		else if (i<.6*numPoints) ns_index=2;
		else if (i<.8*numPoints) ns_index=3;
		for (int j=m_avgindex-1;j>m_avgindex-1-numsamples[ns_index];j--) {
			int n=j; if (n<0) n+=5*numToAvg;
			m_recalc_bend[i]+=m_avgbend[n][i];
			m_recalc_twist[i]+=m_avgtwist[n][i];
		}
	}
	for (i=0;i<numPoints;i++) {
		int ns_index=4;
		if (i<.2*numPoints) ns_index=0;
		else if (i<.4*numPoints) ns_index=1;
		else if (i<.6*numPoints) ns_index=2;
		else if (i<.8*numPoints) ns_index=3;
		m_recalc_bend[i]/=numsamples[ns_index];
		m_recalc_twist[i]/=numsamples[ns_index];
	}
}

/*
Name:		GetOrientationVector
Purpose:	Gets the orientation of a particular segment of the link-modelled ShapeTape. The orientation 
			is described by 3 orthogonal unit vectors, u, n, b (see unb2rpy function).
Accepts:	u = 3-element array corresponding to the xyz components of the tangent to the surface of the
				tape at the location defined by index.
			n = 3-element array corresponding to the xyz components of the normal  to the surface of the
				tape at the location defined by index.
			b = 3-element array corresponding to the 3rd unit axis at the point on the surface of the 
				tape at the location defined by index. b = u X n
			index = Zero-based index of the tape whose orientation is to be found. index must be greater
					or equal to zero, and less than or equal to num_regions * interval.
Returns:	void (u, n, b arrays are filled).
*/
void BodyPart::GetOrientationVector(double *u, double *n, double *b, int index) {
	//function returns the 3 orthogonal unit vectors at a point on the tape defined by the value index
	//index must be in the valid range for the tape (ex: 0 to 80 for a tape with 16 sensing regions
	//and interp of 5).
	for (int i=0;i<3;i++) {
		u[i]=m_u[index][i];
		n[i]=m_n[index][i];
		b[i]=m_b[index][i];
	}
}

/*
Name:		GetPosition
Purpose:	Gets the position of a particular segment of the link-modelled ShapeTape. 
Accepts:	r = 3-element array corresponding to the xyz location of the tape at the location defined 
				by index.
			index = Zero-based index of the tape whose position is to be found. index must be greater
					or equal to zero, and less than or equal to num_regions * interval.
Returns:	void (r array is filled with xyz location).
*/
void BodyPart::GetPosition(double *r, int index) {
	r[0]=m_r[index][0];
	r[1]=m_r[index][1];
	r[2]=m_r[index][2];
}

/*Name:		LoadLinFile
Purpose:	Loads the link file whose name is szFilename and gets all of the necessary link file settings
			from this file. It is necessary to call this function before performing any of the link-model
			calculations described in this api.
Accepts:	szFilename = null-terminated character string which indicates the name of the link file to be 
						 opened.
Returns:	true if the link file was loaded correctly, false otherwise.
*/
bool BodyPart::LoadLinFile(char *szFilename) {
	//function loads a file of limb parameters and sets the various
	//bend and twist regions accordingly

	//allocate memory for re-calculated bend and twist arrays as well as bend and twist offsets
	int maxIndex = interval*num_region;
	int numIndices=maxIndex+1;

	//initialize all bends, twists, and offsets to zero
	for (int i=0;i<numIndices;i++) {
		m_recalc_bend[i]=0.0;
		m_recalc_twist[i]=0.0;
		m_bendoffsets[i]=0.0;
		m_twistoffsets[i]=0.0;
	}
	filedata linfile(szFilename);
	//check to make sure that file contains valid data
	if (linfile.getFileLength(szFilename)<10) {
		linfile.closeDataFile();
		DeleteFile(szFilename);
		return false;
	}
	if (m_limbparts) {
		delete []m_limbparts;
		m_limbparts=NULL;
	}
	if (m_jointnames) {
		delete []m_jointnames;
		m_jointnames=NULL;
	}
	linfile.getString("bonelabel_","string",&m_limbparts);
	linfile.getString("bonelabel","jointlabel_string",&m_jointnames);
	
	int use_joints[MAXITEMS];
	linfile.getInteger("bonelabel","major_joint_mark",MAXITEMS,use_joints);
	for (i=1;i<MAXITEMS;i++) m_majorJoint[i-1]=(BOOL)use_joints[i];
	linfile.getDouble("bonelabel","tape_bonelength_mm",MAXITEMS,m_tape_bonelengths);
	linfile.getDouble("bonelabel","anat_bonelength_mm",MAXBONES,m_anat_bonelengths);
	//find cumulative tape lengths
	double cum_length[MAXITEMS];

	for (i=0;i<MAXITEMS;i++) {
		cum_length[i]=0.0;
		for (int j=0;j<=i;j++) {
			cum_length[i]+=m_tape_bonelengths[j];
		}
	}
	//modify m_tape_bonelengths so that the zero-based index begins with the first real bone
	for (i=0;i<MAXITEMS-1;i++) {
		m_tape_bonelengths[i]=m_tape_bonelengths[i+1];
	}
	m_tape_bonelengths[MAXITEMS-1]=0.0;

	//find parameters relating to the initial orientation of the body part
	m_nOrientBoneNum=(int)linfile.getDouble("bonelabel","orient_bonenum")-2;
	linfile.getDouble("bonelabel","orient_xyz",3,m_orientXYZ);
	m_dOrient_rotax_deg=linfile.getDouble("bonelabel","orient_rotax_deg");

	//modify m_orientXYZ and m_dOrient_rotax_deg based on tapetype:
	if (m_nTapetype==LEFTARM||m_nTapetype==RIGHTARM) {
		m_orientXYZ[0]=1.0; m_orientXYZ[1]=0.0; m_orientXYZ[2]=0.0;
		m_dOrient_rotax_deg = 0.0;
	}
	else if (m_nTapetype==LEFTLEG||m_nTapetype==RIGHTLEG) {
		m_orientXYZ[0]=0.0; m_orientXYZ[1]=-1.0; m_orientXYZ[2]=0.0;
		m_dOrient_rotax_deg = 0.0;
	}
	else if (m_nTapetype==HEADTAPE||m_nTapetype==BACKTAPE) {
		m_orientXYZ[0]=0.0; m_orientXYZ[1]=1.0; m_orientXYZ[2]=0.0;
		m_dOrient_rotax_deg = 0.0;
	}
	
	//get offsets in deg
	double boffset_deg[MAXITEMS], toffset_deg[MAXITEMS];
	linfile.getDouble("bonelabel","b_offset_deg",MAXITEMS,boffset_deg);
	linfile.getDouble("bonelabel","t_offset_deg",MAXITEMS,toffset_deg);
	//modify b_offset_deg & t_offset_deg so that the zero-based index begins with the first real bone
	for (i=0;i<MAXITEMS-1;i++) {
		boffset_deg[i]=boffset_deg[i+1];
		toffset_deg[i]=toffset_deg[i+1];
	}
	boffset_deg[MAXITEMS-1]=0.0;
	toffset_deg[MAXITEMS-1]=0.0;
		
	//convert from deg to rad
	const double degToRad = 0.01745329251994;
	for (i=0;i<MAXITEMS;i++) {
		boffset_rad[i]=boffset_deg[i]*degToRad;
		toffset_rad[i]=toffset_deg[i]*degToRad;
	}

	//get values for member variables in terms of sensor indices (indices range from 0 to 80 for a 
	//32 sensor tape with 16 pairs and the default interp of 5
	
	//sensor indices
	for (i=0;i<MAXITEMS;i++) {
		sensor_indices[i]=(int)(maxIndex*cum_length[i]/tape_length+0.5);
	}
	
	m_nNumsegs=0;
	//make sure that sum of m_tape_bonelengths does not surpass the length of the tape
	double dTestsum=0.0;
	for (i=0;i<MAXITEMS;i++) {
		dTestsum+=m_tape_bonelengths[i];
	}
	if (dTestsum>tape_length) {
		return false; //too much length specified in link file
	}
	for (i=0;i<MAXITEMS;i++) {
		//bend and twist collection and application zones
		//bend at each joint is collected 50% from the previous bone and 50% from the next bone.
		//twist for each bone is collected from 100% of each bone
		//bend is applied at each joint location
		//twist is applied along the length of each bone
		if (m_tape_bonelengths[i]>0) {
			m_nNumsegs++;
			if (i==0) {
				//first bone segment
				bcollect_start[0]=sensor_indices[0];
				bcollect_end[0]=(sensor_indices[0]+sensor_indices[1])/2-1;
				tcollect_start[0]=sensor_indices[0];
				tcollect_end[0]=sensor_indices[1]-1;
				bapply_start[0]=sensor_indices[0];
				bapply_end[0]=sensor_indices[0];
				tapply_start[0]=sensor_indices[0];
				tapply_end[0]=sensor_indices[1]-1;
			}
			else if (i==MAXITEMS-1) {
				//last bone segment
				bcollect_start[i]=(sensor_indices[i-1]+sensor_indices[i])/2;
				bcollect_end[i]=maxIndex; //(end of tape)
				tcollect_start[i]=sensor_indices[i];
				tcollect_end[i]=maxIndex; //(end of tape)
				bapply_start[i]=sensor_indices[i];
				bapply_end[i]=sensor_indices[i];
				tapply_start[i]=sensor_indices[i];
				tapply_end[i]=maxIndex;
			}
			else {//somewhere in the middle of the tape
				bcollect_start[i]=(sensor_indices[i-1]+sensor_indices[i])/2;
				bcollect_end[i]=(sensor_indices[i]+sensor_indices[i+1])/2-1;
				tcollect_start[i]=sensor_indices[i];
				tcollect_end[i]=sensor_indices[i+1]-1;
				bapply_start[i]=sensor_indices[i];
				bapply_end[i]=sensor_indices[i];
				tapply_start[i]=sensor_indices[i];
				tapply_end[i]=sensor_indices[i+1]-1;
			}
		}
		else {
			//not a valid bone
			bcollect_start[i]=0; bcollect_end[i]=-1;
			tcollect_start[i]=0; tcollect_end[i]=-1;
			bapply_start[i]=0; bapply_end[i]=-1;
			tapply_start[i]=0; tapply_end[i]=-1;
		}

	}	

	//calculate offsets for each sensor index
	for (i=0;i<MAXITEMS;i++) {
		if (m_tape_bonelengths[i]>0) {
			//bend offsets
			int numSegs=bapply_end[i]-bapply_start[i]+1;
			for (int j=bapply_start[i];j<=bapply_end[i];j++) {
				m_bendoffsets[j]=boffset_rad[i]/numSegs;
			}
			//twist offsets
			numSegs=tapply_end[i]-tapply_start[i]+1;
			for (j=tapply_start[i];j<=tapply_end[i];j++) {
				m_twistoffsets[j]=toffset_rad[i]/numSegs;
			}
		}
	}
	return true;
}

/*Name:		ReCalculateBendTwist
Purpose:	This virtual function just does the default bend/twist to Cartesian conversion for normal
			ShapeTape. 
Accepts:	bend = double array of (num_region*interval+1) points corresponding to the interpolated bend
					values along the length of the tape.
			twist = double array of (num_region*interval+1) points corresponding to the interpolated twist
					values along the length of the tape.
Returns:	void (fills the m_stU, m_stN, m_stB, and m_stR member variable arrays with the orientation
			and position of interpolated points along the length of the ShapeTape.
*/
void BodyPart::ReCalculateBendTwist(double bend[], double twist[]) {
	//get ShapeTape quantities
	bt2cart(bend,twist,m_stU,m_stN,m_stB,m_stR);
}

/*
Name:		FindLimbAnglesUsingBT
Purpose:	Virtual function that finds all of the limb angles and joint positions, based on the 
			recalculated bend and twist of the tape. 
Accepts:	void
Returns:	void  (fills m_jointpos and m_bone_orient member variable arrays).
*/
void BodyPart::FindLimbAnglesUsingBT() {
	//uses the re-calculated bend and twist arrays (m_recalc_bend and m_recalc_twist
	//to come up with limb angles for arm or leg parts.
	//get u,n,b,r vectors
	if (!m_recalc_bend) return;
	if (!m_recalc_twist) return;
	int numVertices = interval*num_region+1;
	//rotate starting vector by correction matrix
	double temp_u[3], temp_b[3];
	for (int i=0;i<3;i++) {
		temp_u[i]=Uo[i];
		temp_b[i]=Bo[i];
	}
	ApplyBaseCorrection_Vec(Uo);
	ApplyBaseCorrection_Vec(Bo);
		
	bt2cart(m_recalc_bend,m_recalc_twist,m_u,m_n,m_b,m_r);
	//restore starting vectors
	for (i=0;i<3;i++) {
		Uo[i]=temp_u[i];
		Bo[i]=temp_b[i];
	}

	
	//get joint positions
	int nJointcount=0;
	for (i=0;i<MAXITEMS;i++) {
		if (m_majorJoint[i]) {
			for (int j=0;j<3;j++) m_jointpos[nJointcount][j]=m_r[sensor_indices[i]][j];
			nJointcount++;
		}
		//get bone orientations (average tape orientation data between joints
		if (bapply_end[i]-bapply_start[i]<0) continue;
		double avg_u[3]={0.0,0.0,0.0}; //average u orientation vector for bone
		double avg_n[3]={0.0,0.0,0.0}; //average n orientation vector for bone
		double avg_b[3]={0.0,0.0,0.0}; //average b orientation vector for bone
		
		int startIndex=(bapply_start[i]+bapply_end[i])/2;
		int endIndex=0;
		if (i+1<MAXITEMS&&bapply_end[i+1]>0) endIndex=(bapply_start[i+1]+bapply_end[i+1])/2;
		else endIndex=numVertices-1;

		//int midIndex = (startIndex+endIndex)/2;
		int midIndex=endIndex-1;
		for (int k=0;k<3;k++) {
				avg_n[k]=m_n[midIndex][k]; 
				avg_u[k]=m_u[midIndex][k];
		}
		
		//temp hack
		if (m_nTapetype==LEFTARM||m_nTapetype==LEFTLEG) {
			for (int j=0;j<3;j++) {
				avg_n[j]=-avg_n[j];
			}
		}
		
		//normalize
		normalize(avg_u); normalize(avg_n);
		//ensure vector orthogonality
		cross(avg_u,avg_n,avg_b);
		//normalize
		normalize(avg_b);
		cross(avg_b,avg_u,avg_n);
		//normalize
		normalize(avg_n);
		
		//get orientation
		double roll=0,pitch=0,yaw=0;
		unb2rpy(avg_u,avg_n,avg_b,roll,pitch,yaw);
		m_bone_orient[i][0]=roll; m_bone_orient[i][1]=pitch; m_bone_orient[i][2]=yaw;
	}
}

/*Name:		ApplyBaseCorrection_Pt
Purpose:	Used for rotating a given point by the transform described in the m_limbtransform
			matrix. Used to correct the positions of points along the ShapeTape.
Accepts:	point = three element vector describing the point to be rotated. point will be filled with 
					new values when the function returns.
			basepoint = three element vector describing the point about which the rotation will be performed.
						This is typically set to be the base of the tape.
Returns:	void  (point array is filled with it's new location after the rotation).
*/
void BodyPart::ApplyBaseCorrection_Pt(double *point, double *basepoint)
{
	//this function applies the limb transformation to a point about the base basepoint
	//first subtract base point
	for (int i=0;i<3;i++) point[i]-=basepoint[i];
	double newpoint[3]; 
	for (i=0;i<3;i++) {
		newpoint[i]=m_limbtransform[i][0]*point[0] + m_limbtransform[i][1]*point[1] 
			 + m_limbtransform[i][2]*point[2] + basepoint[i];
	}
	for (i=0;i<3;i++) point[i]=newpoint[i];
}


/*Name:		ApplyBaseCorrection_Vec
Purpose:	Used for rotating a given vector by the transform described in the m_limbtransform
			matrix. Used to correct the orientations of vectors along the ShapeTape.
Accepts:	vec = three element vector describing the vector to be rotated. vec will be filled with 
					new values when the function returns.
Returns:	void  (vec array is filled with it's new values after the rotation).
*/
void BodyPart::ApplyBaseCorrection_Vec(double *vec)
{
	//this function applies the limb transformation to a vector
	double newvec[3]; 
	for (int i=0;i<3;i++) {
		newvec[i]=m_limbtransform[i][0]*vec[0] + m_limbtransform[i][1]*vec[1] + m_limbtransform[i][2]*vec[2];
	}
	for (i=0;i<3;i++) vec[i]=newvec[i];
}


/*Name:		unb2rpy
Purpose:	Converts the orthogonal unit orientation vectors u,n,b into Euler angles: roll_angle, 
			pitch_angle, and yaw_angle in radians. u represents a tangent vector to the ShapeTape
			at some point, n is the normal vector, and b is perpendicular to u and n, i.e. b = uXn.
			The yaw rotation is performed first clockwise positive about the +y-axis. The pitch
			rotation is performed next counter-clockwise positive about the rotated +z-axis, and the
			roll rotation is performed last, clockwise positive about the doubly rotated +x-axis.
Accepts:	u = 3 element tangent vector at some point on the ShapeTape.
			n = 3 element normal vector at some point on the ShapeTape.
			b = vector perpendicular to u and n, i.e. b = uXn.
			roll_angle = Euler angle roll in radians.
			pitch_angle = Euler angle pitch in radians.
			yaw_angle = Euler angle yaw in radians.
Returns:	void (roll_angle, pitch_angle, and yaw_angle are set by function).
*/
void BodyPart::unb2rpy(double *u, double *n, double *b, double &roll_angle
					   , double &pitch_angle, double &yaw_angle)
{
	//This function converts U,N, and B vectors into roll pitch 
	//and yaw euler angles.

	//The first two rotations (yaw and pitch) take U from [1 0 0] 
	//to its final orientation. The third rotation affects B and N only.
	//The yaw and pitch rotations are simply the azimuth and elevation
	//angles of U in spherical coordinates. For confirmation, try
	//quatdemo.m in Matlab, and observe the transformation matrices
	//in the reference, particularly the cos(yaw)*cos(pitch) term in A.
	//We have a 'y-up' coordinate system, so the base plane for the
	//spherical coordinates is yz, not xy. Some if statements are required
	//to create yaw and roll that vary from -180 to + 180, and to
	//account for signs in various quadrants.

	double Ux = u[0];
	double Uy = u[1];
	double Uz = u[2];
	double Nx = n[0];
	double Ny = n[1];
	double Nz = n[2];
	double Bx = b[0];
	double By = b[1];
	double Bz = b[2];

	double yaw=0;
	double pitch=0;
	double roll=0;

	//pitch
	if (Uy>1) Uy = 1;
	else if (Uy<-1) Uy = -1;
	pitch = asin(Uy);
	//yaw
	double dTest = Ux / cos(pitch);
	if (dTest>1) dTest=1;
	else if (dTest<-1) dTest = -1;
	yaw = acos(dTest);
	//check sign
	dTest = sin(yaw)*cos(pitch)*Uz;
	if (dTest<0) yaw=-yaw; //check to see if sign of yaw needs to be reversed
	//roll
	dTest = Ny/cos(pitch);
	if (dTest>1) dTest=1;
	else if (dTest<-1) dTest=-1;
	roll=acos(dTest);
	//check sign
	dTest=cos(pitch)*sin(roll)*By;
	if (dTest<0) roll=-roll;

	roll_angle=roll;
	pitch_angle=pitch;
	yaw_angle=yaw;
}

/*
Name:		ZeroTransforms
Purpose:	Resets the m_limbtransform matrix to the identity matrix. The m_limbtransform matrix
			describes how the ShapeTape should be rotated to match up properly with the homing pose.
Accepts:	void
Returns:	void
*/
void BodyPart::ZeroTransforms() {
	//Resets transform for the tape to identity matrix
	m_limbtransform[0][0]=1.0; m_limbtransform[0][1]=0.0; m_limbtransform[0][2]=0.0;
	m_limbtransform[1][0]=0.0; m_limbtransform[1][1]=1.0; m_limbtransform[1][2]=0.0;
	m_limbtransform[2][0]=0.0; m_limbtransform[2][1]=0.0; m_limbtransform[2][2]=1.0;
}

/*
Name:		DistribSensorData
Purpose:	Interpolates either bend or twist data by the interpolation interval.
Accepts:	sensor_data = array of either bend or twist data that is typically num_region long.
			interp_data = array which contains the interpolated values.
Returns:	void	(interp_data array is filled with interpolated values).
*/
void BodyPart::DistribSensorData(double sensor_data[], double interp_data[])
{
	int numVertices = num_region*interval+1;
	for (int i=0;i<num_region;i++) {
		double interp_val = sensor_data[i]/interval;
		for (int j=0;j<interval;j++) interp_data[interval*i+j]=interp_val;
	}
	interp_data[interval*num_region]=0;
}

/*Name:		GetBodyConfig
Purpose:	Finds out what tape type (i.e. arm, back, leg, or head (spine)) this tape is.	
Accepts:	void	
Returns:	true if the configuration file could be properly read, false otherwise.
*/
bool BodyPart::GetBodyConfig() {
	//function gets configuration parameters from the .mst configuration file
	if (!GetLimbTransform()) return false;
	filedata mstfile(m_configfile);
	if (mstfile.getFileLength(m_configfile)<10) {
		//invalid configuration file
		return false;
	}
	m_nTapetype = mstfile.getInteger("[settings]","tapetype");
	return true;
}

/*Name:		GetSTVectors
Purpose:	Gets the normal ShapeTape (i.e. non-link model) parameters for orientation (u,n,b) and 
Accepts:	u = 2 dimensional array of ShapeTape tangent vectors (must be (num_region * interval+1) 
				long by 3 wide).
			n = 2 dimensional array of ShapeTape normal vectors (must be (num_region * interval+1) 
				long by 3 wide).
			b = 2 dimensional array of ShapeTape vectors perpendicular to u and n (must be 
			(num_region * interval+1) long by 3 wide).
			r = 2 dimensional array of interpolated positions along the ShapeTape (must be 
			(num_region * interval+1) long by 3 wide).
Returns:	void (fills the u, n, b, r arrays with the non-link model ShapeTape orientation and 
			position).
*/
void BodyPart::GetSTVectors(double u[][3], double n[][3], double b[][3], double r[][3]) {
	int numVertices = num_region*interval+1;
	for (int i=0;i<numVertices;i++) {
		for (int j=0;j<3;j++) {
			u[i][j]=m_stU[i][j];
			n[i][j]=m_stN[i][j];
			b[i][j]=m_stB[i][j];
			r[i][j]=m_stR[i][j];
		}
	}
}

/*Name:		GetConfigFilename
Purpose:	Gets the name of the configuration file (stored in the m_configfile member variable).
Accepts:	void	
Returns:	pointer to the m_configfile character array (filename of configuration file).
*/
char * BodyPart::GetConfigFilename() {
	return m_configfile;
}

/*Name:		GetLinkTapeVectors
Purpose:	Similar to the GetSTVectors except it gets the corresponding quantities for the 
			link-model based ShapeTape.
Accepts:	u = 2 dimensional array of link model ShapeTape tangent vectors 
			(must be (num_region * interval+1) long by 3 wide).
			n = 2 dimensional array of link model ShapeTape normal vectors 
			(must be (num_region * interval+1) long by 3 wide).
			b = 2 dimensional array of link model ShapeTape vectors perpendicular to u and n 
			(must be (num_region * interval+1) long by 3 wide).
			r = 2 dimensional array of interpolated positions along the link model ShapeTape 
			(must be (num_region * interval+1) long by 3 wide).
Returns:	void (fills the u, n, b, r arrays with the link model ShapeTape orientation and 
			position).
*/

void BodyPart::GetLinkTapeVectors(double u[][3], double n[][3], double b[][3], double r[][3]) {
	int numVertices = num_region*interval+1;
	for (int i=0;i<numVertices;i++) {
		for (int j=0;j<3;j++) {
			u[i][j]=m_u[i][j];
			n[i][j]=m_n[i][j];
			b[i][j]=m_b[i][j];
			r[i][j]=m_r[i][j];
		}
	}
}

/*Name:		ComputeOffsets
Purpose:	Computes the rotational offsets for this ShapeTape.
Accepts:	void
Returns:	void
*/
void BodyPart::ComputeOffsets() {
	GetTransforms();
}

/*Name:		GetBoneLength
Purpose:	Returns the bone length of the bone specified by the zero-based value of index.
Accepts:	index = zero-based index of bone segment whose length is to be returned.
Returns:	length of the bone specified by the zero-based value of index.
*/
double BodyPart::GetBoneLength(int index) {
	return m_anat_bonelengths[index];
}

/*Name:		GetCalibrationFilename
Purpose:	Gets the calibration name for this ShapeTape
Accepts:	void
Returns:	Null-terminated character string corresponding to the calibration filename for this tape.
*/
char * BodyPart::GetCalibrationFilename() {
	return cal_file_name;
}
