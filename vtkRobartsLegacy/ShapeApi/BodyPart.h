// BodyPart.h: interface for the BodyPart class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_BODYPART_H__375FF030_A17A_4BC3_B720_0448D7D68A64__INCLUDED_)
#define AFX_BODYPART_H__375FF030_A17A_4BC3_B720_0448D7D68A64__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "tapeAPI.h"
#define MAXREGIONS 24
#define MAXSUBREGIONS 120 //based on max interp interval of 5
#define MAXBONES 8
#define MAXJOINTS 7
#define MAXITEMS 15
#define MAXAVGBUFFER 100

//tape body part types
#define RIGHTARM 0
#define LEFTARM 1
#define RIGHTLEG 2
#define LEFTLEG 3
#define HEADTAPE 4
#define BACKTAPE 13

class SHAPEAPI_API BodyPart : public tapeAPI  
{
public:
	char * GetCalibrationFilename();
	double GetBoneLength(int index);
	void GetTransforms();
	void GetLinkTapeVectors(double u[][3], double n[][3], double b[][3], double r[][3]);
	char * GetConfigFilename();
	void GetSTVectors(double u[][3], double n[][3], double b[][3],double r[][3]);
	BodyPart(char *configfile, char *rootfolder);
	virtual ~BodyPart();
	void ResetAverages();
	double GetJointBendAngle(int jointnum);
	double GetJointTwistAngle(int jointnum);
	void GetOrientationVector(double *u, double *n, double *b, int index);
	void GetPosition(double *r, int index);
	virtual void ComputeOffsets();
	void unb2rpy(double *u, double *n, double *b, double &roll_angle
			   , double &pitch_angle, double &yaw_angle);
protected:
	char *m_rootfolder;
	int m_nTapetype;
	BOOL m_majorJoint[MAXITEMS];
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
	double m_recalc_bend[MAXSUBREGIONS];
	double m_recalc_twist[MAXSUBREGIONS];
	char *m_configfile;
	char *m_limbparts;
	char *m_jointnames;
	double m_twistoffsets[MAXSUBREGIONS];
	double m_bendoffsets[MAXSUBREGIONS];
	double m_u[MAXSUBREGIONS][3];
	double m_n[MAXSUBREGIONS][3];
	double m_b[MAXSUBREGIONS][3];
	double m_r[MAXSUBREGIONS][3];
	double m_limbtransform[3][3];
	double m_jointpos[MAXJOINTS][3]; //positions of limb joints [0]=x, [1]=y, [2]=z
	double m_bone_orient[MAXITEMS][3]; //orientations of bone segments, 
								    	//[0]=roll, [1]=pitch, [2]=yaw
	int m_numAveraged;
	int m_avgindex;
	int m_nNumsegs;
	double m_avgbend[MAXAVGBUFFER][MAXSUBREGIONS];
	double m_avgtwist[MAXAVGBUFFER][MAXSUBREGIONS];
	double m_jtangles[MAXJOINTS];
	double m_jbangles[MAXJOINTS];
	
	double m_anat_bonelengths[MAXBONES];

	bool GetBodyConfig();
	virtual void FindLimbAnglesUsingBT();
	virtual void ReCalculateBendTwist(double bend[], double twist[]);
	void AverageBendTwist(int numToAvg);
	void GetPosFromOrientation(double pos[], double yaw, double pitch, double length);
	bool GetLimbTransform();
	bool LoadLinFile(char *szFilename);
	void ZeroTransforms();
	void DistribSensorData(double sensor_data[], double interp_data[]);
private:
	double m_dOrient_rotax_deg;
	double m_orientXYZ[3];
	int m_nOrientBoneNum;

	//shapetape Cartesian info
	double m_stR[MAXSUBREGIONS][3];
	double m_stU[MAXSUBREGIONS][3];
	double m_stN[MAXSUBREGIONS][3];
	double m_stB[MAXSUBREGIONS][3];

	void ApplyBaseCorrection_Pt(double *point, double *basepoint);
	void ApplyBaseCorrection_Vec(double *vec);
};

#endif // !defined(AFX_BODYPART_H__375FF030_A17A_4BC3_B720_0448D7D68A64__INCLUDED_)
