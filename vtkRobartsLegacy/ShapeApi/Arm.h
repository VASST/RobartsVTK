// Arm.h: interface for the Arm class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ARM_H__A6837929_7BAE_48EF_AEBC_751C40181DFB__INCLUDED_)
#define AFX_ARM_H__A6837929_7BAE_48EF_AEBC_751C40181DFB__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "BodyPart.h"

struct ARMINFO
{
	double dShoulderposition[3];
	double dElbowposition[3];
	double dWrist[3];
	double dHand_roll;
	double dHand_pitch;
	double dHand_yaw;
	double dKnuckleposition[3];
	double dFinger_roll;
	double dFinger_pitch;
	double dFinger_yaw;
	double dUpperarm_roll;
	double dUpperarm_pitch;
	double dUpperarm_yaw;
	double dForearm_roll;
	double dForearm_pitch;
	double dForearm_yaw;
	double dChest_roll;
	double dChest_pitch;
	double dChest_yaw;
};

class SHAPEAPI_API Arm : public BodyPart  
{
public:
	bool UpdateLinkFile(char *subjectFile, int &nClampindex);
	void GetShoulderOffsets(double *shoulder);
	ARMINFO arminfo;
	Arm(char *configfile, char *rootfolder);
	virtual ~Arm();
	void CaptureShoulder();
	void CalculateShoulderOffsets();
	ARMINFO *GetArmData(int numToAvg=1);
	virtual void ComputeOffsets();
protected:
	virtual void ReCalculateBendTwist(double bend[], double twist[], int numToAvg=1);
	virtual void FindLimbAnglesUsingBT();
private:
	bool GetArmConfig();
	double m_dShoulderoffsets[3];
	double m_defShoulderBend[MAXREGIONS];
	double m_defShoulderTwist[MAXREGIONS];
	bool m_bUsecapturedshoulder;
	double m_dShoulderwidth;
	double m_dShoulderheight;
	double m_dUpperarmlength;
	double m_dForearmlength;

	void AssignCapturedOffsets();
	void CalculateShoulderOffsets(double *shoulder);
};

#endif // !defined(AFX_ARM_H__A6837929_7BAE_48EF_AEBC_751C40181DFB__INCLUDED_)
