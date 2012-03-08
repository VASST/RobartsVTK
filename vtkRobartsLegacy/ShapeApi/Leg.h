// Leg.h: interface for the Leg class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_LEG_H__8FDA6950_525E_4CD1_8488_C64F42AE49BA__INCLUDED_)
#define AFX_LEG_H__8FDA6950_525E_4CD1_8488_C64F42AE49BA__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "BodyPart.h"

struct LEGINFO {
	double dUpperthighposition[3];
	double dThigh_roll;
	double dThigh_pitch;
	double dThigh_yaw;
	double dKneeposition[3];
	double dShin_roll;
	double dShin_pitch;
	double dShin_yaw;
	double dFoot_roll; 
	double dFoot_pitch;
	double dFoot_yaw;
	double dAnkle[3];
	double dKnuckle[3];
	double dToe_roll;
	double dToe_pitch;
	double dToe_yaw;
};


class SHAPEAPI_API Leg : public BodyPart  
{
public:
	bool UpdateLinkFile(char *subjectFile, int &nHipclampindex, double &dKneetapeindex);
	void GetHipOffsets(double *hip);
	LEGINFO leginfo;
	void CalculateHipOffsets();
	Leg(char *configfile, char *rootfolder);
	virtual ~Leg();
	LEGINFO * GetLegData(int numToAvg=1);
	virtual void ComputeOffsets();

private:
	double m_dLegseparation;
	bool GetLegConfig();
	virtual void FindLimbAnglesUsingBT();
	virtual void ReCalculateBendTwist(double bend[], double twist[], int numToAvg=1);
	double m_hippos[3];
	double m_dFootlength;
	double m_dToelength;
	double m_dThighlength;
	double m_dShinlength;
	double m_dThighwidth;
	double m_dHipwidth;
	double m_dHipoffsets[3];
};

#endif // !defined(AFX_LEG_H__8FDA6950_525E_4CD1_8488_C64F42AE49BA__INCLUDED_)
