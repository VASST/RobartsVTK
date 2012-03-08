// Head.h: interface for the Head class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_HEAD_H__C57F949D_78D5_40FB_879D_1FBA81F1FEB4__INCLUDED_)
#define AFX_HEAD_H__C57F949D_78D5_40FB_879D_1FBA81F1FEB4__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "BodyPart.h"

struct HEADINFO {
	double dHeadroll;
	double dHeadpitch;
	double dHeadyaw;
};

class SHAPEAPI_API Head : public BodyPart  
{
public:
	bool UpdateLinkFile(char *subjectFile, int &nClampindex);
	bool m_bIncludeback;
	HEADINFO headinfo;
	void CaptureSpine();
	HEADINFO * GetHeadData(int numToAvg=1);
	int GetSpineData(double r[][3], double roll[], double pitch[], double yaw[], int numToAvg=1);
	int GetSpineAndHeadData(double r[][3], double roll[], double pitch[], double yaw[], HEADINFO *hi, 
						int numToAvg=1);
	Head(char *configfile, char *rootfolder);
	virtual ~Head();
	virtual void ComputeOffsets();

private:
	bool GetHeadConfig();
	bool m_bSpinecaptured;
	double m_defSpine[MAXREGIONS];
	void AssignCapturedOffsets(); 
	int GetThoracicIndex();
	virtual void FindLimbAnglesUsingBT();
	virtual void ReCalculateBendTwist(double bend[], double twist[], int numToAvg=1);
};

#endif // !defined(AFX_HEAD_H__C57F949D_78D5_40FB_879D_1FBA81F1FEB4__INCLUDED_)
