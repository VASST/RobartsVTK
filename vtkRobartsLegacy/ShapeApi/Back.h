// Back.h: interface for the Back class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_BACK_H__5D9101A8_743C_484C_A34C_B18D84FAB602__INCLUDED_)
#define AFX_BACK_H__5D9101A8_743C_484C_A34C_B18D84FAB602__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "BodyPart.h"


class SHAPEAPI_API Back : public BodyPart  
{
public:
	Back(char *configfile, char *rootfolder);
	virtual ~Back();
	void CaptureSpine();
	int GetThoracicIndex();
	void GetBackData(double r[][3], double roll[], double pitch[], double yaw[], int numToAvg=1);
	virtual void ComputeOffsets();
	bool UpdateLinkFile(char *subjectFile, int &nClampindex);

private:
	bool GetBackConfig();
	bool m_bIncludeback;
	bool m_bFlatoffsets;
	bool m_bDoBackCalc;
	bool m_bSpinecaptured;
	double m_defSpine[MAXREGIONS];
	void AssignCapturedOffsets();
	virtual void ReCalculateBendTwist(double bend[], double twist[], int numToAvg=1);
	virtual void FindLimbAnglesUsingBT();
};

#endif // !defined(AFX_BACK_H__5D9101A8_743C_484C_A34C_B18D84FAB602__INCLUDED_)
