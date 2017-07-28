#ifndef __L_U_RANGE_H__
#define __L_U_RANGE_H__

#include <vector>
#include <iostream>

//////////////////////////////////////////////////////////////////////////
class LURange
{
public:
	LURange();
	void SetList(std::vector< double > l){this->mList=l;};
	double GetLT()const{return this->mLT;};
	double GetUT()const{return this->mUT;};
	double GetMedian()const{return this->mMedian;};
	void Update();
protected:
	std::vector<double> mList;
	double mLT;
	double mUT;
	double	mMedian;
};

#endif