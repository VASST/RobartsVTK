#include "LURange.h"
#include <math.h>
 //////////////////////////////////////////////////////////////////////////
 void dsort(std::vector<double>& x)
{
int i,j;
double A;
int n=x.size();
for(j=1; j<n; ++j)
{
A=x[j];
        for(i=j-1; i>=0; --i)
        {
               if(x[i]<=A)goto gotit;
                x[i+1]=x[i];
        }
i=-1;
gotit: x[i+1]=A;
}
}
//////////////////////////////////////////////////////////////////////////
double dmedian(std::vector<double>& x)
{
int n=x.size();
	 if(n==0)return 0.0;
	if(n==1)return x[0];

int m;
double ret;
m=n/2;
dsort(x);
if(2*m==n)
{
ret=0.5*(x[m-1]+x[m]);
}else
{
ret =x[m];
}
return ret;
}
 
//////////////////////////////////////////////////////////////////////////
double daverage(std::vector<double>& x)
{
double av;

int n=x.size();
	 if(n==0)return 0.0;
	if(n==1)return x[0];

int i;
av=0.0;


for( i=0; i<n; ++i)av+=x[i];
av/=n;

return av;
}

//////////////////////////////////////////////////////////////////////////
LURange::LURange()
{
this->mLT=0.0;
this->mUT=0.0;
this->mMedian=0.0;
}
//////////////////////////////////////////////////////////////////////////
void LURange::Update()
{

int i;
std::vector<double> l;
std::vector<double>  u;

//calculate the median
this->mMedian=dmedian(mList);
//this->mMedian=daverage(mList);

l.push_back(this->mMedian);
u.push_back(this->mMedian);

int s;
s=mList.size();
for(i=0; i<s; ++i)
{
if(this->mList[i]>this->mMedian)
{
	u.push_back(this->mList[i]);

}else{
	if(this->mList[i]<this->mMedian)
	{
	l.push_back(this->mList[i]);

	}
}
}

s=l.size();
this->mLT=0.0;
for(i=0; i<s; ++i)
{
this->mLT+=fabs(this->mMedian-l[i]);
}
if(l.size()>1)this->mLT/=(l.size()-1);

s=u.size();
this->mUT=0.0;
for(i=0; i<s; ++i)
{
this->mUT+=fabs(this->mMedian-u[i]);
}
if(u.size()>1)this->mUT/=(u.size()-1);



	
}
