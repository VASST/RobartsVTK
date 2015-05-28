//==============================================================================
//
//==============================================================================



#include "vtkImplicitSplineForce.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

//----------------------------------------------------------------------------
vtkImplicitSplineForce* vtkImplicitSplineForce::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkImplicitSplineForce");
  if(ret)
    {
    return (vtkImplicitSplineForce*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkImplicitSplineForce;
}


//////////////////////////////////////////////////////////////////////
vtkImplicitSplineForce::vtkImplicitSplineForce()
{
  // Constant for sigmoid function
  this->gammaSigmoid = 2;
  this->scaleForce = 20.0;
  strcpy(this->SplineKnots,"knot3DHeart.txt");
  strcpy(this->ControlPoints," ");

}
////////////////////////////////////////////////////////////////////////////////

void vtkImplicitSplineForce::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent.GetNextIndent());
  os << indent.GetNextIndent() << "Gamma Sigmoid: " << this->gammaSigmoid << endl;
}



////////////////////////////////////////////////////////////////////////////////
vtkImplicitSplineForce::~vtkImplicitSplineForce()
{

}
////////////////////////////////////////////////////////////////////////////////


void vtkImplicitSplineForce::SetInput(char * controlPnt) {
  strcpy(this->ControlPoints,controlPnt);
  ReadFileControlPoints(controlPnt);
}

void vtkImplicitSplineForce::SetInput(int splineId) {
  
  ReadFile3DBSplineKnots(this->SplineKnots);
  
  // Read B-spline control points (coefficients) from file
  if (splineId <= 0 || splineId > 20)
    return;

  switch(splineId){
  case 1:
    ReadFileControlPoints("control3DLSHeart01.txt");
    break;
  case 2:
    ReadFileControlPoints("control3DLSHeart02.txt");
    break;
  case 3:
    ReadFileControlPoints("control3DLSHeart03.txt");
    break;
  case 4:
    ReadFileControlPoints("control3DLSHeart04.txt");
    break;
  case 5:
    ReadFileControlPoints("control3DLSHeart05.txt");
    break;
  case 6:
    ReadFileControlPoints("control3DLSHeart06.txt");
    break;
  case 7:
    ReadFileControlPoints("control3DLSHeart07.txt");
    break;
  case 8:
    ReadFileControlPoints("control3DLSHeart08.txt");
    break;
  case 9:
    ReadFileControlPoints("control3DLSHeart09.txt");
    break;
  case 10:
    ReadFileControlPoints("control3DLSHeart10.txt");
    break;
  case 11:
    ReadFileControlPoints("control3DLSHeart11.txt");
    break;
  case 12:
    ReadFileControlPoints("control3DLSHeart12.txt");
    break;
  case 13:
    ReadFileControlPoints("control3DLSHeart13.txt");
    break;
  case 14:
    ReadFileControlPoints("control3DLSHeart14.txt");
    break;
  case 15:
    ReadFileControlPoints("control3DLSHeart15.txt");
    break;
  case 16:
    ReadFileControlPoints("control3DLSHeart16.txt");
    break;
  case 17:
    ReadFileControlPoints("control3DLSHeart17.txt");
    break;
  case 18:
    ReadFileControlPoints("control3DLSHeart18.txt");
    break;
  case 19:
    ReadFileControlPoints("control3DLSHeart19.txt");
    break;
  case 20:
    ReadFileControlPoints("control3DLSHeart20.txt");
    break;
  }
}


////////////////////////////////////////////////////////////////////////////////
int vtkImplicitSplineForce::GenerateForce(vtkMatrix4x4 * transformMatrix, double force[3])
{
  int n=3; //Cubic spline

  int i,j;

  int flag = 0;

  double dSpline, value, deriv, gradient[3];

  // Calculate distance (dSpline) to B-spline surface
  dSpline = CalculateDistanceBasis(transformMatrix->GetElement(0,3), transformMatrix->GetElement(1,3), transformMatrix->GetElement(2,3), n, 3);
  //changes========
  //cout << "Distance To Spline:  " << dSpline << endl;
  
  if(dSpline > -1.0){
    flag = 1;
  }

  CalculateDistanceDerivativeBasis(transformMatrix->GetElement(0,3), transformMatrix->GetElement(1,3), transformMatrix->GetElement(2,3), n, 3, gradient);
  //changes========
  fnGaussValueDeriv(gammaSigmoid, dSpline, value, deriv);

  for(i=0;i<3;i++){
    force[i] = 0;
    for(j=0;j<3;j++){
      force[i] += gradient[i];//*this->transformMatrix->GetElement(j,i) // tool translation;  // Transpose of Jacobian matrix
      //changes========
    }
    force[i] *= -deriv;
  }

  //======== set Magnitude of the force to the values of potential fields
  double fMag = sqrt(force[0]*force[0]+force[1]*force[1]+force[2]*force[2]);
  if(fMag>1e-10){
    for(i=0;i<3;i++){
      force[i]=value*force[i]/fMag;
    }
  }

  return flag;
}


int vtkImplicitSplineForce::fnSigmoidValueDeriv(double a, double x, double& value, double& deriv)
{
  double tmp = exp(-a*x);
  
  value = 1/(1+tmp);

  if(tmp>1.0e20)  // avoid overflow???
    deriv = 0;
  else
    deriv = a*tmp/( (1+tmp)*(1+tmp) );
  
  return 0;
}

int vtkImplicitSplineForce::fnGaussValueDeriv(double a, double x, double& value, double& deriv)
{
  int m=2;
  a = 2;

  double tmp = exp(-pow(x*x/(a*a), m));
    
  value = 1-tmp;

  deriv = tmp*2*m*pow(x,2*m-1)/pow(a*a, m);
  
  return 0;
}

int vtkImplicitSplineForce::SetGamma(double gamma)
{
  gammaSigmoid = gamma;

  return 0;
}


int vtkImplicitSplineForce::CalculateDistanceDerivativeBasis(double x, double y, double z, int n, int direction, double *gradient)
{
  double u = x;
  double v = y;
  double w = z;
  int Nk;
  int Iu, Iv, Iw;
  int i,j,k;

  Iu = DimKnot_U-1-n;
  Iv = DimKnot_V-1-n;
  Iw = DimKnot_W-1-n;

  //   Calculate Iu, Iv, Iw;
  for(i=0;i<DimKnot_U;i++){
    if(u<knot1b[i]){
      Iu = i-1;
      break;
    }
  }
  if(Iu<n-1) Iu = n-1;
  else if(Iu>DimKnot_U-1-n) Iu = DimKnot_U-1-n;

  for(i=0;i<DimKnot_V;i++){
    if(v<knot2b[i]){
      Iv = i-1;
      break;
    }
  }
  if(Iv<n-1) Iv = n-1;
  else if(Iv>DimKnot_V-1-n) Iv = DimKnot_V-1-n;

  for(i=0;i<DimKnot_W;i++){
    if(w<knot3b[i]){
      Iw = i-1;
      break;
    }
  }
  if(Iw<n-1) Iw = n-1;
  else if(Iw>DimKnot_W-1-n) Iw = DimKnot_W-1-n;

  double Nu[10], Nv[10], Nw[10];
  double dNu[10], dNv[10], dNw[10];
  for(i=0;i<=n;i++){
    Nk = Iu - 2 + i;
    Nu[i] = BasisFunction3(Nk, knot1b, u, DimKnot_U-1);
    dNu[i] = BasisFunction3DerivativeD(Nk, knot1b, u, DimKnot_U-1);
  }
  for(i=0;i<=n;i++){
    Nk = Iv - 2 + i;
    Nv[i] = BasisFunction3(Nk, knot2b, v, DimKnot_V-1);
    dNv[i] = BasisFunction3DerivativeD(Nk, knot2b, v, DimKnot_V-1);
  }
  for(i=0;i<=n;i++){
    Nk = Iw - 2 + i;
    Nw[i] = BasisFunction3(Nk, knot3b, w, DimKnot_W-1);
    dNw[i] = BasisFunction3DerivativeD(Nk, knot3b, w, DimKnot_W-1);
  }

  double distDT;
  distDT = 0;
  for(k=0;k<=n;k++){
    for(i=0;i<=n;i++){
      for(j=0;j<=n;j++){
        distDT += controlQ3D[Iw-2+k][Iv-2+i][Iu-2+j]*Nw[k]*Nv[i]*dNu[j];
      }
    }
  }
  gradient[0] = distDT;

  distDT = 0;
  for(k=0;k<=n;k++){
    for(i=0;i<=n;i++){
      for(j=0;j<=n;j++){
        distDT += controlQ3D[Iw-2+k][Iv-2+i][Iu-2+j]*Nw[k]*dNv[i]*Nu[j];
      }
    }
  }
  gradient[1] = distDT;

  distDT = 0;
  for(k=0;k<=n;k++){
    for(i=0;i<=n;i++){
      for(j=0;j<=n;j++){
        distDT += controlQ3D[Iw-2+k][Iv-2+i][Iu-2+j]*dNw[k]*Nv[i]*Nu[j];
      }
    }
  }
  gradient[2] = distDT;

  return 0;
}


// B-Spline basis function:
double vtkImplicitSplineForce::BasisFunction3(int k, double *knot, double u, int K)
{
  int n=3;
  double Nu = 0;
  double tmp1, tmp2, tmp3;
  
  // If u is outside of knot sequence, return 0;
  if( u<knot[0] || u>knot[K]) return 0;  //???

  //exception k=0
  if(k==0){  
    if(u<knot[k+1]){
      tmp1 = (u-knot[k-1])/(knot[k+1]-knot[k-1])*(knot[k+1]-u)/(knot[k+1]-knot[k]);
      tmp2 = (knot[k+2]-u)/(knot[k+2]-knot[k])*(u-knot[k])/(knot[k+1]-knot[k]);
      tmp3 = (knot[k+3]-u)/(knot[k+3]-knot[k])*(u-knot[k])/(knot[k+2]-knot[k])*(u-knot[k])/(knot[k+1]-knot[k]);
      Nu = (u-knot[k-1])/(knot[k+2]-knot[k-1])*(tmp1+tmp2) + tmp3;
    }
    else if(u<knot[k+2]){
      tmp1 = (u-knot[k])/(knot[k+2]-knot[k])*(knot[k+2]-u)/(knot[k+2]-knot[k+1]);
      tmp2 = (knot[k+3]-u)/(knot[k+3]-knot[k+1])*(u-knot[k+1])/(knot[k+2]-knot[k+1]);
      tmp3 = (u-knot[k-1])/(knot[k+2]-knot[k-1])*(knot[k+2]-u)/(knot[k+2]-knot[k])*(knot[k+2]-u)/(knot[k+2]-knot[k+1]);
      Nu = (knot[k+3]-u)/(knot[k+3]-knot[k])*(tmp1+tmp2) + tmp3;
    }
    else if(u<knot[k+3]){
      Nu = (knot[k+3]-u)*(knot[k+3]-u)*(knot[k+3]-u)/( (knot[k+3]-knot[k])*(knot[k+3]-knot[k+1])*(knot[k+3]-knot[k+2]) );
    }
  }

  //exception k=K-2
  else if(k==K-2){ 
    if(u<=knot[k]){
      Nu = (u - knot[k-1])*(u - knot[k-1])*(u - knot[k-1])/( (knot[k+2]-knot[k-1])*(knot[k+1]-knot[k-1])*(knot[k]-knot[k-1]) );
    }
    else if(u<knot[k+1]){
      tmp1 = (u-knot[k-1])/(knot[k+1]-knot[k-1])*(knot[k+1]-u)/(knot[k+1]-knot[k]);
      tmp2 = (knot[k+2]-u)/(knot[k+2]-knot[k])*(u-knot[k])/(knot[k+1]-knot[k]);
      //tmp3 = (knot[k+3]-u)/(knot[k+3]-knot[k])*(u-knot[k])/(knot[k+2]-knot[k])*(u-knot[k])/(knot[k+1]-knot[k]);
      //Nu = (u-knot[k-1])/(knot[k+2]-knot[k-1])*(tmp1+tmp2) + tmp3;
      Nu = (u-knot[k-1])/(knot[k+2]-knot[k-1])*(tmp1+tmp2);
    }
    else if(u<knot[k+2]){
      //tmp1 = (u-knot[k])/(knot[k+2]-knot[k])*(knot[k+2]-u)/(knot[k+2]-knot[k+1]);
      //tmp2 = (knot[k+3]-u)/(knot[k+3]-knot[k+1])*(u-knot[k+1])/(knot[k+2]-knot[k+1]);
      tmp3 = (u-knot[k-1])/(knot[k+2]-knot[k-1])*(knot[k+2]-u)/(knot[k+2]-knot[k])*(knot[k+2]-u)/(knot[k+2]-knot[k+1]);
      //Nu = (knot[k+3]-u)/(knot[k+3]-knot[k])*(tmp1+tmp2) + tmp3;
      Nu = tmp3;
    }
  }


  // Normal: 1<=k<=K-3
  else{  
    if(u<knot[k]){
      Nu = (u - knot[k-1])*(u - knot[k-1])*(u - knot[k-1])/( (knot[k+2]-knot[k-1])*(knot[k+1]-knot[k-1])*(knot[k]-knot[k-1]) );
    }
    else if(u<knot[k+1]){
      tmp1 = (u-knot[k-1])/(knot[k+1]-knot[k-1])*(knot[k+1]-u)/(knot[k+1]-knot[k]);
      tmp2 = (knot[k+2]-u)/(knot[k+2]-knot[k])*(u-knot[k])/(knot[k+1]-knot[k]);
      tmp3 = (knot[k+3]-u)/(knot[k+3]-knot[k])*(u-knot[k])/(knot[k+2]-knot[k])*(u-knot[k])/(knot[k+1]-knot[k]);
      Nu = (u-knot[k-1])/(knot[k+2]-knot[k-1])*(tmp1+tmp2) + tmp3;
    }
    else if(u<knot[k+2]){
      tmp1 = (u-knot[k])/(knot[k+2]-knot[k])*(knot[k+2]-u)/(knot[k+2]-knot[k+1]);
      tmp2 = (knot[k+3]-u)/(knot[k+3]-knot[k+1])*(u-knot[k+1])/(knot[k+2]-knot[k+1]);
      tmp3 = (u-knot[k-1])/(knot[k+2]-knot[k-1])*(knot[k+2]-u)/(knot[k+2]-knot[k])*(knot[k+2]-u)/(knot[k+2]-knot[k+1]);
      Nu = (knot[k+3]-u)/(knot[k+3]-knot[k])*(tmp1+tmp2) + tmp3;
    }
    else if(u<knot[k+3]){
      Nu = (knot[k+3]-u)*(knot[k+3]-u)*(knot[k+3]-u)/( (knot[k+3]-knot[k])*(knot[k+3]-knot[k+1])*(knot[k+3]-knot[k+2]) );
    }
  }

  return Nu;
}



// Calculate first derivative of cubic B-Spline basis function:
double vtkImplicitSplineForce::BasisFunction3DerivativeD(int k, double *knot, double u, int K)
{
  int n=3;
  double dNu = 0;
  double tmp1, tmp2;
  double dtmp,dtmp1,dtmp2,dtmp3;
  
  // If u is outside of knot sequence, return 0;
  if( u<knot[0] || u>knot[K]) return 0;  //???

  //exception k=0
  if(k==0){  
    if(u<knot[k+1]){
      tmp1 = (u-knot[k-1])/(knot[k+1]-knot[k-1])*(knot[k+1]-u)/(knot[k+1]-knot[k]);
      tmp2 = (knot[k+2]-u)/(knot[k+2]-knot[k])*(u-knot[k])/(knot[k+1]-knot[k]);

      dtmp1 = ( (knot[k+1]-u) + (u-knot[k-1])*(-1) )/(knot[k+1]-knot[k-1])/(knot[k+1]-knot[k]);
      dtmp2 = ( (-1)*(u-knot[k]) + (knot[k+2]-u) )/(knot[k+2]-knot[k])/(knot[k+1]-knot[k]);

      dtmp = (knot[k+3]-knot[k])*(knot[k+2]-knot[k])*(knot[k+1]-knot[k]);
      dtmp3 = ( (-1)*(u-knot[k])*(u-knot[k]) + (knot[k+3]-u)*2*(u-knot[k]) )/dtmp;

      dNu = 1/(knot[k+2]-knot[k-1])*(tmp1+tmp2) + (u-knot[k-1])/(knot[k+2]-knot[k-1])*(dtmp1+dtmp2)+dtmp3;
    }
    else if(u<knot[k+2]){
      tmp1 = (u-knot[k])/(knot[k+2]-knot[k])*(knot[k+2]-u)/(knot[k+2]-knot[k+1]);
      tmp2 = (knot[k+3]-u)/(knot[k+3]-knot[k+1])*(u-knot[k+1])/(knot[k+2]-knot[k+1]);

      dtmp1 = ( (knot[k+2]-u) + (u-knot[k])*(-1) )/(knot[k+2]-knot[k])/(knot[k+2]-knot[k+1]);
      dtmp2 = ( (-1)*(u-knot[k+1]) + (knot[k+3]-u) )/(knot[k+3]-knot[k+1])/(knot[k+2]-knot[k+1]);

      dtmp  = (knot[k+2]-knot[k-1])*(knot[k+2]-knot[k])*(knot[k+2]-knot[k+1]);
      dtmp3 = ( (knot[k+2]-u)*(knot[k+2]-u) + (u-knot[k-1])*(-2)*(knot[k+2]-u) )/dtmp;

      dNu = (-1)/(knot[k+3]-knot[k])*(tmp1+tmp2) + (knot[k+3]-u)/(knot[k+3]-knot[k])*(dtmp1+dtmp2)+dtmp3;
    }
    else if(u<knot[k+3]){
      dNu = (-3)*(knot[k+3]-u)*(knot[k+3]-u)/( (knot[k+3]-knot[k])*(knot[k+3]-knot[k+1])*(knot[k+3]-knot[k+2]) );
    }
  }

  //exception k=K-2
  else if(k==K-2){ 
    if(u<=knot[k]){
       dNu = 3*(u - knot[k-1])*(u - knot[k-1])/( (knot[k+2]-knot[k-1])*(knot[k+1]-knot[k-1])*(knot[k]-knot[k-1]) );
    }
    else if(u<knot[k+1]){
      tmp1 = (u-knot[k-1])/(knot[k+1]-knot[k-1])*(knot[k+1]-u)/(knot[k+1]-knot[k]);
      tmp2 = (knot[k+2]-u)/(knot[k+2]-knot[k])*(u-knot[k])/(knot[k+1]-knot[k]);

      dtmp1 = ( (knot[k+1]-u) + (u-knot[k-1])*(-1) )/(knot[k+1]-knot[k-1])/(knot[k+1]-knot[k]);
      dtmp2 = ( (-1)*(u-knot[k]) + (knot[k+2]-u) )/(knot[k+2]-knot[k])/(knot[k+1]-knot[k]);

      dNu = 1/(knot[k+2]-knot[k-1])*(tmp1+tmp2) + (u-knot[k-1])/(knot[k+2]-knot[k-1])*(dtmp1+dtmp2);
    }
    else if(u<knot[k+2]){
      dtmp  = (knot[k+2]-knot[k-1])*(knot[k+2]-knot[k])*(knot[k+2]-knot[k+1]);
      dtmp3 = ( (knot[k+2]-u)*(knot[k+2]-u) + (u-knot[k-1])*(-2)*(knot[k+2]-u) )/dtmp;

      dNu = dtmp3;
    }
  }


  // Normal: 1<=k<=K-3
  else{  
    if(u<knot[k]){
       dNu = 3*(u - knot[k-1])*(u - knot[k-1])/( (knot[k+2]-knot[k-1])*(knot[k+1]-knot[k-1])*(knot[k]-knot[k-1]) );
    }
    else if(u<knot[k+1]){
      tmp1 = (u-knot[k-1])/(knot[k+1]-knot[k-1])*(knot[k+1]-u)/(knot[k+1]-knot[k]);
      tmp2 = (knot[k+2]-u)/(knot[k+2]-knot[k])*(u-knot[k])/(knot[k+1]-knot[k]);

      dtmp1 = ( (knot[k+1]-u) + (u-knot[k-1])*(-1) )/(knot[k+1]-knot[k-1])/(knot[k+1]-knot[k]);
      dtmp2 = ( (-1)*(u-knot[k]) + (knot[k+2]-u) )/(knot[k+2]-knot[k])/(knot[k+1]-knot[k]);

      dtmp = (knot[k+3]-knot[k])*(knot[k+2]-knot[k])*(knot[k+1]-knot[k]);
      dtmp3 = ( (-1)*(u-knot[k])*(u-knot[k]) + (knot[k+3]-u)*2*(u-knot[k]) )/dtmp;

      dNu = 1/(knot[k+2]-knot[k-1])*(tmp1+tmp2) + (u-knot[k-1])/(knot[k+2]-knot[k-1])*(dtmp1+dtmp2)+dtmp3;
    }
    else if(u<knot[k+2]){
      tmp1 = (u-knot[k])/(knot[k+2]-knot[k])*(knot[k+2]-u)/(knot[k+2]-knot[k+1]);
      tmp2 = (knot[k+3]-u)/(knot[k+3]-knot[k+1])*(u-knot[k+1])/(knot[k+2]-knot[k+1]);

      dtmp1 = ( (knot[k+2]-u) + (u-knot[k])*(-1) )/(knot[k+2]-knot[k])/(knot[k+2]-knot[k+1]);
      dtmp2 = ( (-1)*(u-knot[k+1]) + (knot[k+3]-u) )/(knot[k+3]-knot[k+1])/(knot[k+2]-knot[k+1]);

      dtmp  = (knot[k+2]-knot[k-1])*(knot[k+2]-knot[k])*(knot[k+2]-knot[k+1]);
      dtmp3 = ( (knot[k+2]-u)*(knot[k+2]-u) + (u-knot[k-1])*(-2)*(knot[k+2]-u) )/dtmp;

      dNu = (-1)/(knot[k+3]-knot[k])*(tmp1+tmp2) + (knot[k+3]-u)/(knot[k+3]-knot[k])*(dtmp1+dtmp2)+dtmp3;
    }
    else if(u<knot[k+3]){
      dNu = (-3)*(knot[k+3]-u)*(knot[k+3]-u)/( (knot[k+3]-knot[k])*(knot[k+3]-knot[k+1])*(knot[k+3]-knot[k+2]) );
    }
  }

  return dNu;
}

///////////////////////////////////////////////////////////////////////////////////
double vtkImplicitSplineForce::CalculateDistanceBasis(double x, double y, double z, int n, int direction)
{
  double u = x;
  double v = y;
  double w = z;
  int Nk;
  int Iu, Iv, Iw;
  int i,j,k;

  //   Calculate Iu, Iv, Iw;
  Iu = CalculateKnotIu(u, knot1b, DimKnot_U-1, n);
  Iv = CalculateKnotIu(v, knot2b, DimKnot_V-1, n);
  Iw = CalculateKnotIu(w, knot3b, DimKnot_W-1, n);

  //double distance = BasisFunction3(Nk, knot, u, K);
  double Nu[10], Nv[10], Nw[10];
  for(i=0;i<=n;i++){
    Nk = Iu - 2 + i;
    Nu[i] = BasisFunction3(Nk, knot1b, u, DimKnot_U-1);
  }
  for(i=0;i<=n;i++){
    Nk = Iv - 2 + i;
    Nv[i] = BasisFunction3(Nk, knot2b, v, DimKnot_V-1);
  }
  for(i=0;i<=n;i++){
    Nk = Iw - 2 + i;
    Nw[i] = BasisFunction3(Nk, knot3b, w, DimKnot_W-1);
  }

  double distT = 0;
  for(k=0;k<=n;k++){
    for(i=0;i<=n;i++){
      for(j=0;j<=n;j++){
        distT += controlQ3D[Iw-2+k][Iv-2+i][Iu-2+j]*Nw[k]*Nv[i]*Nu[j];
      }
    }
  }

  return distT;
}

int vtkImplicitSplineForce::CalculateKnotIu(double u, double *knot, int K, int n)
{
  int Iu = K-n;

  for(int i=0;i<=K;i++){
    if(u<knot[i]){
      Iu = i-1;
      break;
    }
  }

  if(Iu<n-1) Iu = n-1;
  else if(Iu>K-n) Iu = K-n;

  return Iu;
}

int vtkImplicitSplineForce::ReadFile3DBSplineKnots(char *fname)
{
  // Read knots from file: fname
  //============================================================================
  int k;
  FILE *fpInKnot = fopen(fname, "r");

  // knot1/2/3
  for(k=0;k<=NUM_INTERVALU_S;k++){
    fscanf(fpInKnot, "%lf", &knot1[k]);
  }

  for(k=0;k<=NUM_INTERVALV_S;k++){
    fscanf(fpInKnot, "%lf", &knot2[k]);
  }

  for(k=0;k<=NUM_INTERVALW_S;k++){
    fscanf(fpInKnot, "%lf", &knot3[k]);
  }
  
  // knot1/2/3b
  for(k=0;k<=DimKnot_U-1;k++){
    fscanf(fpInKnot, "%lf", &knot1b[k]);
  }
  for(k=0;k<=DimKnot_V-1;k++){
    fscanf(fpInKnot, "%lf", &knot2b[k]);
  }
  for(k=0;k<=DimKnot_W-1;k++){
    fscanf(fpInKnot, "%lf", &knot3b[k]);
  }

//  fscanf(fpInKnot, "%lf %lf %lf %lf %lf %lf", &knot1Min,&knot1Max,&knot2Min,&knot2Max,&knot3Min,&knot3Max);
  
  fclose(fpInKnot);
  //============================================================================

  return 0;
}

int vtkImplicitSplineForce::ReadFileControlPoints(char *fname)
{
  // Read control points from file
  //============================================================================
  FILE *fpIn = fopen(fname, "r");

  int i,j,k;
  for(k=0;k<DimCPoint_W;k++){
    for(i=0;i<DimCPoint_V;i++){
      for(j=0;j<DimCPoint_U;j++){
        fscanf(fpIn,"%lf", &controlQ3D[k][i][j]);
      }
    }
  }

  fclose(fpIn);
  //============================================================================

  return 0;
}
