#include "vtkPowellMinimizer.h"
#include "vtkObjectFactory.h"


#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d) // for mnbrak and brent


// Modified mnbrak which knows about the whole multidimensional problem and
// computes its own parameter sets parametrized by the minimization factor.
// This requires the multiplication of each of the parameters passed in curParms
// by the actual value for the single variable function, placing the results in
// The *parms vector which can be accessed by the objective function when it is
// called. It places its result into the *retParm parameter.
#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
static double maxarg1, maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b), (maxarg1) > (maxarg2) ? (maxarg1) : (maxarg2))
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a): -fabs(a))

static void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb, double *fc,
       double *xi, void (*funk)(void *data), double *curParms, double *parms, 
       double *retParm, int n, void *data)
{
  double ulim, u,r,q,fu,dum;
  int i;

  for (i=0; i<n; i++) 
    {
    parms[i] = curParms[i]+xi[i]*(*ax);
    }
  (*funk)(data);
  *fa = (*retParm);

  for (i=0;i<n;i++) 
    {
    parms[i] = curParms[i]+xi[i]*(*bx);
    }
  (*funk)(data);
  *fb = (*retParm);
  
  if (*fb > *fa) 
    {
    SHFT(dum,*ax,*bx,dum);
    SHFT(dum,*fb,*fa,dum);
    }

  *cx = (*bx) + GOLD*(*bx-*ax);
  for (i=0;i<n;i++) 
    {
    parms[i] = curParms[i]+xi[i]*(*cx);
    }
  (*funk)(data);
  *fc = (*retParm);
  
  //  cout << "curParms (brak)("<<curParms[0]<<","<<curParms[1]<<")";
  while (*fb > *fc) 
    {
    r = (*bx-*ax)*(*fb-*fc);
    q = (*bx-*cx)*(*fb-*fa);
    u = (*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/
      (2.0*SIGN(FMAX(fabs(q-r),TINY),q-r));
    ulim=(*bx)+GLIMIT*(*cx-*bx);
    if ((*bx-u)*(u-*cx) > 0.0) 
      {
      for (i=0;i<n;i++) 
  {
  parms[i] = curParms[i]+xi[i]*u;
  }
      (*funk)(data);
      fu = (*retParm);
      if (fu < *fc) 
  {
  *ax = (*bx);
  *bx = u;
  *fa=(*fb);
  *fb = fu;
  return;
  } 
      else if (fu > *fb) 
  {
  *cx = u;
  *fc = fu;
  return;
  }
      u = (*cx)+GOLD*(*cx-*bx);
      for (i=0;i<n;i++)
  {
  parms[i] = curParms[i]+xi[i]*u;
  }
      (*funk)(data);
      fu = (*retParm);
      } 
    else if ((*cx-u)*(u-ulim) > 0.0) {
      for (i=0;i<n;i++) parms[i] = curParms[i]+xi[i]*u;
      (*funk)(data);
      fu = (*retParm);
      if (fu < *fc) {
  SHFT (*bx, *cx, u, *cx+GOLD*(*cx-*bx));
  for (i=0;i<n;i++) parms[i] = curParms[i]+xi[i]*u;
  (*funk)(data);
  SHFT (*fb, *fc, fu, (*retParm));
      }
    } else if ((u-ulim)*(ulim-*cx) >= 0.0) {
      u = ulim;
      for (i=0;i<n;i++) parms[i] = curParms[i]+xi[i]*u;
      (*funk)(data);
      fu = (*retParm);
    } else {
      u = (*cx) + GOLD*(*cx-*bx);
      for (i=0;i<n;i++) parms[i] = curParms[i]+xi[i]*u;
      (*funk)(data);
      fu = (*retParm);
    }
    SHFT(*ax, *bx, *cx, u);
    SHFT(*fa, *fb, *fc, fu);
  }
}
      
       



// a modified brent which knows about the whole multidimensional problem
// and computes its own parameter sets parametrized by the minimization
// factor.
#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10;
static double brent (double ax, double bx, double cx, double tol, int n,
        double *xmin, double *xi, void (*funk)(void *data), double *curParms,
        double *parms, double *retParm, void *data)
{
  int i,iter;
  double a,b,d,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
  double e=0.0;

  a=(ax < cx ? ax : cx);
  b=(ax > cx ? ax : cx);
  x=w=v=bx;
  //    cout << "curParms ("<<curParms[0]<<","<<curParms[1]<<")";
  //   cout << "a="<<a<<" b="<<b<<" x="<<x<<"\n";
  for (i=0;i<n;i++) parms[i] = curParms[i]+xi[i]*x;
  //  cout << "call 1\n";
  (*funk)(data);
  fw=fv=fx=(*retParm);
  for(iter=1;iter<=ITMAX;iter++){
    xm=0.5*(a+b);
    tol1=tol*fabs(x)+ZEPS;
    tol2=2.0*tol1;
    //    cout << "Done yet? x-xm="<<fabs(x-xm)<<" and tol2..="<<tol2-0.5*(b-a)<<"\n";
    if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
      *xmin=x;
      return fx;
    }
    if (fabs(e) > tol1) {
      r=(x-w)*(fx-fv);
      q=(x-v)*(fx-fw);
      p=(x-v)*q-(x-w)*r;
      q=2.0*(q-r);
      if (q > 0.0) p = -p;
      q = fabs(q);
      etemp = e;
      e=d;
      if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x)) {
  e = (x >= xm ? a-x : b-x);
  d = CGOLD*e;
      } else {
  d=p/q;
  u=x+d;
  if (u-a < tol2 || b-u < tol2)
    d = SIGN(tol1,xm-x);
      }
    } else {
      e=(x >= xm ? a-x : b-x);
      d=CGOLD*e;
    }
    u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
    //    cout << "u is now "<<u<<"\n";
    for (i=0;i<n;i++) parms[i] = curParms[i]+xi[i]*u;
    //  cout << "call 2\n";
    (*funk)(data);
    fu = (*retParm);
    //    cout << "and the associated fu is "<<fu<<"while fx is "<<fx<<"and x="<<x<<"\n";
    if (fu <= fx) {
      //cout << "hi\n";
      if (u >= x) a=x; else b=x;
      SHFT(v,w,x,u);
      SHFT(fv,fw,fx,fu);
    } else {
      if (u < x) a=u; else b=u;
      if (fu <= fw || w==x) {
  v=w;
  w=u;
  fv=fw;
  fw=fu;
      } else if (fu <= fv || v== x || v == w) {
  v=u;
  fv=fu;
      }
    }
  }
  cout << "too many iterations in brent\n";
  *xmin=x;
  return fx;
}


#define TOL 0.005
static void linmin(double *p, double *xi, int n, double *fret,
       void (*funk)(void *data), double *parms, double *retParm,
       void *data)
{
  int j;
  double xx, xmin, fx, fb, fa, bx, ax;
  
  ax = 0.0;
  xx = 1.0;
  //    cout << "before mnbrak\n";
  mnbrak (&ax, &xx, &bx, &fa, &fx, &fb, xi, funk, p, parms, retParm, n, data);
  //    cout << "before brent\n";
  *fret = brent(ax, xx, bx, TOL, n, &xmin, xi, funk, p, parms, retParm, data);
  for (j=0;j<n;j++){
    xi[j] *=xmin;
    p[j] += xi[j];
  }
}


static void powell(double *p, double **xi, int n, double ftol, 
       int *iter, double *retParm, void (*funk)(void *data), 
       double *parms, void *data)
{
  int i, ibig, j;
  double del, fp, fptt, t;
  double *pt = new double [n];
  double *ptt = new double [n];
  double *xit = new double[n];

  double fret;

  for (j=0; j<n; j++) pt[j] = p[j]; // save initial point

  (*funk)(data); // evaluates function on p and puts result in fret
  fret = *retParm;


  for (*iter=1;;++(*iter)) {
    fp = fret;
    ibig = 0;
    del = 0.0;
    for (i=0;i<n;i++){
      for(j=0;j<n;j++) xit[j] = xi[j][i];
      fptt=fret;
      //           cout << "fret before linmin "<< fret<<"\n";
      linmin (p, xit, n, &fret, funk, parms, retParm, data);
      //    cout << "fret after linmin "<< fret<<"\n";
      if (fabs(fptt-fret) > del) {
  del = fabs(fptt-fret);
  ibig = i;
      }
    }
    if (2.0*fabs(fp-fret) <= ftol*(fabs(fp)+fabs(fret))) {
      delete [] pt;
      delete [] ptt;
      delete [] xit;
      return;
    }
    if (*iter == ITMAX) cout << "powell exceeding ITMAX\n";
    for (j=0;j<n; j++) {
      ptt[j] = 2.0*p[j]-pt[j];
      xit[j] = p[j] - pt[j];
      pt[j] = p[j];
    }

    for (j=0; j<n; j++) parms[j] = ptt[j]; // load the parms
    (*funk)(data);
    fptt = *retParm;
    if (fptt < fp) {
      t=2.0*(fp-2.0*fret+fptt)*sqrt(fp-fret-del)-del*sqrt(fp-fptt);
      if (t < 0.0) {
  linmin(p, xit, n, &fret, funk, parms, retParm, data);
  if (ibig>2) cout << "bad shit going down\n";
  for (j=0;j<n;j++) {
    xi[j][ibig] = xi[j][n-1];
    xi[j][n-1]=xit[j];
  }
      }
    }
  }
}
  


void vtkPowellMinimizerFunction(void *data)
{
  vtkPowellMinimizer *self = (vtkPowellMinimizer *)data;
  if (self->Function)
    {
    self->Function(self->FunctionArg);
    }
}             

//----------------------------------------------------------------------------
vtkPowellMinimizer *vtkPowellMinimizer::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkPowellMinimizer");
  if(ret)
    {
    return (vtkPowellMinimizer*)ret;
    }
  // If the factory was unable to create the object, then create it here.
  return new vtkPowellMinimizer;
}

//----------------------------------------------------------------------------
vtkPowellMinimizer::vtkPowellMinimizer()
{
  this->Function = NULL;
  this->FunctionArg = NULL;
  this->FunctionArgDelete = NULL;

  this->NumberOfParameters = 0;
  this->ParameterNames = NULL;
  //  this->ParameterIndices = NULL;
  this->Parameters = NULL;
  this->ParameterBrackets = NULL;

  this->ScalarResult = 0.0;

  this->Tolerance = 0.005;
  this->MaxIterations = 1000;
  this->Iterations = 0;
}
  
//----------------------------------------------------------------------------
vtkPowellMinimizer::~vtkPowellMinimizer()
{
  cout << "deleting the minimizer!\n";
  if ((this->FunctionArg) && (this->FunctionArgDelete)) 
    {
    (*this->FunctionArgDelete)(this->FunctionArg);
    }
  this->FunctionArg = NULL;
  this->FunctionArgDelete = NULL;
  this->Function = NULL;
  //  cout << this->FunctionArg << "is functionarg\n";

  if (this->ParameterNames)
    {
    for (int i = 0; i < this->NumberOfParameters; i++)
      {
      if (this->ParameterNames[i])
  {
        delete [] this->ParameterNames[i];
  }
      }
    delete [] this->ParameterNames;
    this->ParameterNames = NULL;
    }
  if (this->Parameters)
    {
    delete [] this->Parameters;
    this->Parameters = NULL;
    }
  if (this->ParameterBrackets)
    {
    delete [] this->ParameterBrackets;
    this->ParameterBrackets = NULL;
    }

  this->NumberOfParameters = 0;
} 

//----------------------------------------------------------------------------
void vtkPowellMinimizer::PrintSelf(ostream& os, vtkIndent indent)
{
  this->vtkObject::PrintSelf(os, indent);
  os << indent << "ScalarResult: " << this->ScalarResult << "\n";
  os << indent << "MaxIterations: " << this->MaxIterations << "\n";
  os << indent << "Iterations: " << this->Iterations << "\n";
  os << indent << "Tolerance: " << this->Tolerance << "\n";
}

//----------------------------------------------------------------------------
void vtkPowellMinimizer::SetFunction(void (*f)(void *), void *arg)
{
  if ( f != this->Function || arg != this->FunctionArg )
    {
    // delete the current arg if there is one and a delete meth
    if ((this->FunctionArg) && (this->FunctionArgDelete))
      {
      (*this->FunctionArgDelete)(this->FunctionArg);
      }
    this->Function = f;
    this->FunctionArg = arg;
    this->Modified();
    }
}

//----------------------------------------------------------------------------
void vtkPowellMinimizer::SetFunctionArgDelete(void (*f)(void *))
{
  if ( f != this->FunctionArgDelete)
    {
    this->FunctionArgDelete = f;
    this->Modified();
    }
}

//----------------------------------------------------------------------------
double *vtkPowellMinimizer::GetScalarVariableBracket(const char *name)
{
  static double errval[2] = { 0.0, 0.0 };

  for (int i = 0; i < this->NumberOfParameters; i++)
    {
    if (strcmp(name,this->ParameterNames[i]) == 0)
      {
      return this->ParameterBrackets[i];
      }
    }

  vtkErrorMacro("GetScalarVariableBracket: no parameter named " << name);
  return errval;
}

//----------------------------------------------------------------------------
double vtkPowellMinimizer::GetScalarVariableValue(const char *name)
{
  for (int i = 0; i < this->NumberOfParameters; i++)
    {
    if (strcmp(name,this->ParameterNames[i]) == 0)
      {
      return this->Parameters[i];
      }
    }
  vtkErrorMacro("GetScalarVariableValue: no parameter named " << name);
  return 0.0;
}

//----------------------------------------------------------------------------
// initialize the simplex, also find the indices of the variables
int vtkPowellMinimizer::Initialize()
{
  if (!this->Function)
    {
    vtkErrorMacro("Initialize: Function is NULL!");
    return 0;
    }

  this->Iterations = 0;
  
  return 1;
}

//----------------------------------------------------------------------------
void vtkPowellMinimizer::SetScalarVariableBracket(const char *name, 
                double bmin, double bmax)
{
  int i;

  for (i = 0; i < this->NumberOfParameters; i++)
    {
    if (strcmp(name,this->ParameterNames[i]) == 0)
      {
      if (this->ParameterBrackets[i][0] != bmin ||
    this->ParameterBrackets[i][1] != bmax)
  {
  this->ParameterBrackets[i][0] = bmin;
  this->ParameterBrackets[i][1] = bmax;
  this->Modified();
  }
      return;
      }
    }

  int n = this->NumberOfParameters + 1;
  char **newParameterNames = new char *[n];
  //  int *newParameterIndices = new int[n];
  double *newParameters = new double[n];
  double (*newParameterBrackets)[2] = new double[n][2];

  for (i = 0; i < this->NumberOfParameters; i++)
    {
    newParameterNames[i] = this->ParameterNames[i];
    newParameterBrackets[i][0] = this->ParameterBrackets[i][0];
    newParameterBrackets[i][1] = this->ParameterBrackets[i][1];
    }

  char *cp = new char[strlen(name)+8];
  strcpy(cp,name);
  newParameterNames[n-1] = cp;
  newParameterBrackets[n-1][0] = bmin;
  newParameterBrackets[n-1][1] = bmax;

  if (this->ParameterNames)
    {
    delete [] this->ParameterNames;
    }
  if (this->Parameters)
    {
    delete [] this->Parameters;
    }
  if (this->ParameterBrackets)
    {
    delete [] this->ParameterBrackets;
    }

  this->NumberOfParameters = n;
  this->ParameterNames = newParameterNames;
  this->Parameters = newParameters;
  this->ParameterBrackets = newParameterBrackets;

  this->Modified();
}

//----------------------------------------------------------------------------
void vtkPowellMinimizer::Minimize()
{
  if (!this->Initialize())
    {
    return;
    }

  double **Vertices = new double *[this->NumberOfParameters];
  //  double *mem = new double[this->NumberOfParameters * this->NumberOfParameters];
  for (int l = 0; l < this->NumberOfParameters; l++)
    {
      // initial parameter values are bottom of bracket
      this->Parameters[l] = this->ParameterBrackets[l][0];
      Vertices[l] = new double[this->NumberOfParameters];
      // set up the initial matrix
      for (int m = 0; m < this->NumberOfParameters; m++)
  {
    if (m-l)
      Vertices[l][m] = 0.0f;
    else
      Vertices[l][m] = 1.0f;
  }
    }

  double *p = new double[this->NumberOfParameters];
  for (int i=0;i<this->NumberOfParameters;i++) p[i] = this->Parameters[i];

  powell(p, Vertices, this->NumberOfParameters, 
   this->Tolerance, &this->Iterations, &this->ScalarResult, 
   &vtkPowellMinimizerFunction, this->Parameters, this);

  for (int iGN=0; iGN<this->NumberOfParameters;iGN++) 
    {
    this->Parameters[iGN] = p[iGN];
    }
  delete [] p;
  for (int lGN=0; lGN < this->NumberOfParameters; lGN++)
    delete [] Vertices[lGN];
  delete [] Vertices;

}

