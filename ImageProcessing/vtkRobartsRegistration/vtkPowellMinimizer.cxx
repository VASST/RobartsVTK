/*=========================================================================

Robarts Visualization Toolkit

Copyright (c) 2016 Virtual Augmentation and Simulation for Surgery and Therapy, Robarts Research Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

=========================================================================*/

// Plus includes
#include <PlusCommon.h>

#include "vtkPowellMinimizer.h"
#include "vtkObjectFactory.h"

//----------------------------------------------------------------------------

#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d) // for mnbrak and brent

//----------------------------------------------------------------------------
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

//----------------------------------------------------------------------------
static void mnbrak(double* ax, double* bx, double* cx, double* fa, double* fb, double* fc,
                   std::vector<double>& xi, void (*funk)(void* data), std::vector<double>& curParms, std::vector<double>& parms,
                   double* retParm, int n, void* data)
{
  double ulim, u, r, q, fu, dum;
  int i;

  for (i = 0; i < n; i++)
  {
    parms[i] = curParms[i] + xi[i] * (*ax);
  }
  (*funk)(data);
  *fa = (*retParm);

  for (i = 0; i < n; i++)
  {
    parms[i] = curParms[i] + xi[i] * (*bx);
  }
  (*funk)(data);
  *fb = (*retParm);

  if (*fb > *fa)
  {
    SHFT(dum, *ax, *bx, dum);
    SHFT(dum, *fb, *fa, dum);
  }

  *cx = (*bx) + GOLD * (*bx - *ax);
  for (i = 0; i < n; i++)
  {
    parms[i] = curParms[i] + xi[i] * (*cx);
  }
  (*funk)(data);
  *fc = (*retParm);

  while (*fb > *fc)
  {
    r = (*bx - *ax) * (*fb - *fc);
    q = (*bx - *cx) * (*fb - *fa);
    u = (*bx) - ((*bx - *cx) * q - (*bx - *ax) * r) / (2.0 * SIGN(FMAX(fabs(q - r), TINY), q - r));
    ulim = (*bx) + GLIMIT * (*cx - *bx);
    if ((*bx - u) * (u - *cx) > 0.0)
    {
      for (i = 0; i < n; i++)
      {
        parms[i] = curParms[i] + xi[i] * u;
      }
      (*funk)(data);
      fu = (*retParm);
      if (fu < *fc)
      {
        *ax = (*bx);
        *bx = u;
        *fa = (*fb);
        *fb = fu;
        return;
      }
      else if (fu > *fb)
      {
        *cx = u;
        *fc = fu;
        return;
      }
      u = (*cx) + GOLD * (*cx - *bx);
      for (i = 0; i < n; i++)
      {
        parms[i] = curParms[i] + xi[i] * u;
      }
      (*funk)(data);
      fu = (*retParm);
    }
    else if ((*cx - u) * (u - ulim) > 0.0)
    {
      for (i = 0; i < n; i++)
      {
        parms[i] = curParms[i] + xi[i] * u;
      }
      (*funk)(data);
      fu = (*retParm);
      if (fu < *fc)
      {
        SHFT(*bx, *cx, u, *cx + GOLD * (*cx - *bx));
        for (i = 0; i < n; i++)
        {
          parms[i] = curParms[i] + xi[i] * u;
        }
        (*funk)(data);
        SHFT(*fb, *fc, fu, (*retParm));
      }
    }
    else if ((u - ulim) * (ulim - *cx) >= 0.0)
    {
      u = ulim;
      for (i = 0; i < n; i++)
      {
        parms[i] = curParms[i] + xi[i] * u;
      }
      (*funk)(data);
      fu = (*retParm);
    }
    else
    {
      u = (*cx) + GOLD * (*cx - *bx);
      for (i = 0; i < n; i++)
      {
        parms[i] = curParms[i] + xi[i] * u;
      }
      (*funk)(data);
      fu = (*retParm);
    }
    SHFT(*ax, *bx, *cx, u);
    SHFT(*fa, *fb, *fc, fu);
  }
}

//----------------------------------------------------------------------------
// a modified brent which knows about the whole multidimensional problem
// and computes its own parameter sets parametrized by the minimization
// factor.
#define ITMAX 100
#define CGOLD 0.3819660
#define ZEPS 1.0e-10;
static double brent(double ax, double bx, double cx, double tol, int n,
                    double* xmin, std::vector<double>& xi, void (*funk)(void* data), std::vector<double>& curParms,
                    std::vector<double>& parms, double* retParm, void* data)
{
  int i, iter;
  double a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
  double e = 0.0;

  a = (ax < cx ? ax : cx);
  b = (ax > cx ? ax : cx);
  x = w = v = bx;
  for (i = 0; i < n; i++)
  {
    parms[i] = curParms[i] + xi[i] * x;
  }

  (*funk)(data);
  fw = fv = fx = (*retParm);
  for (iter = 1; iter <= ITMAX; iter++)
  {
    xm = 0.5 * (a + b);
    tol1 = tol * fabs(x) + ZEPS;
    tol2 = 2.0 * tol1;
    if (fabs(x - xm) <= (tol2 - 0.5 * (b - a)))
    {
      *xmin = x;
      return fx;
    }
    if (fabs(e) > tol1)
    {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0)
      {
        p = -p;
      }
      q = fabs(q);
      etemp = e;
      e = d;
      if (fabs(p) >= fabs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x))
      {
        e = (x >= xm ? a - x : b - x);
        d = CGOLD * e;
      }
      else
      {
        d = p / q;
        u = x + d;
        if (u - a < tol2 || b - u < tol2)
        {
          d = SIGN(tol1, xm - x);
        }
      }
    }
    else
    {
      e = (x >= xm ? a - x : b - x);
      d = CGOLD * e;
    }
    u = (fabs(d) >= tol1 ? x + d : x + SIGN(tol1, d));

    for (i = 0; i < n; i++)
    {
      parms[i] = curParms[i] + xi[i] * u;
    }

    (*funk)(data);
    fu = (*retParm);
    if (fu <= fx)
    {
      if (u >= x)
      {
        a = x;
      }
      else
      {
        b = x;
      }
      SHFT(v, w, x, u);
      SHFT(fv, fw, fx, fu);
    }
    else
    {
      if (u < x)
      {
        a = u;
      }
      else
      {
        b = u;
      }
      if (fu <= fw || w == x)
      {
        v = w;
        w = u;
        fv = fw;
        fw = fu;
      }
      else if (fu <= fv || v == x || v == w)
      {
        v = u;
        fv = fu;
      }
    }
  }
  cout << "too many iterations in brent\n";
  *xmin = x;
  return fx;
}

//----------------------------------------------------------------------------
#define TOL 0.005
static void linmin(std::vector<double>& p, std::vector<double>& xi, int n, double* fret,
                   void (*funk)(void* data), std::vector<double>& parms, double* retParm, void* data)
{
  int j;
  double xx, xmin, fx, fb, fa, bx, ax;

  ax = 0.0;
  xx = 1.0;
  mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, xi, funk, p, parms, retParm, n, data);
  *fret = brent(ax, xx, bx, TOL, n, &xmin, xi, funk, p, parms, retParm, data);
  for (j = 0; j < n; j++)
  {
    xi[j] *= xmin;
    p[j] += xi[j];
  }
}

//----------------------------------------------------------------------------
static void powell(std::vector<double>& p, std::vector<std::vector<double>>& xi, int n, double ftol,
                   int* iter, double* retParm, void (*funk)(void* data),
                   std::vector<double>& parms, void* data)
{
  int i, ibig, j;
  double del, fp, fptt, t;
  std::vector<double> pt(n);
  std::vector<double> ptt(n);
  std::vector<double> xit(n);

  double fret;

  for (j = 0; j < n; j++)
  {
    pt[j] = p[j];  // save initial point
  }

  (*funk)(data); // evaluates function on p and puts result in fret
  fret = *retParm;

  for (*iter = 1; ; ++(*iter))
  {
    fp = fret;
    ibig = 0;
    del = 0.0;
    for (i = 0; i < n; i++)
    {
      fptt = fret;
      xit = xi[i];
      linmin(p, xit, n, &fret, funk, parms, retParm, data);
      if (fabs(fptt - fret) > del)
      {
        del = fabs(fptt - fret);
        ibig = i;
      }
    }

    if (2.0 * fabs(fp - fret) <= ftol * (fabs(fp) + fabs(fret)))
    {
      return;
    }

    if (*iter == ITMAX)
    {
      cout << "powell exceeding ITMAX\n";
    }
    for (j = 0; j < n; j++)
    {
      ptt[j] = 2.0 * p[j] - pt[j];
      xit[j] = p[j] - pt[j];
      pt[j] = p[j];
    }

    for (j = 0; j < n; j++)
    {
      parms[j] = ptt[j];  // load the parms
    }
    (*funk)(data);
    fptt = *retParm;
    if (fptt < fp)
    {
      t = 2.0 * (fp - 2.0 * fret + fptt) * sqrt(fp - fret - del) - del * sqrt(fp - fptt);
      if (t < 0.0)
      {
        linmin(p, xit, n, &fret, funk, parms, retParm, data);
        if (ibig > 2)
        {
          cout << "bad shit going down\n";
        }
        for (j = 0; j < n; j++)
        {
          xi[j][ibig] = xi[j][n - 1];
          xi[j][n - 1] = xit[j];
        }
      }
    }
  }
}

//----------------------------------------------------------------------------
void vtkPowellMinimizerFunction(void* data)
{
  vtkPowellMinimizer* self = (vtkPowellMinimizer*)data;
  if (self->Function)
  {
    self->Function(self->FunctionArg);
  }
}

//----------------------------------------------------------------------------
vtkPowellMinimizer* vtkPowellMinimizer::New()
{
  // First try to create the object from the vtkObjectFactory
  vtkObject* ret = vtkObjectFactory::CreateInstance("vtkPowellMinimizer");
  if (ret)
  {
    return (vtkPowellMinimizer*)ret;
  }
  // If the factory was unable to create the object, then create it here.
  return new vtkPowellMinimizer;
}

//----------------------------------------------------------------------------
vtkPowellMinimizer::vtkPowellMinimizer()
  : Function(NULL)
  , FunctionArg(NULL)
  , FunctionArgDelete(NULL)
  , ScalarResult(0.0)
  , Tolerance(0.005)
  , MaxIterations(1000)
  , Iterations(0)
{

}

//----------------------------------------------------------------------------
vtkPowellMinimizer::~vtkPowellMinimizer()
{
  if ((this->FunctionArg) && (this->FunctionArgDelete))
  {
    (*this->FunctionArgDelete)(this->FunctionArg);
  }
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
void vtkPowellMinimizer::SetFunction(void (*f)(void*), void* arg)
{
  if (f != this->Function || arg != this->FunctionArg)
  {
    // delete the current arg if there is one and a delete method
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
void vtkPowellMinimizer::SetFunctionArgDelete(void (*f)(void*))
{
  if (f != this->FunctionArgDelete)
  {
    this->FunctionArgDelete = f;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
std::pair<double, double> vtkPowellMinimizer::GetScalarVariableBracket(const std::string& name)
{
  static double errval[2] = { 0.0, 0.0 };

  for (int i = 0; i < this->ParameterNames.size(); i++)
  {
    if (PlusCommon::IsEqualInsensitive(name, this->ParameterNames[i]))
    {
      return this->ParameterBrackets[i];
    }
  }

  vtkErrorMacro("GetScalarVariableBracket: no parameter named " << name);
  return std::pair<double, double>(-1.0, -1.0);
}

//----------------------------------------------------------------------------
void vtkPowellMinimizer::GetScalarVariableBracket(const std::string& name, std::pair<double, double>& range)
{
  range = this->GetScalarVariableBracket(name);
}

//----------------------------------------------------------------------------
double vtkPowellMinimizer::GetScalarVariableValue(const char* name)
{
  for (int i = 0; i < this->ParameterNames.size(); i++)
  {
    if (PlusCommon::IsEqualInsensitive(name, this->ParameterNames[i]))
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
double vtkPowellMinimizer::GetScalarResult()
{
  return this->ScalarResult;
}

//----------------------------------------------------------------------------
void vtkPowellMinimizer::SetScalarVariableBracket(const std::string& name, double bmin, double bmax)
{
  for (auto i = 0; i < this->ParameterNames.size(); ++i)
  {
    if (PlusCommon::IsEqualInsensitive(name, this->ParameterNames[i]))
    {
      if (this->ParameterBrackets[i].first != bmin ||
          this->ParameterBrackets[i].second != bmax)
      {
        this->ParameterBrackets[i].first = bmin;
        this->ParameterBrackets[i].second = bmax;
        this->Modified();
      }
      return;
    }
  }

  this->ParameterNames.push_back(name);
  this->Parameters.push_back(0.0);
  this->ParameterBrackets.push_back(std::pair<double, double>(bmin, bmax));

  this->Modified();
}

//----------------------------------------------------------------------------
void vtkPowellMinimizer::SetScalarVariableBracket(const std::string& name, const double range[2])
{
  this->SetScalarVariableBracket(name, range[0], range[1]);
}

//----------------------------------------------------------------------------
void vtkPowellMinimizer::Minimize()
{
  if (!this->Initialize())
  {
    return;
  }

  std::vector<std::vector<double>> vertices;
  for (unsigned int l = 0; l < this->ParameterNames.size(); l++)
  {
    // initial parameter values are bottom of bracket
    this->Parameters[l] = this->ParameterBrackets[l].first;
    std::vector<double> inside;
    inside.resize(this->Parameters.size());
    vertices.push_back(inside);

    // set up the initial matrix
    for (int m = 0; m < this->ParameterNames.size(); m++)
    {
      if (m - l)
      {
        vertices[l][m] = 0.0f;
      }
      else
      {
        vertices[l][m] = 1.0f;
      }
    }
  }

  std::vector<double> p(this->ParameterNames.size());
  for (int i = 0; i < this->ParameterNames.size(); i++)
  {
    p[i] = this->Parameters[i];
  }

  powell(p, vertices, this->Parameters.size(), this->Tolerance, &this->Iterations, &this->ScalarResult, &vtkPowellMinimizerFunction, this->Parameters, this);
}