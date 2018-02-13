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

#include "vtkFunctionMinimizer.h"
#include "vtkRobartsCommon.h"

// VTK includes
#include <vtkObjectFactory.h>

namespace
{
  //----------------------------------------------------------------------------
  static double amotry(std::vector<std::vector<double>>& p,
                       std::vector<double>& y,
                       std::vector<double>& ptry,
                       std::vector<double>& psum,
                       int ndim,
                       void(*funk)(void* data),
                       void* data,
                       double* result,
                       int ihi,
                       double fac)
  {
    int j;
    double fac1, fac2, ytry;

    fac1 = (1.0 - fac) / (double)ndim;
    fac2 = fac1 - fac;
    for (j = 0; j < ndim; j++)
    {
      ptry[j] = psum[j] * fac1 - p[ihi][j] * fac2;
    }
    (*funk)(data);
    ytry = *result;
    if (ytry < y[ihi])
    {
      y[ihi] = ytry;
      for (j = 0; j < ndim; j++)
      {
        psum[j] += ptry[j] - p[ihi][j];
        p[ihi][j] = ptry[j];
      }
    }
    return ytry;
  }

  //----------------------------------------------------------------------------
  static int amoeba(std::vector<std::vector<double>>& p,
                    std::vector<double>& y,
                    std::vector<double>& ptry,
                    int ndim,
                    double ftol,
                    void(*funk)(void* data),
                    void* data,
                    double* result,
                    int* nfunk,
                    int maxnfunk)
  {
    int i, ihi, ilo, inhi, j, mpts;
    double rtol, sum, swap, ysave, ytry;
    std::vector<double> psum(ndim);

    mpts = ndim + 1;
    *nfunk = 0;

    for (j = 0; j < ndim; j++)
    {
      sum = 0.0;
      for (i = 0; i < mpts; i++)
      {
        sum += p[i][j];
      }
      psum[j] = sum;
    }

    for (;;)
    {
      ilo = 0;
      if (y[0] > y[1])
      {
        ihi = 0;
        inhi = 1;
      }
      else
      {
        ihi = 1;
        inhi = 0;
      }
      for (i = 0; i < mpts; i++)
      {
        if (y[i] <= y[ilo])
        {
          ilo = i;
        }
        if (y[i] > y[ihi])
        {
          inhi = ihi;
          ihi = i;
        }
        else if (y[i] > y[inhi] && i != ihi)
        {
          inhi = i;
        }
      }

      if (fabs(y[ihi]) + fabs(y[ilo]) < ftol)
      {
        rtol = double(2.0 * fabs(y[ihi] - y[ilo]));
      }
      else
      {
        rtol = double(2.0 * fabs(y[ihi] - y[ilo]) / (fabs(y[ihi]) + fabs(y[ilo])));
      }

      if (rtol < ftol)
      {
        swap = y[1];
        y[1] = y[ilo];
        y[ilo] = swap;
        for (i = 0; i < ndim; i++)
        {
          swap = p[0][i];
          p[0][i] = p[ilo][i];
          p[ilo][i] = swap;
        }
        break;
      }
      if (*nfunk >= maxnfunk)
      {
        return -1;      /* break if greater than max number of func evals */
      }

      *nfunk += 2;
      ytry = amotry(p, y, ptry, psum, ndim, funk, data, result, ihi, double(-1.0));
      if (ytry <= y[ilo])
      {
        ytry = amotry(p, y, ptry, psum, ndim, funk, data, result, ihi, double(2.0));
      }
      else if (ytry >= y[inhi])
      {
        ysave = y[ihi];
        ytry = amotry(p, y, ptry, psum, ndim, funk, data, result, ihi, double(0.5));
        if (ytry >= ysave)
        {
          for (i = 0; i < mpts; i++)
          {
            if (i != ilo)
            {
              for (j = 0; j < ndim; j++)
              {
                p[i][j] = ptry[j] = psum[j] = (p[i][j] + p[ilo][j]) / double(2.0);
              }
              (*funk)(data);
              y[i] = *result;
            }
          }
          *nfunk += ndim;

          for (j = 0; j < ndim; j++)
          {
            sum = 0;
            for (i = 0; i < mpts; i++)
            {
              sum += p[i][j];
            }
            psum[j] = sum;
          }
        }
      }
      else
      {
        --(*nfunk);
      }
    }

    return 0;
  }

  //----------------------------------------------------------------------------
  static double minimize(std::vector<double>& parameters,
                         std::vector<std::vector<double>> vertices,
                         int ndim,
                         void(*function)(void* data),
                         void* data,
                         double* result,
                         double tolerance,
                         int maxiterations,
                         int* iterations)
  {
    std::vector<double> y(ndim + 1);

    for (int k = 0; k < ndim + 1; k++)
    {
      for (int l = 0; l < ndim; l++)
      {
        parameters[l] = vertices[k][l];
      }

      (*function)(data);
      y[k] = *result;
    }

    amoeba(vertices, y, parameters, ndim, tolerance, function, data, result, iterations, maxiterations);
    *result = y[1]; // copy the lowest result in the *result

    //set x equal to lowest of amoeba vertices
    for (int j = 0; j < ndim; j++)
    {
      parameters[j] = vertices[0][j];
    }

    return *result;
  }
}

//----------------------------------------------------------------------------

vtkStandardNewMacro(vtkFunctionMinimizer);

//----------------------------------------------------------------------------
void vtkFunctionMinimizerFunction(void* data)
{
  vtkFunctionMinimizer* self = (vtkFunctionMinimizer*)data;
  if (self->Function)
  {
    self->Function(self->FunctionArg);
  }
}

//----------------------------------------------------------------------------
vtkFunctionMinimizer::vtkFunctionMinimizer()
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
vtkFunctionMinimizer::~vtkFunctionMinimizer()
{
  if ((this->FunctionArg) && (this->FunctionArgDelete))
  {
    (*this->FunctionArgDelete)(this->FunctionArg);
  }
}

//----------------------------------------------------------------------------
void vtkFunctionMinimizer::PrintSelf(ostream& os, vtkIndent indent)
{
  this->vtkObject::PrintSelf(os, indent);
  os << indent << "ScalarResult: " << this->ScalarResult << "\n";
  os << indent << "MaxIterations: " << this->MaxIterations << "\n";
  os << indent << "Iterations: " << this->Iterations << "\n";
  os << indent << "Tolerance: " << this->Tolerance << "\n";
}

//----------------------------------------------------------------------------
void vtkFunctionMinimizer::SetFunction(void (*f)(void*), void* arg)
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
void vtkFunctionMinimizer::SetFunctionArgDelete(void (*f)(void*))
{
  if (f != this->FunctionArgDelete)
  {
    this->FunctionArgDelete = f;
    this->Modified();
  }
}

//----------------------------------------------------------------------------
std::pair<double, double> vtkFunctionMinimizer::GetScalarVariableBracket(const std::string& name)
{
  static double errval[2] = { 0.0, 0.0 };

  for (int i = 0; i < this->ParameterNames.size(); i++)
  {
    if (IsEqualInsensitive(name, this->ParameterNames[i]))
    {
      return this->ParameterBrackets[i];
    }
  }

  vtkErrorMacro("GetScalarVariableBracket: no parameter named " << name);
  return std::pair<double, double>(-1.0, -1.0);
}

//----------------------------------------------------------------------------
void vtkFunctionMinimizer::GetScalarVariableBracket(const std::string& name, std::pair<double, double>& range)
{
  range = this->GetScalarVariableBracket(name);
}

//----------------------------------------------------------------------------
double vtkFunctionMinimizer::GetScalarVariableValue(const std::string& name)
{
  for (int i = 0; i < this->ParameterNames.size(); i++)
  {
    if (IsEqualInsensitive(name, this->ParameterNames[i]))
    {
      return this->Parameters[i];
    }
  }
  vtkErrorMacro("GetScalarVariableValue: no parameter named " << name);
  return 0.0;
}

//----------------------------------------------------------------------------
// initialize the simplex, also find the indices of the variables
int vtkFunctionMinimizer::Initialize()
{
  if (!this->Function)
  {
    vtkErrorMacro("Initialize: Function is NULL");
    return 0;
  }

  this->Vertices.resize(this->ParameterNames.size());
  for (int l = 0; l < this->ParameterNames.size(); l++)
  {
    this->Vertices[l] = std::vector<double>(this->ParameterNames.size());
    // initial parameter values are middle of bracket
    this->Vertices[0][l] = 0.5 * (this->ParameterBrackets[l].first + this->ParameterBrackets[l].second);

    // set up the simplex vertices
    for (int m = 1; m <= this->ParameterNames.size(); m++)
    {
      this->Vertices[m][l] = this->Vertices[0][l];
      if ((m - 1) == l)
      {
        this->Vertices[m][l] = this->ParameterBrackets[l].second;
      }
    }
  }

  this->Iterations = 0;

  return 1;
}

//----------------------------------------------------------------------------
void vtkFunctionMinimizer::SetScalarVariableBracket(const std::string& name, double bmin, double bmax)
{
  for (auto i = 0; i < this->ParameterNames.size(); i++)
  {
    if (IsEqualInsensitive(name, this->ParameterNames[i]))
    {
      if (this->ParameterBrackets[i].first != bmin || this->ParameterBrackets[i].second != bmax)
      {
        this->ParameterBrackets[i].first = bmin;
        this->ParameterBrackets[i].second = bmax;

        this->Modified();
      }
      return;
    }
  }

  this->ParameterNames.push_back(name);
  this->ParameterBrackets.push_back(std::pair<double, double>(bmin, bmax));
  this->Parameters.push_back(0.0);
  this->Vertices.push_back(std::vector<double>(this->ParameterNames.size()));

  this->Modified();
}

//----------------------------------------------------------------------------
void vtkFunctionMinimizer::SetScalarVariableBracket(const std::string& name, const std::pair<double, double>& range)
{
  this->SetScalarVariableBracket(name, range.first, range.second);
}

//----------------------------------------------------------------------------
void vtkFunctionMinimizer::Minimize()
{
  if (!this->Initialize())
  {
    return;
  }
  minimize(this->Parameters, this->Vertices, this->ParameterNames.size(), &vtkFunctionMinimizerFunction, this, &this->ScalarResult, this->Tolerance, this->MaxIterations, &this->Iterations);
}

