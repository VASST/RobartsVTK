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

#ifndef __vtkPowellMinimizer_h
#define __vtkPowellMinimizer_h

#include "vtkRobartsRegistrationExport.h"
#include "vtkObject.h"

#include <vector>

class vtkRobartsRegistrationExport vtkPowellMinimizer : public vtkObject
{
public:
  static vtkPowellMinimizer* New();
  vtkTypeMacro(vtkPowellMinimizer, vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Specify the function to be minimized.
  void SetFunction(void (*f)(void*), void* arg);

  // Description:
  // Set a function to call when a void* argument is being discarded.
  void SetFunctionArgDelete(void (*f)(void*));

  // Description:
  // Specify a variable to modify during the minimization.  Only the
  // variable you specify will be modified.  You must specify estimated
  // min and max possible values for each variable.
  void SetScalarVariableBracket(const std::string& name, double min, double max);
  void SetScalarVariableBracket(const std::string& name, const double range[2]);
  std::pair<double, double> GetScalarVariableBracket(const std::string& name);
  void GetScalarVariableBracket(const std::string& name, std::pair<double, double>& range);

  // Description:
  // Get the value of a variable at the current stage of the minimization.
  double GetScalarVariableValue(const char* name);

  // Description:
  // Iterate until the minimum is found to within the specified tolerance.
  void Minimize();

  // Description:
  // Initialize the minimization (this must be called before Iterate,
  // but is not necessary before Minimize).
  int Initialize();

  // Description:
  // Perform one iteration of minimization.
  // void Iterate();

  // Description:
  // Get the current value resulting from the minimization.
  //
  vtkSetMacro(ScalarResult, double);
  double GetScalarResult();

  // Description:
  // Specify the fractional tolerance to aim for during the minimization.
  vtkSetMacro(Tolerance, double);
  vtkGetMacro(Tolerance, double);

  // Description:
  // Specify the maximum number of iterations to try before
  // printing an error and aborting.
  vtkSetMacro(MaxIterations, int);
  vtkGetMacro(MaxIterations, int);

  // Description:
  // Return the number of interactions required for the last
  // minimization that was performed.
  vtkGetMacro(Iterations, int);

protected:
  vtkPowellMinimizer();
  ~vtkPowellMinimizer();

protected:
  vtkPowellMinimizer(const vtkPowellMinimizer&) {};
  void operator=(const vtkPowellMinimizer&) {};

  void (*Function)(void*);
  void (*FunctionArgDelete)(void*);
  void* FunctionArg;

  std::vector<std::string>                ParameterNames;
  std::vector<double>                     Parameters;
  std::vector<std::pair<double, double>>  ParameterBrackets;

  double                                  ScalarResult;

  double                                  Tolerance;
  int                                     MaxIterations;
  int                                     Iterations;

  friend void vtkPowellMinimizerFunction(void* data);
};

#endif