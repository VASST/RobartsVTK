#ifndef __vtkPowellMinimizer_h
#define __vtkPowellMinimizer_h

#include "vtkRobartsRegistrationExport.h"

#include "vtkObject.h"

class vtkRobartsRegistrationExport vtkPowellMinimizer : public vtkObject
{
public:
  static vtkPowellMinimizer *New();

  vtkTypeMacro(vtkPowellMinimizer,vtkObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Specify the function to be minimized.
  void SetFunction(void (*f)(void *), void *arg);

  // Description:
  // Set a function to call when a void* argument is being discarded.
  void SetFunctionArgDelete(void (*f)(void *));

  // Description:
  // Specify a variable to modify during the minimization.  Only the
  // variable you specify will be modified.  You must specify estimated
  // min and max possible values for each variable.
  void SetScalarVariableBracket(const char *name, double min, double max);
  void SetScalarVariableBracket(const char *name, const double range[2]);;
  double *GetScalarVariableBracket(const char *name);
  void GetScalarVariableBracket(const char *name, double range[2]);;

  // Description:
  // Get the value of a variable at the current stage of the minimization.
  double GetScalarVariableValue(const char *name);

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
  vtkSetMacro(ScalarResult,double);
  double GetScalarResult();;

  // Description:
  // Specify the fractional tolerance to aim for during the minimization.
  vtkSetMacro(Tolerance,double);
  vtkGetMacro(Tolerance,double);

  // Description:
  // Specify the maximum number of iterations to try before
  // printing an error and aborting.
  vtkSetMacro(MaxIterations,int);
  vtkGetMacro(MaxIterations,int);

  // Description:
  // Return the number of interactions required for the last
  // minimization that was performed.
  vtkGetMacro(Iterations,int);

protected:
  vtkPowellMinimizer();
  ~vtkPowellMinimizer();
  vtkPowellMinimizer(const vtkPowellMinimizer&) {};
  void operator=(const vtkPowellMinimizer&) {};

//BTX
  void (*Function)(void *);
  void (*FunctionArgDelete)(void *);
  void *FunctionArg;
//ETX

  int NumberOfParameters;
  char **ParameterNames;
  double *Parameters;
//BTX
  double (*ParameterBrackets)[2];
//ETX

  double ScalarResult;

  double Tolerance;
  int MaxIterations;
  int Iterations;

//BTX
  friend void vtkPowellMinimizerFunction(void *data);
//ETX
};

#endif
