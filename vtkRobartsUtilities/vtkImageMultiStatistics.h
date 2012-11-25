#ifndef __VTKIMAGEMULTISTATISTICS_H__
#define __VTKIMAGEMULTISTATISTICS_H__

#include "vtkProcessObject.h"
#include "vtkImageData.h"

class vtkImageMultiStatistics : public vtkProcessObject
{
public:
  static vtkImageMultiStatistics *New();

  vtkTypeMacro(vtkImageMultiStatistics,vtkProcessObject);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Compute and return the average magnitude.
  double GetAverageMagnitude(int component = 0);

  // Description:
  // Compute and return the standard deviation.
  double GetStandardDeviation(int component = 0);

  // Description:
  // Compute and return the standard deviation.
  long int GetCount();

  void Update();

  void SetInput(vtkImageData *input);
  vtkImageData *GetInput();


protected:
  vtkImageMultiStatistics();
  ~vtkImageMultiStatistics();

  void Execute();

  double* AverageMagnitude;
  double* StandardDeviation;
  long int Count;
  int NumberOfComponents;

  vtkTimeStamp ExecuteTime;

private:
  vtkImageMultiStatistics(const vtkImageMultiStatistics&) {}; //not implemented
  void operator=(const vtkImageMultiStatistics&) {}; //not implemented
};

#endif \\__VTKIMAGEMULTISTATISTICS_H__



