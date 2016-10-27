#ifndef __vtkCudaFuzzyConnectednessFilter_H__
#define __vtkCudaFuzzyConnectednessFilter_H__

#include "vtkCudaImageAnalyticsExport.h"

#include "CUDA_fuzzyconnectednessfilter.h"
#include "CudaObject.h"
#include "vtkImageAlgorithm.h"

class vtkImageData;
class vtkInformation;
class vtkInformationVector;
struct vtkCudaFuzzyConnectednessFilterInformation;

class vtkCudaImageAnalyticsExport vtkCudaFuzzyConnectednessFilter : public vtkImageAlgorithm, public CudaObject
{
public:

  vtkTypeMacro(vtkCudaFuzzyConnectednessFilter, vtkImageAlgorithm)

  static vtkCudaFuzzyConnectednessFilter* New();

  // Description:
  // Get/Set the t-Norm and s-Norm type
  vtkSetClampMacro(TNorm, int, 0, 2);
  vtkGetMacro(TNorm, int);
  vtkSetClampMacro(SNorm, int, 0, 2);
  vtkGetMacro(SNorm, int);

protected:

  virtual void Reinitialize(bool withData = false);
  virtual void Deinitialize(bool withData = false);

  int RequestData(vtkInformation* request,
                  vtkInformationVector** inputVector,
                  vtkInformationVector* outputVector);

  vtkCudaFuzzyConnectednessFilter();
  virtual ~vtkCudaFuzzyConnectednessFilter();

private:
  vtkCudaFuzzyConnectednessFilter operator=(const vtkCudaFuzzyConnectednessFilter&);
  vtkCudaFuzzyConnectednessFilter(const vtkCudaFuzzyConnectednessFilter&);

  int TNorm;
  int SNorm;

  Fuzzy_Connectedness_Information* Information;
};

#endif