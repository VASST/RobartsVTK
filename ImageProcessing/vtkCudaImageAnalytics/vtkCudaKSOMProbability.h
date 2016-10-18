#ifndef __VTKCUDAKSOMPROBABILITY_H__
#define __VTKCUDAKSOMPROBABILITY_H__

#include "vtkCudaImageAnalyticsExport.h"
#include "vtkVersionMacros.h"

#include "CUDA_KSOMProbability.h"
#include "vtkImageAlgorithm.h"
#include "CudaObject.h"
#include "float.h"

class vtkAlgorithmOutput;
class vtkImageCast;
class vtkImageData;
class vtkInformation;
class vtkInformationVector;
class vtkTransform;

class vtkCudaImageAnalyticsExport vtkCudaKSOMProbability : public vtkImageAlgorithm, public CudaObject
{
public:
  vtkTypeMacro( vtkCudaKSOMProbability, vtkImageAlgorithm );

  static vtkCudaKSOMProbability *New();

  vtkSetClampMacro(Scale,double,0,FLT_MAX);
  vtkGetMacro(Scale,double);

  vtkSetMacro(Entropy,bool);
  vtkGetMacro(Entropy,bool);
  void SetImageInputConnection(vtkAlgorithmOutput* in);
  void SetMapInputConnection(vtkAlgorithmOutput* in);
  void SetProbabilityInputConnection(vtkAlgorithmOutput* in, int index);

  // Description:
  // If the subclass does not define an Execute method, then the task
  // will be broken up, multiple threads will be spawned, and each thread
  // will call this method. It is public so that the thread functions
  // can call this method.
  virtual int RequestData(vtkInformation *request, 
               vtkInformationVector **inputVector, 
               vtkInformationVector *outputVector);
  virtual int RequestInformation( vtkInformation* request,
               vtkInformationVector** inputVector,
               vtkInformationVector* outputVector);
  virtual int RequestUpdateExtent( vtkInformation* request,
               vtkInformationVector** inputVector,
               vtkInformationVector* outputVector);
  virtual int FillInputPortInformation(int i, vtkInformation* info);

protected:
  vtkCudaKSOMProbability();
  virtual ~vtkCudaKSOMProbability();
  
  void Reinitialize(int withData);
  void Deinitialize(int withData);

  double Scale;

  bool Entropy;

  Kohonen_Probability_Information Info;

private:
  vtkCudaKSOMProbability operator=(const vtkCudaKSOMProbability&);
  vtkCudaKSOMProbability(const vtkCudaKSOMProbability&);
};

#endif
