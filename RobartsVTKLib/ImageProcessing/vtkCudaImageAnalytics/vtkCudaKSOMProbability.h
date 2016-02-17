#ifndef __VTKCUDAKSOMPROBABILITY_H__
#define __VTKCUDAKSOMPROBABILITY_H__

#include "CUDA_KSOMProbability.h"
#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"
#include "vtkCudaObject.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"
#include "vtkSetGet.h"

#include "float.h"
#include <vtkVersion.h> // for VTK_MAJOR_VERSION

class vtkCudaKSOMProbability : public vtkImageAlgorithm, public vtkCudaObject
{
public:
  vtkTypeMacro( vtkCudaKSOMProbability, vtkImageAlgorithm );

  static vtkCudaKSOMProbability *New();

  vtkSetClampMacro(Scale,double,0,FLT_MAX);
  vtkGetMacro(Scale,double);

  vtkSetMacro(Entropy,bool);
  vtkGetMacro(Entropy,bool);
#if (VTK_MAJOR_VERSION < 6)  
  void SetImageInput(vtkImageData* in);
  void SetMapInput(vtkImageData* in);
  void SetProbabilityInput(vtkImageData* in, int index);
#else
  void SetImageInputConnection(vtkAlgorithmOutput* in);
  void SetMapInputConnection(vtkAlgorithmOutput* in);
  void SetProbabilityInputConnection(vtkAlgorithmOutput* in, int index);
#endif

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

private:
  vtkCudaKSOMProbability operator=(const vtkCudaKSOMProbability&){}
  vtkCudaKSOMProbability(const vtkCudaKSOMProbability&){}

  double Scale;

  bool Entropy;

  Kohonen_Probability_Information info;
};

#endif
