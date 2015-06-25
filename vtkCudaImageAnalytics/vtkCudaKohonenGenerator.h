#ifndef __VTKCUDAKOHONENGENERATOR_H__
#define __VTKCUDAKOHONENGENERATOR_H__

#include "CUDA_kohonengenerator.h"
#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageCast.h"
#include "vtkTransform.h"
#include "vtkCudaObject.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"

#include "vtkPiecewiseFunction.h"

#include <vtkVersion.h> // for VTK_MAJOR_VERSION

class vtkCudaKohonenGenerator : public vtkImageAlgorithm, public vtkCudaObject
{
public:
  vtkTypeMacro( vtkCudaKohonenGenerator, vtkImageAlgorithm );

  static vtkCudaKohonenGenerator *New();
  
  vtkSetObjectMacro(MeansAlphaSchedule,vtkPiecewiseFunction);
  vtkGetObjectMacro(MeansAlphaSchedule,vtkPiecewiseFunction);
  vtkSetObjectMacro(MeansWidthSchedule,vtkPiecewiseFunction);
  vtkGetObjectMacro(MeansWidthSchedule,vtkPiecewiseFunction);
  vtkSetObjectMacro(VarsAlphaSchedule,vtkPiecewiseFunction);
  vtkGetObjectMacro(VarsAlphaSchedule,vtkPiecewiseFunction);
  vtkSetObjectMacro(VarsWidthSchedule,vtkPiecewiseFunction);
  vtkGetObjectMacro(VarsWidthSchedule,vtkPiecewiseFunction);
  vtkSetObjectMacro(WeightsAlphaSchedule,vtkPiecewiseFunction);
  vtkGetObjectMacro(WeightsAlphaSchedule,vtkPiecewiseFunction);
  vtkSetObjectMacro(WeightsWidthSchedule,vtkPiecewiseFunction);
  vtkGetObjectMacro(WeightsWidthSchedule,vtkPiecewiseFunction);
  
  void SetNumberOfIterations(int number);
  int GetNumberOfIterations();

  void SetBatchSize(double fraction);
  double GetBatchSize();

  void SetKohonenMapSize(int SizeX, int SizeY);
  
  vtkDataObject* GetInput(int idx);
#if (VTK_MAJOR_VERSION <= 5)
  void SetInput(int idx, vtkDataObject *input);
#else
  void SetInputConnection(int idx, vtkAlgorithmOutput *input);
#endif

  bool GetUseMaskFlag();
  void SetUseMaskFlag(bool t);

  bool GetUseAllVoxelsFlag();
  void SetUseAllVoxelsFlag(bool t);

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
  vtkCudaKohonenGenerator();
  virtual ~vtkCudaKohonenGenerator();
  
  void Reinitialize(int withData);
  void Deinitialize(int withData);

private:

  Kohonen_Generator_Information& GetCudaInformation(){ return this->info; }

  vtkCudaKohonenGenerator operator=(const vtkCudaKohonenGenerator&){}
  vtkCudaKohonenGenerator(const vtkCudaKohonenGenerator&){}
  
  vtkPiecewiseFunction* MeansAlphaSchedule;
  vtkPiecewiseFunction* MeansWidthSchedule;
  vtkPiecewiseFunction* VarsAlphaSchedule;
  vtkPiecewiseFunction* VarsWidthSchedule;
  vtkPiecewiseFunction* WeightsAlphaSchedule;
  vtkPiecewiseFunction* WeightsWidthSchedule;

  int outExt[6];

  Kohonen_Generator_Information info;

  int    MaxEpochs;
  double  BatchPercent;
  bool  UseAllVoxels;

  bool  UseMask;

};

#endif