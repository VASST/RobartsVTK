#ifndef __VTKCUDAKOHONENGENERATOR_H__
#define __VTKCUDAKOHONENGENERATOR_H__

#include "vtkCudaImageAnalyticsExport.h"
#include "vtkVersionMacros.h"

#include "CUDA_kohonengenerator.h"
#include "CudaObject.h"
#include "vtkImageAlgorithm.h"
#include "vtkPiecewiseFunction.h"

class vtkAlgorithmOutput;
class vtkImageCast;
class vtkImageData;
class vtkInformation;
class vtkInformationVector;
class vtkTransform;

class vtkCudaImageAnalyticsExport vtkCudaKohonenGenerator : public vtkImageAlgorithm, public CudaObject
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
  void SetInputConnection(int idx, vtkAlgorithmOutput *input);

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
  vtkPiecewiseFunction* MeansAlphaSchedule;
  vtkPiecewiseFunction* MeansWidthSchedule;
  vtkPiecewiseFunction* VarsAlphaSchedule;
  vtkPiecewiseFunction* VarsWidthSchedule;
  vtkPiecewiseFunction* WeightsAlphaSchedule;
  vtkPiecewiseFunction* WeightsWidthSchedule;

  int outExt[6];

  Kohonen_Generator_Information Info;

  int    MaxEpochs;
  double  BatchPercent;
  bool  UseAllVoxels;

  bool  UseMask;

  Kohonen_Generator_Information& GetCudaInformation();

private:
  vtkCudaKohonenGenerator operator=(const vtkCudaKohonenGenerator&);
  vtkCudaKohonenGenerator(const vtkCudaKohonenGenerator&);
};

#endif