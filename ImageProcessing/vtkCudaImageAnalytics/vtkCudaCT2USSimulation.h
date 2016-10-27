#ifndef __vtkCudaCT2USSimulation_H__
#define __vtkCudaCT2USSimulation_H__

#include "vtkCudaImageAnalyticsExport.h"

#include "CUDA_cttoussimulation.h"
#include "CudaObject.h"
#include "vtkAlgorithm.h"

class vtkImageCast;
class vtkImageData;
class vtkTransform;

class vtkCudaImageAnalyticsExport vtkCudaCT2USSimulation : public vtkAlgorithm, public CudaObject
{
public:

  vtkTypeMacro(vtkCudaCT2USSimulation, vtkAlgorithm)

  static vtkCudaCT2USSimulation* New();

  void SetInput(vtkImageData*);
  void SetInput(vtkImageData*, int i);
  void SetTransform(vtkTransform*);
  void Update();
  vtkImageData* GetOutput();
  vtkImageData* GetOutput(int);

  //output parameters
  void SetOutputResolution(int x, int y, int z);
  void SetLogarithmicScaleFactor(float factor);
  void SetTotalReflectionThreshold(float threshold);
  void SetLinearCombinationAlpha(float a); //weighting for the reflection
  void SetLinearCombinationBeta(float b); //weighting for the density
  void SetLinearCombinationBias(float bias); //bias amount
  float GetLinearCombinationAlpha(); //weighting for the reflection
  float GetLinearCombinationBeta(); //weighting for the density
  float GetLinearCombinationBias(); //bias amount
  void SetDensityScaleModel(float scale, float offset);

  //probe geometry
  void SetProbeWidth(float x, float y);
  void SetFanAngle(float xAngle, float yAngle);
  void SetNearClippingDepth(float depth);
  void SetFarClippingDepth(float depth);

  //metric parameters
  float GetCrossCorrelation();

protected:
  vtkCudaCT2USSimulation();
  virtual ~vtkCudaCT2USSimulation();

  virtual void Reinitialize(bool withData = false);
  virtual void Deinitialize(bool withData = false);

  CT_To_US_Information Information;
  vtkTransform* UsTransform;

  vtkImageCast* Caster;

  vtkImageData* UsOutput;
  vtkImageData* DensOutput;
  vtkImageData* TransOutput;
  vtkImageData* ReflOutput;

  vtkImageData* InputUltrasound;

  float Alpha;
  float Beta;
  float Bias;

  bool AutoGenerateLinearCombination;

private:
  vtkCudaCT2USSimulation operator=(const vtkCudaCT2USSimulation&);
  vtkCudaCT2USSimulation(const vtkCudaCT2USSimulation&);
};

#endif