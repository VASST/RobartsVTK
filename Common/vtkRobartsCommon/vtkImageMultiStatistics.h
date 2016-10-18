/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkImageMultiStatistics.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __VTKIMAGEMULTISTATISTICS_H__
#define __VTKIMAGEMULTISTATISTICS_H__

#include "vtkRobartsCommonExport.h"

#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkInformation.h"

class vtkRobartsCommonExport vtkImageMultiStatistics : public vtkAlgorithm
{
public:
  static vtkImageMultiStatistics *New();

  vtkTypeMacro(vtkImageMultiStatistics,vtkAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Compute and return the average magnitude.
  double GetAverageMagnitude(int component = 0);

  // Description:
  // Compute and return the average magnitude.
  double GetMeanSquared(int component = 0);

  // Description:
  // Compute and return the standard deviation.
  double GetStandardDeviation(int component = 0);

  // Description:
  // Compute and return the covariance.
  double GetCovariance(int component1 = 0, int component2 = 0);

  // Description:
  // Compute and return the entropy of the image in terms of a given component.
  double GetSingleEntropy(int component = 0);

  // Description:
  // Compute and return the joint entropy of the image in terms of the given components
  double GetJointEntropy(int component1 = 0, int component2 = 0);

  // Description:
  // Compute and return the entropy of the image over all components.
  double GetTotalEntropy();

  // Description:
  // Compute and return the specified component of the nth significant PCA weighting vector
  double GetPCAWeight(int significance = 0, int component = 0);

  // Description:
  // Compute and return the nth significance in PCA
  double GetPCAVariance(int significance = 0);

  // Description:
  // Compute and return the number of pixels.
  long int GetCount();

  // Description:
  // Set the number of bins used in PDF estimation for entropy
  void SetEntropyResolution(int bins);
  int GetEntropyResolution();

  void Update();

  void SetInputData(int port, vtkImageData *input);
  void SetInputData(vtkImageData *input)
  {
    SetInputData(0,input);
  }
  void SetInputConnection(int port, vtkAlgorithmOutput *input);
  void SetInputConnection(vtkAlgorithmOutput  *input);

  vtkImageData *GetInput(int port);
  vtkImageData *GetInput();

  int FillInputPortInformation(int i, vtkInformation* info);

  vtkSetMacro(UseMask, bool);
  vtkGetMacro(UseMask, bool);

protected:
  vtkImageMultiStatistics();
  ~vtkImageMultiStatistics();

  void Execute();

  double* AverageMagnitude;
  double* MeanSquared;
  double** Covariance;
  double** JointEntropy;
  double TotalEntropy;
  long int Count;
  int NumberOfComponents;
  int NumberOfBins;

  double* PCAVariance;
  double** PCAAxisVectors;

  bool UseMask;

  vtkTimeStamp ExecuteTime;

private:
  vtkImageMultiStatistics(const vtkImageMultiStatistics&) {}; //not implemented
  void operator=(const vtkImageMultiStatistics&) {}; //not implemented
};

#endif
