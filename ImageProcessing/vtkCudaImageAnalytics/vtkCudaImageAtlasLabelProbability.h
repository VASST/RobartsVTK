/*=========================================================================

Program:   Visualization Toolkit
Module:    vtkCudaImageAtlasLabelProbability.h

Copyright (c) John SH Baxter, Robarts Research Institute

    This software is distributed WITHOUT ANY WARRANTY; without even
    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaImageAtlasLabelProbability.h
*
*  @brief Header file with definitions for the CUDA accelerated label agreement data term. This
*      generates entropy or probability data terms based on how many label maps agree with a particular
*      labeling of each voxel.
*
*  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
*
*  @note August 27th 2013 - Documentation first compiled.
*
*/

#ifndef __vtkCudaImageAtlasLabelProbability_h
#define __vtkCudaImageAtlasLabelProbability_h

#include "vtkCudaImageAnalyticsExport.h"
#include "vtkVersionMacros.h"

#include "CudaObject.h"
#include "vtkImageAlgorithm.h"
#include <float.h>

class vtkDataObject;

class vtkCudaImageAnalyticsExport vtkCudaImageAtlasLabelProbability : public vtkImageAlgorithm, public CudaObject
{
public:
  static vtkCudaImageAtlasLabelProbability* New();
  vtkTypeMacro(vtkCudaImageAtlasLabelProbability, vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set a collection of label maps for the seeding operation.
  virtual void SetInputLabelMapConnection(vtkAlgorithmOutput* in, int number);

  // Description:
  // Determine whether to normalize entropy data terms over [0,1] or [0,inf). This
  // does not effect probability terms.
  void SetNormalizeDataTermOn();
  void SetNormalizeDataTermOff();
  int GetNormalizeDataTerm();

  // Description:
  // Determine which label is being used as the seed.
  vtkSetClampMacro(LabelID, int, 0, INT_MAX);
  vtkGetMacro(LabelID, int);

  // Description:
  // Determine whether or not to use entropy rather than probability in the output
  // image.
  vtkSetMacro(Entropy, bool);
  vtkGetMacro(Entropy, bool);
  void SetOutputToEntropy();
  void SetOutputToProbability();

  // Description:
  // If no labels seed a particular voxel, theoretically, the entropy cost is infinity,
  // here is where you define the cut off, which does effect the scaling of the normalized
  // terms.
  vtkSetClampMacro(MaxValueToGive, double, 0.0, DBL_MAX);
  vtkGetMacro(MaxValueToGive, double);

  // Description:
  // Determine if the results should be spatially blurred (as probabilities) before being
  // returned. Helps account for some possible registration or alignment errors.
  vtkSetMacro(GaussianBlurOn, bool);
  vtkGetMacro(GaussianBlurOn, bool);

  // Description:
  // Determine how much blurring should occur.
  void SetStDevX(double val);
  void SetStDevY(double val);
  void SetStDevZ(double val);
  double GetStDevX();
  double GetStDevY();
  double GetStDevZ();

protected:
  vtkCudaImageAtlasLabelProbability();
  ~vtkCudaImageAtlasLabelProbability();

  virtual void Reinitialize(bool withData = false) {} // not implemented;
  virtual void Deinitialize(bool withData = false) {} // not implemented;

  int LabelID;
  int NormalizeDataTerm;
  bool Entropy;
  int NumberOfLabelMaps;
  double MaxValueToGive;

  bool GaussianBlurOn;
  double GaussianDevs[3];

  virtual int RequestInformation(vtkInformation*,
                                 vtkInformationVector**,
                                 vtkInformationVector*);

  virtual int RequestData(vtkInformation* request,
                          vtkInformationVector** inputVector,
                          vtkInformationVector* outputVector);

  virtual int FillInputPortInformation(int port, vtkInformation* info);

private:
  vtkCudaImageAtlasLabelProbability(const vtkCudaImageAtlasLabelProbability&);  // Not implemented.
  void operator=(const vtkCudaImageAtlasLabelProbability&);  // Not implemented.
};

#endif
