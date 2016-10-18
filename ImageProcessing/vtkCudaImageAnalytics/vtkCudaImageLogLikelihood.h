/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkCudaImageLogLikelihood.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaImageLogLikelihood.h
 *
 *  @brief Header file with definitions for the CUDA accelerated log likelihood data term. This
 *      generates entropy data terms based on the histogram of a set of provided seeds.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */


#ifndef __vtkCudaImageLogLikelihood_h
#define __vtkCudaImageLogLikelihood_h

#include "vtkCudaImageAnalyticsExport.h"
#include "vtkVersionMacros.h"

#include "CudaObject.h"
#include "vtkImageAlgorithm.h"

class vtkDataObject;

class vtkCudaImageAnalyticsExport vtkCudaImageLogLikelihood : public vtkImageAlgorithm, public CudaObject
{
public:
  static vtkCudaImageLogLikelihood *New();
  vtkTypeMacro(vtkCudaImageLogLikelihood,vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set the input image who you want to define a log likelihood data term for.
  virtual void SetInputImageConnection(vtkAlgorithmOutput *in)
  {
    this->SetInputConnection(0,in);
  }

  // Description:
  // Set a collection of label maps for the seeding operation.
  virtual void SetInputLabelMap(vtkAlgorithmOutput *in, int number)
  {
    if(number >= 0)
    {
      this->SetNthInputConnection(1,number,in);
    }
  }

  // Description:
  // Determine whether to normalize entropy data terms over [0,1] (on) or [0,inf) (off).
  void SetNormalizeDataTermOn()
  {
    this->NormalizeDataTerm = 1;
  }
  void SetNormalizeDataTermOff()
  {
    this->NormalizeDataTerm = 0;
  }
  int GetNormalizeDataTerm()
  {
    return (this->NormalizeDataTerm);
  }

  // Description:
  // Determine which label is being used as the seed.
  vtkSetClampMacro(LabelID,int, 0, INT_MAX);
  vtkGetMacro(LabelID,int);

  // Description:
  // Determine the resolution of the histogram used for the data term. Cannot exceed 512 bins for
  // computability reasons.
  vtkSetClampMacro(HistogramSize,int, 1, 512);
  vtkGetMacro(HistogramSize,int);

  // Description:
  // Determine what fraction of the input labels need to agree before a seed is considered valid. For
  // example, if RequiredAgreement=0.5, then at least half of the input label maps must have the same
  // value at the same pixel for it to be considered a seed pixel.
  vtkSetClampMacro(RequiredAgreement,double,0.0,1.0);
  vtkGetMacro(RequiredAgreement,double);

protected:
  vtkCudaImageLogLikelihood();
  ~vtkCudaImageLogLikelihood();

  int LabelID;
  int NormalizeDataTerm;
  int HistogramSize;
  double RequiredAgreement;
  int NumberOfLabelMaps;

  virtual int RequestInformation (vtkInformation *,
                                  vtkInformationVector **,
                                  vtkInformationVector *);

  virtual int RequestData(vtkInformation *request,
                          vtkInformationVector **inputVector,
                          vtkInformationVector *outputVector );

  virtual int FillInputPortInformation(int port, vtkInformation* info);

private:
  vtkCudaImageLogLikelihood(const vtkCudaImageLogLikelihood&);  // Not implemented.
  void operator=(const vtkCudaImageLogLikelihood&);  // Not implemented.
  void Reinitialize(int withData) {}; // Not implemented.
  void Deinitialize(int withData) {}; // Not implemented.
};

#endif
