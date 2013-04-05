/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImageLogLikelihood.h

  Copyright (c) Martin Rajchl, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __vtkImageLogLikelihood_h
#define __vtkImageLogLikelihood_h


#include "vtkThreadedImageAlgorithm.h"

#include <float.h>
#include <limits.h>

class vtkImageLogLikelihood : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageLogLikelihood *New();
  vtkTypeMacro(vtkImageLogLikelihood,vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual void SetInput1(vtkDataObject *in) { this->SetInput(0,in); }
  virtual void SetInput2(vtkDataObject *in) { this->SetInput(1,in); }

  void SetNormalizeDataTerm() {this->NormalizeDataTerm = 1; }
  int GetNormalizeDataTerm() {return (this->NormalizeDataTerm); }

  vtkSetClampMacro(LabelID,int, 0, INT_MAX);
  vtkGetMacro(LabelID,int);

  vtkSetClampMacro(HistogramResolution,float, 1, 1024);
  vtkGetMacro(HistogramResolution,float);

protected:
  vtkImageLogLikelihood();
  ~vtkImageLogLikelihood();

  int LabelID;
  int NormalizeDataTerm;
  float HistogramResolution;

  virtual int RequestInformation (vtkInformation *,
                                    vtkInformationVector **,
                                    vtkInformationVector *);

  virtual void ThreadedRequestData(vtkInformation *request,
                                     vtkInformationVector **inputVector,
                                     vtkInformationVector *outputVector,
                                     vtkImageData ***inData,
                                     vtkImageData **outData,
                                     int extent[6], int threadId);

  virtual int FillInputPortInformation(int port, vtkInformation* info);


private:
  vtkImageLogLikelihood(const vtkImageLogLikelihood&);  // Not implemented.
  void operator=(const vtkImageLogLikelihood&);  // Not implemented.



};

#endif
