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
#include "vtkAlgorithmOutput.h"
#include "vtkDataObject.h"
#include "vtkImageData.h"

#include <float.h>
#include <limits.h>

class vtkImageLogLikelihood : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageLogLikelihood *New();
  vtkTypeMacro(vtkImageLogLikelihood,vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual void SetInputImage(vtkDataObject *in) { this->SetInput(0,in); }
  virtual void SetInputLabelMap(vtkDataObject *in, int number) { if(number >= 0) this->SetNthInputConnection(1,number,in->GetProducerPort()); }

  void SetNormalizeDataTerm() {this->NormalizeDataTerm = 1; }
  int GetNormalizeDataTerm() {return (this->NormalizeDataTerm); }

  vtkSetClampMacro(LabelID,int, 0, INT_MAX);
  vtkGetMacro(LabelID,int);

  vtkSetClampMacro(HistogramResolution,float, 1, 1024);
  vtkGetMacro(HistogramResolution,float);

  vtkSetClampMacro(RequiredAgreement,double,0.0,1.0);
  vtkGetMacro(RequiredAgreement,double);

protected:
  vtkImageLogLikelihood();
  ~vtkImageLogLikelihood();

  int LabelID;
  int NormalizeDataTerm;
  float HistogramResolution;
  double RequiredAgreement;
  int NumberOfLabelMaps;

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
