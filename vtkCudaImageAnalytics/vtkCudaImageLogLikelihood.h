/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaImageAtlasLabelProbability.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __vtkCudaImageLogLikelihood_h
#define __vtkCudaImageLogLikelihood_h

#include "vtkImageAlgorithm.h"
#include "vtkAlgorithmOutput.h"
#include "vtkDataObject.h"
#include "vtkCudaObject.h"

#include <float.h>
#include <limits.h>

class vtkCudaImageLogLikelihood : public vtkImageAlgorithm, public vtkCudaObject
{
public:
  static vtkCudaImageLogLikelihood *New();
  vtkTypeMacro(vtkCudaImageLogLikelihood,vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual void SetInputImage(vtkDataObject *in) { this->SetInput(0,in); }
  virtual void SetInputLabelMap(vtkDataObject *in, int number) { if(number >= 0) this->SetNthInputConnection(1,number,in->GetProducerPort()); }

  void SetNormalizeDataTerm() {this->NormalizeDataTerm = 1; }
  int GetNormalizeDataTerm() {return (this->NormalizeDataTerm); }

  vtkSetClampMacro(LabelID,int, 0, INT_MAX);
  vtkGetMacro(LabelID,int);

  vtkSetClampMacro(HistogramSize,int, 1, 512);
  vtkGetMacro(HistogramSize,int);

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
  void Reinitialize(int withData){};  // Not implemented.
  void Deinitialize(int withData){};  // Not implemented.



};

#endif
