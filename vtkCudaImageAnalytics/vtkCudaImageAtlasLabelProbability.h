/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaImageAtlasLabelProbability.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __vtkCudaImageAtlasLabelProbability_h
#define __vtkCudaImageAtlasLabelProbability_h


#include "vtkImageAlgorithm.h"
#include "vtkAlgorithmOutput.h"
#include "vtkDataObject.h"
#include "vtkCudaObject.h"

#include <float.h>
#include <limits.h>

class vtkCudaImageAtlasLabelProbability : public vtkImageAlgorithm, public vtkCudaObject
{
public:
  static vtkCudaImageAtlasLabelProbability *New();
  vtkTypeMacro(vtkCudaImageAtlasLabelProbability,vtkImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual void SetInputLabelMap(vtkDataObject *in, int number) { if(number >= 0) this->SetNthInputConnection(0,number,in->GetProducerPort()); }

  void SetNormalizeDataTerm() {this->NormalizeDataTerm = 1; }
  int GetNormalizeDataTerm() {return (this->NormalizeDataTerm); }

  vtkSetClampMacro(LabelID,int, 0, INT_MAX);
  vtkGetMacro(LabelID,int);

  vtkSetMacro(Entropy,bool);
  vtkGetMacro(Entropy,bool);
  void SetOutputToEntropy(){ this->SetEntropy(true); }
  void SetOutputToProbability(){ this->SetEntropy(false); }

  vtkSetClampMacro(MaxValueToGive,double,0.0,DBL_MAX);
  vtkGetMacro(MaxValueToGive,double);

protected:
  vtkCudaImageAtlasLabelProbability();
  ~vtkCudaImageAtlasLabelProbability();
  
  void Reinitialize(int withData){} // not implemented;
  void Deinitialize(int withData){} // not implemented;

  int LabelID;
  int NormalizeDataTerm;
  bool Entropy;
  int NumberOfLabelMaps;
  double MaxValueToGive;

  virtual int RequestInformation (vtkInformation *,
                                    vtkInformationVector **,
                                    vtkInformationVector *);

  virtual int RequestData(vtkInformation *request,
                                     vtkInformationVector **inputVector,
                                     vtkInformationVector *outputVector );
  
  virtual int FillInputPortInformation(int port, vtkInformation* info);

private:
  vtkCudaImageAtlasLabelProbability(const vtkCudaImageAtlasLabelProbability&);  // Not implemented.
  void operator=(const vtkCudaImageAtlasLabelProbability&);  // Not implemented.



};

#endif
