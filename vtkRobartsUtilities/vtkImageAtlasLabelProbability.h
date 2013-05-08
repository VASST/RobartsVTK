/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImageAtlasLabelProbability.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __vtkImageAtlasLabelProbability_h
#define __vtkImageAtlasLabelProbability_h


#include "vtkThreadedImageAlgorithm.h"
#include "vtkAlgorithmOutput.h"
#include "vtkDataObject.h"

#include <float.h>
#include <limits.h>

class vtkImageAtlasLabelProbability : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageAtlasLabelProbability *New();
  vtkTypeMacro(vtkImageAtlasLabelProbability,vtkThreadedImageAlgorithm);
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
  vtkImageAtlasLabelProbability();
  ~vtkImageAtlasLabelProbability();

  int LabelID;
  int NormalizeDataTerm;
  bool Entropy;
  int NumberOfLabelMaps;
  double MaxValueToGive;

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
  vtkImageAtlasLabelProbability(const vtkImageAtlasLabelProbability&);  // Not implemented.
  void operator=(const vtkImageAtlasLabelProbability&);  // Not implemented.



};

#endif
