/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkImageDataTerm.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkImageDataTerm - Add, subtract, multiply, divide, invert, sin,
// cos, exp, log.
// .SECTION Description
// vtkImageDataTerm implements more complex mathematic operations commonly
// used in definining data terms. This filter takes 


#ifndef __vtkImageDataTerm_h
#define __vtkImageDataTerm_h


// Operation options.
#define VTK_CONSTANT           0
#define VTK_LOGISTIC            1
#define VTK_GAUSSIAN           2

#include "vtkThreadedImageAlgorithm.h"

class VTK_IMAGING_EXPORT vtkImageDataTerm : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageDataTerm *New();
  vtkTypeRevisionMacro(vtkImageDataTerm,vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set/Get the Operation to perform.
  vtkSetMacro(Operation,int);
  vtkGetMacro(Operation,int);

  // Description:
  // Set each pixel in the output image to the constant C. Image
  // 2 is not used
  void SetOperationToConstant() {this->SetOperation(VTK_CONSTANT);};

  // Description:
  // Set each pixel to sigmoid(K1*x1+C1)*sigmoid(K2*x2+C2) where K and C are the constants
  // and x is the pixel value. If image 2 is not provided, sigm0id(K2*x2+C2) <= 1.
  void SetOperationToLogistic() {this->SetOperation(VTK_LOGISTIC);};
  
  // Description:
  // Set each pixel to gaussian(x-C,K^2) where K and C are the constants
  // and x is the pixel value. Image 2 can be used.
  void SetOperationToGaussian() {this->SetOperation(VTK_GAUSSIAN);};

  // Description:
  // A constant used by some operations (typically multiplicative). Default is 1.
  vtkSetMacro(ConstantK1,double);
  vtkGetMacro(ConstantK1,double);
  vtkSetMacro(ConstantK2,double);
  vtkGetMacro(ConstantK2,double);

  // Description:
  // A constant used by some operations (typically additive). Default is 0.
  vtkSetMacro(ConstantC1,double);
  vtkGetMacro(ConstantC1,double);
  vtkSetMacro(ConstantC2,double);
  vtkGetMacro(ConstantC2,double);

  // Description:
  // Set the two inputs to this filter. For some operations, the second input
  // is not used.
  virtual void SetInput1(vtkDataObject *in) { this->SetInput(0,in); }
  virtual void SetInput2(vtkDataObject *in) { this->SetInput(1,in); }

  // Description:
  // Take the negative log of the output to convert from a probability to an
  // entropy. These methods are toggles between those two modes. Default:
  // probability.
  void SetOutputEntropy(){ this->Entropy = true; };
  void SetOutputProbability(){ this->Entropy = false; };
  bool GetEntropyUsed() { return this->Entropy; };
  bool GetProbabilityUsed() { return !this->Entropy; };

protected:
  vtkImageDataTerm();
  ~vtkImageDataTerm() {};

  int Operation;
  double ConstantK1;
  double ConstantC1;
  double ConstantK2;
  double ConstantC2;
  bool Entropy;

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
  vtkImageDataTerm(const vtkImageDataTerm&);  // Not implemented.
  void operator=(const vtkImageDataTerm&);  // Not implemented.
};

#endif

