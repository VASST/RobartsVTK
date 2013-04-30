/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkDiceCoefficient.h

  Copyright (c) Martin Rajchl, Robarts Research Institute

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

  =========================================================================*/

#ifndef __vtkDiceCoefficient_h
#define __vtkDiceCoefficient_h


#include "vtkThreadedImageAlgorithm.h"
#include "vtkDataObject.h"
#include "vtkImageData.h"


#include <float.h>
#include <limits.h>

class vtkDiceCoefficient : public vtkThreadedImageAlgorithm
{
public:
    static vtkDiceCoefficient *New();
    vtkTypeMacro(vtkDiceCoefficient,vtkThreadedImageAlgorithm);
    void PrintSelf(ostream& os, vtkIndent indent);

    virtual void SetInput1(vtkDataObject *in) { this->SetInput(0,in); }
    virtual void SetInput2(vtkDataObject *in) { this->SetInput(1,in); }

    vtkSetClampMacro(LabelID,int, 0, INT_MAX);
    vtkGetMacro(LabelID,int);

    vtkGetMacro(DiceCoefficient,double);


protected:
    vtkDiceCoefficient();
    ~vtkDiceCoefficient();

    vtkSetClampMacro(DiceCoefficient,double, 0, 1);

    int LabelID;
    double DiceCoefficient;
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

    template <class T> void vtkDiceCoefficientExecute(vtkDiceCoefficient *self,
                                      vtkImageData *in1Data, T *in1Ptr,
                                      vtkImageData *in2Data, T *in2Ptr,
                                      vtkImageData *outData, T *outPtr,
                                      int outExt[6], int id);

private:
    vtkDiceCoefficient(const vtkDiceCoefficient&);  // Not implemented.
    void operator=(const vtkDiceCoefficient&);  // Not implemented.



};

#endif
