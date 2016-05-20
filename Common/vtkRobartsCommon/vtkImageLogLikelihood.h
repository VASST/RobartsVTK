/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImageLogLikelihood.h

  Copyright (c) Martin Rajchl, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkImageLogLikelihood.h
 *
 *  @brief Header file with definitions for the log likelihood data term. This
 *      generates entropy data terms based on the histogram of a set of provided seeds.
 *
 *  @author Martin Rajchl (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled. (jshbaxter)
 *
 */

#ifndef __vtkImageLogLikelihood_h
#define __vtkImageLogLikelihood_h

#include "vtkRobartsCommonModule.h"

#include "vtkThreadedImageAlgorithm.h"
#include "vtkDataObject.h"
#include "vtkImageData.h"

#include <float.h>
#include <limits.h>

class VTKROBARTSCOMMON_EXPORT vtkImageLogLikelihood : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageLogLikelihood *New();
  vtkTypeMacro(vtkImageLogLikelihood,vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set the input image who you want to define a log likelihood data term for.
  virtual void SetInputDataImage(vtkDataObject *in);
  virtual void SetInputConnection(vtkAlgorithmOutput *in);

  // Description:
  // Set a collection of label maps for the seeding operation.
  virtual void SetInputLabelMapConnection(vtkAlgorithmOutput *in, int number);
  virtual void SetInputLabelMapData(vtkDataObject *in, int number);

  // Description:
  // Determine whether to normalize entropy data terms over [0,1] (on) or [0,inf) (off).
  void SetNormalizeDataTermOn();
  void SetNormalizeDataTermOff();
  int GetNormalizeDataTerm();

  // Description:
  // Determine which label is being used as the seed.
  vtkSetClampMacro(LabelID,int, 0, INT_MAX);
  vtkGetMacro(LabelID,int);

  // Description:
  // Determine the resolution of the histogram used for the data term.
  vtkSetClampMacro(HistogramResolution,float, FLT_MIN, FLT_MAX);
  vtkGetMacro(HistogramResolution,float);

  // Description:
  // Determine what fraction of the input labels need to agree before a seed is considered valid. For
  // example, if RequiredAgreement=0.5, then at least half of the input label maps must have the same
  // value at the same pixel for it to be considered a seed pixel.
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
