/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkImageAtlasLabelProbability.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkImageAtlasLabelProbability.h
*
*  @brief Header file with definitions for the CPU-based label agreement data term. This generates
*      entropy or probability data terms based on how many label maps agree with a particular
*      labelling of each voxel.
*
*  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
*
*  @note August 27th 2013 - Documentation first compiled.
*
*/

#ifndef __vtkImageAtlasLabelProbability_H__
#define __vtkImageAtlasLabelProbability_H__

#include "vtkRobartsCommonModule.h"
#include "vtkVersionMacros.h"

#include "vtkThreadedImageAlgorithm.h"
#include "vtkDataObject.h"

#include <float.h>
#include <limits.h>

class VTKROBARTSCOMMON_EXPORT vtkImageAtlasLabelProbability : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageAtlasLabelProbability *New();
  vtkTypeMacro(vtkImageAtlasLabelProbability,vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Set a collection of label maps for the seeding operation.
  virtual void SetInputLabelMapConnection(vtkAlgorithmOutput *in, int number)
  {
    if(number >= 0)
    {
      this->SetNthInputConnection(0,number,in);
    }
  }

  // Description:
  // Determine whether to normalize entropy data terms over [0,1] or [0,inf). This
  // does not effect probability terms.
  void SetNormalizeDataTermOn();
  void SetNormalizeDataTermOff();
  int GetNormalizeDataTerm();

  // Description:
  // Determine which label is being used as the seed.
  vtkSetClampMacro(LabelID,int, 0, INT_MAX);
  vtkGetMacro(LabelID,int);

  // Description:
  // Determine whether or not to use entropy rather than probability in the output
  // image.
  vtkSetMacro(Entropy,bool);
  vtkGetMacro(Entropy,bool);
  void SetOutputToEntropy();
  void SetOutputToProbability();

  // Description:
  // If no labels seed a particular voxel, theoretically, the entropy cost is infinity,
  // here is where you define the cut off, which does effect the scaling of the normalized
  // terms.
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
