/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkImageEntropyPlaneSelection.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkImageEntropyPlaneSelection.h
 *
 *  @brief Header file with definitions for a class to select orthogonal, axis-alligned
 *      planes for plane selection purposes based on the entropy of the probabilistic
 *      segmentation.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note September 5th 2013 - Documentation first compiled. (jshbaxter)
 *
 */

#ifndef _VTKIMAGEENTROPYPLANESELECTION_H__
#define _VTKIMAGEENTROPYPLANESELECTION_H__

#include "vtkRobartsCommonExport.h"

#include "vtkAlgorithm.h"
#include "vtkImageData.h"
#include "vtkImageAlgorithm.h"
#include "vtkTransform.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkAlgorithmOutput.h"

#include <map>

class vtkRobartsCommonExport vtkImageEntropyPlaneSelection : public vtkImageAlgorithm
{
public:
  vtkTypeMacro( vtkImageEntropyPlaneSelection, vtkImageAlgorithm );
  static vtkImageEntropyPlaneSelection *New();

  vtkDataObject* GetInput(int idx);
  void SetInput(int idx, vtkDataObject *input);

  double GetEntropyInX(int slice);
  double GetEntropyInY(int slice);
  double GetEntropyInZ(int slice);
  int GetSliceInX();
  int GetSliceInY();
  int GetSliceInZ();

  // Description:
  // If the subclass does not define an Execute method, then the task
  // will be broken up, multiple threads will be spawned, and each thread
  // will call this method. It is public so that the thread functions
  // can call this method.
  virtual int RequestData(vtkInformation *request,
                          vtkInformationVector **inputVector,
                          vtkInformationVector *outputVector);
  virtual int RequestInformation( vtkInformation* request,
                                  vtkInformationVector** inputVector,
                                  vtkInformationVector* outputVector);
  virtual int RequestUpdateExtent( vtkInformation* request,
                                   vtkInformationVector** inputVector,
                                   vtkInformationVector* outputVector);
  virtual int RequestDataObject( vtkInformation* request,
                                 vtkInformationVector** inputVector,
                                 vtkInformationVector* outputVector);
  virtual int FillInputPortInformation(int i, vtkInformation* info);

protected:
  vtkImageEntropyPlaneSelection();
  virtual ~vtkImageEntropyPlaneSelection();

  int Extent[6];
  double* EntropyInX;
  double* EntropyInY;
  double* EntropyInZ;

  std::map<vtkIdType,int> InputDataPortMapping;
  std::map<int,vtkIdType> BackwardsInputDataPortMapping;
  int FirstUnusedDataPort;

private:
  vtkImageEntropyPlaneSelection operator=(const vtkImageEntropyPlaneSelection&);
  vtkImageEntropyPlaneSelection(const vtkImageEntropyPlaneSelection&);
};

#endif //_VTKIMAGEENTROPYPLANESELECTION_H__