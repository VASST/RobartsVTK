/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaImageVote.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaImageVote.h
 *
 *  @brief Header file with definitions for the CUDA accelerated voting operation. This module
 *      Takes a probabilistic or weighted image, and replaces each voxel with a label corresponding
 *      to the input image with the highest value at that location. ( argmax{} operation )
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#ifndef __VTKCUDAIMAGEVOTE_H__
#define __VTKCUDAIMAGEVOTE_H__

#include "vtkImageAlgorithm.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkSetGet.h"
#include "vtkCudaObject.h"

#include <map>
#include <limits.h>

class vtkCudaImageVote : public vtkImageAlgorithm, public vtkCudaObject
{
public:
  vtkTypeMacro( vtkCudaImageVote, vtkImageAlgorithm );
  static vtkCudaImageVote *New();
  
  // Description:
  // Set the input to the filter associated with an integer
  // label to be given.
  vtkDataObject* GetInput(int idx);
  void SetInput(int idx, vtkDataObject *input);
  
  // Description:
  // Set what scalar type the output is expected to be.
  vtkSetClampMacro(OutputDataType,int,1,20);
  vtkGetMacro(OutputDataType,int);

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
  virtual int FillInputPortInformation(int i, vtkInformation* info);

  template<class T>
  T GetMappedTerm(int i){ return (T)(BackwardsInputPortMapping.find(i) == BackwardsInputPortMapping.end() ? 0: BackwardsInputPortMapping[i]); }

protected:
  vtkCudaImageVote();
  virtual ~vtkCudaImageVote();

private:
  vtkCudaImageVote operator=(const vtkCudaImageVote&){}
  vtkCudaImageVote(const vtkCudaImageVote&){}
  void Reinitialize(int withData){};
  void Deinitialize(int withData){};
  
  int CheckInputConsistency( vtkInformationVector** inputVector, int* Extent, int* NumLabels, int* DataType, int* NumComponents);

  std::map<vtkIdType,int> InputPortMapping;
  std::map<int,vtkIdType> BackwardsInputPortMapping;
  int FirstUnusedPort;

  int OutputDataType;
};

#endif
