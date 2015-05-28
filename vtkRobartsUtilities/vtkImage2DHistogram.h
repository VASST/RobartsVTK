/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkImage2DHistogram.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __vtkImage2DHistogram_H__
#define __vtkImage2DHistogram_H__

#include "vtkImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkMultiThreader.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"

class vtkImage2DHistogram : public vtkImageAlgorithm
{
public:

  vtkTypeMacro( vtkImage2DHistogram, vtkImageAlgorithm )

  static vtkImage2DHistogram *New();
  
  // Description:
  // Get/Set the number of threads to create when rendering
  vtkSetClampMacro( NumberOfThreads, int, 1, VTK_MAX_THREADS );
  vtkGetMacro( NumberOfThreads, int );
  
  // Description:
  // Get/Set the resolution of the returned histogram
  void SetResolution( int res[2] );
  vtkGetMacro( Resolution, int* );
  
  virtual int RequestData(vtkInformation *request, 
               vtkInformationVector **inputVector, 
               vtkInformationVector *outputVector);
  virtual int RequestInformation( vtkInformation* request,
               vtkInformationVector** inputVector,
               vtkInformationVector* outputVector);
  virtual int RequestUpdateExtent( vtkInformation* request,
               vtkInformationVector** inputVector,
               vtkInformationVector* outputVector);
  
  void ThreadedExecute(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads);

protected:

  // The method that starts the multithreading
  template< class T >
  void ThreadedExecuteCasted(vtkImageData *inData, vtkImageData *outData, int threadId, int numThreads);

  int Resolution[2];

  vtkImage2DHistogram();
  virtual ~vtkImage2DHistogram();
  
  vtkMultiThreader* Threader;
  int NumberOfThreads;

private:
  vtkImage2DHistogram operator=(const vtkImage2DHistogram&){} //not implemented
  vtkImage2DHistogram(const vtkImage2DHistogram&){} //not implemented

};

#endif