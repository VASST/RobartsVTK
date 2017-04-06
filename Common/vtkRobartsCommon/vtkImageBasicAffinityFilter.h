/*=========================================================================

Robarts Visualization Toolkit

Copyright (c) 2016 Virtual Augmentation and Simulation for Surgery and Therapy, Robarts Research Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

=========================================================================*/

#ifndef __vtkImageBasicAffinityFilter_H__
#define __vtkImageBasicAffinityFilter_H__

#include "vtkRobartsCommonExport.h"

#include "vtkThreadedImageAlgorithm.h"
#include "vtkImageData.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkMultiThreader.h"

class vtkRobartsCommonExport vtkImageBasicAffinityFilter : public vtkThreadedImageAlgorithm
{
public:
  vtkTypeMacro(vtkImageBasicAffinityFilter, vtkThreadedImageAlgorithm)
  static vtkImageBasicAffinityFilter* New();

  // Description:
  // Get/Set the number of threads to create when rendering
  vtkSetClampMacro(NumberOfThreads, int, 1, VTK_MAX_THREADS);
  vtkGetMacro(NumberOfThreads, int);

  // Description:
  // Get/Set the weights for the basic affinity function
  vtkSetClampMacro(DistanceWeight, double, 0.0, 1000000.0);
  vtkGetMacro(DistanceWeight, double);
  vtkSetClampMacro(IntensityWeight, double, 0.0, 1000000.0);
  vtkGetMacro(IntensityWeight, double);

  // The method that starts the multithreading
  template<class T>
  void ThreadedExecuteCasted(vtkImageData* inData, vtkImageData* outData, int threadId, int numThreads);
  void ThreadedExecute(vtkImageData* inData, vtkImageData* outData, int threadId, int numThreads);

protected:
  int RequestData(vtkInformation* request, vtkInformationVector** inputVector, vtkInformationVector* outputVector);

protected:
  vtkImageBasicAffinityFilter();
  virtual ~vtkImageBasicAffinityFilter();

private:
  vtkImageBasicAffinityFilter operator=(const vtkImageBasicAffinityFilter&);
  vtkImageBasicAffinityFilter(const vtkImageBasicAffinityFilter&);

  vtkMultiThreader* Threader;
  int NumberOfThreads;

  double DistanceWeight;
  double IntensityWeight;

};

#endif