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

/** @file vtkCudaMaxFlowSegmentationWorker.h
 *
 *  @brief Implementation file with class that runs each individual GPU for GHMF.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *
 *  @note August 27th 2013 - Documentation first compiled.
 *
 *  @note This is not a front-end class.
 *
 */

#ifndef __VTKCUDAMAXFLOWSEGMENTATIONWORKER_H__
#define __VTKCUDAMAXFLOWSEGMENTATIONWORKER_H__

#include "vtkCudaImageAnalyticsExport.h"

#include "CudaObject.h"
#include <list>
#include <set>
#include <vector>

class vtkCudaMaxFlowSegmentationScheduler;

class vtkCudaImageAnalyticsExport vtkCudaMaxFlowSegmentationWorker : public CudaObject
{
public:
  void UpdateBuffersInUse();
  void AddToStack(float* CPUBuffer);
  void RemoveFromStack(float* CPUBuffer);
  void BuildStackUpToPriority(unsigned int priority);
  void TakeDownPriorityStacks();
  int LowestBufferShift(unsigned int n);
  void ReturnLeafLabels();
  void ReturnBuffer(float* CPUBuffer);

  virtual void Reinitialize(bool withData = false) {};
  virtual void Deinitialize(bool withData = false) {};

public:
  vtkCudaMaxFlowSegmentationScheduler* const Parent;
  const int GPU;
  int NumBuffers;
  std::map<float*, float*> CPU2GPUMap;
  std::map<float*, float*> GPU2CPUMap;
  std::set<float*> CPUInUse;
  std::list<float*> UnusedGPUBuffers;
  std::list<float*> AllGPUBufferBlocks;
  std::vector< std::list< float* > > PriorityStacks;
  vtkCudaMaxFlowSegmentationWorker(int g, double usage, vtkCudaMaxFlowSegmentationScheduler* p);
  ~vtkCudaMaxFlowSegmentationWorker();
};

#endif