/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaMaxFlowSegmentationWorker.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

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

#include <set>
#include <list>
#include <vector>

#include "vtkCudaMaxFlowSegmentationScheduler.h"
#include "vtkCudaObject.h"

class vtkCudaMaxFlowSegmentationWorker : public vtkCudaObject {
public:

  vtkCudaMaxFlowSegmentationScheduler* const Parent;
  const int GPU;
    int NumBuffers;
  std::map<float*,float*> CPU2GPUMap;
  std::map<float*,float*> GPU2CPUMap;
  std::set<float*> CPUInUse;
  std::list<float*> UnusedGPUBuffers;
  std::list<float*> AllGPUBufferBlocks;
  std::vector< std::list< float* > > PriorityStacks;
  vtkCudaMaxFlowSegmentationWorker(int g, double usage, vtkCudaMaxFlowSegmentationScheduler* p );
  ~vtkCudaMaxFlowSegmentationWorker();
  void UpdateBuffersInUse();
  void AddToStack( float* CPUBuffer );
  void RemoveFromStack( float* CPUBuffer );
  void BuildStackUpToPriority( int priority );
  void TakeDownPriorityStacks();
  int LowestBufferShift(int n);
  void ReturnLeafLabels();
  void ReturnBuffer(float* CPUBuffer);
  void Reinitialize(int withData){} // not used
  void Deinitialize(int withData){} // not used

};

#endif