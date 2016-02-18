/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaHierarchicalMaxFlowSegmentation.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaHierarchicalMaxFlowSegmentation.h
 *
 *  @brief Header file with definitions of GPU-based solver for generalized hierarchical max-flow
 *      segmentation problems with a priori known scheduling over a single GPU. See
 *      vtkHierarchicalMaxFlowSegmentation.h for most of the interface documentation.
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  
 *  @note August 27th 2013 - Documentation first compiled.
 *
 */

#ifndef __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION_H__
#define __VTKCUDAHIERARCHICALMAXFLOWSEGMENTATION_H__

#include "vtkHierarchicalMaxFlowSegmentation.h"
#include "CudaObject.h"

#include <map>
#include <list>
#include <set>
#include <limits.h>
#include <float.h>

class vtkCudaHierarchicalMaxFlowSegmentation : public vtkHierarchicalMaxFlowSegmentation, public CudaObject
{
public:
  vtkTypeMacro( vtkCudaHierarchicalMaxFlowSegmentation, vtkHierarchicalMaxFlowSegmentation );
  static vtkCudaHierarchicalMaxFlowSegmentation *New();

protected:
  vtkCudaHierarchicalMaxFlowSegmentation();
  virtual ~vtkCudaHierarchicalMaxFlowSegmentation();

  void Reinitialize(int withData);
  void Deinitialize(int withData);

  virtual int InitializeAlgorithm();
  virtual int RunAlgorithm();

  double  MaxGPUUsage;
  void PropogateLabels( vtkIdType currNode );
  void SolveMaxFlow( vtkIdType currNode, int* timeStep );
  void UpdateLabel( vtkIdType node, int* timeStep );

  //Mappings for CPU-GPU buffer sharing
  void ReturnBufferGPU2CPU(float* CPUBuffer, float* GPUBuffer);
  void MoveBufferCPU2GPU(float* CPUBuffer, float* GPUBuffer);
  void GetGPUBuffersV2(int reference);
  std::list<float*> AllGPUBufferBlocks;
  std::map<float*,float*> CPU2GPUMap;
  std::map<float*,float*> GPU2CPUMap;
  std::set<float*> CPUInUse;
  std::list<float*> UnusedGPUBuffers;
  std::set<float*> ReadOnly;
  std::set<float*> NoCopyBack;

  //Prioirty structure
  class CircListNode;
  std::map< float*, CircListNode* > PrioritySet;
  std::map< float*, int > PrioritySetNumUses;
  void ClearBufferOrdering( vtkIdType currNode );
  void SimulateIterationForBufferOrdering( vtkIdType currNode, int* reference );
  void SimulateIterationForBufferOrderingUpdateLabelStep( vtkIdType currNode, int* reference );
  void UpdateBufferOrderingAt( float* buffer, int reference );
  void DeallocatePrioritySet();

  int    NumMemCpies;
  int    NumKernelRuns;

private:
  vtkCudaHierarchicalMaxFlowSegmentation operator=(const vtkCudaHierarchicalMaxFlowSegmentation&){} //not implemented
  vtkCudaHierarchicalMaxFlowSegmentation(const vtkCudaHierarchicalMaxFlowSegmentation&){} //not implemented
};

#endif
