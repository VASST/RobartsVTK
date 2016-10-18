/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkCudaCommon.h

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __vtkCudaCommon_h__
#define __vtkCudaCommon_h__

#include "vtkCudaCommonExport.h"
#include "vector_types.h"

//-- COMMON STATEMENTS -------------------------

#define MAX_GRID_SIZE 65535
#define NUMTHREADS 512
#define CUDASTDOFFSET threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z))

vtkCudaCommonExport dim3 GetGrid(int size);


#endif // __vtkCudaCommon_h__