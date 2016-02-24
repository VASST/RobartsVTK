/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaCommon.h

  Copyright (c) Adam Rankin, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "vtkCudaCommon.h"

#include <math.h>

//-----------------------------------------------------------------
dim3 GetGrid(int size)
{
  size = (size-1) / NUMTHREADS + 1;
  dim3 grid( size, 1, 1 );
  if( grid.x > MAX_GRID_SIZE )
  {
    grid.x = grid.y = (int) sqrt( (double)(size-1) ) + 1;
  }
  else if( grid.y > MAX_GRID_SIZE )
  {
    grid.x = grid.y = grid.z = (int) pow( (double)(size-1), (double)1.0/3.0 ) + 1;
  }
  return grid;
}