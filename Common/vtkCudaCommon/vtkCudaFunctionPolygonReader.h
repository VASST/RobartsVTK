/*=========================================================================

  Program:   Robarts Visualization Toolkit
  Module:    vtkCudaFunctionPolygonReader.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaFunctionPolygonReader.h
 *
 *  @brief Header file defining a reader for polygonal TF objects
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on December 14, 2011
 *
 */
#ifndef VTKCUDAFUNCTIONPOLYGONREADER_H
#define VTKCUDAFUNCTIONPOLYGONREADER_H

#include "vtkCudaCommonModule.h"

#include "vtkCudaFunctionPolygon.h"
#include <string>
#include <list>

/**
 *  @brief Class that writes function polygons to file in a set format
 *  @see vtkCudaFunctionPolygon vtkCudaFunctionPolygonReader
 *
 */
class VTKCUDACOMMON_EXPORT vtkCudaFunctionPolygonReader : public vtkObject
{
public:
  static vtkCudaFunctionPolygonReader* New();
  vtkTypeMacro( vtkCudaFunctionPolygonReader, vtkObject );

  void SetFileName( const std::string& filename );
  void Read();
  void Clear();
  vtkCudaFunctionPolygon* GetOutput( unsigned int n );
  size_t GetNumberOfOutputs();

protected:
  vtkCudaFunctionPolygonReader();
  ~vtkCudaFunctionPolygonReader();

  vtkCudaFunctionPolygon* readTFPolygon();

  std::list<vtkCudaFunctionPolygon*> objects;
  std::string filename;
  bool fileNameSet;

  std::ifstream* file;
};
#endif