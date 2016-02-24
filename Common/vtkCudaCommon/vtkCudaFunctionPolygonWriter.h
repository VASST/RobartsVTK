/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaFunctionPolygonWriter.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCudaFunctionPolygonWriter.h
 *
 *  @brief Header file defining a writer for polygonal TF objects
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on December 14, 2011
 *
 */
#ifndef VTKCUDAFUNCTIONPOLYGONWRITER_H
#define VTKCUDAFUNCTIONPOLYGONWRITER_H

#include "vtkCudaFunctionPolygon.h"
#include <string>
#include <list>

/** @brief Class that writes function polygons to file in a set format
 *
 *  @see vtkCudaFunctionPolygon vtkCudaFunctionPolygonReader
 *
 */
class VTKCUDACOMMON_EXPORT vtkCudaFunctionPolygonWriter : public vtkObject
{
public:

  /** @brief VTK compatible constructor method
   *
   */
  static vtkCudaFunctionPolygonWriter* New();

  void SetFileName( std::string filename );
  void Write();
  void Clear();
  void AddInput( vtkCudaFunctionPolygon* object );
  void RemoveInput( vtkCudaFunctionPolygon* object );

protected:
  vtkCudaFunctionPolygonWriter();
  ~vtkCudaFunctionPolygonWriter();

  void printTFPolygon( vtkCudaFunctionPolygon* e );

  std::list<vtkCudaFunctionPolygon*> objects;
  std::string filename;
  bool fileNameSet;

  std::ofstream* file;

};
#endif