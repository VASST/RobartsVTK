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

#include "vtkCudaFunctionPolygon.h"
#include <string>
#include <list>

/**
 *  @brief Class that writes function polygons to file in a set format
 *  @see vtkCudaFunctionPolygon vtkCudaFunctionPolygonReader
 *
 */
class vtkCudaFunctionPolygonReader : public vtkObject
{
public:

  vtkTypeMacro( vtkCudaFunctionPolygonReader, vtkObject );

  /**
   *  @brief VTK compatible constructor method
   */
  static vtkCudaFunctionPolygonReader* New();

  void SetFileName( std::string filename );
  void Read();
  void Clear();
  vtkCudaFunctionPolygon* GetOutput( unsigned int n );
  size_t GetNumberOfOutputs( );

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