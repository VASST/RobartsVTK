/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkCudaObject.h

  Copyright (c) John SH Baxter, Robarts Research Institute

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/** @file vtkCuda2DTransferFunction.h
 *
 *  @brief Header file defining a 2 dimensional transfer function composed of function objects
 *
 *  @author John Stuart Haberl Baxter (Dr. Peters' Lab (VASST) at Robarts Research Institute)
 *  @note First documented on March 28, 2011
 *
 */
#ifndef VTKCUDA2DTRANSFERFUNCTION_H
#define VTKCUDA2DTRANSFERFUNCTION_H

#include "RobartsVTKConfigure.h"
#include "vtkCudaCommonExport.h"

#include "vtkCudaFunctionObject.h"
#include <vector>

/** @brief 2D transfer functions are composed of a list of function objects which define which parts of the transfer function are given which attributes.
 *
 *
 */
class vtkCudaCommonExport vtkCuda2DTransferFunction : public vtkObject {
public:
  
  vtkTypeMacro( vtkCuda2DTransferFunction, vtkObject );

  /** @brief VTK compatible constructor method
   *
   */
  static vtkCuda2DTransferFunction*  New();

  /** @brief Given a buffer, fill the buffer with the label values defined by the function objects in the transfer function (the buffer becomes a lookup table for the classification function)
   *
   *  @param outputTable The classification function lookup table housing the label values
   *  @param sizeI The size of each transfer function lookup table in the intensity dimension
   *  @param sizeG The size of each transfer function lookup table in the gradient dimension
   *  @param lowI The minimum intensity represented by this table (often the minimum intensity of the image)
   *  @param highI The maximum intensity represented by this table (often the minimum maximum of the image)
   *  @param lowG The minimum logarithmically scaled gradient (including offset) represented by this table
   *  @param highG The maximum logarithmically scaled gradient (including offset) represented by this table
   *  @param offsetG The offset for logarithmically scaling the gradient
   *
   *  @pre outputTable is of size sizeI*sizeG
   *
   *  @post outputTable is populated with values between 0 and GetNumberOfClassifications()
   *  @post If the list of function objects is empty, outputTable will consist solely of 0s
   *
   *  @note The responsibility for both allocation and deallocation of the buffer is given to the caller
   */
  void GetClassifyTable(  short* outputTable, int sizeI, int sizeG,
              float lowI, float highI, float offsetI, float lowG, float highG, float offsetG, int logUsed);

  /** @brief Given a buffer for each of the RGBA, fill these buffers with the colour and opacity values defined by the function objects in the transfer function (the buffer becomes a lookup table for the transfer function)
   *
   *  @param outputRTable The transfer function lookup table housing the red colour component
   *  @param outputGTable The transfer function lookup table housing the green colour component
   *  @param outputBTable The transfer function lookup table housing the blue colour component
   *  @param outputATable The transfer function lookup table housing the opacity component
   *  @param sizeI The size of each transfer function lookup table in the intensity dimension
   *  @param sizeG The size of each transfer function lookup table in the gradient dimension
   *  @param lowI The minimum intensity represented by this table (often the minimum intensity of the image)
   *  @param highI The maximum intensity represented by this table (often the minimum maximum of the image)
   *  @param lowG The minimum logarithmically scaled gradient (including offset) represented by this table
   *  @param highG The maximum logarithmically scaled gradient (including offset) represented by this table
   *  @param offsetG The offset for logarithmically scaling the gradient
   *
   *  @pre outputRTable, outputGTable, outputBTable, outputATable are all of size sizeI*sizeG
   *
   *  @post outputRTable, outputGTable, outputBTable, outputATable are all populated with values between 0.0f and 1.0f
   *  @post If the list of function objects is empty, outputRTable, outputGTable, outputBTable, outputATable will all consist solely of 0.0fs
   *
   *  @note The responsibility for both allocation and deallocation of the buffers is given to the caller
   */
  void GetTransferTable(  float* outputRTable, float* outputGTable, float* outputBTable, float* outputATable,
              int sizeI, int sizeG, float lowI, float offsetI, float highI, float lowG, float highG, float offsetG, int logUsed);
  
  /** @brief Given a buffer for each of the ADSP, fill these buffers with the colour and opacity values defined by the function objects in the transfer function (the buffer becomes a lookup table for the transfer function)
   *
   *  @param outputATable The transfer function lookup table housing the ambient shading component
   *  @param outputDTable The transfer function lookup table housing the diffuse shading component
   *  @param outputSTable The transfer function lookup table housing the specular shading component
   *  @param outputPTable The transfer function lookup table housing the specular power component
   *  @param sizeI The size of each transfer function lookup table in the intensity dimension
   *  @param sizeG The size of each transfer function lookup table in the gradient dimension
   *  @param lowI The minimum intensity represented by this table (often the minimum intensity of the image)
   *  @param highI The maximum intensity represented by this table (often the minimum maximum of the image)
   *  @param lowG The minimum logarithmically scaled gradient (including offset) represented by this table
   *  @param highG The maximum logarithmically scaled gradient (including offset) represented by this table
   *  @param offsetG The offset for logarithmically scaling the gradient
   *
   *  @pre outputRTable, outputGTable, outputBTable, outputATable are all of size sizeI*sizeG
   *
   *  @post outputRTable, outputGTable, outputBTable, outputATable are all populated with values between 0.0f and 1.0f
   *  @post If the list of function objects is empty, outputRTable, outputGTable, outputBTable, outputATable will all consist solely of 0.0fs
   *
   *  @note The responsibility for both allocation and deallocation of the buffers is given to the caller
   */
  void GetShadingTable(  float* outputATable, float* outputDTable, float* outputSTable, float* outputPTable,
              int sizeI, int sizeG, float lowI, float offsetI, float highI, float lowG, float highG, float offsetG, int logUsed);
  
  /** @brief Gets the maximum number of classifications this transfer function currently has
   *
   *  @note This is currently just the maximum identifier found in the set of function objects, even if there are gaps in how these numbers are distributed over the objects
   */
  short GetNumberOfClassifications();
  
  /** @brief Gets the maximum gradient which this transfer function assigns attributes to
   *
   *  @pre The transfer function includes at least one function object
   */
  double getMaxGradient();
  
  /** @brief Gets the minimum gradient which this transfer function assigns attributes to
   *
   *  @pre The transfer function includes at least one function object
   */
  double getMinGradient();

  /** @brief Gets the maximum intensity which this transfer function assigns attributes to
   *
   *  @pre The transfer function includes at least one function object
   */
  double getMaxIntensity();

  /** @brief Gets the minimum intensity which this transfer function assigns attributes to
   *
   *  @pre The transfer function includes at least one function object
   */
  double getMinIntensity();
  
  /** @brief Adds a new function object to the list of function objects (note that order is ideally irrelevant)
   *
   *  @param object The non-null function object to be added
   *
   *  @note The responsibility for deallocation is given to the vtkCuda2DTransferFunction for this object
   */
  void AddFunctionObject(vtkCudaFunctionObject* object);

  /** @brief Removes a given function object from the list of function objects
   *
   *  @param object The non-null function object to be removed from the transfer function
   *
   *  @pre object is in the list of function objects for this transfer function
   *
   *  @note If the object is not found in the list, this method is safe, but does nothing
   *  @note The responsibility for deallocation for this object is given to the caller
   */
  void RemoveFunctionObject(vtkCudaFunctionObject* object);

  /** @brief Retrives a given function object from the list of function objects by it's index
   *
   *  @param index The index of the function object to be retrieved
   *
   *  @pre index is between 0 and the number of objects - 1 inclusive
   *
   *  @post Responsibility for deallocation remains with the 2D transfer function
   *
   */
  vtkCudaFunctionObject* GetFunctionObject(unsigned int index);

  /** @brief Retrives the current number of function objects associated with the function
   *
   */
  int GetNumberOfFunctionObjects();



protected:
  
  /** @brief Constructor which creates an empty transfer function, one with no function objects in its list
   *
   */
  vtkCuda2DTransferFunction();
  
  /** @brief Destructor which deallocates all the function objects in its list
   *
   */
  ~vtkCuda2DTransferFunction();

private:
  std::vector<vtkCudaFunctionObject*>* components; /**< The list of function objects making up this transfer function*/

};

#endif