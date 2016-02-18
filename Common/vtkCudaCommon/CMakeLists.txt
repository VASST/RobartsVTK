PROJECT(vtkCudaCommon)

include(module.cmake)

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

IF( MSVC )
  SET(Module_HDRS 
    vtkCudaObject.h 
    vtkCudaDeviceManager.h 
    vtkCudaMemoryTexture.h 
    CUDA_commonKernels.h 
    vtkCuda2DTransferFunction.h 
    vtkCudaFunctionObject.h 
    vtkCudaFunctionPolygon.h 
    vtkCudaFunctionPolygonReader.h 
    vtkCudaFunctionPolygonWriter.h 
    )
ELSE()
  SET(Module_HDRS)
ENDIF()

SET ( Module_SRCS
  vtkCudaObject.cxx
  vtkCudaDeviceManager.cxx
  vtkCudaMemoryTexture.cxx
  vtkCuda2DTransferFunction.cxx
  vtkCudaFunctionObject.cxx
  vtkCudaFunctionPolygon.cxx
  vtkCudaFunctionPolygonReader.cxx
  vtkCudaFunctionPolygonWriter.cxx
)

set( Module_CUDA_SRCS
  CUDA_commonKernels.cu
  )

SET_SOURCE_FILES_PROPERTIES(
  CUDA_commonKernels.h
  vtkCudaObject.h 
  WRAP_EXCLUDE)

# -----------------------------------------------------------------
# Build the library
include_directories(${${PROJECT_NAME}_INCLUDE_DIRS})
cuda_vtk_add_library( ${PROJECT_NAME} ${Module_CUDA_SRCS} ${Module_SRCS} ${Module_HDRS} )

# -----------------------------------------------------------------
# Build the python wrapping if enabled
IF( VTK_WRAP_PYTHON )
  vtk_module_load(${PROJECT_NAME})
  vtk_module_headers_load(${PROJECT_NAME})
  vtk_include_recurse(${PROJECT_NAME})
  vtk_add_python_wrapping(${PROJECT_NAME})
ENDIF()