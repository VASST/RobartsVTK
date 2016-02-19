PROJECT(vtkCudaCommon)

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

SET( Module_SRCS
  CudaObject.cxx
  vtkCudaDeviceManager.cxx
  vtkCudaMemoryTexture.cxx
  vtkCuda2DTransferFunction.cxx
  vtkCudaFunctionObject.cxx
  vtkCudaFunctionPolygon.cxx
  vtkCudaFunctionPolygonReader.cxx
  vtkCudaFunctionPolygonWriter.cxx
)

SET( Module_CUDA_SRCS
  CUDA_commonKernels.cu
  )

IF(BUILD_SHARED_LIBS)
  SET(GPGPU_BUILD_TYPE SHARED)
ENDIF()

REMOVE_VTK_DEFINITIONS()

#------------------------------------------------------------------
# See VTK/Accelerators/Pistons
vtk_module_impl()

# -----------------------------------------------------------------
# Generate the cuda sources
INCLUDE_DIRECTORIES(${${vtk-module}_DEPENDS_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
cuda_compile(Module_CUDA_C_SRCS ${Module_CUDA_SRCS} ${GPGPU_BUILD_TYPE})

SET_SOURCE_FILES_PROPERTIES(
  CudaObject.cxx 
  ${Module_CUDA_C_SRCS} 
  WRAP_EXCLUDE
  )
LIST(APPEND Module_SRCS ${Module_CUDA_C_SRCS})

# -----------------------------------------------------------------
# Build the library
INCLUDE_DIRECTORIES(${${PROJECT_NAME}_INCLUDE_DIRS})
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PRIVATE ${CUDA_LIBRARIES})

# -----------------------------------------------------------------
# Build the python wrapping if enabled
IF( VTK_WRAP_PYTHON )
  vtk_module_load(${PROJECT_NAME})
  vtk_module_headers_load(${PROJECT_NAME})
  vtk_include_recurse(${PROJECT_NAME})
  vtk_add_python_wrapping(${PROJECT_NAME})
ENDIF()

LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)