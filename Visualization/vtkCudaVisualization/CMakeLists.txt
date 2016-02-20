PROJECT(vtkCudaVisualization)

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

SET( Module_SRCS
  vtkCudaVolumeMapper.cxx
  vtkCudaRendererInformationHandler.cxx
  vtkCudaVolumeInformationHandler.cxx
  vtkCudaOutputImageInformationHandler.cxx
  vtkCuda1DVolumeMapper.cxx
  vtkCuda1DTransferFunctionInformationHandler.cxx
  vtkCuda2DVolumeMapper.cxx
  vtkCuda2DTransferFunctionInformationHandler.cxx
  vtkCuda2DInExLogicVolumeMapper.cxx
  vtkCuda2DInExLogicTransferFunctionInformationHandler.cxx
  vtkCudaDualImageVolumeMapper.cxx
  vtkCudaDualImageTransferFunctionInformationHandler.cxx
  vtkCudaDRRImageVolumeMapper.cxx
)

SET( Module_CUDA_SRCS
  CUDA_vtkCudaVolumeMapper_renderAlgo.cu
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda2DVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.cuh
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
IF( RobartsVTK_WRAP_PYTHON AND VTK_WRAP_PYTHON )
  vtk_module_load(${PROJECT_NAME})
  vtk_module_headers_load(${PROJECT_NAME})
  vtk_include_recurse(${PROJECT_NAME})
  vtk_add_python_wrapping(${PROJECT_NAME})
ENDIF()

LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)