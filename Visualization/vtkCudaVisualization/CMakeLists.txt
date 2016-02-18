PROJECT(vtkCudaVisualization)

INCLUDE(module.cmake)

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

IF( MSVC )
  SET(Module_HDRS 
    vtkCudaVolumeMapper.h
    vtkCudaRendererInformationHandler.h
    vtkCudaVolumeInformationHandler.h
    vtkCudaOutputImageInformationHandler.h
    CUDA_containerRendererInformation.h
    CUDA_containerVolumeInformation.h
    CUDA_containerOutputImageInformation.h
    CUDA_vtkCudaVolumeMapper_renderAlgo.h
    vtkCuda1DVolumeMapper.h
    vtkCuda1DTransferFunctionInformationHandler.h
    CUDA_container1DTransferFunctionInformation.h
    CUDA_vtkCuda1DVolumeMapper_renderAlgo.h
    vtkCuda2DVolumeMapper.h
    vtkCuda2DTransferFunctionInformationHandler.h
    CUDA_container2DTransferFunctionInformation.h
    CUDA_vtkCuda2DVolumeMapper_renderAlgo.h
    vtkCuda2DInExLogicVolumeMapper.h
    CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.h
    CUDA_container2DInExTransferFunctionInformation.h
    vtkCuda2DInExLogicTransferFunctionInformationHandler.h
    vtkCudaDualImageVolumeMapper.h
    vtkCudaDualImageTransferFunctionInformationHandler.h
    CUDA_containerDualImageTransferFunctionInformation.h
    CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.h
    vtkCudaDRRImageVolumeMapper.h
    CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.h
    )
ELSE()
  SET(Module_HDRS)
ENDIF()

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
  )

SET( Module_CUDA_HDRS
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda2DVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.cuh
  )

SET_SOURCE_FILES_PROPERTIES(
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.h
  CUDA_vtkCudaVolumeMapper_renderAlgo.cu
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.cuh
  WRAP_EXCLUDE
  )

# -----------------------------------------------------------------
# Build the library
include_directories(${${PROJECT_NAME}_INCLUDE_DIRS})
cuda_vtk_add_library( ${PROJECT_NAME} ${Module_CUDA_SRCS} ${Module_CUDA_HDRS} ${Module_SRCS} ${Module_HDRS} )

# -----------------------------------------------------------------
# Build the python wrapping if enabled
IF( VTK_WRAP_PYTHON AND RobartsVTK_CUDA_WRAP_PYTHON )
  vtk_module_load(${PROJECT_NAME})
  vtk_module_headers_load(${PROJECT_NAME})
  vtk_include_recurse(${PROJECT_NAME})
  vtk_add_python_wrapping(${PROJECT_NAME})
ENDIF()