PROJECT(vtkCudaVisualization)

INCLUDE_DIRECTORIES(${vtkCudaCommon_INCLUDE_DIRS})

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

SET(${PROJECT_NAME}_LIBS vtkCudaCommon)

IF( MSVC )
  SET(${PROJECT_NAME}_HDRS 
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
  SET(${PROJECT_NAME}_HDRS)
ENDIF()

SET ( ${PROJECT_NAME}_SRCS
  #Volume rendering base classes
  vtkCudaVolumeMapper.cxx
  vtkCudaRendererInformationHandler.cxx
  vtkCudaVolumeInformationHandler.cxx
  vtkCudaOutputImageInformationHandler.cxx
  CUDA_vtkCudaVolumeMapper_renderAlgo.cu

  #1D Transfer function rendering subclass
  vtkCuda1DVolumeMapper.cxx
  vtkCuda1DTransferFunctionInformationHandler.cxx
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.cuh

  #2D Transfer function rendering subclass
  vtkCuda2DVolumeMapper.cxx
  vtkCuda2DTransferFunctionInformationHandler.cxx
  CUDA_vtkCuda2DVolumeMapper_renderAlgo.cuh

  #2D transfer function rendering with inclusion/exclusion logic with slicing planes
  vtkCuda2DInExLogicVolumeMapper.cxx
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.cuh
  vtkCuda2DInExLogicTransferFunctionInformationHandler.cxx

  #Dual image rendering subclass
  vtkCudaDualImageVolumeMapper.cxx
  vtkCudaDualImageTransferFunctionInformationHandler.cxx
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.cuh

  #Digitally reconstructed radiograph rendering subclass
  vtkCudaDRRImageVolumeMapper.cxx
  CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.cuh
)

REMOVE_VTK_DEFINITIONS()
CUDA_ADD_LIBRARY( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} ${${PROJECT_NAME}_HDRS} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} )