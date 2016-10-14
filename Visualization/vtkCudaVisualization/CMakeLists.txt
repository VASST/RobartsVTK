PROJECT(vtkCudaVisualization)

SET( ${PROJECT_NAME}_SRCS
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

SET( ${PROJECT_NAME}_CUDA_SRCS
  CUDA_vtkCudaVolumeMapper_renderAlgo.cu
  CUDA_vtkCuda1DVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda2DVolumeMapper_renderAlgo.cuh
  CUDA_vtkCuda2DInExLogicVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaDualImageVolumeMapper_renderAlgo.cuh
  CUDA_vtkCudaDRRImageVolumeMapper_renderAlgo.cuh
  )

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
    vtkCudaVolumeMapper.h
    vtkCudaRendererInformationHandler.h
    vtkCudaVolumeInformationHandler.h
    vtkCudaOutputImageInformationHandler.h
    vtkCuda1DVolumeMapper.h
    vtkCuda1DTransferFunctionInformationHandler.h
    vtkCuda2DVolumeMapper.h
    vtkCuda2DTransferFunctionInformationHandler.h
    vtkCuda2DInExLogicVolumeMapper.h
    vtkCuda2DInExLogicTransferFunctionInformationHandler.h
    vtkCudaDualImageVolumeMapper.h
    vtkCudaDualImageTransferFunctionInformationHandler.h
    vtkCudaDRRImageVolumeMapper.h
  )
ENDIF()

IF(BUILD_SHARED_LIBS)
  SET(GPGPU_BUILD_TYPE SHARED)
ENDIF()

REMOVE_VTK_DEFINITIONS()

# -----------------------------------------------------------------
# Generate the cuda sources
cuda_compile(${PROJECT_NAME}_CUDA_C_SRCS ${${PROJECT_NAME}_CUDA_SRCS} ${GPGPU_BUILD_TYPE})
LIST(APPEND ${PROJECT_NAME}_SRCS ${${PROJECT_NAME}_CUDA_C_SRCS})

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${VTK_INCLUDE_DIRS} 
  ${CUDA_INCLUDE_DIRS}
  )
target_link_libraries(${PROJECT_NAME} PUBLIC
  ${CUDA_LIBRARIES}
  ${OPENGL_LIBRARIES}
  vtkCudaCommon
  vtkCudaCommonStatic
  vtkRenderingVolume
  vtkRenderingVolume${VTK_RENDERING_BACKEND}
  )
GENERATE_EXPORT_DIRECTIVE_FILE(${PROJECT_NAME})