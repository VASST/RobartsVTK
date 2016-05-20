PROJECT(vtkCudaVisualization)

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
SET(${PROJECT_NAME}_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})
vtk_module_impl()

# -----------------------------------------------------------------
# Generate the cuda sources
cuda_compile(Module_CUDA_C_SRCS ${Module_CUDA_SRCS} ${GPGPU_BUILD_TYPE})

SET_SOURCE_FILES_PROPERTIES(
  ${Module_CUDA_C_SRCS} 
  WRAP_EXCLUDE
  )
LIST(APPEND Module_SRCS ${Module_CUDA_C_SRCS})

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PRIVATE ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES})

# -----------------------------------------------------------------
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES module_uses_cuda ON)
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)