PROJECT(vtkCudaCommonStatic)

SET( Module_SRCS
  CUDA_commonKernels.h
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
SET(${PROJECT_NAME}_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
vtk_module_impl()

# -----------------------------------------------------------------
# Generate the cuda sources
cuda_compile(Module_CUDA_C_SRCS ${Module_CUDA_SRCS} ${GPGPU_BUILD_TYPE})

SET_SOURCE_FILES_PROPERTIES( ${Module_CUDA_C_SRCS} CudaObject.cxx CUDA_commonKernels.h
  PROPERTIES WRAP_EXCLUDE ON
  )
LIST(APPEND Module_SRCS ${Module_CUDA_C_SRCS})

# -----------------------------------------------------------------
# Build the library
# Override shared libs settings, due to template functions in CUDA file, dll linkage is not possible
SET(BUILD_SHARED_LIBS OFF)
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PRIVATE ${CUDA_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
SET(BUILD_SHARED_LIBS ON)

# -----------------------------------------------------------------
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES module_uses_cuda "ON")
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)