PROJECT(vtkCudaCommonStatic)

include(module.cmake)

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

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
vtk_module_impl()

# -----------------------------------------------------------------
# Generate the cuda sources
INCLUDE_DIRECTORIES(${${vtk-module}_DEPENDS_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
cuda_compile(Module_CUDA_C_SRCS ${Module_CUDA_SRCS} ${GPGPU_BUILD_TYPE})

SET_SOURCE_FILES_PROPERTIES( ${Module_CUDA_C_SRCS} CudaObject.cxx CUDA_commonKernels.h
  PROPERTIES WRAP_EXCLUDE ON
  )
LIST(APPEND Module_SRCS ${Module_CUDA_C_SRCS})

# -----------------------------------------------------------------
# Build the library
INCLUDE_DIRECTORIES(${${PROJECT_NAME}_INCLUDE_DIRS})
# Override shared libs settings, due to template functions in CUDA file, dll linkage is not possible
SET(BUILD_SHARED_LIBS OFF)
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PRIVATE ${CUDA_LIBRARIES})
SET(BUILD_SHARED_LIBS ON)

# -----------------------------------------------------------------
SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES module_uses_cuda "ON")
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)