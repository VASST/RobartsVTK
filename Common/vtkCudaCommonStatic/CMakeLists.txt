PROJECT(CudaCommonStatic)

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS
    CUDA_commonKernels.h
    )
ENDIF()

SET( ${PROJECT_NAME}_CUDA_SRCS
  CUDA_commonKernels.cu
  )

IF(BUILD_SHARED_LIBS)
  SET(GPGPU_BUILD_TYPE SHARED)
ENDIF()

# Enable other dependencies to include the files necessary for cuda builds
SET(vtk${PROJECT_NAME}_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${VTK_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  CACHE INTERNAL "vtk${PROJECT_NAME}_INCLUDE_DIRS")

REMOVE_VTK_DEFINITIONS()

# -----------------------------------------------------------------
# Generate the cuda sources
LIST(APPEND CUDA_NVCC_FLAGS -I${CMAKE_CURRENT_BINARY_DIR})
foreach(dir ${vtkCudaCommon_INCLUDE_DIRS})
  LIST(APPEND CUDA_NVCC_FLAGS -I${dir})
endforeach()

cuda_compile(${PROJECT_NAME}_CUDA_C_SRCS ${${PROJECT_NAME}_CUDA_SRCS} ${GPGPU_BUILD_TYPE})
SET(${PROJECT_NAME}_SRCS ${${PROJECT_NAME}_CUDA_C_SRCS})

# -----------------------------------------------------------------
# Build the library
# Override shared libs settings, due to template functions in CUDA file, dll linkage is not possible
ADD_LIBRARY(vtk${PROJECT_NAME} STATIC ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(vtk${PROJECT_NAME} PUBLIC
  ${vtk${PROJECT_NAME}_INCLUDE_DIRS}
  )
target_link_libraries(vtk${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})
set_target_properties(vtk${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
GENERATE_EXPORT_DIRECTIVE_FILE(vtk${PROJECT_NAME})

RobartsVTKInstallLibrary(vtk${PROJECT_NAME} ${PROJECT_NAME})
