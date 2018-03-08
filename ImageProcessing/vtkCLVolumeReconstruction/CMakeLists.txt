PROJECT(CLVolumeReconstruction)

SET( ${PROJECT_NAME}_SRCS
  vtkCLVolumeReconstruction.cpp
  )

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS
    vtkCLVolumeReconstruction.h
  )
ENDIF()

SET(vtk${PROJECT_NAME}_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  )

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(vtk${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(vtk${PROJECT_NAME} PUBLIC
  ${vtk${PROJECT_NAME}_INCLUDE_DIRS}
  )
target_compile_definitions(vtk${PROJECT_NAME} PUBLIC
  KERNEL_CL_LOCATION="${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl"
  )
target_link_libraries(vtk${PROJECT_NAME} PUBLIC
  OpenCL
  vtkCommonCore
  vtkCommonDataModel
  vtkCommonExecutionModel
  )
GENERATE_EXPORT_DIRECTIVE_FILE(vtk${PROJECT_NAME})

RobartsVTKInstallLibrary(vtk${PROJECT_NAME} ${PROJECT_NAME})
