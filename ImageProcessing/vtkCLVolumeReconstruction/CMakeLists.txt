PROJECT(vtkCLVolumeReconstruction)

SET( ${PROJECT_NAME}_SRCS
  vtkCLVolumeReconstruction.cpp
  )

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
    vtkCLVolumeReconstruction.h
  )
ENDIF()

SET(${PROJECT_NAME}_INCLUDE_DIRS 
  ${CMAKE_SOURCE_DIR} 
  ${OpenCL_INCLUDE_DIRS} 
  ${CUDA_INCLUDE_DIRS} 
  ${VTK_INCLUDE_DIRS}
  )

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCL_INCLUDE_DIRS} 
  ${CUDA_INCLUDE_DIRS} 
  ${VTK_INCLUDE_DIRS}
  )
target_compile_definitions(${PROJECT_NAME} PUBLIC
  KERNEL_CL_LOCATION="${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl"
  )
target_link_libraries(${PROJECT_NAME} PUBLIC 
  OpenCL 
  vtkCommonCore
  vtkCommonDataModel
  vtkCommonExecutionModel
  )
GENERATE_EXPORT_DIRECTIVE_FILE(${PROJECT_NAME})