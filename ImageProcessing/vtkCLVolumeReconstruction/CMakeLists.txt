PROJECT(vtkCLVolumeReconstruction)

include(module.cmake)

SET( Module_SRCS
  vtkCLVolumeReconstruction.cpp
  )

# TODO : try to determine if there is an OpenCL equivalent to the fucntions called by vector_math.h
SET(${PROJECT_NAME}_INCLUDE_DIRS 
  ${CMAKE_SOURCE_DIR} 
  ${OpenCL_INCLUDE_DIRS} 
  ${CUDA_INCLUDE_DIRS} 
  ${VTK_INCLUDE_DIRS}
  )

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${OpenCL_LIBRARY})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)