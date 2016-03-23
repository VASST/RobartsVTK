PROJECT(CudaReconstruction)

include(module.cmake)

SET( Module_SRCS
  kernels.cl 
  CudaReconstruction.h
  CudaReconstruction.cpp
  )

SET_SOURCE_FILES_PROPERTIES(kernels.cl PROPERTIES HEADER_FILE_ONLY TRUE)
SET_SOURCE_FILES_PROPERTIES(kernels.cl PROPERTIES WRAP_EXCLUDE TRUE)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR} ${OpenCL_INCLUDE_DIRS})

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${OpenCL_LIBRARIES})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)