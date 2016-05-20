PROJECT(vtkCudaCommon)

include(module.cmake)

SET( Module_SRCS
  CudaObject.cxx
  vtkCudaDeviceManager.cxx
  vtkCudaMemoryTexture.cxx
  vtkCuda2DTransferFunction.cxx
  vtkCudaFunctionObject.cxx
  vtkCudaFunctionPolygon.cxx
  vtkCudaFunctionPolygonReader.cxx
  vtkCudaFunctionPolygonWriter.cxx
  vtkCudaCommon.cxx
)

# -----------------------------------------------------------------
# Build the library
SET(${PROJECT_NAME}_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PRIVATE ${CUDA_LIBRARIES} ${VTK_LIBRARIES} ${OPENGL_LIBRARIES})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)