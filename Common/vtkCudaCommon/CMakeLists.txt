PROJECT(vtkCudaCommon)

SET(${PROJECT_NAME}_LIBS vtkCommonCore)

# Add entries to project details
SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_SOURCE_DIR} ${${PROJECT_NAME}_BINARY_DIR} CACHE INTERNAL "" FORCE)

IF( MSVC )
  SET(${PROJECT_NAME}_HDRS 
    vtkCudaObject.h 
    vtkCudaDeviceManager.h 
    vtkCudaMemoryTexture.h 
    CUDA_commonKernels.h 
    vtkCuda2DTransferFunction.h 
    vtkCudaFunctionObject.h 
    vtkCudaFunctionPolygon.h 
    vtkCudaFunctionPolygonReader.h 
    vtkCudaFunctionPolygonWriter.h 
    )
ELSE()
  SET(${PROJECT_NAME}_HDRS)
ENDIF()

SET ( ${PROJECT_NAME}_SRCS
  vtkCudaObject.cxx
  vtkCudaDeviceManager.cxx
  vtkCudaMemoryTexture.cxx
  CUDA_commonKernels.cu
  vtkCuda2DTransferFunction.cxx
  vtkCudaFunctionObject.cxx
  vtkCudaFunctionPolygon.cxx
  vtkCudaFunctionPolygonReader.cxx
  vtkCudaFunctionPolygonWriter.cxx
)

GENERATE_EXPORT_DIRECTIVE_FILE(${PROJECT_NAME})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

REMOVE_VTK_DEFINITIONS()
CUDA_ADD_LIBRARY( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} ${${PROJECT_NAME}_HDRS} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} )