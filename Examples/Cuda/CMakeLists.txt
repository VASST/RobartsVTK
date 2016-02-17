PROJECT( Cuda2DTFVolumeRendering )

SET(${PROJECT_NAME}_LIBS 
  vtkRenderingVolumeOpenGL 
  vtkRenderingFreeTypeOpenGL 
  vtkFiltersCore 
  vtkImagingCore 
  vtkInteractionWidgets 
  vtkIOImage 
  vtkCommonCore 
  vtkCudaCommon 
  vtkCudaVisualization 
  )

SET ( ${PROJECT_NAME}_SRCS
  Cuda2DTFVolumeRendering.cxx
)

INCLUDE_DIRECTORIES ( ${vtkCudaVisualization_INCLUDE_DIRS} ${vtkCudaCommon_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} )
