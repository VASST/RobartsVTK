PROJECT( MaxFlowExample )

SET(${PROJECT_NAME}_LIBS 
  vtkRenderingCore 
  vtkFiltersCore 
  vtkImagingCore 
  vtkIOImage 
  vtkCommonCore 
  vtkRobartsCommon 
  vtkCudaImageAnalytics 
  vtkCudaCommon
  vtksys
  )

SET ( ${PROJECT_NAME}_SRCS
  MaxFlowExample.cxx
)

INCLUDE_DIRECTORIES ( ${vtkCudaCommon_INCLUDE_DIRS} ${vtkCudaImageAnalytics_INCLUDE_DIRS} ${vtkRobartsCommon_INCLUDE_DIRS} )

CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} )
