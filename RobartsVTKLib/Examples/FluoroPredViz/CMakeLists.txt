PROJECT( FluoroPredViz )

IF(QT4_FOUND)
  INCLUDE( ${QT_USE_FILE} )
ELSE()
  SET(QT5_LIBRARIES ${Qt5Widgets_LIBRARIES})
ENDIF()

SET(${PROJECT_NAME}_LIBS 
  vtkCudaCommon 
  vtkCudaVisualization 
  vtkRobartsCommon 
  vtkRenderingVolumeOpenGL 
  vtkRenderingFreeTypeOpenGL 
  vtkRenderingCore 
  vtkFiltersCore 
  vtkImagingCore 
  vtkInteractionWidgets 
  vtkGUISupportQt 
  vtkIOCore 
  vtkCommonCore 
  vtkIOMINC 
  )

SET ( ${PROJECT_NAME}_SRCS
  main.cxx
  FluoroPredViz.h
  FluoroPredViz.cxx
  ResizableQVTKWidget.h
  ResizableQVTKWidget.cxx
)

INCLUDE_DIRECTORIES ( ${vtkRobartsCommon_INCLUDE_DIRS} ${vtkCudaVisualization_INCLUDE_DIRS} ${vtkCudaCommon_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} ${QT5_LIBRARIES})