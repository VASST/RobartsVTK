PROJECT( FluoroPredViz )

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

IF( QT4_FOUND )
  INCLUDE(${QT_USE_FILE})
ELSE()
  INCLUDE_DIRECTORIES(${Qt5Widgets_INCLUDE_DIRS})
  LIST(APPEND ${PROJECT_NAME}_LIBS ${_Qt5Widgets_LIB_DEPENDENCIES})
ENDIF()

INCLUDE_DIRECTORIES ( ${vtkRobartsCommon_INCLUDE_DIRS} ${vtkCudaVisualization_INCLUDE_DIRS} ${vtkCudaCommon_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS})