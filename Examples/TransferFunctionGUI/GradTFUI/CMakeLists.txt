PROJECT(TFUI-Gradient)

SET(${PROJECT_NAME}_LIBS 
  TFUI-Common 
  vtkCudaVisualization 
  vtkCudaImageAnalytics 
  vtkCudaCommon 
  vtkRenderingVolume 
  vtkRenderingCore 
  vtkFiltersCore 
  vtkImagingCore 
  vtkIOImage 
  vtkCommonCore 
  vtkGUISupportQt
  )

IF( MSVC )
  SET(${PROJECT_NAME}_HDRS 
    qTransferFunctionWindowWidget.h
    qHistogramHolderLabel.h
    qTransferFunctionDefinitionWidget.h
    qSegmentationWidget.h
  )
ENDIF()

SET ( ${PROJECT_NAME}_SRCS
  main.cpp
  qTransferFunctionWindowWidget.cxx 
  qHistogramHolderLabel.cxx 
  qTransferFunctionDefinitionWidget.cxx
  qSegmentationWidget.cxx
)

IF( QT4_FOUND )
  INCLUDE(${QT_USE_FILE})
ELSE()
  INCLUDE_DIRECTORIES(${Qt5Widgets_INCLUDE_DIRS})
  LIST(APPEND ${PROJECT_NAME}_LIBS ${_Qt5Widgets_LIB_DEPENDENCIES})
ENDIF()

INCLUDE_DIRECTORIES(${TFUI-Common_SOURCE_DIR} ${vtkCudaCommon_INCLUDE_DIRS} ${vtkCudaVisualization_INCLUDE_DIRS} ${vtkCudaImageAnalytics_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} ${${PROJECT_NAME}_HDRS})
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} )