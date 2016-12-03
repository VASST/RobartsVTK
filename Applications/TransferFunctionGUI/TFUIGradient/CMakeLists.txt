PROJECT(TFUIGradient)

SET ( ${PROJECT_NAME}_SRCS 
  qTransferFunctionWindowWidget.cxx 
  qHistogramHolderLabel.cxx 
  qTransferFunctionDefinitionWidget.cxx
  qSegmentationWidget.cxx
  main.cxx
)

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
    qTransferFunctionWindowWidget.h 
    qHistogramHolderLabel.h 
    qTransferFunctionDefinitionWidget.h
    qSegmentationWidget.h
  )
ENDIF()

SET(${PROJECT_NAME}_INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  CACHE INTERNAL "${PROJECT_NAME}_INCLUDE_DIRS"
  )

# -----------------------------------------------------------------
# Build the executable
ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${VTK_INCLUDE_DIRS}
  ${${PROJECT_NAME}_INCLUDE_DIRS}
  )
target_link_libraries(${PROJECT_NAME} PUBLIC 
  Qt5::Widgets
  TFUICommon 
  vtkCudaVisualization 
  vtkCudaImageAnalytics 
  vtkCudaCommon 
  vtkRenderingGL2PS${VTK_RENDERING_BACKEND}
  vtkRenderingVolume${VTK_RENDERING_BACKEND}
  vtkRenderingCore 
  vtkFiltersCore 
  vtkImagingCore 
  vtkImagingGeneral 
  vtkIOImage 
  vtkCommonCore 
  vtkGUISupportQt
  )