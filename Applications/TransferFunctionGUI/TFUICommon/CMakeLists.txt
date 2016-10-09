PROJECT(TFUICommon)

SET ( ${PROJECT_NAME}_SRCS 
  qTransferFunctionWindowWidgetInterface.cxx
  qFileManagementWidget.cxx
  qStereoRenderingWidget.cxx
  qVirtualToolWidget.cxx
  qShadingWidget.cxx
  qDeviceManagementWidget.cxx
  vtkImageHackedPlaneWidget.cxx
)

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
    qTransferFunctionWindowWidgetInterface.h
    qFileManagementWidget.h
    qStereoRenderingWidget.h
    qVirtualToolWidget.h
    qShadingWidget.h
    qDeviceManagementWidget.h
    vtkImageHackedPlaneWidget.h
  )
ENDIF()

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${VTK_INCLUDE_DIRS}
  )
target_link_libraries(${PROJECT_NAME} PUBLIC
  Qt5::Widgets
  vtkCudaCommon
  vtkCudaVisualization
  vtkRenderingVolume${VTK_RENDERING_BACKEND}
  vtkInteractionWidgets
  vtksys
  vtkIOLegacy
  vtkIOMINC
  vtkImagingMath
  )
GENERATE_EXPORT_DIRECTIVE_FILE(${PROJECT_NAME})