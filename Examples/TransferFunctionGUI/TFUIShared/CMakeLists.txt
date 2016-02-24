PROJECT(TFUICommon)

include(module.cmake)

SET ( Module_SRCS 
  qTransferFunctionWindowWidgetInterface.cxx
  qFileManagementWidget.cxx
  qStereoRenderingWidget.cxx
  qVirtualToolWidget.cxx
  qShadingWidget.cxx
  qDeviceManagementWidget.cxx
  vtkImageHackedPlaneWidget.cxx
)

IF( QT4_FOUND )
  INCLUDE(${QT_USE_FILE})
ELSE()
  LIST(APPEND ${PROJECT_NAME}_INCLUDE_DIRS ${Qt5Widgets_INCLUDE_DIRS})
  SET(${PROJECT_NAME}_APPEND_LIBS Qt5::Widgets ${_Qt5Widgets_LIB_DEPENDENCIES})
  LIST(APPEND ${PROJECT_NAME}_LIBS ${PROJECT_NAME}_APPEND_LIBS)
ENDIF()

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${${PROJECT_NAME}_APPEND_LIBS})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)