PROJECT(TFUI-Common)

SET(${PROJECT_NAME}_LIBS 
  vtkCudaCommon 
  vtkCudaVisualization  
  vtkRenderingVolumeOpenGL 
  vtkRenderingFreeTypeOpenGL 
  vtkInteractionWidgets 
  vtksys 
  vtkIOMINC 
  vtkImagingMath 
  )

INCLUDE_DIRECTORIES(${vtkCudaCommon_INCLUDE_DIRS} ${vtkCudaVisualization_INCLUDE_DIRS})

SET ( ${PROJECT_NAME}_SRCS
  qTransferFunctionWindowWidgetInterface.cxx
  qFileManagementWidget.cxx
  qStereoRenderingWidget.cxx
  qVirtualToolWidget.cxx
  qShadingWidget.cxx
  qDeviceManagementWidget.cxx
  vtkImageHackedPlaneWidget.cxx
)

IF( MSVC )
  SET( ${PROJECT_NAME}_HDRS 
    qTransferFunctionWindowWidgetInterface.h
    qFileManagementWidget.h
    qStereoRenderingWidget.h
    qVirtualToolWidget.h
    qShadingWidget.h
    qDeviceManagementWidget.h
    vtkImageHackedPlaneWidget.h
  )
ENDIF()

IF( QT4_FOUND )
  INCLUDE(${QT_USE_FILE})
ELSE()
  INCLUDE_DIRECTORIES(${Qt5Widgets_INCLUDE_DIRS})
  LIST(APPEND ${PROJECT_NAME}_LIBS ${_Qt5Widgets_LIB_DEPENDENCIES})
ENDIF()

CUDA_ADD_LIBRARY( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} ${${PROJECT_NAME}_HDRS})
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} )