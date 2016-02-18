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

IF( QT4_FOUND )
  INCLUDE(${QT_USE_FILE})
ELSE()
  INCLUDE_DIRECTORIES(${Qt5Widgets_INCLUDE_DIRS})
ENDIF()

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

if( MSVC )
  SET( ${PROJECT_NAME}_HDRS 
    qTransferFunctionWindowWidgetInterface.h
    qFileManagementWidget.h
    qStereoRenderingWidget.h
    qVirtualToolWidget.h
    qShadingWidget.h
    qDeviceManagementWidget.h
    vtkImageHackedPlaneWidget.h
  )
endif()

CUDA_ADD_LIBRARY( ${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} ${${PROJECT_NAME}_HDRS})
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} )