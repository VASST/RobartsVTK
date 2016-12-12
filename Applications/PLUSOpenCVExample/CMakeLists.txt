PROJECT(PLUSOpenCVExample)

FIND_PACKAGE(PlusApp REQUIRED)
FIND_PACKAGE(PlusLib REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

SET(${PROJECT_NAME}_APPTYPE MACOSX_BUNDLE)
IF(WIN32)
  SET(${PROJECT_NAME}_APPTYPE WIN32)
ENDIF()

ADD_EXECUTABLE( ${PROJECT_NAME} ${${PROJECT_NAME}_APPTYPE}
  main.cpp
  PLUSOpenCVExampleMainWindow.h
  PLUSOpenCVExampleMainWindow.cpp
  PLUSOpenCVExampleMainWindow.ui
  ${PlusApp_CommonWidgets_QRC}
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${CMAKE_CURRENT_SOURCE_DIR}
  )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} PRIVATE
  opencv_core
  opencv_imgproc
  opencv_videoio
  opencv_highgui
  opencv_features2d
  opencv_calib3d
  PlusCommonWidgets
  vtkPlusCommon
  vtkPlusDataCollection
  vtkCommonCore
  vtkRenderingCore
  vtkGUISupportQt
  vtkRenderingGL2PS${VTK_RENDERING_BACKEND}
  Qt5::Widgets 
  Qt5::Core 
  Qt5::Gui 
)