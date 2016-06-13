PROJECT( CameraCalibration )

CMAKE_POLICY(SET CMP0020 NEW)
SET(CMAKE_AUTOMOC ON)
SET(CMAKE_AUTOUIC ON)
CMAKE_MINIMUM_REQUIRED( VERSION 3.0 )

IF(NOT RobartsVTK_USE_QT5_ONLY)
  FIND_PACKAGE( Qt4 QUIET )
ENDIF()

IF( NOT Qt4_FOUND )
  FIND_PACKAGE( Qt5 COMPONENTS Gui Widgets Xml)
ENDIF()

IF( Qt4_FOUND )
  SET( QT_USE_OPENGL TRUE )
  INCLUDE( ${QT_USE_FILE} )
ELSE()
  INCLUDE_DIRECTORIES(${Qt5Gui_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS})
ENDIF()

FIND_PACKAGE(OpenCV 3.1 REQUIRED)

FIND_PACKAGE(VTK 6.3 REQUIRED)
INCLUDE_DIRECTORIES(${VTK_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

FIND_PACKAGE(PlusLib REQUIRED)
INCLUDE(${PlusLib_USE_FILE})

FIND_PACKAGE(PlusApp REQUIRED)
INCLUDE_DIRECTORIES(${PlusApp_INCLUDE_DIRS})

IF(WIN32)
  SET(CameraCalib_WIN32 WIN32)
ENDIF()

ADD_EXECUTABLE( CameraCalibration ${CameraCalib_WIN32}
  main.cpp
  CameraCalibrationMainWindow.h
  CameraCalibrationMainWindow.cpp
  CameraCalibrationMainWidget.h
  CameraCalibrationMainWidget.cpp
  CameraCalibrationMainWidget.ui
  OpenCVInternals.h
  OpenCVInternals.cpp
  QScaledView.h
  QScaledView.cpp
  QCaptureThread.h
  QCaptureThread.cpp
  ${PlusApp_CommonWidgets_QRC}
  ${UISrcs}
)

IF( Qt5_FOUND )
  TARGET_LINK_LIBRARIES( CameraCalibration Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml)
ELSEIF( Qt4_FOUND )
  TARGET_LINK_LIBRARIES( CameraCalibration ${QT_LIBRARIES} )
ENDIF()

IF(WIN32)
  # From PLUS, use MSMF capture library to query webcams
  TARGET_LINK_LIBRARIES( CameraCalibration MfVideoCaptureLibrary )
ENDIF()

TARGET_LINK_LIBRARIES( CameraCalibration
  ${OpenCV_LIBS}
  vtkndicapi
  vtkPlusDataCollection
  vtkCommonCore
  vtkRenderingCore
  vtkGUISupportQt
  PlusCommonWidgets
  ${OPENGL_LIBRARIES}
)
