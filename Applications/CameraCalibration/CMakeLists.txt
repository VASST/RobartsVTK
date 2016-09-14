PROJECT( CameraCalibration )

CMAKE_POLICY(SET CMP0020 NEW)
SET(CMAKE_AUTOMOC ON)
SET(CMAKE_AUTOUIC ON)
CMAKE_MINIMUM_REQUIRED( VERSION 3.0 )

FIND_PACKAGE( Qt5 COMPONENTS Gui Widgets Xml)

INCLUDE_DIRECTORIES(${Qt5Gui_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS})

FIND_PACKAGE(OpenCV 3.1 REQUIRED)

FIND_PACKAGE(VTK REQUIRED)
INCLUDE_DIRECTORIES(${VTK_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

FIND_PACKAGE(PlusLib REQUIRED)
INCLUDE(${PlusLib_USE_FILE})

FIND_PACKAGE(PlusApp REQUIRED)
INCLUDE_DIRECTORIES(${PlusApp_INCLUDE_DIRS})

IF(WIN32)
  SET(CameraCalib_WIN32 WIN32)
ENDIF()

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/Threads)

ADD_EXECUTABLE( CameraCalibration ${CameraCalib_WIN32}
  main.cpp
  CameraCalibrationMainWindow.h
  CameraCalibrationMainWindow.cpp
  CameraCalibrationMainWidget.h
  CameraCalibrationMainWidget.cpp
  CameraCalibrationMainWidget.ui
  OpenCVCameraCapture.h
  OpenCVCameraCapture.cpp
  QScaledView.h
  QScaledView.cpp
  Threads/QComputeThread.h
  Threads/QComputeThread.cpp
  Threads/QCaptureThread.h
  Threads/QCaptureThread.cpp
  ${PlusApp_CommonWidgets_QRC}
  ${UISrcs}
)

TARGET_LINK_LIBRARIES( CameraCalibration Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Xml)

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
