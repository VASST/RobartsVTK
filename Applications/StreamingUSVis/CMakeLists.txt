CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

PROJECT(StreamingUSVis)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

#-------------------------------------
# Retrieve source files
#-------------------------------------
SET(${PROJECT_NAME}_SRCS
  main.cpp
  mainwindow.cpp
  vtkBoxRepresentation2.cpp
  )
  
IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
  vtkBoxRepresentation2.h
  mainwindow.h
  )
ENDIF()

SET(${PROJECT_NAME}_UI_FILES
  mainwindow.ui
)

#-------------------------------------
# Create executable
#-------------------------------------
ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_SRCS} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_UI_FILES} )
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCV_INCLUDE_DIRS} 
  ${VTK_INCLUDE_DIRS}
  ${ITK_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
  ${TFUIDual_INCLUDE_DIRS}
  )
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC 
  Qt5::Widgets
  OpenCL
  vtkCudaCommon
  vtkCudaVisualization
  vtkRobartsVisualization
  vtkCudaImageAnalytics
  vtkCLVolumeReconstruction
  vtksys
  TFUICommon
  vtkInteractionWidgets
  vtkInteractionImage
  vtkGUISupportQt
  vtkPlusVolumeReconstruction
  vtkPlusCommon
  vtkPlusDataCollection
  opencv_core
  opencv_imgproc
  opencv_videoio
  )