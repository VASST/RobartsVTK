CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

PROJECT( KeyholeRenderingMultipleViewPorts )

FIND_PACKAGE(OpenCV 3.1.0 REQUIRED NO_MODULE)

#-------------------------------------
# Retrieve source files
#-------------------------------------
SET(${PROJECT_NAME}_SRCS
  main.cpp
  )

#-------------------------------------
# Create executable
#-------------------------------------
ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCV_INCLUDE_DIRS} 
  ${VTK_INCLUDE_DIRS}
  )
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PUBLIC 
  Qt5::Widgets
  vtkRobartsVisualization
  ${OPENGL_LIBRARIES}
  vtkIOImage
  vtksys
  vtkFiltersTexture
  opencv_core
  opencv_imgproc
  opencv_videoio
  opencv_imgcodecs
  opencv_imgproc
  opencv_highgui
  )
target_compile_definitions(${PROJECT_NAME} PUBLIC
  DEFAULT_BACKGROUND_FILE="${RobartsVTK_Data_DIR}/Images/CAP_201587T184044_UND.avi"
  DEFAULT_MASK_FILE="${RobartsVTK_Data_DIR}/Images/mask_2.png"
  )