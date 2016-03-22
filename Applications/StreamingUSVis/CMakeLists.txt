CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

PROJECT(StreamingUSVis)

IF(UNIX)
  SET(EXECUTABLE_STYLE "")
ELSEIF(APPLE)
  SET(EXECUTABLE_STYLE MACOSX_BUNDLE)
ELSEIF(WIN32)
  SET(EXECUTABLE_STYLE WIN32)
ENDIF()

#-------------------------------------
# Find OpenCL
#-------------------------------------
FIND_PACKAGE(OpenCL REQUIRED)

#-------------------------------------
# Add CudaReconstruction
#-------------------------------------
ADD_SUBDIRECTORY(CudaReconstruction)

SET(StreamingUSVis_INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCV_INCLUDE_DIR} 
  ${VTK_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIR}
  ${OpenCL_INCLUDE_DIRS}
  ${vtkCudaCommon_INCLUDE_DIRS}
  ${vtkCudaVisualization_INCLUDE_DIRS}
  ${vtkCudaImageAnalytics_INCLUDE_DIRS}
  ${TFUIGradient_INCLUDE_DIRS}
  ${TFUICommon_INCLUDE_DIRS}
  ${CudaReconstruction_INCLUDE_DIRS}
  )

IF( VTK_QT_VERSION VERSION_LESS "5" )
  INCLUDE(${QT_USE_FILE})
ELSE()
  SET(StreamingUSVis_INCLUDE_DIRS ${StreamingUSVis_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS})
ENDIF()

SET(StreamingUSVis_LIBRARIES 
  ${GLUT_LIBRARIES}
  ${OpenCL_LIBRARIES}
  ${vtkCudaCommon_LIBRARIES}
  ${vtkCudaVisualization_LIBRARIES}
  ${vtkCudaImageAnalytics_LIBRARIES}
  ${TFUIGradient_LIBRARIES}
  ${TFUICommon_LIBRARIES}
  ${CudaReconstruction_LIBRARIES}
  )

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

#-------------------------------------
# Add PLUS
#-------------------------------------
INCLUDE(${PlusLib_USE_FILE})

#-------------------------------------
# Retrieve source files
#-------------------------------------
FILE(GLOB UI_FILES *.ui)
FILE(GLOB QT_WRAP *.h)
FILE(GLOB CXX_FILES *.cxx)
FILE(GLOB CPP_FILES *.cpp)

INCLUDE_DIRECTORIES(${StreamingUSVis_INCLUDE_DIRS})

#-------------------------------------
# Create executable
#-------------------------------------
if( VTK_QT_VERSION VERSION_GREATER "4" )
  qt5_wrap_ui(UISrcs ${UI_FILES} )

  ADD_EXECUTABLE(${PROJECT_NAME} ${EXECUTABLE_STYLE}
    ${CXX_FILES} 
    ${CPP_FILES} 
    ${UISrcs} 
    ${QT_WRAP}
  )

  TARGET_LINK_LIBRARIES(${PROJECT_NAME} Qt5::Widgets)
else()
  QT4_WRAP_UI(UISrcs ${UI_FILES})

  ADD_EXECUTABLE(${PROJECT_NAME} ${EXECUTABLE_STYLE} 
    ${CXX_FILES} 
    ${CPP_FILES} 
    ${UISrcs} 
    ${QT_WRAP}
  )
 
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${QT_LIBRARIES})
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${StreamingUSVis_LIBRARIES})