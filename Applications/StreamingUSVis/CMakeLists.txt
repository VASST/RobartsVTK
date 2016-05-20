CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

PROJECT(StreamingUSVis)

#-------------------------------------
# Find OpenCL
#-------------------------------------
FIND_PACKAGE(OpenCL REQUIRED)

#-------------------------------------
# Load module to access include lists
#-------------------------------------
vtk_module_load(TFUIDual)
vtk_module_load(TFUICommon)
vtk_module_load(vtkCudaCommon)
vtk_module_load(vtkCudaVisualization)
vtk_module_load(vtkRobartsVisualization)
vtk_module_load(vtkCudaImageAnalytics)
vtk_module_load(vtkCLVolumeReconstruction)

SET(StreamingUSVis_INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCV_INCLUDE_DIRS} 
  ${VTK_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${PLUSLIB_INCLUDE_DIRS}
  ${TFUIDual_INCLUDE_DIRS}
  ${TFUICommon_INCLUDE_DIRS}
  ${OpenCL_INCLUDE_DIRS}
  ${vtkCudaCommon_INCLUDE_DIRS}
  ${vtkCudaVisualization_INCLUDE_DIRS}
  ${vtkRobartsVisualization_INCLUDE_DIRS}
  ${vtkCudaImageAnalytics_INCLUDE_DIRS}
  ${vtkCLVolumeReconstruction_INCLUDE_DIRS}
  ${PlusLib_INCLUDE_DIRS})
LIST(REMOVE_DUPLICATES StreamingUSVis_INCLUDE_DIRS)

IF( VTK_QT_VERSION VERSION_LESS "5" )
  INCLUDE(${QT_USE_FILE})
ELSE()
  SET(StreamingUSVis_INCLUDE_DIRS ${StreamingUSVis_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS})
ENDIF()

SET(StreamingUSVis_LIBRARIES 
  ${OpenCL_LIBRARIES}
  ${vtkCudaCommon_LIBRARIES}
  ${vtkCudaVisualization_LIBRARIES}
  ${vtkRobartsVisualization_LIBRARIES}
  ${vtkCudaImageAnalytics_LIBRARIES}
  ${vtkCLVolumeReconstruction_LIBRARIES}
  vtksys
  vtkInteractionWidgets
  vtkInteractionImage
  vtkGUISupportQt
  vtkPlusVolumeReconstruction
  vtkPlusCommon
  vtkPlusDataCollection
  opencv_core
  opencv_imgproc
  opencv_videoio)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

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

  ADD_EXECUTABLE(${PROJECT_NAME}
    ${CXX_FILES} 
    ${CPP_FILES} 
    ${UISrcs} 
    ${QT_WRAP}
  )

  TARGET_LINK_LIBRARIES(${PROJECT_NAME} Qt5::Widgets)
else()
  QT4_WRAP_UI(UISrcs ${UI_FILES})

  ADD_EXECUTABLE(${PROJECT_NAME}
    ${CXX_FILES} 
    ${CPP_FILES} 
    ${UISrcs} 
    ${QT_WRAP}
  )
 
  TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${QT_LIBRARIES})
endif()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${StreamingUSVis_LIBRARIES})
