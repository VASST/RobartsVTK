CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

PROJECT( KeyholeRendering )

FIND_PACKAGE(OpenCV 3.1.0 REQUIRED NO_MODULE)

vtk_module_load(vtkRobartsVisualization)

SET(KeyholeRendering_INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCV_INCLUDE_DIRS} 
  ${VTK_INCLUDE_DIRS}
  ${vtkRobartsVisualization_INCLUDE_DIRS}
  )
LIST(REMOVE_DUPLICATES KeyholeRendering_INCLUDE_DIRS)

SET(KeyholeRendering_LIBRARIES 
  ${vtkRobartsVisualization_LIBRARIES}
  ${VTK_LIBRARIES} 
  ${OpenCV_LIBS}
  ${OPENGL_LIBRARIES}
  )

INCLUDE_DIRECTORIES(${KeyholeRendering_INCLUDE_DIRS})
#-------------------------------------
# Retrieve source files
#-------------------------------------
FILE(GLOB UI_FILES *.ui)
FILE(GLOB QT_WRAP *.h)
FILE(GLOB CXX_FILES *.cxx)
FILE(GLOB CPP_FILES *.cpp)

#-------------------------------------
# Create executable
#-------------------------------------
qt5_wrap_ui(UISrcs ${UI_FILES})

ADD_EXECUTABLE(${PROJECT_NAME}
  ${CXX_FILES} 
  ${CPP_FILES} 
  ${UISrcs} 
  ${QT_WRAP}
)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} Qt5::Widgets ${KeyholeRendering_LIBRARIES})
