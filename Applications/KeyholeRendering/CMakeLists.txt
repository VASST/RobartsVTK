CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

PROJECT( KeyholeRendering )

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

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${KeyholeRendering_LIBRARIES})
