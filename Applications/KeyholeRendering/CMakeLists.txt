CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

PROJECT( KeyholeRendering )

vtk_module_load(vtkRobartsVisualization)


SET(KeyholeRendering_INCLUDE_DIRS 
  ${CMAKE_CURRENT_SOURCE_DIR} 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCV_INCLUDE_DIRS} 
  ${VTK_INCLUDE_DIRS}
  ${vtkRobartsVisualization_INCLUDE_DIRS}
  CACHE INTERNAL ""
  )
LIST(REMOVE_DUPLICATES KeyholeRendering_INCLUDE_DIRS)
MARK_AS_ADVANCED(KeyholeRendering_INCLUDE_DIRS)

SET(KeyholeRendering_LIBRARIES 
  ${vtkRobartsVisualization_LIBRARIES}
  ${VTK_LIBRARIES} 
  ${OpenCV_LIBS}
  CACHE INTERNAL ""
  )

MARK_AS_ADVANCED(KeyholeRendering_LIBRARIES)

INCLUDE_DIRECTORIES(${KeyholeRendering_INCLUDE_DIRS})
#-------------------------------------
# Create executable
#-------------------------------------
ADD_EXECUTABLE(${PROJECT_NAME} main.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${KeyholeRendering_LIBRARIES})
