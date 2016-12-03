PROJECT(vtkRobartsVisualization)

SET ( ${PROJECT_NAME}_SRCS
  vtkKeyholePass.cpp
)

IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET ( ${PROJECT_NAME}_HDRS 
    vtkKeyholePass.h
  )
ENDIF()

SET(_${PROJECT_NAME}_INCLUDE_DIRS ${VTK_INCLUDE_DIRS})
SET(_${PROJECT_NAME}_LIBRARIES ${OPENGL_LIBRARIES})

IF(FFTW_FOUND AND PNG_FOUND)
  LIST(APPEND ${PROJECT_NAME}_SRCS
    vtkRetinex.cpp
    #vtkDisparityMap.cpp #TODO disable until more time can be spent to fix these up
    #vtkXBLImage.cpp
  )
  IF( MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
    LIST (APPEND ${PROJECT_NAME}_HDRS vtkRetinex.h  )
  ENDIF()

  SET(_${PROJECT_NAME}_INCLUDE_DIRS _${${PROJECT_NAME}_INCLUDE_DIRS}
    ${FFTW_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
  )
  SET(_${PROJECT_NAME}_LIBRARIES _${${PROJECT_NAME}_LIBRARIES} ${PNG_LIBRARIES} ${FFTW_LIBRARIES})
ENDIF()

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${_${PROJECT_NAME}_INCLUDE_DIRS}
  )
target_link_libraries(${PROJECT_NAME} PUBLIC 
  ${_${PROJECT_NAME}_LIBRARIES}
  vtkRobartsCommon
  vtkInteractionStyle
  vtkRenderingFreeType
  vtkRenderingGL2PS${VTK_RENDERING_BACKEND}
  vtkRendering${VTK_RENDERING_BACKEND}
  )
GENERATE_EXPORT_DIRECTIVE_FILE(${PROJECT_NAME})