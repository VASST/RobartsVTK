PROJECT(vtkRobartsVisualization)

SET(${PROJECT_NAME}_SRCS
  vtkMultiViewportImageProcessingPass.cxx
  )

IF(NOT ${VTK_RENDERING_BACKEND} STREQUAL "OpenGL")
  LIST(APPEND ${PROJECT_NAME}_SRCS vtkKeyholePass.cpp)
ENDIF()

IF(MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET(${PROJECT_NAME}_HDRS 
    vtkMultiViewportImageProcessingPass.h
  )
  IF(NOT ${VTK_RENDERING_BACKEND} STREQUAL "OpenGL")
    LIST(APPEND ${PROJECT_NAME}_HDRS vtkKeyholePass.h)
  ENDIF()
ENDIF()

SET(${PROJECT_NAME}_INCLUDE_DIRS ${VTK_INCLUDE_DIRS})

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${${PROJECT_NAME}_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME} PUBLIC 
  ${OPENGL_LIBRARIES}
  vtkRobartsCommon
  vtkInteractionStyle
  vtkRenderingFreeType
  vtkRendering${VTK_RENDERING_BACKEND}
  vtkIOImage
  )

IF(NOT ${VTK_RENDERING_BACKEND} STREQUAL "OpenGL")
  target_link_libraries(${PROJECT_NAME} PUBLIC
    vtkRenderingGL2PS${VTK_RENDERING_BACKEND}
    )
ENDIF()

target_compile_definitions(${PROJECT_NAME} PUBLIC
  SHADER_DIRECTORY="${CMAKE_SOURCE_DIR}/Visualization/Shaders/"
  )

GENERATE_EXPORT_DIRECTIVE_FILE(${PROJECT_NAME})