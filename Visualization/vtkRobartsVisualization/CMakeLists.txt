PROJECT(vtkRobartsVisualization)

SET(${PROJECT_NAME}_SRCS
  vtkKeyholePass.cpp
  vtkMultiViewportImageProcessingPass.cxx
  )

IF(MSVC OR ${CMAKE_GENERATOR} MATCHES "Xcode")
  SET(${PROJECT_NAME}_HDRS 
    vtkKeyholePass.h
    vtkMultiViewportImageProcessingPass.h
  )
ENDIF()

SET(${PROJECT_NAME}_INCLUDE_DIRS ${VTK_INCLUDE_DIRS})
SET(${PROJECT_NAME}_LIBRARIES ${OPENGL_LIBRARIES})

# -----------------------------------------------------------------
# Build the library
ADD_LIBRARY(${PROJECT_NAME} ${${PROJECT_NAME}_HDRS} ${${PROJECT_NAME}_SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR} 
  ${${PROJECT_NAME}_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME} PUBLIC 
  ${_${PROJECT_NAME}_LIBRARIES}
  vtkRobartsCommon
  vtkInteractionStyle
  vtkRenderingFreeType
  vtkRenderingGL2PS${VTK_RENDERING_BACKEND}
  vtkRendering${VTK_RENDERING_BACKEND}
  vtkIOImage
  )

target_compile_definitions(${PROJECT_NAME} PUBLIC
  SHADER_DIRECTORY="${CMAKE_SOURCE_DIR}/Visualization/Shaders/"
  )

GENERATE_EXPORT_DIRECTIVE_FILE(${PROJECT_NAME})