PROJECT(vtkRobartsVisualization)

include(module.cmake)

SET ( Module_SRCS
  vtkKeyholePass.cpp
)

SET(${PROJECT_NAME}_INCLUDE_DIRS 
  ${VTK_INCLUDE_DIRS}
  )

SET(${PROJECT_NAME}_LIBRARIES ${OPENGL_LIBRARIES})
  
IF(FFTW_FOUND AND PNG_FOUND)
  SET( Module_SRCS ${Module_SRCS}
    vtkRetinex.cpp
  )
  SET(${PROJECT_NAME}_INCLUDE_DIRS ${${PROJECT_NAME}_INCLUDE_DIRS}
    ${fftw_INCLUDE_DIRS}
    ${PNG_INCLUDE_DIRS}
  )
  SET(${PROJECT_NAME}_LIBRARIES ${${PROJECT_NAME}_LIBRARIES} ${PNG_LIBRARIES} ${fftw_LIBRARIES})
ENDIF()

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})
vtk_module_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${${PROJECT_NAME}_LIBRARIES})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)