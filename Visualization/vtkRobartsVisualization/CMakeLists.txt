PROJECT(vtkRobartsVisualization)

include(module.cmake)

SET ( Module_SRCS
  gradientMagPass1.fs 
  gradientMagPass2.fs
  vtkKeyhole.fs
  
  vtkKeyholePass.h vtkKeyholePass.cpp
)

# -----------------------------------------------------------------
# Build the library
vtk_module_library(${PROJECT_NAME} ${Module_SRCS})

# -----------------------------------------------------------------
LIST(APPEND VTK_MODULES_ENABLED ${PROJECT_NAME})
SET(VTK_MODULES_ENABLED ${VTK_MODULES_ENABLED} PARENT_SCOPE)