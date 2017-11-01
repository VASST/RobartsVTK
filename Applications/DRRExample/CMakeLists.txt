PROJECT( DRRExample )

SET ( ${PROJECT_NAME}_SRCS
  main.cxx
)

# -----------------------------------------------------------------
# Build the executable
ADD_EXECUTABLE(${PROJECT_NAME} ${${PROJECT_NAME}_SRCS})
target_link_libraries(${PROJECT_NAME}
  vtkCudaCommon
  vtkRobartsCommon
  vtkCudaVisualization
  vtkRenderingGL2PS${VTK_RENDERING_BACKEND}
  vtkRenderingVolume${VTK_RENDERING_BACKEND}
  vtkRenderingCore 
  vtkFiltersCore 
  vtkImagingCore 
  vtkInteractionWidgets 
  vtkIOCore 
  vtkCommonCore 
  )