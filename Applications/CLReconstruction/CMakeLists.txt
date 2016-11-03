PROJECT( CLReconstruction )

SET ( Module_SRCS 
  CLReconstruction.cpp
)

# -----------------------------------------------------------------
# Build the executable
add_executable(${PROJECT_NAME} ${Module_SRCS})
target_link_libraries(${PROJECT_NAME} PUBLIC
  vtkRenderingVolume${VTK_RENDERING_BACKEND}
  vtkFiltersCore
  vtkImagingCore
  vtkInteractionWidgets
  vtkIOImage
  vtkPlusDataCollection
  vtkCLVolumeReconstruction
  vtkCudaCommon
  vtkCudaVisualization
  vtkRobartsVisualization
  vtkCudaImageAnalytics
  )
