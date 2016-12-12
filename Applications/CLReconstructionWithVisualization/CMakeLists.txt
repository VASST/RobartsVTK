PROJECT( CLReconstructionWithVisualization )

SET ( Module_SRCS 
  main.cpp
)

# -----------------------------------------------------------------
# Build the executable
add_executable(${PROJECT_NAME} ${Module_SRCS})
target_link_libraries(${PROJECT_NAME} PUBLIC
  vtkRenderingVolume${VTK_RENDERING_BACKEND}
  vtkRenderingGL2PS${VTK_RENDERING_BACKEND}
  vtkFiltersCore
  vtkImagingCore
  vtkInteractionWidgets
  vtkIOImage
  vtkCommonCore
  vtkCudaCommon
  vtkCudaVisualization
  vtkCLVolumeReconstruction
  vtkPlusDataCollection
  )
