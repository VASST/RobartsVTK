PROJECT( Cuda1DTFVolumeRendering )

SET ( Module_SRCS 
  Cuda1DTFVolumeRendering.cpp
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
  vtksys
  vtkCommonCore
  vtkCudaCommon
  vtkCudaVisualization
  )
target_compile_definitions(${PROJECT_NAME} PUBLIC
  DEFAULT_VISUALIZATION_SEQ_FILE="${RobartsVTK_Data_DIR}/Sequences/3DUS.mha"
  )