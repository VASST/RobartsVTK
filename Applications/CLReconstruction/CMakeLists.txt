PROJECT( CLReconstruction )

SET ( ${PROJECT_NAME}_SRCS 
  CLReconstruction.cpp
)

# -----------------------------------------------------------------
# Build the executable
add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SRCS})
target_link_libraries(${PROJECT_NAME} PUBLIC
  vtkRenderingVolume${VTK_RENDERING_BACKEND}
  vtkFiltersCore
  vtkImagingCore
  vtkInteractionWidgets
  vtkIOImage
  vtkPlusDataCollection
  vtkCLVolumeReconstruction
  )
target_compile_definitions(${PROJECT_NAME} PUBLIC
  KERNEL_CL_LOCATION="${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl"
  )