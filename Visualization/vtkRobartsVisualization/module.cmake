vtk_module(vtkRobartsVisualization
  GROUPS
    Visualization
  DEPENDS
    vtkInteractionStyle
    vtkRenderingFreeType
    vtkRendering${VTK_RENDERING_BACKEND}
  )