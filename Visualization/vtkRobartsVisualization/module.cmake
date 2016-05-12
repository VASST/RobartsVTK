vtk_module(vtkRobartsVisualization
  GROUPS
    Visualization
  DEPENDS
    vtkRendering${VTK_RENDERING_BACKEND}
  )