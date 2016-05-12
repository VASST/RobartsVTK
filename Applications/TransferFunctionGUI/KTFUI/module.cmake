vtk_module(TFUIKohonen
  GROUPS
    Examples
  DEPENDS
    TFUICommon 
    vtkCudaVisualization 
    vtkCudaImageAnalytics 
    vtkCudaCommon 
    vtkRenderingVolume${VTK_RENDERING_BACKEND} 
    vtkRenderingCore 
    vtkFiltersCore 
    vtkImagingCore 
    vtkIOImage 
    vtkCommonCore 
    vtkGUISupportQt
  EXCLUDE_FROM_WRAPPING
  )