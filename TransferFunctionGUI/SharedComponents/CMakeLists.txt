PROJECT(SharedComponents)

SET(KIT SharedComponents)
SET(UKIT SharedComponents)
SET(KIT_LIBS)
SET (KIT_EXTRA_LIBS vtkCudaCommon vtkCudaVisualization)

SET ( Kit_SRCS
  transferFunctionWindowWidgetInterface.h
  transferFunctionWindowWidgetInterface.cxx
  fileManagementWidget.cxx
  fileManagementWidget.h
  stereoRenderingWidget.cxx
  stereoRenderingWidget.h
  virtualToolWidget.cxx
  virtualToolWidget.h
  shadingWidget.cxx
  shadingWidget.h
  deviceManagementWidget.h
  deviceManagementWidget.cxx
  vtkImageHackedPlaneWidget.cxx
  vtkImageHackedPlaneWidget.h
)

SET( mocHeaders  
  fileManagementWidget.h
  stereoRenderingWidget.h
  virtualToolWidget.h
  shadingWidget.h
  deviceManagementWidget.h
)

QT4_WRAP_CPP( Moc_SRCS ${mocHeaders} )

CUDA_ADD_LIBRARY( SharedComponents ${Kit_SRCS} ${Moc_SRCS} )